#
# # SafeSense on CUHK-X (Skeleton + IMU)
#
# Trains SafeSense on the prepared CUHK-X arrays (`cuhkx_prepared.npz` from the preprocessing notebook) and evaluates it on held-out users 8, 9, 23, 24 with clean metrics and the 25-scenario corruption suite.
#
# Same architecture and recipe as the UTD-MHAD paper model, adapted only at the inputs:
# - Skeleton: 17 joints (Human3.6M layout) -> positions + first differences + bone vectors = 150 features per step; 17-joint body graph.
# - IMU: 5 sensors x (3 acc + 3 gyro) = 30 channels + first differences = 60 features per step.
# - 64 time steps (10 Hz clips), so the temporal reducer goes 64 -> 32 -> 16 tokens.
#
# Settings: GPU T4, no internet. Set `PREP` to your input path. `MODE = "dev"` holds out 2 training users for validation; `MODE = "final"` trains on all 14 training users and reports the test users.
#
import os, json, math, time, copy, random
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

PREP = Path("/home/nnmax/Desktop/final-year-project/cuhk-training/v2/cuhkx_prepared_v2.npz")  # <- set to the preprocessing output
if not PREP.exists():
    hits = []; PREP = hits[0] if hits else PREP
import os; OUT = Path(os.environ["OUT"]); OUT.mkdir(exist_ok=True)
MODE = "final"            # "dev" or "final"
PROTOCOL = os.environ["PROTOCOL"]  # "cross_subject": test people never seen in training (users 8, 9, 23, 24)
                            # "same_subject": every person in training, 20% of each person's clips per class held out
                            #                 (the protocol the CUHK-X paper reports; MODE is ignored)
VAL_USERS = {"user7", "user22"}
SEEDS = [int(x) for x in os.environ.get("SEEDS", "22").split(",")]              # add more seeds, e.g. [22, 23, 24], if time allows
EPOCHS, BATCH, LR, WD = 60, 32, 1e-3, 5e-2
SWA_START = 45            # average weights from this epoch to the end (inclusive)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print(PREP, DEV)
#
# ## 1. Load data and build features
#
D = np.load(PREP, allow_pickle=True)
skel_raw = D["skel"].astype(np.float32)        # [N,T,17,3]
smask = D["skel_mask"].astype(bool)            # [N,T]
imu_raw = D["imu"].astype(np.float32)          # [N,T,5,6]
imask = D["imu_mask"].astype(bool)             # [N,T]
idev = D["imu_dev"].astype(bool)               # [N,5]
label = D["label"].astype(np.int64); user = D["user"].astype(str); split = D["split"].astype(str)
ACTIONS = [str(a) for a in D["actions"]]; NC = len(ACTIONS)
N, T, J, _ = skel_raw.shape
print(skel_raw.shape, imu_raw.shape, "classes", NC)

# Human3.6M 17-joint layout: 0 pelvis, 1-3 right leg, 4-6 left leg, 7 spine, 8 thorax, 9 neck, 10 head, 11-13 left arm, 14-16 right arm
EDGES = [(1,0),(2,1),(3,2),(4,0),(5,4),(6,5),(7,0),(8,7),(9,8),(10,9),(11,8),(12,11),(13,12),(14,8),(15,14),(16,15)]
v = smask
bl = lambda a, b: np.linalg.norm(skel_raw[:, :, a] - skel_raw[:, :, b], axis=-1)[v].mean()
print("bone length check (left vs right should match): thigh %.3f/%.3f shin %.3f/%.3f upper arm %.3f/%.3f forearm %.3f/%.3f" %
      (bl(2,1), bl(5,4), bl(3,2), bl(6,5), bl(12,11), bl(15,14), bl(13,12), bl(16,15)))
print("vertical axis guess (mean head - pelvis):", (skel_raw[:, :, 10] - skel_raw[:, :, 0])[v].mean(0).round(3))

# Yaw normalisation: rotate each clip about the vertical axis so its mean hip line (right hip -> left hip) points along +x.
# Removes camera/viewpoint differences between the two homes; the style view still adds +-30 deg back as augmentation.
NORMALISE_YAW = True
if NORMALISE_YAW:
    hips = (skel_raw[:, :, 4] - skel_raw[:, :, 1]) * smask[..., None]
    h = hips.sum(1) / np.maximum(smask.sum(1, keepdims=True), 1)
    yaw = -np.arctan2(h[:, 1], h[:, 0]); c, sn = np.cos(yaw)[:, None, None], np.sin(yaw)[:, None, None]
    x, y = skel_raw[..., 0].copy(), skel_raw[..., 1].copy()
    skel_raw[..., 0] = c * x - sn * y; skel_raw[..., 1] = sn * x + c * y
    print("yaw normalised; hip-line angle spread before (deg): %.1f" % np.degrees(np.std(yaw)))
# zero devices that are absent
imu_raw = imu_raw * idev[:, None, :, None]
train_idx = np.where(split == "train")[0]; test_idx = np.where(split == "test")[0]
if PROTOCOL == "same_subject":
    rs = np.random.RandomState(0); held = np.zeros(N, bool)
    for key in np.unique(np.char.add(np.char.add(user, "_"), label.astype(str))):
        w = np.where(np.char.add(np.char.add(user, "_"), label.astype(str)) == key)[0]
        held[rs.choice(w, max(1, len(w) // 5), replace=False)] = True
    train_idx, test_idx = np.where(~held)[0], np.where(held)[0]; MODE = "final"
if MODE == "dev":
    fit_idx = np.array([i for i in train_idx if user[i] not in VAL_USERS]); eval_idx = np.array([i for i in train_idx if user[i] in VAL_USERS])
else:
    fit_idx, eval_idx = train_idx, test_idx
print(PROTOCOL, "| fit", len(fit_idx), "eval", len(eval_idx), "| eval users", sorted(set(user[eval_idx])))

def diff(x, m):
    d = np.zeros_like(x); d[:, 1:] = x[:, 1:] - x[:, :-1]
    ok = np.zeros_like(m); ok[:, 1:] = m[:, 1:] & m[:, :-1]
    return d * ok.reshape(ok.shape + (1,) * (x.ndim - 2))

def skel_features(s, m):
    pos = s.reshape(*s.shape[:2], -1)
    bones = np.stack([s[:, :, c] - s[:, :, p] for c, p in EDGES], 2).reshape(*s.shape[:2], -1)
    return np.concatenate([pos, diff(pos, m), bones], -1) * m[..., None]

def imu_features(x, m):
    flat = x.reshape(*x.shape[:2], -1)
    return np.concatenate([flat, diff(flat, m)], -1) * m[..., None]

# per-channel normalisation statistics from the fitting users only
def stats(feat, m, idx):
    vals = feat[idx][m[idx]]; mu = vals.mean(0); sd = vals.std(0) + 1e-6
    return mu.astype(np.float32), sd.astype(np.float32)
S_FEAT = skel_features(skel_raw, smask); I_FEAT = imu_features(imu_raw, imask)
s_mu, s_sd = stats(S_FEAT, smask, fit_idx); i_mu, i_sd = stats(I_FEAT, imask, fit_idx)
SK_IN, IM_IN = S_FEAT.shape[-1], I_FEAT.shape[-1]
print("skeleton features", SK_IN, "imu features", IM_IN)
np.savez(OUT / "norm_stats.npz", s_mu=s_mu, s_sd=s_sd, i_mu=i_mu, i_sd=i_sd)
# raw noise scale per stream (for corruption views/suite), from fitting users
S_STD = skel_raw[fit_idx][smask[fit_idx]].std(0).mean(); I_STD = imu_raw[fit_idx][imask[fit_idx]].reshape(-1, 30).std(0)
#
# ## 2. SafeSense model (ported from `src/safesense/model.py`, inputs parameterised)
#
def pool(seq, mask):
    valid = mask.unsqueeze(1).bool(); cnt = valid.sum(-1).clamp_min(1)
    mean = seq.masked_fill(~valid, 0).sum(-1) / cnt
    mx = seq.masked_fill(~valid, -torch.inf).amax(-1); mx = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))
    return torch.cat((mean, mx), 1)

def groups_for(c):
    g = max(1, min(16, c))
    while c % g: g -= 1
    return g

class TCNBlock(nn.Module):
    def __init__(s, c, d, p):
        super().__init__(); s.dw = nn.Conv1d(c, c, 5, padding=2 * d, dilation=d, groups=c); s.pw = nn.Conv1d(c, c, 1)
        s.n = nn.GroupNorm(groups_for(c), c); s.dp = nn.Dropout(p)
    def forward(s, x): return x + s.dp(F.gelu(s.n(s.pw(s.dw(x)))))

class TCN(nn.Module):
    def __init__(s, i, w, p=0.15, dil=(1, 2, 4)):
        super().__init__(); s.inp = nn.Conv1d(i, w, 1); s.b = nn.Sequential(*[TCNBlock(w, d, p) for d in dil])
    def forward(s, x): return s.b(s.inp(x))

class GraphBlock(nn.Module):
    def __init__(s, c, p=0.1):
        super().__init__(); s.a = nn.Linear(c, c); s.b = nn.Linear(c, c, bias=False); s.n = nn.LayerNorm(c); s.dp = nn.Dropout(p)
    def forward(s, x, A): return x + s.dp(F.gelu(s.n(s.a(x) + s.b(torch.einsum("ij,btjc->btic", A, x)))))

class SkeletonGraph(nn.Module):
    def __init__(s, J, edges):
        super().__init__(); A = torch.eye(J)
        for c, p in edges: A[c, p] = A[p, c] = 1
        d = A.sum(1).rsqrt(); s.register_buffer("A", d[:, None] * A * d[None]); s.J = J
        s.inp = nn.Sequential(nn.Linear(6, 96), nn.LayerNorm(96), nn.GELU())
        s.blocks = nn.ModuleList([GraphBlock(96) for _ in range(3)]); s.att = nn.Linear(96, 1); s.t = TCN(96, 96)
    def forward(s, x, m):
        B, T_, _ = x.shape; J = s.J
        pos = x[:, :, :3 * J].reshape(B, T_, J, 3); mot = x[:, :, 3 * J:6 * J].reshape(B, T_, J, 3)
        n = s.inp(torch.cat((pos, mot), -1))
        for b in s.blocks: n = b(n, s.A)
        w = torch.softmax(s.att(n).squeeze(-1), 2)
        return s.t(((n * w.unsqueeze(-1)).sum(2) * m.unsqueeze(-1)).transpose(1, 2))

class Reducer(nn.Module):
    def __init__(s, c):
        super().__init__(); s.steps = nn.ModuleList([nn.Sequential(nn.Conv1d(c, c, 5, stride=2, padding=2, groups=c), nn.Conv1d(c, c, 1), nn.GroupNorm(groups_for(c), c), nn.GELU()) for _ in range(2)])
    def forward(s, x, m):
        v = m.float()
        for st in s.steps:
            x = st(x); v = (F.max_pool1d(v.unsqueeze(1), 3, stride=2, padding=1).squeeze(1) > 0).float(); x = x * v.unsqueeze(1)
        return x, v

class FactorizedBlock(nn.Module):
    def __init__(s, w=96, h=4, p=0.1):
        super().__init__()
        s.tn = nn.LayerNorm(w); s.ta = nn.MultiheadAttention(w, h, dropout=p, batch_first=True)
        s.mn = nn.LayerNorm(w); s.ma = nn.MultiheadAttention(w, h, dropout=p, batch_first=True)
        s.fn = nn.LayerNorm(w); s.ff = nn.Sequential(nn.Linear(w, 2 * w), nn.GELU(), nn.Dropout(p), nn.Linear(2 * w, w), nn.Dropout(p)); s.dp = nn.Dropout(p)
    @staticmethod
    def safe(v):
        v = v.clone(); miss = ~v.any(1); v[miss, 0] = True; return v
    def forward(s, tok, valid):
        B, M, S, W = tok.shape
        t = tok.reshape(B * M, S, W); tv = valid.reshape(B * M, S); n = s.tn(t)
        u, _ = s.ta(n, n, n, key_padding_mask=~s.safe(tv), need_weights=False); t = (t + s.dp(u)) * tv.unsqueeze(-1)
        tok = t.reshape(B, M, S, W)
        m = tok.permute(0, 2, 1, 3).reshape(B * S, M, W); mv = valid.permute(0, 2, 1).reshape(B * S, M); n = s.mn(m)
        u, _ = s.ma(n, n, n, key_padding_mask=~s.safe(mv), need_weights=False); m = (m + s.dp(u)) * mv.unsqueeze(-1)
        tok = m.reshape(B, S, M, W).permute(0, 2, 1, 3)
        return (tok + s.ff(s.fn(tok))) * valid.unsqueeze(-1)

class SafeSenseCUHK(nn.Module):
    def __init__(s, sk_in, im_in, J, edges, nc, width=96, blocks=2):
        super().__init__()
        s.sk_flat = TCN(sk_in, 64); s.sk_graph = SkeletonGraph(J, edges); s.sk_fuse = nn.Conv1d(160, width, 1)
        s.imu = TCN(im_in, 64); s.imu_proj = nn.Conv1d(64, width, 1)
        s.sk_red = Reducer(width); s.im_red = Reducer(width)
        s.blocks = nn.ModuleList([FactorizedBlock(width) for _ in range(blocks)])
        s.cls = nn.Sequential(nn.LayerNorm(2 * width), nn.Linear(2 * width, nc))
        s.sk_exp = nn.Sequential(nn.LayerNorm(2 * width), nn.Linear(2 * width, nc)); s.im_exp = nn.Sequential(nn.LayerNorm(2 * width), nn.Linear(2 * width, nc))
    def forward(s, sk, im, sm, imk):
        avail = torch.stack((sm.any(1), imk.any(1)), 1).float()
        sk = sk * sm.unsqueeze(-1); im = im * imk.unsqueeze(-1)
        a = s.sk_fuse(torch.cat((s.sk_flat(sk.transpose(1, 2)), s.sk_graph(sk, sm.float())), 1))
        b = s.imu_proj(s.imu(im.transpose(1, 2)))
        a, sv = s.sk_red(a, sm); b, iv = s.im_red(b, imk)
        sv = sv * avail[:, :1]; iv = iv * avail[:, 1:]
        tok = torch.stack((a.transpose(1, 2), b.transpose(1, 2)), 1); valid = torch.stack((sv.bool(), iv.bool()), 1)
        for bl in s.blocks: tok = bl(tok, valid)
        se, ie = pool(tok[:, 0].transpose(1, 2), sv), pool(tok[:, 1].transpose(1, 2), iv)
        cnt = valid.float().sum(1).clamp_min(1).unsqueeze(-1)
        fused = (tok * valid.unsqueeze(-1)).sum(1) / cnt
        logits = s.cls(pool(fused.transpose(1, 2), valid.any(1).float()))
        extras = dict(sk_logits=s.sk_exp(se) * avail[:, :1], im_logits=s.im_exp(ie) * avail[:, 1:],
                      sk_emb=se[:, :se.shape[1] // 2], im_emb=ie[:, :ie.shape[1] // 2], avail=avail)
        return logits, extras

m0 = SafeSenseCUHK(SK_IN, IM_IN, J, EDGES, NC)
print("parameters:", sum(p.numel() for p in m0.parameters() if p.requires_grad))
#
# ## 3. Corruptions (shared by training views and the evaluation suite)
#
# All corruptions act on the raw skeleton / IMU arrays and masks; features are rebuilt afterwards, exactly as in the UTD-MHAD pipeline.
#
def shift(x, m, k):
    if k == 0: return x, m
    x2 = np.zeros_like(x); m2 = np.zeros_like(m)
    if k > 0: x2[:, k:] = x[:, :-k]; m2[:, k:] = m[:, :-k]
    else: x2[:, :k] = x[:, -k:]; m2[:, :k] = m[:, -k:]
    return x2, m2

def frame_loss(m, rate, rng):
    keep = rng.random(m.shape) >= rate; return m & keep

def corrupt(s, sm, i, im, kind, level, target, rng):
    s, sm, i, im = s.copy(), sm.copy(), i.copy(), im.copy()
    def on(t): return target in (t, "both")
    if kind == "shift":
        if on("skel"): s, sm = shift(s, sm, level)
        if on("imu"): i, im = shift(i, im, level)
    elif kind == "loss":
        if on("skel"): sm = frame_loss(sm, level, rng)
        if on("imu"): im = frame_loss(im, level, rng)
    elif kind == "noise":
        if on("skel"): s = s + rng.normal(0, level * S_STD, s.shape).astype(np.float32) * sm[..., None, None]
        if on("imu"): i = i + (rng.normal(0, 1, i.shape) * level * I_STD.reshape(5, 6)).astype(np.float32) * im[..., None, None]
    elif kind == "missing":
        if on("skel"): sm[:] = False
        if on("imu"): im[:] = False
    elif kind == "composite":
        i, im = shift(i, im, 4); im = frame_loss(im, 0.3, rng)
        i = i + (rng.normal(0, 1, i.shape) * 0.3 * I_STD.reshape(5, 6)).astype(np.float32) * im[..., None, None]
    return s, sm, i, im

SUITE = [("clean", "clean", 0, "both")]
for tgt in ("skel", "imu"):
    for k in (-4, -2, 2, 4): SUITE.append(("shift", "shift", k, tgt))
    for r in (0.1, 0.3, 0.5): SUITE.append(("frame_loss", "loss", r, tgt))
    for n in (0.1, 0.3, 0.5): SUITE.append(("noise", "noise", n, tgt))
for tgt in ("skel", "imu", "both"): SUITE.append(("missing", "missing", 0, tgt))
SUITE.append(("composite", "composite", 0, "imu"))
print(len(SUITE), "scenarios")

def to_tensors(s, sm, i, im):
    sf = (skel_features(s, sm) - s_mu) / s_sd * sm[..., None]
    imf = (imu_features(i, im) - i_mu) / i_sd * im[..., None]
    f = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(DEV)
    return f(sf.astype(np.float32)), f(imf.astype(np.float32)), f(sm), f(im)
#
# ## 4. Four-view corruption-aware training (same weights as the paper)
#
W_CLEAN, W_TEMP, W_STRUCT, W_STYLE = 0.55, 0.15, 0.15, 0.15
W_EXPERT, W_ALIGN, W_CONS, TEMP = 0.15, 0.15, 0.10, 2.0

def rot_z(s, deg):
    a = np.deg2rad(deg)[:, None, None, None]; c, sn = np.cos(a), np.sin(a)
    x, y, z = s[..., 0:1], s[..., 1:2], s[..., 2:3]
    return np.concatenate([c * x - sn * y, sn * x + c * y, z], -1)

def make_views(idx, rng):
    s, sm, i, im = skel_raw[idx], smask[idx], imu_raw[idx], imask[idx]
    views = {"clean": (s, sm, i, im)}
    # temporal view: random per-modality shift up to 4 steps + random frame loss
    ts, tsm = shift(s, sm, int(rng.integers(-4, 5))); ti, tim = shift(i, im, int(rng.integers(-4, 5)))
    views["temporal"] = (ts, frame_loss(tsm, rng.uniform(0, 0.5), rng), ti, frame_loss(tim, rng.uniform(0, 0.5), rng))
    # structured view: noise on one or both streams, sometimes a whole stream dropped
    ss, ssm, si, sim = corrupt(s, sm, i, im, "noise", rng.choice([0.1, 0.3, 0.5]), rng.choice(["skel", "imu", "both"]), rng)
    r = rng.random(len(idx)); ssm = ssm & (r >= 0.15)[:, None]; sim = sim & ((r < 0.15) | (r >= 0.30))[:, None]
    views["structured"] = (ss, ssm, si, sim)
    # style view: subject/sensor style changes that keep the label
    scale = rng.uniform(0.85, 1.15, (len(idx), 1, 1, 1)).astype(np.float32)
    ys = (rot_z(s, rng.uniform(-30, 30, len(idx))) * scale).astype(np.float32)
    gain = rng.uniform(0.9, 1.1, (len(idx), 1, 5, 6)).astype(np.float32); bias = (rng.normal(0, 0.05, (len(idx), 1, 5, 6)) * I_STD.reshape(1, 1, 5, 6)).astype(np.float32)
    views["style"] = (ys, sm, i * gain + bias * im[..., None, None], im)
    return views

counts = np.bincount(label[fit_idx], minlength=NC).astype(np.float32)
cw = torch.tensor((counts.max() / np.maximum(counts, 1)) ** 0.5, device=DEV); cw = cw / cw.mean()
def ce(logits, y): return F.cross_entropy(logits, y, weight=cw, label_smoothing=0.1)

def train_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed); rng = np.random.default_rng(seed)
    model = SafeSenseCUHK(SK_IN, IM_IN, J, EDGES, NC).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    steps = EPOCHS * math.ceil(len(fit_idx) / BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.1)
    swa, n_swa = None, 0; log = []
    for ep in range(1, EPOCHS + 1):
        model.train(); perm = rng.permutation(fit_idx); tot = 0.0; t0 = time.time()
        for b in range(0, len(perm), BATCH):
            idx = perm[b:b + BATCH]; y = torch.from_numpy(label[idx]).to(DEV)
            views = make_views(idx, rng); out = {}
            for name, v in views.items(): out[name] = model(*to_tensors(*v))
            lc, ec = out["clean"]
            loss = W_CLEAN * ce(lc, y) + W_TEMP * ce(out["temporal"][0], y) + W_STRUCT * ce(out["structured"][0], y) + W_STYLE * ce(out["style"][0], y)
            av = ec["avail"]
            exp_loss = (F.cross_entropy(ec["sk_logits"], y, reduction="none") * av[:, 0]).sum() / av[:, 0].sum().clamp_min(1) + \
                       (F.cross_entropy(ec["im_logits"], y, reduction="none") * av[:, 1]).sum() / av[:, 1].sum().clamp_min(1)
            both = av[:, 0] * av[:, 1]
            align = ((1 - F.cosine_similarity(ec["sk_emb"], ec["im_emb"], dim=1)) * both).sum() / both.sum().clamp_min(1)
            teacher = F.softmax(lc.detach() / TEMP, 1)
            cons = sum(F.kl_div(F.log_softmax(out[k][0] / TEMP, 1), teacher, reduction="batchmean") for k in ("temporal", "structured", "style")) / 3 * TEMP ** 2
            loss = loss + W_EXPERT * exp_loss + W_ALIGN * align + W_CONS * cons
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            tot += loss.item() * len(idx)
        if ep >= SWA_START:
            sd = {k: v.detach().float().clone() for k, v in model.state_dict().items()}
            if swa is None: swa = sd
            else:
                for k in swa: swa[k] += (sd[k] - swa[k]) / (n_swa + 1)
            n_swa += 1
        row = dict(epoch=ep, loss=tot / len(perm), sec=round(time.time() - t0, 1))
        if ep % 5 == 0 or ep == EPOCHS:
            tr = evaluate(model, fit_idx); row.update(train_acc=tr["acc"], **evaluate(model, eval_idx)); print(row)
        log.append(row)
        torch.save(dict(model=model.state_dict(), epoch=ep), OUT / f"last_seed{seed}.pt")  # survives a session timeout
    final = SafeSenseCUHK(SK_IN, IM_IN, J, EDGES, NC).to(DEV); final.load_state_dict({k: v.to(final.state_dict()[k].dtype) for k, v in swa.items()})
    torch.save(dict(model=final.state_dict(), J=J, edges=EDGES, actions=ACTIONS, sk_in=SK_IN, im_in=IM_IN, T=T, seed=seed, mode=MODE), OUT / f"safesense_cuhkx_seed{seed}_swa.pt")
    json.dump(log, open(OUT / f"train_log_seed{seed}.json", "w"), indent=1)
    return final
#
# ## 5. Evaluation helpers
#
@torch.no_grad()
def predict(model, s, sm, i, im, bs=256):
    model.eval(); outs = []
    for b in range(0, len(s), bs):
        outs.append(model(*to_tensors(s[b:b + bs], sm[b:b + bs], i[b:b + bs], im[b:b + bs]))[0].softmax(1).cpu())
    return torch.cat(outs).numpy()

def metrics(y, p):
    present = np.unique(y)
    return dict(acc=round(float(accuracy_score(y, p)), 4), macro_f1=round(float(f1_score(y, p, labels=present, average="macro")), 4))

def evaluate(model, idx):
    return metrics(label[idx], predict(model, skel_raw[idx], smask[idx], imu_raw[idx], imask[idx]).argmax(1))

def robustness(model, idx, seed=20260814):
    rows = []
    for cat, kind, lvl, tgt in SUITE:
        rng = np.random.default_rng(seed)
        if kind == "clean": s, sm, i, im = skel_raw[idx], smask[idx], imu_raw[idx], imask[idx]
        else: s, sm, i, im = corrupt(skel_raw[idx], smask[idx], imu_raw[idx], imask[idx], kind, lvl, tgt, rng)
        r = metrics(label[idx], predict(model, s, sm, i, im).argmax(1)); rows.append(dict(category=cat, kind=kind, level=lvl, target=tgt, **r))
    import pandas as pd
    df = pd.DataFrame(rows); cat = df[df.category != "clean"].groupby("category")[["acc", "macro_f1"]].mean()
    return df, cat, dict(equal_category_acc=round(float(cat.acc.mean()), 4), equal_category_f1=round(float(cat.macro_f1.mean()), 4))
#
# ## 6. Train and evaluate
#
results = {}
for seed in SEEDS:
    print(f"===== seed {seed} ({MODE}) =====")
    model = train_seed(seed)
    clean = evaluate(model, eval_idx)
    suite, cat, agg = robustness(model, eval_idx)
    print("clean:", clean); print(cat.round(4)); print("aggregate:", agg)
    suite.to_csv(OUT / f"robustness_seed{seed}.csv", index=False)
    p = predict(model, skel_raw[eval_idx], smask[eval_idx], imu_raw[eval_idx], imask[eval_idx]).argmax(1)
    np.save(OUT / f"confusion_seed{seed}.npy", confusion_matrix(label[eval_idx], p, labels=range(NC)))
    results[seed] = dict(clean=clean, categories=cat.round(4).to_dict(), aggregate=agg)
json.dump(dict(protocol=PROTOCOL, mode=MODE, eval_users=sorted(set(user[eval_idx])), n_eval=int(len(eval_idx)), params=int(sum(p.numel() for p in m0.parameters())), results=results),
          open(OUT / f"results_{PROTOCOL}_{MODE}.json", "w"), indent=1)
print(json.dumps(results, indent=1))
#
# ## 7. Single-modality check (for the comparison with the paper's skeleton-only 79.1% and IMU-only 45.5%)
#
# Same trained model, one stream removed at test time. These are the "missing" scenarios above, shown side by side with the clean fused result.
#
for seed in SEEDS:
    df = __import__("pandas").read_csv(OUT / f"robustness_seed{seed}.csv")
    print(seed, df[df.category.isin(["clean", "missing"])][["category", "target", "acc", "macro_f1"]].to_string(index=False))

#
# ## 8. Diagnostics: where the errors are
#
# Per-class accuracy on the evaluation users and the most frequent confusions. Fine-grained hand/object actions (brush teeth, comb hair, type, write, read, use phone) are expected to be the hard ones for a 17-joint skeleton without hand joints.
#
import pandas as pd
cm = np.load(OUT / f"confusion_seed{SEEDS[0]}.npy")
support = cm.sum(1); acc_c = np.where(support > 0, np.diag(cm) / np.maximum(support, 1), np.nan)
pc = pd.DataFrame(dict(action=ACTIONS, support=support, acc=acc_c.round(3))).sort_values("acc")
print(pc.to_string(index=False))
off = cm.copy(); np.fill_diagonal(off, 0)
top = np.dstack(np.unravel_index(np.argsort(off.ravel())[::-1][:15], off.shape))[0]
print("\nmost frequent confusions (true -> predicted, count):")
for a, b in top: print(f"  {ACTIONS[a]:<28} -> {ACTIONS[b]:<28} {off[a, b]}")
#
# ## 9. Reference baselines on the same split (sanity check on the data, not the model)
#
# A plain 2-layer bidirectional GRU per modality, no corruption training. If these land close to SafeSense, the ceiling comes from the data (short sparse clips, fine-grained actions, 14 training users), not from the SafeSense architecture.
#
class GRUBase(nn.Module):
    def __init__(s, d, nc):
        super().__init__(); s.g = nn.GRU(d, 128, 2, batch_first=True, bidirectional=True, dropout=0.3); s.h = nn.Linear(512, nc)
    def forward(s, x, m):
        o, _ = s.g(x); mf = m.float().unsqueeze(-1); mean = (o * mf).sum(1) / mf.sum(1).clamp_min(1)
        mx = o.masked_fill(~m.unsqueeze(-1), -1e4).amax(1); return s.h(torch.cat((mean, mx), 1))

def baseline(which, epochs=40):
    torch.manual_seed(0); rng = np.random.default_rng(0)
    d = SK_IN if which == "skel" else IM_IN; net = GRUBase(d, NC).to(DEV); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=5e-2)
    def batch(idx):
        sf, imf, sm_, im_ = to_tensors(skel_raw[idx], smask[idx], imu_raw[idx], imask[idx])
        return (sf, sm_) if which == "skel" else (imf, im_)
    for ep in range(epochs):
        net.train()
        for b in range(0, len(fit_idx), BATCH):
            idx = fit_idx[rng.permutation(len(fit_idx))[:BATCH]]
            x, m = batch(idx); loss = ce(net(x, m), torch.from_numpy(label[idx]).to(DEV))
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval(); P = []
    with torch.no_grad():
        for b in range(0, len(eval_idx), 256):
            x, m = batch(eval_idx[b:b + 256]); P.append(net(x, m).argmax(1).cpu())
    return metrics(label[eval_idx], torch.cat(P).numpy())
print("GRU skeleton-only:", baseline("skel")); print("GRU IMU-only:", baseline("imu"))
