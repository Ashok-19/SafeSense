# %% [md]
# # Original SafeSense on NTU RGB+D 120 (skeleton only) for the live demo
#
# Trains the **original SafeSense architecture** (the same network as the UTD-MHAD and CUHK-X notebooks: skeleton graph + flat TCN, IMU TCN, temporal reducers, factorized temporal/modality attention, fused and per-modality heads) on NTU-120 2D skeletons, and exports a checkpoint the live demo loads.
#
# - **What changed from the CUHK-X model:** only the skeleton input. Each of the 17 joints is COCO-17 (from a camera pose estimator) with (x, y, confidence) instead of Human3.6M (x, y, z); features are the same positions + first differences + bone vectors (156 per step instead of 150). The IMU branch keeps its CUHK-X shape and is marked missing, because NTU has no IMU. Parameters: about 496k; the extra ~47k over 449,049 are only the three 120-class output heads (CUHK-X has 40 classes).
# - **Training recipe:** the original four-view corruption-aware training (clean, temporal, structured, style views; per-modality expert loss; consistency distillation), AdamW + OneCycle, SWA over the last epochs. Without IMU, the IMU-only terms (alignment, IMU expert, dropping the whole skeleton stream) have nothing to act on and are off. NTU clips also get the same affine, mirror, jitter and joint-drop augmentation as the SafeSense-Pose run.
# - **Data / split:** PYSKL `ntu120_hrnet.pkl`, NTU-120 cross-subject (test people never seen in training). One person per clip (the most confident), as in the original model; the live demo also uses the main person.
# - **Output:** `/kaggle/working/safesense_orig_ntu120.pt` and `results_safesense_orig_ntu120.json`. Put the `.pt` and `safesense_orig.py` next to the demo's `app.py` and run `python app.py --checkpoint safesense_orig_ntu120.pt`.
#
# Kaggle settings: **GPU T4, Internet ON**. Saves a checkpoint every epoch and resumes after an interruption.
# %% [md]
# ## 1. Modules shared with the demo
# %%
%%writefile safesense_pose.py
"""SafeSense-Pose: skeleton-only action model for 2D COCO-17 keypoints.

Shared by the training notebook and the live demo, so both use exactly the same
preprocessing and network. Input to the model: [B, M, T, V, C] with M=2 bodies,
T=48 frames, V=17 COCO joints, C=5 channels (x, y, score, dx, dy).
"""
import numpy as np, torch, torch.nn as nn

T_STEPS, MAX_BODIES, V = 48, 2, 17
# COCO-17: 0 nose, 1-2 eyes, 3-4 ears, 5-6 shoulders, 7-8 elbows, 9-10 wrists, 11-12 hips, 13-14 knees, 15-16 ankles
COCO_EDGES = [(0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 7), (7, 9), (6, 8), (8, 10),
              (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16), (5, 6), (11, 12)]


def sample_frames(n, t=T_STEPS, rng=None):
    """Indices of t frames spread evenly over n frames; random offset per segment when rng is given."""
    if n <= 0: return np.zeros(t, int)
    edges = np.linspace(0, n, t + 1)
    if rng is None: idx = (edges[:-1] + edges[1:]) / 2
    else: idx = edges[:-1] + rng.random(t) * np.maximum(edges[1:] - edges[:-1], 1e-6)
    return np.clip(np.floor(idx).astype(int), 0, n - 1)


def normalise(kp, score):
    """kp [M,T,V,2] pixels, score [M,T,V] -> [M,T,V,5] centred on the main body's hips and scaled by its torso.

    The same function runs on NTU clips and on live camera windows, so camera distance, framing and
    image size stop mattering. Missing joints (score 0) stay at zero.
    """
    kp = kp.astype(np.float32); score = score.astype(np.float32)
    ok = score > 0.05
    hip = (kp[0, :, 11] + kp[0, :, 12]) / 2; sho = (kp[0, :, 5] + kp[0, :, 6]) / 2
    good = ok[0, :, [5, 6, 11, 12]].all(0)
    if good.any():
        centre = hip[good].mean(0); scale = np.linalg.norm(sho[good] - hip[good], axis=-1).mean()
    else:  # fall back to the bounding box of whatever joints were seen
        pts = kp[ok] if ok.any() else kp.reshape(-1, 2)
        centre = pts.mean(0); scale = max(np.ptp(pts[:, 1]) / 3, 1.0)
    xy = (kp - centre) / max(scale, 1e-3) * ok[..., None]
    vel = np.diff(xy, axis=1, prepend=xy[:, :1]) * (ok & np.roll(ok, 1, axis=1))[..., None]
    return np.concatenate([xy, score[..., None] * ok[..., None], vel], -1)


def pack(kp, score, idx):
    """Pick frames, keep the 2 most confident bodies, pad to MAX_BODIES, normalise -> [M,T,V,5] float32."""
    kp, score = kp[:, idx], score[:, idx]
    order = np.argsort(-score.mean((1, 2)))[:MAX_BODIES]; kp, score = kp[order], score[order]
    if kp.shape[0] < MAX_BODIES:
        pad = MAX_BODIES - kp.shape[0]
        kp = np.concatenate([kp, np.zeros((pad,) + kp.shape[1:], kp.dtype)])
        score = np.concatenate([score, np.zeros((pad,) + score.shape[1:], score.dtype)])
    return normalise(kp, score)


def adjacency():
    A = np.eye(V, dtype=np.float32)
    for i, j in COCO_EDGES: A[i, j] = A[j, i] = 1
    d = A.sum(1) ** -0.5
    return torch.tensor(d[:, None] * A * d[None])


class GraphBlock(nn.Module):
    """Spatial graph convolution over the body (fixed skeleton + learned links) followed by a temporal convolution."""
    def __init__(s, cin, cout, stride=1, drop=0.1):
        super().__init__()
        s.register_buffer("A", adjacency()); s.B = nn.Parameter(torch.zeros(V, V))
        s.gcn = nn.Conv2d(cin, cout, 1); s.self_ = nn.Conv2d(cin, cout, 1); s.bn1 = nn.BatchNorm2d(cout)
        s.tcn = nn.Sequential(nn.Conv2d(cout, cout, (9, 1), (stride, 1), (4, 0)), nn.BatchNorm2d(cout), nn.Dropout(drop))
        s.res = nn.Identity() if (cin == cout and stride == 1) else nn.Sequential(nn.Conv2d(cin, cout, 1, (stride, 1)), nn.BatchNorm2d(cout))
        s.act = nn.ReLU(inplace=True)
    def forward(s, x):  # x [N,C,T,V]
        h = torch.einsum("nctv,vw->nctw", s.gcn(x), s.A + s.B) + s.self_(x)
        return s.act(s.tcn(s.act(s.bn1(h))) + s.res(x))


class SafeSensePose(nn.Module):
    def __init__(s, num_classes, cin=5, widths=(64, 64, 64, 128, 128, 128), drop=0.1):
        super().__init__()
        s.data_bn = nn.BatchNorm1d(cin * V)
        layers, c = [], cin
        for k, w in enumerate(widths):
            layers.append(GraphBlock(c, w, stride=2 if k == 3 else 1, drop=drop)); c = w
        s.blocks = nn.Sequential(*layers); s.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(c, num_classes))
    def forward(s, x):  # x [B,M,T,V,C]
        B, M, Tt, Vv, C = x.shape
        present = (x[..., 2].abs().sum((2, 3)) > 0).float()                       # [B,M] body seen at all
        h = x.permute(0, 1, 4, 3, 2).reshape(B * M, C * Vv, Tt)
        h = s.data_bn(h).reshape(B * M, C, Vv, Tt).permute(0, 1, 3, 2)          # [BM,C,T,V]
        h = s.blocks(h).mean((2, 3)).reshape(B, M, -1)
        w = present / present.sum(1, keepdim=True).clamp_min(1)
        return s.head((h * w[..., None]).sum(1))


NTU120_CLASSES = [
    "drink water", "eat meal/snack", "brushing teeth", "brushing hair", "drop", "pickup", "throw", "sitting down",
    "standing up", "clapping", "reading", "writing", "tear up paper", "wear jacket", "take off jacket", "wear a shoe",
    "take off a shoe", "wear on glasses", "take off glasses", "put on a hat/cap", "take off a hat/cap", "cheer up",
    "hand waving", "kicking something", "reach into pocket", "hopping (one foot jumping)", "jump up",
    "make a phone call/answer phone", "playing with phone/tablet", "typing on a keyboard", "pointing to something with finger",
    "taking a selfie", "check time (from watch)", "rub two hands together", "nod head/bow", "shake head", "wipe face",
    "salute", "put the palms together", "cross hands in front (say stop)", "sneeze/cough", "staggering", "falling",
    "touch head (headache)", "touch chest (stomachache/heart pain)", "touch back (backache)", "touch neck (neckache)",
    "nausea or vomiting condition", "use a fan (with hand or paper)/feeling warm", "punching/slapping other person",
    "kicking other person", "pushing other person", "pat on back of other person", "point finger at the other person",
    "hugging other person", "giving something to other person", "touch other person's pocket", "handshaking",
    "walking towards each other", "walking apart from each other", "put on headphone", "take off headphone",
    "shoot at the basket", "bounce ball", "tennis bat swing", "juggling table tennis balls", "hush (quite)", "flick hair",
    "thumb up", "thumb down", "make ok sign", "make victory sign", "staple book", "counting money", "cutting nails",
    "cutting paper (using scissors)", "snapping fingers", "open bottle", "sniff (smell)", "squat down", "toss a coin",
    "fold paper", "ball up paper", "play magic cube", "apply cream on face", "apply cream on hand back", "put on bag",
    "take off bag", "put something into a bag", "take something out of a bag", "open a box", "move heavy objects",
    "shake fist", "throw up cap/hat", "hands up (both hands)", "cross arms", "arm circles", "arm swings",
    "running on the spot", "butt kicks (kick backward)", "cross toe touch", "side kick", "yawn", "stretch oneself",
    "blow nose", "hit other person with something", "wield knife towards other person",
    "knock over other person (hit with body)", "grab other person's stuff", "shoot at other person with a gun",
    "step on foot", "high-five", "cheers and drink", "carry something with other person", "take a photo of other person",
    "follow other person", "whisper in other person's ear", "exchange things with other person",
    "support somebody with hand", "finger-guessing game (playing rock-paper-scissors)"]
assert len(NTU120_CLASSES) == 120

# %%
%%writefile safesense_orig.py
"""Original SafeSense architecture (skeleton + IMU, factorized temporal/modality attention), on 2D COCO-17 skeletons.

Identical to the CUHK-X / UTD-MHAD SafeSense model; only the skeleton input changes: each of the 17 joints carries
(x, y, confidence) instead of (x, y, z), from the same normalisation the live demo uses (safesense_pose.normalise).
The IMU branch is kept with the CUHK-X shape (5 sensors x 6 channels) and is marked missing when there is no IMU,
so the same weights can later be fine-tuned with skeleton + IMU on CUHK-X.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from safesense_pose import COCO_EDGES, T_STEPS, pack, sample_frames, NTU120_CLASSES

J, EDGES, IMU_DEV, IMU_CH = 17, COCO_EDGES, 5, 6
SK_IN = 3 * J * 2 + 3 * len(EDGES)      # positions + first differences + bone vectors = 156
IM_IN = IMU_DEV * IMU_CH * 2              # 60


def body_window(kp, score, idx):
    """kp [M,n,17,2] pixels, score [M,n,17] -> main person's [T,17,3] (x, y, conf) and frame mask [T]."""
    x = pack(kp, score, idx)[0]                        # most confident body, normalised
    raw = np.ascontiguousarray(x[..., :3]); mask = (x[..., 2] > 0).any(-1)
    return raw.astype(np.float32), mask


def _diff(x, m):
    d = torch.zeros_like(x); d[:, 1:] = x[:, 1:] - x[:, :-1]
    ok = torch.zeros_like(m); ok[:, 1:] = m[:, 1:] & m[:, :-1]
    return d * ok.unsqueeze(-1).to(x.dtype)


def skel_features(s, m):
    """s [B,T,J,3], m [B,T] bool -> [B,T,SK_IN] (same feature recipe as the CUHK-X / UTD notebooks)."""
    B, T = s.shape[:2]; pos = s.reshape(B, T, -1)
    c, p = zip(*EDGES); bones = (s[:, :, list(c)] - s[:, :, list(p)]).reshape(B, T, -1)
    return torch.cat((pos, _diff(pos, m), bones), -1) * m.unsqueeze(-1).to(s.dtype)


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

class SafeSense(nn.Module):
    """The original SafeSense network (unchanged from the CUHK-X notebook's SafeSenseCUHK)."""
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


class SafeSenseCOCO(nn.Module):
    """SafeSense plus its feature normalisation, taking raw windows: raw [B,T,17,3], mask [B,T] (and optional IMU)."""
    def __init__(s, nc):
        super().__init__(); s.net = SafeSense(SK_IN, IM_IN, J, EDGES, nc)
        s.register_buffer("s_mu", torch.zeros(SK_IN)); s.register_buffer("s_sd", torch.ones(SK_IN))
    def forward(s, raw, mask, imu=None, imask=None):
        B, T = raw.shape[:2]
        sk = (skel_features(raw, mask) - s.s_mu) / s.s_sd * mask.unsqueeze(-1).to(raw.dtype)
        if imu is None:
            imu = raw.new_zeros(B, T, IM_IN); imask = torch.zeros(B, T, dtype=torch.bool, device=raw.device)
        return s.net(sk, imu, mask, imask)

# %%
import os, sys, time, json, math, pickle, subprocess, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, os.getcwd())
import importlib, safesense_pose, safesense_orig; importlib.reload(safesense_pose); importlib.reload(safesense_orig)
from safesense_pose import pack, sample_frames, NTU120_CLASSES, T_STEPS
from safesense_orig import SafeSenseCOCO, body_window, skel_features, SK_IN

OUT = Path("/kaggle/working"); OUT.mkdir(exist_ok=True)
URL = "https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu120_hrnet.pkl"
EPOCHS, BATCH, LR, WD = 60, 128, 2e-3, 5e-2
SWA_START = 46            # average weights from this epoch to the end
WORKERS = 4
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print(DEV, torch.__version__)
# %% [md]
# ## 2. Get the data
# %%
hits = list(Path("/kaggle/input").rglob("ntu120_hrnet.pkl")) if Path("/kaggle/input").exists() else []
PKL = hits[0] if hits else Path("/kaggle/temp/ntu120_hrnet.pkl")
if not PKL.exists():
    PKL.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["wget", "-q", "--show-progress", "-O", str(PKL), URL], check=True)
print(PKL, round(PKL.stat().st_size / 1e9, 2), "GB")
# %%
t0 = time.time()
with open(PKL, "rb") as f: data = pickle.load(f)
split, ann = data["split"], data["annotations"]
print("splits:", {k: len(v) for k, v in split.items()}, "| clips:", len(ann), "| %.0fs" % (time.time() - t0))
a0 = ann[0]; print({k: (getattr(v, "shape", v) if k != "frame_dir" else v) for k, v in a0.items()})
# Keep only what training needs, as compact float16 arrays.
clips = {}
for a in ann:
    kp = np.asarray(a["keypoint"], np.float16); sc = np.asarray(a["keypoint_score"], np.float16)
    if kp.ndim == 3: kp, sc = kp[None], sc[None]
    clips[a["frame_dir"]] = (kp, sc, int(a["label"]))
del data, ann
train_ids = [k for k in split["xsub_train"] if k in clips]; val_ids = [k for k in split["xsub_val"] if k in clips]
NC = 1 + max(c[2] for c in clips.values()); assert NC == 120, NC
lens = np.array([clips[k][0].shape[1] for k in train_ids[:5000]])
print("train", len(train_ids), "val", len(val_ids), "| frames per clip (median, p10, p90):", np.median(lens), np.percentile(lens, 10), np.percentile(lens, 90))
# %% [md]
# ## 3. Dataset
#
# Each clip becomes the main person's window: 48 frames, 17 joints, (x, y, confidence), centred on the hips and scaled by the torso, the same function the demo runs on live camera frames.
# %%
FLIP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
class NTU(torch.utils.data.Dataset):
    def __init__(s, ids, train): s.ids, s.train = ids, train
    def __len__(s): return len(s.ids)
    def __getitem__(s, i):
        kp, sc, y = clips[s.ids[i]]; kp = kp.astype(np.float32); sc = sc.astype(np.float32)
        if s.train:
            rng = np.random.default_rng()
            idx = sample_frames(kp.shape[1], T_STEPS, rng)
            ok = sc > 0.05; c = kp[ok].mean(0) if ok.any() else np.zeros(2, np.float32)
            th = np.deg2rad(rng.uniform(-10, 10)); sx, sy = rng.uniform(0.9, 1.1, 2); sh = rng.uniform(-0.1, 0.1)
            Mx = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]) @ np.array([[sx, sh], [0, sy]])
            kp = (kp - c) @ Mx.T.astype(np.float32) + c
            if rng.random() < 0.5: kp = kp[:, :, FLIP]; kp[..., 0] = 2 * c[0] - kp[..., 0]; sc = sc[:, :, FLIP]
            span = np.ptp(kp[ok][:, 1]) if ok.any() else 100.0
            kp = kp + rng.normal(0, 0.01 * span, kp.shape).astype(np.float32)
            sc = sc * (rng.random(sc.shape) > 0.05)
        else:
            idx = sample_frames(kp.shape[1], T_STEPS)
        raw, m = body_window(kp, sc, idx)
        return torch.from_numpy(raw), torch.from_numpy(m), y
dl_tr = torch.utils.data.DataLoader(NTU(train_ids, True), BATCH, shuffle=True, num_workers=WORKERS, drop_last=True, pin_memory=True, persistent_workers=WORKERS > 0)
dl_va = torch.utils.data.DataLoader(NTU(val_ids, False), 256, num_workers=WORKERS, pin_memory=True)
r, m, y = next(iter(dl_tr)); print(r.shape, m.shape, y[:8])

# Feature normalisation statistics from training clips (stored inside the model, so the demo needs nothing extra)
R, Mk = [], []
for k, (r, m, _) in enumerate(torch.utils.data.DataLoader(NTU(train_ids, False), 512, shuffle=True, num_workers=WORKERS)):
    R.append(r); Mk.append(m)
    if k == 15: break
R, Mk = torch.cat(R), torch.cat(Mk)
F_ = skel_features(R, Mk)[Mk]; s_mu, s_sd = F_.mean(0), F_.std(0) + 1e-6
S_STD = float(R[..., :2][Mk].std())   # raw coordinate spread, scales the noise in the structured view
print("feature stats from", len(R), "clips; coordinate std %.3f" % S_STD)
# %% [md]
# ## 4. Original SafeSense four-view training (skeleton only)
#
# Views per batch, as in the original recipe: clean; temporal (shift up to 4 steps, random frame loss up to 50%); structured (Gaussian noise at 0.1/0.3/0.5 of the coordinate spread); style (scale 0.85 to 1.15 and a horizontal squeeze standing in for the +-30 degree yaw rotation, which a 2D skeleton cannot do). Loss weights are the paper's.
# %%
W_CLEAN, W_TEMP, W_STRUCT, W_STYLE = 0.55, 0.15, 0.15, 0.15
W_EXPERT, W_CONS, TEMP = 0.15, 0.10, 2.0

def shift_batch(r, m, k):
    out_r, out_m = torch.zeros_like(r), torch.zeros_like(m)
    for v in k.unique().tolist():
        sel = k == v; rr, mm = r[sel], m[sel]
        if v > 0: out_r[sel, v:], out_m[sel, v:] = rr[:, :-v], mm[:, :-v]
        elif v < 0: out_r[sel, :v], out_m[sel, :v] = rr[:, -v:], mm[:, -v:]
        else: out_r[sel], out_m[sel] = rr, mm
    return out_r, out_m

def make_views(r, m):
    B = r.shape[0]; dev = r.device; views = {"clean": (r, m)}
    tr, tm = shift_batch(r, m, torch.randint(-4, 5, (B,), device=dev))
    tm = tm & (torch.rand(B, r.shape[1], device=dev) >= torch.rand(B, 1, device=dev) * 0.5)
    views["temporal"] = (tr * tm[..., None, None], tm)
    lvl = torch.tensor([0.1, 0.3, 0.5], device=dev)[torch.randint(0, 3, (B,), device=dev)].view(B, 1, 1, 1)
    sr = r.clone(); sr[..., :2] = sr[..., :2] + torch.randn_like(sr[..., :2]) * lvl * S_STD * m[..., None, None]
    views["structured"] = (sr, m)
    yr = r.clone(); sc = torch.empty(B, 1, 1, 1, device=dev).uniform_(0.85, 1.15)
    yr[..., :2] = yr[..., :2] * sc; yr[..., 0:1] = yr[..., 0:1] * torch.cos(torch.deg2rad(torch.empty(B, 1, 1, 1, device=dev).uniform_(-30, 30)))
    views["style"] = (yr, m)
    return views

def ce(z, y): return F.cross_entropy(z, y, label_smoothing=0.1)

model = SafeSenseCOCO(NC).to(DEV); model.s_mu.copy_(s_mu); model.s_sd.copy_(s_sd)
print("parameters:", sum(p.numel() for p in model.parameters()))
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=EPOCHS * len(dl_tr), pct_start=0.1)
scaler = torch.amp.GradScaler(enabled=DEV == "cuda")
start, log, best, swa, n_swa = 1, [], 0.0, None, 0
CK = OUT / "last_orig.pt"
if CK.exists():  # resume after an interrupted session
    ck = torch.load(CK, map_location=DEV, weights_only=False)
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
    start, log, best, swa, n_swa = ck["epoch"] + 1, ck["log"], ck["best"], ck["swa"], ck["n_swa"]; print("resumed at epoch", start)

def evaluate(m_, loader):
    m_.eval(); P, Y = [], []
    with torch.no_grad(), torch.autocast(DEV, enabled=DEV == "cuda"):
        for r, mk, y in loader: P.append(m_(r.to(DEV, non_blocking=True), mk.to(DEV, non_blocking=True))[0].float().cpu()); Y.append(y)
    P, Y = torch.cat(P), torch.cat(Y)
    return P, Y, dict(top1=float((P.argmax(1) == Y).float().mean()), top5=float((P.topk(5, 1).indices == Y[:, None]).any(1).float().mean()))

for ep in range(start, EPOCHS + 1):
    model.train(); t0 = time.time(); tot = n = cor = 0
    for r, mk, y in dl_tr:
        r, mk, y = r.to(DEV, non_blocking=True), mk.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True)
        views = make_views(r, mk)
        with torch.autocast(DEV, enabled=DEV == "cuda"):
            out = {k: model(*v) for k, v in views.items()}
            lc, ec = out["clean"]
            loss = W_CLEAN * ce(lc, y) + W_TEMP * ce(out["temporal"][0], y) + W_STRUCT * ce(out["structured"][0], y) + W_STYLE * ce(out["style"][0], y)
            av = ec["avail"][:, 0]
            loss = loss + W_EXPERT * (F.cross_entropy(ec["sk_logits"].float(), y, reduction="none") * av).sum() / av.sum().clamp_min(1)
            teacher = F.softmax(lc.detach().float() / TEMP, 1)
            loss = loss + W_CONS * sum(F.kl_div(F.log_softmax(out[k][0].float() / TEMP, 1), teacher, reduction="batchmean") for k in ("temporal", "structured", "style")) / 3 * TEMP ** 2
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sched.step()
        tot += loss.item() * len(y); n += len(y); cor += int((lc.argmax(1) == y).sum())
    if ep >= SWA_START:
        sd = {k: v.detach().float().clone() for k, v in model.state_dict().items()}
        if swa is None: swa = sd
        else:
            for k in swa: swa[k] += (sd[k] - swa[k]) / (n_swa + 1)
        n_swa += 1
    row = dict(epoch=ep, loss=round(tot / n, 4), train_acc=round(cor / n, 4), sec=round(time.time() - t0))
    if ep % 5 == 0 or ep == EPOCHS:
        row.update({k: round(v, 4) for k, v in evaluate(model, dl_va)[2].items()})
        if row["top1"] > best:
            best = row["top1"]; torch.save(dict(model=model.state_dict(), epoch=ep, top1=best), OUT / "best_orig.pt")
    log.append(row); print(row)
    torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), epoch=ep, log=log, best=best, swa=swa, n_swa=n_swa), CK)
# %% [md]
# ## 5. Results on unseen people (NTU-120 cross-subject)
#
# The SWA-averaged weights (the original recipe's final model) are compared with the last-epoch weights; the better one on this split is exported.
# %%
P, Y, res_last = evaluate(model, dl_va); print("last epoch:", {k: round(v, 4) for k, v in res_last.items()})
if swa is not None:
    swa_model = SafeSenseCOCO(NC).to(DEV); swa_model.load_state_dict({k: v.to(swa_model.state_dict()[k].dtype) for k, v in swa.items()})
    P2, Y2, res_swa = evaluate(swa_model, dl_va); print("SWA (epochs %d-%d):" % (SWA_START, EPOCHS), {k: round(v, 4) for k, v in res_swa.items()})
    if res_swa["top1"] >= res_last["top1"]: model, P, Y = swa_model, P2, Y2
res = evaluate(model, dl_va)[2]; print("exported model, cross-subject:", {k: round(v, 4) for k, v in res.items()})
acc_c = np.array([float((P[Y == c].argmax(1) == c).float().mean()) if (Y == c).any() else np.nan for c in range(NC)])
DEMO = ["drink water", "sitting down", "standing up", "clapping", "hand waving", "jump up", "make a phone call/answer phone",
        "falling", "touch head (headache)", "touch chest (stomachache/heart pain)", "staggering", "squat down",
        "hands up (both hands)", "arm circles", "running on the spot", "kicking something", "pickup", "cross arms"]
di = [NTU120_CLASSES.index(a) for a in DEMO]
print("\nper-class accuracy on the demo candidates:")
for a, i in sorted(zip(DEMO, di), key=lambda t: -acc_c[t[1]]): print(f"  {a:<40} {acc_c[i]:.3f}")
mm = np.isin(Y.numpy(), di); Pd = P[mm][:, di]; Yd = torch.tensor([di.index(int(v)) for v in Y[mm]])
demo_top1 = float((Pd.argmax(1) == Yd).float().mean())
print("\nrestricted to these %d actions: top1 %.4f on %d clips" % (len(di), demo_top1, int(mm.sum())))
worst = np.argsort(acc_c)[:10]; print("\nhardest classes overall:", [(NTU120_CLASSES[i], round(acc_c[i], 3)) for i in worst])
# %% [md]
# ## 6. Export for the live demo
# %%
torch.save(dict(arch="safesense_orig", model=model.state_dict(), classes=NTU120_CLASSES, num_classes=NC, t_steps=T_STEPS, layout="coco17",
                cross_subject=res, demo_top1=demo_top1, demo_actions=DEMO, log=log), OUT / "safesense_orig_ntu120.pt")
json.dump(dict(cross_subject=res, demo_top1=demo_top1, per_class={NTU120_CLASSES[i]: (None if np.isnan(acc_c[i]) else round(float(acc_c[i]), 4)) for i in range(NC)}, log=log),
          open(OUT / "results_safesense_orig_ntu120.json", "w"), indent=1)
print(sorted(p.name for p in OUT.iterdir()))
ck = torch.load(OUT / "safesense_orig_ntu120.pt", map_location="cpu", weights_only=False)
m2 = SafeSenseCOCO(ck["num_classes"]); m2.load_state_dict(ck["model"]); m2.eval()
raw, mk = body_window(np.random.rand(1, 75, 17, 2).astype(np.float32) * 400, np.ones((1, 75, 17), np.float32), sample_frames(75))
print("demo-style input", raw.shape, "->", ck["classes"][int(m2(torch.from_numpy(raw)[None], torch.from_numpy(mk)[None])[0].argmax())])
