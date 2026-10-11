# %% [md]
# # SafeSense-Pose (449k, original SafeSense budget): train on NTU RGB+D 120 (2D skeletons)
#
# Trains a skeleton-only SafeSense model on NTU-120 2D keypoints and exports a checkpoint the live demo loads directly.
#
# - **Data:** PYSKL's `ntu120_hrnet.pkl`, 2D COCO-17 keypoints extracted with HRNet, the same format a phone or webcam pose estimator (RTMPose, or MediaPipe remapped to COCO-17) produces. 106 people, 120 actions. Downloaded directly from OpenMMLab; NTU's dataset terms apply, so do not re-share the file publicly.
# - **Split:** NTU-120 cross-subject (`xsub_train` / `xsub_val`): test people are never seen in training.
# - **Model:** graph convolutions over the 17-joint body plus temporal convolutions, 2 bodies per clip, 48 frames, inputs x, y, confidence and motion. This notebook trains the 449,448-parameter version, `widths=(64, 64, 64, 96, 96, 96)`, the same budget as the original SafeSense (449,049). The earlier 682k run used `(64, 64, 64, 128, 128, 128)`; everything else (data, split, augmentation, schedule) is identical so the two can be compared directly.
# - **Output:** `/kaggle/working/safesense_pose_ntu120_449k.pt` (weights, class names, settings) and `safesense_pose.py` (model and preprocessing). The demo imports both, so training and live inference share the same code.
#
# Kaggle settings: **GPU T4, Internet ON** (needed for the download). The notebook saves a checkpoint every epoch and resumes from it if a run is interrupted.
# %% [md]
# ## 1. Model and preprocessing module (shared with the demo)
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
import os, sys, time, json, pickle, subprocess, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, os.getcwd())
import importlib, safesense_pose; importlib.reload(safesense_pose)
from safesense_pose import SafeSensePose, pack, sample_frames, NTU120_CLASSES, T_STEPS

OUT = Path("/kaggle/working"); OUT.mkdir(exist_ok=True)
URL = "https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu120_hrnet.pkl"
WIDTHS = (64, 64, 64, 96, 96, 96)  # 449,448 parameters
EPOCHS, BATCH, LR, WD, WARMUP = 100, 64, 0.05, 5e-4, 2
WORKERS = 4
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print(DEV, torch.__version__)
# %% [md]
# ## 2. Get the data
#
# Uses a copy under `/kaggle/input` if you attached one (for example a private dataset you saved from an earlier run), otherwise downloads it.
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
# ## 3. Dataset and augmentation
#
# Training clips get a random affine (rotation, scale, shear), a random left-right mirror, keypoint jitter and randomly dropped joints. The last two imitate a live pose estimator, which is noisier than HRNet on NTU video and sometimes misses a joint.
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
        return torch.from_numpy(pack(kp, sc, idx)), y
dl_tr = torch.utils.data.DataLoader(NTU(train_ids, True), BATCH, shuffle=True, num_workers=WORKERS, drop_last=True, pin_memory=True, persistent_workers=WORKERS > 0)
dl_va = torch.utils.data.DataLoader(NTU(val_ids, False), 256, num_workers=WORKERS, pin_memory=True)
x, y = next(iter(dl_tr)); print(x.shape, y[:8])
# %% [md]
# ## 4. Train
# %%
model = SafeSensePose(NC, widths=WIDTHS).to(DEV)
print("parameters:", sum(p.numel() for p in model.parameters()))
opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, nesterov=True, weight_decay=WD)
steps_ep = len(dl_tr); total = EPOCHS * steps_ep
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / (WARMUP * steps_ep)) * 0.5 * (1 + np.cos(np.pi * min(s, total) / total)))
scaler = torch.amp.GradScaler(enabled=DEV == "cuda")
start, log, best = 1, [], 0.0
CK = OUT / "last.pt"
if CK.exists():  # resume after an interrupted session
    ck = torch.load(CK, map_location=DEV, weights_only=False)
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
    start, log, best = ck["epoch"] + 1, ck["log"], ck["best"]; print("resumed at epoch", start)

def evaluate(m, loader):
    m.eval(); P, Y = [], []
    with torch.no_grad(), torch.autocast(DEV, enabled=DEV == "cuda"):
        for x, y in loader: P.append(m(x.to(DEV, non_blocking=True)).float().cpu()); Y.append(y)
    P, Y = torch.cat(P), torch.cat(Y)
    return P, Y, dict(top1=float((P.argmax(1) == Y).float().mean()), top5=float((P.topk(5, 1).indices == Y[:, None]).any(1).float().mean()))

for ep in range(start, EPOCHS + 1):
    model.train(); t0 = time.time(); tot = n = cor = 0
    for x, y in dl_tr:
        x, y = x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True)
        with torch.autocast(DEV, enabled=DEV == "cuda"):
            z = model(x); loss = F.cross_entropy(z, y, label_smoothing=0.1)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sched.step()
        tot += loss.item() * len(y); n += len(y); cor += int((z.argmax(1) == y).sum())
    row = dict(epoch=ep, loss=round(tot / n, 4), train_acc=round(cor / n, 4), sec=round(time.time() - t0))
    if ep % 5 == 0 or ep == EPOCHS:
        row.update({k: round(v, 4) for k, v in evaluate(model, dl_va)[2].items()})
        if row["top1"] > best:
            best = row["top1"]; torch.save(dict(model=model.state_dict(), epoch=ep, top1=best), OUT / "best.pt")
    log.append(row); print(row)
    torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), epoch=ep, log=log, best=best), CK)
# %% [md]
# ## 5. Results on unseen people (NTU-120 cross-subject)
#
# Overall accuracy, then accuracy on a candidate set of demo actions. For the demo, predictions can be restricted to that set, which is reported separately below.
# %%
P, Y, res = evaluate(model, dl_va); print("final model, cross-subject:", {k: round(v, 4) for k, v in res.items()})
acc_c = np.array([float((P[Y == c].argmax(1) == c).float().mean()) if (Y == c).any() else np.nan for c in range(NC)])
DEMO = ["drink water", "sitting down", "standing up", "clapping", "hand waving", "jump up", "make a phone call/answer phone",
        "falling", "touch head (headache)", "touch chest (stomachache/heart pain)", "staggering", "squat down",
        "hands up (both hands)", "arm circles", "running on the spot", "kicking something", "pickup", "cross arms"]
di = [NTU120_CLASSES.index(a) for a in DEMO]
print("\nper-class accuracy on the demo candidates:")
for a, i in sorted(zip(DEMO, di), key=lambda t: -acc_c[t[1]]): print(f"  {a:<40} {acc_c[i]:.3f}")
m = np.isin(Y.numpy(), di); Pd = P[m][:, di]; Yd = torch.tensor([di.index(int(v)) for v in Y[m]])
print("\nrestricted to these %d actions: top1 %.4f on %d clips" % (len(di), float((Pd.argmax(1) == Yd).float().mean()), int(m.sum())))
worst = np.argsort(acc_c)[:10]; print("\nhardest classes overall:", [(NTU120_CLASSES[i], round(acc_c[i], 3)) for i in worst])
# %% [md]
# ## 6. Export for the live demo
# %%
import shutil
torch.save(dict(model=model.state_dict(), classes=NTU120_CLASSES, num_classes=NC, widths=WIDTHS, t_steps=T_STEPS, layout="coco17",
                cross_subject=res, demo_actions=DEMO, log=log), OUT / "safesense_pose_ntu120_449k.pt")
json.dump(dict(cross_subject=res, per_class={NTU120_CLASSES[i]: (None if np.isnan(acc_c[i]) else round(float(acc_c[i]), 4)) for i in range(NC)}, log=log),
          open(OUT / "results_ntu120_449k.json", "w"), indent=1)
print(sorted(p.name for p in OUT.iterdir()))
# Quick check that the exported file loads the way the demo will load it
ck = torch.load(OUT / "safesense_pose_ntu120_449k.pt", map_location="cpu", weights_only=False)
m2 = SafeSensePose(ck["num_classes"], widths=tuple(ck["widths"])); m2.load_state_dict(ck["model"]); m2.eval()
live = pack(np.random.rand(1, 75, 17, 2).astype(np.float32) * 400, np.ones((1, 75, 17), np.float32), sample_frames(75))
print("demo-style input", live.shape, "->", ck["classes"][int(m2(torch.from_numpy(live)[None]).argmax())])
