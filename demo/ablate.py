"""Which step of the live pipeline loses accuracy? Each variant changes ONE thing from the offline path, on whole clips
at 30 fps with training's own frame sampling, so the drop of each variant is that step's cost.

  python ablate.py --pkl data/ntu120_hrnet.pkl --checkpoint safesense_pose_ntu120.pt --n 1000
"""
import argparse, sys, pickle, collections
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--pkl", required=True); ap.add_argument("--checkpoint", default="safesense_pose_ntu120.pt")
ap.add_argument("--n", type=int, default=1000)
a = ap.parse_args()
sys.argv = ["app.py", "--checkpoint", a.checkpoint, "--actions", "all"]
import app, torch
from safesense_pose import pack, sample_frames

data = pickle.load(open(a.pkl, "rb"))
ann = {x["frame_dir"]: x for x in data["annotations"]}
ids = [k for k in data["split"]["xsub_val"] if k in ann]
rng = np.random.default_rng(0)
if a.n: ids = list(rng.choice(ids, min(a.n, len(ids)), replace=False))


def per_frame(kp, sc, f, clip, filt, sort):
    k, s = kp[:, f], sc[:, f]
    if clip: s = np.clip(s, 0, 1)
    if filt:
        keep = s.mean(1) > 0.3; k, s = k[keep], s[keep]
    if sort and len(k):
        area = np.ptp(k[..., 0], 1) * np.ptp(k[..., 1], 1); o = np.argsort(-area)[:2]; k, s = k[o], s[o]
    return k, s


def live_like(kp, sc, clip=False, filt=False, sort=False, trk=False, drop_empty=False):
    T = kp.shape[1]
    items = [(f, *per_frame(kp, sc, f, clip, filt, sort)) for f in range(T)]
    if drop_empty: items = [it for it in items if len(it[1])]
    if trk: k2, s2 = app.track([it for it in items if len(it[1])]) if drop_empty else track_keep_empty(items)
    else:   # keep pkl slots, zero-filled where a body was removed
        n = len(items); k2 = np.zeros((2, n, 17, 2), np.float32); s2 = np.zeros((2, n, 17), np.float32)
        for i, (_, k, s) in enumerate(items):
            for p in range(min(2, len(k))): k2[p, i], s2[p, i] = k[p], s[p]
    return pack(k2, s2, sample_frames(k2.shape[1]))


def track_keep_empty(items):   # app.track, but frames with nobody stay in place as zeros
    full = [i for i, it in enumerate(items) if len(it[1])]
    k, s = app.track([items[i] for i in full])
    K = np.zeros((2, len(items), 17, 2), np.float32); S = np.zeros((2, len(items), 17), np.float32)
    K[:, full], S[:, full] = k, s
    return K, S


V = {"offline (pkl as is)": dict(), "clip scores to 0..1": dict(clip=True), "drop bodies with mean score <= 0.3": dict(filt=True),
     "sort by area each frame": dict(sort=True), "sort by area + track": dict(sort=True, trk=True),
     "drop + sort + track": dict(filt=True, sort=True, trk=True),
     "everything live does": dict(clip=True, filt=True, sort=True, trk=True, drop_empty=True)}
hit = collections.Counter(); st = collections.Counter()
for k, fid in enumerate(ids):
    x = ann[fid]; kp = np.asarray(x["keypoint"], np.float32); sc = np.asarray(x["keypoint_score"], np.float32)
    if kp.ndim == 3: kp, sc = kp[None], sc[None]
    y = int(x["label"]); M = kp.shape[0]
    st["clips"] += 1; st[f"bodies_in_pkl={M}"] += 1
    st["scores>1"] += int((sc > 1).sum()); st["scores"] += sc.size
    st["body-frames"] += M * kp.shape[1]; st["body-frames dropped by 0.3 filter"] += int((sc.mean(2) <= 0.3).sum())
    with torch.no_grad():
        for name, kw in V.items():
            hit[name] += int(app.model(torch.from_numpy(live_like(kp, sc, **kw))[None])[0].argmax()) == y
    if (k + 1) % 200 == 0: print(f"{k + 1}/{len(ids)}", flush=True)
n = len(ids)
print("\nwhole clips, 30 fps, training sampling; each line changes one step:")
for name in V: print(f"  {name:38s} {hit[name] / n:.4f}")
print("stats:", dict(st), f"score max {max(float(np.max(ann[i]['keypoint_score'])) for i in ids):.3f}")
