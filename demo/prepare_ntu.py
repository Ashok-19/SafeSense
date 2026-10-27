"""One-time NTU reference for the dashboard's Record & check tab -> ntu_reference.npz (small).

For each of the 120 actions it replays NTU-120 cross-subject test clips (people the model never saw) through the
live window code in 3 s windows, and keeps:
  - how much the NTU people move doing the action (median movement, torso lengths)
  - how often the model gets it right in those windows (the best you can expect live)
  - one example window as the model sees it, for the side-by-side skeleton player

  python prepare_ntu.py --pkl data/ntu120_hrnet.pkl --checkpoint safesense_pose_ntu120.pt
"""
import argparse, sys, pickle, time, collections
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--pkl", default="data/ntu120_hrnet.pkl"); ap.add_argument("--checkpoint", default="safesense_pose_ntu120.pt")
ap.add_argument("--clips", type=int, default=15, help="test clips per action")
a = ap.parse_args()
sys.argv = ["app.py", "--checkpoint", a.checkpoint, "--actions", "all", "--window", "3"]
import app, analysis

t0 = time.time()
print(f"loading {a.pkl} (1.2 GB, takes a while)...", flush=True)
data = pickle.load(open(a.pkl, "rb"))
val = set(data["split"]["xsub_val"]); by = collections.defaultdict(list)
for x in data["annotations"]:
    if x["frame_dir"] in val: by[int(x["label"])].append(x)
del data
C = app.CLASSES; nC = len(C)
motion = np.zeros(nC, np.float32); acc = np.zeros(nC, np.float32); clips = np.zeros(nC, np.int32)
view = np.zeros((nC, 48, 17, 3), np.float32)
rng = np.random.default_rng(0)
for c in range(nC):
    xs = by.get(c, [])
    xs = [xs[i] for i in rng.choice(len(xs), min(a.clips, len(xs)), replace=False)] if xs else []
    mv, hit, cands = [], [], []
    for x in xs:
        kp = np.asarray(x["keypoint"], np.float32); sc = np.asarray(x["keypoint_score"], np.float32)
        if kp.ndim == 3: kp, sc = kp[None], sc[None]
        T = kp.shape[1]; ts = np.arange(T, dtype=np.float32) / 30.0
        k2 = np.zeros((T, 2, 17, 2), np.float32); s2 = np.zeros((T, 2, 17), np.float32)
        for f in range(T):
            k, s = app.select_people(kp[:, f], sc[:, f]); m = min(len(k), 2); k2[f, :m], s2[f, :m] = k[:m], s[:m]
        W = analysis.windows(app, ts, k2, s2, step=1.5)
        for w in W: mv.append(w["motion"]); hit.append(int(np.argmax(w["p"]) == c))
        if W: cands.append((W[len(W) // 2]["motion"], W[len(W) // 2]["x"][0][..., :3]))
    if mv:
        motion[c] = np.median(mv); acc[c] = np.mean(hit); clips[c] = len(xs)
        view[c] = min(cands, key=lambda t: abs(t[0] - motion[c]))[1]   # the example closest to the typical movement
    print(f"{c + 1:3d}/{nC} {C[c]:45s} movement {motion[c]:.2f}  model right {acc[c]:.0%} of windows  ({len(xs)} clips)  {time.time() - t0:.0f}s", flush=True)
np.savez(analysis.REF_FILE, classes=np.array(C), motion=motion, acc=acc, clips=clips, view=view,
         model=np.array(app.MODEL_NAME), time=np.array(time.time()))
print(f"saved {analysis.REF_FILE} in {time.time() - t0:.0f}s", flush=True)
