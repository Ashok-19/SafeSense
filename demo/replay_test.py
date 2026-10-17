"""Check the live pipeline on real NTU-120 test clips (people never seen in training).

Each clip's HRNet keypoints are replayed through the exact code the live demo runs (app.select_people, the time
buffer, app.window_tensor, the model and the action list), at a camera/pose rate you choose, and the accuracy is
compared with the offline evaluation. If the two match, the pipeline is sound and live errors come from the camera /
pose estimate; if the replay is much worse, the pipeline itself is at fault.

  python replay_test.py --pkl ntu120_hrnet.pkl --checkpoint safesense_pose_ntu120.pt --fps 10 --n 2000
"""
import argparse, sys, pickle, time, collections
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--pkl", required=True); ap.add_argument("--checkpoint", default="safesense_pose_ntu120.pt")
ap.add_argument("--fps", type=float, default=10, help="simulated pose rate on the laptop")
ap.add_argument("--n", type=int, default=2000, help="test clips to replay (0 = all)")
ap.add_argument("--jitter", type=float, default=0.0, help="extra keypoint noise in pixels, to imitate a live pose estimator")
ap.add_argument("--stgcnpp", action="store_true")
ap.add_argument("--window", type=float, default=3.0)
a = ap.parse_args()

sys.argv = ["app.py", "--checkpoint", a.checkpoint, "--actions", "all", "--window", str(a.window)] + (["--stgcnpp"] if a.stgcnpp else [])
import app, torch                                     # the demo's own code, without starting its threads

data = pickle.load(open(a.pkl, "rb"))
ann = {x["frame_dir"]: x for x in data["annotations"]}
ids = [k for k in data["split"]["xsub_val"] if k in ann]
rng = np.random.default_rng(0)
if a.n: ids = list(rng.choice(ids, min(a.n, len(ids)), replace=False))
C = app.CLASSES
strong = [C.index(x) for x in app.STRONG]
from safesense_pose import pack, sample_frames
hit_off = 0; hit = hit_s = n_s = 0; per = collections.defaultdict(lambda: [0, 0]); t0 = time.time()
for k, fid in enumerate(ids):
    x = ann[fid]; kp = np.asarray(x["keypoint"], np.float32); sc = np.asarray(x["keypoint_score"], np.float32)
    if kp.ndim == 3: kp, sc = kp[None], sc[None]
    h, w = x.get("img_shape", (1080, 1920))
    T = kp.shape[1]; dur = T / 30.0
    app.buf.clear(); app.S["frame_wh"] = (w, h)
    for t in np.arange(0, dur, 1 / a.fps):              # the pose estimator only sees some frames
        f = min(int(round(t * 30)), T - 1)   # round: int(k/30*30) can land on k-1, repeating a frame
        kk = kp[:, f] + (rng.normal(0, a.jitter, kp[:, f].shape) if a.jitter else 0)
        p_kp, p_sc = app.select_people(kk, sc[:, f])
        app.buf.append((t, p_kp, p_sc))
        while app.buf and app.buf[0][0] < t - app.args.window: app.buf.popleft()
    inp, motion = app.window_tensor()
    if inp is None: pred = -1
    else:
        with torch.no_grad():
            if app.PYSKL: z = torch.log(torch.stack([torch.softmax(m(xi)[0], 0) for (_, m), xi in zip(app.PY_MODELS, inp)]).mean(0))
            else:
                z = app.model(*inp); z = (z[0] if app.ORIG else z)[0]
        pred = int(z.argmax())
    y = int(x["label"])
    if not (app.PYSKL or app.ORIG):   # control: the offline evaluation path on the same clip
        with torch.no_grad(): hit_off += int(app.model(torch.from_numpy(pack(kp, sc, sample_frames(T)))[None])[0].argmax()) == y
    hit += pred == y; per[y][0] += pred == y; per[y][1] += 1
    if y in strong:
        n_s += 1; hit_s += (inp is not None) and strong[int(z[strong].argmax())] == y
    if (k + 1) % 200 == 0: print(f"{k + 1}/{len(ids)}  all-120 acc {hit / (k + 1):.3f}  strong-list acc {hit_s / max(n_s, 1):.3f}  {time.time() - t0:.0f}s", flush=True)
print(f"\nreplay at {a.fps:g} fps, jitter {a.jitter:g}px, {len(ids)} clips: all-120 top-1 {hit / len(ids):.4f} "
      f"(offline path on the same clips: {hit_off / len(ids):.4f}); default action list ({len(strong)} actions) {hit_s / max(n_s, 1):.4f} on {n_s} clips")
worst = sorted((v[0] / v[1], C[c]) for c, v in per.items() if v[1] >= 5 and c in strong)[:8]
print("weakest actions in the default list:", [(n, round(r, 2)) for r, n in worst])
