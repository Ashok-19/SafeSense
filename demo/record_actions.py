"""Record yourself doing NTU actions through the live camera and see what the model makes of each one.

For every action you are prompted on the terminal ("arm swings in 3, 2, 1, go"), you do it for a few seconds, and the
poses are saved to recordings/<action>.npz. Each recording is then scored the way the live demo scores it (the same
app.select_people / app.window_tensor / model), and compared with the NTU test clips of that action:

  - rank of the right action among all 120, and the top 3 guesses, for 3 s windows across the recording
  - how many joints the pose model sees, how sure it is, how big you are in the frame, the pose rate
  - how much you move (torso lengths), next to how much the NTU people move doing the same action

  python record_actions.py --source http://10.32.210.252:8080/video --checkpoint safesense_pose_ntu120.pt --pkl data/ntu120_hrnet.pkl
  python record_actions.py --analyse-only --checkpoint safesense_pose_ntu120.pt --pkl data/ntu120_hrnet.pkl   (re-score saved recordings)
"""
import argparse, sys, time, threading, pickle, json
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--source", default="0"); ap.add_argument("--checkpoint", default="safesense_pose_ntu120.pt")
ap.add_argument("--pkl", default="", help="NTU-120 HRNet pkl, to compare with how NTU people do each action")
ap.add_argument("--actions", default="arm swings,nod head/bow,salute,touch chest (stomachache/heart pain),hand waving,clapping,jump up,squat down,arm circles,cross arms")
ap.add_argument("--secs", type=float, default=6.0, help="seconds to record each action")
ap.add_argument("--pose", default="lightweight", choices=["lightweight", "balanced", "performance"])
ap.add_argument("--out", default="recordings"); ap.add_argument("--analyse-only", action="store_true")
a = ap.parse_args()
sys.argv = ["app.py", "--checkpoint", a.checkpoint, "--actions", "all", "--window", "3"]
import app, torch                                       # the demo's own code, without starting its threads

ACTS = [x.strip() for x in a.actions.split(",") if x.strip()]
bad = [x for x in ACTS if x not in app.CLASSES]
if bad: raise SystemExit(f"not NTU-120 class names: {bad}")
out = Path(a.out); out.mkdir(exist_ok=True)
JOINTS = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist",
          "r_wrist", "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle"]
fname = lambda act: out / (act.split(" (")[0].replace("/", "_").replace(" ", "_") + ".npz")


def record():
    import cv2
    from rtmlib import Body
    src = int(a.source) if a.source.isdigit() else a.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened(): raise SystemExit(f"cannot open {a.source}")
    latest = {"f": None}
    def grab():                                         # always keep only the newest frame, as the live demo does
        while True:
            ok, f = cap.read()
            if ok: latest["f"] = f
    threading.Thread(target=grab, daemon=True).start()
    est = Body(mode=a.pose, backend="onnxruntime", device="cpu")
    while latest["f"] is None: time.sleep(0.05)
    print(f"camera ok, frame {latest['f'].shape[1]}x{latest['f'].shape[0]}. Stand where you will stand for the demo, whole body in view.")
    for act in ACTS:
        input(f"\nNext: {act.upper()}. Press Enter, then start when it says GO (keep doing it until STOP) ")
        for c in (3, 2, 1): print(c, flush=True); time.sleep(1)
        print("GO", flush=True)
        ts, kps, scs, t0 = [], [], [], time.time()
        while time.time() - t0 < a.secs:
            f = latest["f"]; t = time.time()
            kp, sc = est(f); kp, sc = app.select_people(kp, sc)
            k2 = np.zeros((2, 17, 2), np.float32); s2 = np.zeros((2, 17), np.float32); n = min(len(kp), 2)
            k2[:n], s2[:n] = kp[:n], sc[:n]
            ts.append(t - t0); kps.append(k2); scs.append(s2)
        print("STOP", flush=True)
        np.savez(fname(act), ts=np.array(ts), kp=np.array(kps), sc=np.array(scs), wh=np.array(f.shape[1::-1]), action=act)
        print(f"saved {fname(act)}: {len(ts)} poses, {len(ts) / a.secs:.1f} per second")


def windows(ts, kp, sc, step=0.5):
    """Feed a recording through the live buffer and score a 3 s window every `step` seconds -> list of softmax over 120."""
    res = []
    for end in np.arange(min(3.0, ts[-1]), ts[-1] + 1e-6, step):
        app.buf.clear()
        for t, k, s in zip(ts, kp, sc):
            if end - 3.0 <= t <= end:
                keep = s.mean(1) > 0; app.buf.append((t, k[keep], s[keep]))
        x, motion = app.window_tensor()
        if x is None: continue
        with torch.no_grad(): z = app.model(*x); z = (z[0] if app.ORIG else z)[0]
        res.append((torch.softmax(z, 0).numpy(), motion))
    return res


def ntu_reference(acts):
    """Median movement and joint visibility of NTU xsub test clips of each action, through the same window code."""
    if not a.pkl: return {}
    data = pickle.load(open(a.pkl, "rb")); val = set(data["split"]["xsub_val"]); ref = {}
    for act in acts:
        y = app.CLASSES.index(act); mv = []
        clips = [x for x in data["annotations"] if x["label"] == y and x["frame_dir"] in val][:40]
        for x in clips:
            kp = np.asarray(x["keypoint"], np.float32); sc = np.asarray(x["keypoint_score"], np.float32)
            T = kp.shape[1]; ts = np.arange(T) / 30.0
            k2 = np.zeros((T, 2, 17, 2), np.float32); s2 = np.zeros((T, 2, 17), np.float32)
            for f in range(T):
                k, s = app.select_people(kp[:, f], sc[:, f]); n = min(len(k), 2); k2[f, :n], s2[f, :n] = k[:n], s[:n]
            r = windows(ts, k2, s2, step=1.0)
            mv += [m for _, m in r]
        ref[act] = dict(motion=float(np.median(mv)) if mv else float("nan"), clips=len(clips))
    return ref


if not a.analyse_only: record()
ref = ntu_reference(ACTS)
print("\n" + "=" * 100)
summary = {}
for act in ACTS:
    p = fname(act)
    if not p.exists(): print(f"{act}: no recording"); continue
    d = np.load(p); ts, kp, sc = d["ts"], d["kp"], d["sc"]; y = app.CLASSES.index(act)
    r = windows(ts, kp, sc)
    if not r: print(f"{act}: too few poses with a person in view"); continue
    P = np.array([q for q, _ in r]); mean = P.mean(0)
    ranks = [int((q > q[y]).sum()) + 1 for q in P]
    top3 = [(app.CLASSES[i], round(float(mean[i]), 2)) for i in np.argsort(-mean)[:3]]
    s0 = sc[:, 0]; seen = s0 > 0.3
    h = np.array([np.ptp(k[0][s[0] > 0.3, 1]) if (s[0] > 0.3).sum() > 3 else 0 for k, s in zip(kp, sc)])
    info = dict(windows=len(r), right_top1=f"{np.mean([k == 1 for k in ranks]):.0%}", median_rank=int(np.median(ranks)),
                p_right=round(float(mean[y]), 3), top3=top3, pose_per_s=round(len(ts) / max(ts[-1], 1e-6), 1),
                joints_seen=round(float(seen.sum(1).mean()), 1), mean_score=round(float(s0[s0 > 0].mean()) if (s0 > 0).any() else 0, 2),
                body_height_frac=round(float(np.median(h) / d["wh"][1]), 2), motion=round(float(np.median([m for _, m in r])), 3),
                ntu_motion=round(ref.get(act, {}).get("motion", float("nan")), 3))
    unseen = [JOINTS[v] for v in range(17) if seen[:, v].mean() < 0.5]
    info["joints_mostly_missing"] = unseen
    summary[act] = info
    print(f"\n{act}:"); [print(f"   {k:22s} {v}") for k, v in info.items()]
json.dump(summary, open(out / "summary.json", "w"), indent=1)
print(f"\nsaved {out / 'summary.json'}")
