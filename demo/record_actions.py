"""Terminal version of the dashboard's Record & check tab (the dashboard is easier: bash start.sh).

Prompts you for each action, records it through the camera, saves recordings/<action>.npz and scores it with the
same code as the dashboard (analysis.py). Run prepare_ntu.py once first for the NTU comparison.

  python record_actions.py --source http://<phone-ip>:8080/video --checkpoint safesense_pose_ntu120.pt
  python record_actions.py --analyse-only --checkpoint safesense_pose_ntu120.pt      (re-score saved recordings)
"""
import argparse, sys, time, threading, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--source", default="0"); ap.add_argument("--checkpoint", default="safesense_pose_ntu120.pt")
ap.add_argument("--actions", default="arm swings,nod head/bow,salute,touch chest (stomachache/heart pain),hand waving,clapping,jump up,squat down,arm circles,cross arms")
ap.add_argument("--secs", type=float, default=6.0, help="seconds to record each action")
ap.add_argument("--countdown", type=int, default=5, help="seconds between Enter and GO, to get into place")
ap.add_argument("--pose", default="lightweight", choices=["lightweight", "balanced", "performance"])
ap.add_argument("--analyse-only", action="store_true")
a = ap.parse_args()
sys.argv = ["app.py", "--checkpoint", a.checkpoint, "--actions", "all", "--window", "3"]
import app, analysis                                   # the demo's own code, without starting its threads

ACTS = [x.strip() for x in a.actions.split(",") if x.strip()]
bad = [x for x in ACTS if x not in app.CLASSES]
if bad: raise SystemExit(f"not NTU-120 class names: {bad}")
analysis.REC_DIR.mkdir(exist_ok=True)


def record():
    import cv2
    from rtmlib import Body
    cap = app.open_capture(app.norm_source(a.source))
    if not cap.isOpened(): raise SystemExit(app.camera_check(a.source)["msg"])
    latest = {"f": None, "run": True}
    def grab():                                         # always keep only the newest frame, as the live demo does
        while latest["run"]:
            ok, f = cap.read()
            if ok: latest["f"] = f
    th = threading.Thread(target=grab, daemon=True); th.start()
    est = Body(mode=a.pose, backend="onnxruntime", device="cpu")
    while latest["f"] is None: time.sleep(0.05)
    print(f"camera ok, frame {latest['f'].shape[1]}x{latest['f'].shape[0]}. Stand where you will stand for the demo, whole body in view.")
    for act in ACTS:
        input(f"\nNext: {act.upper()}. Press Enter, get into place, start at GO and keep going until STOP ")
        for c in range(a.countdown, 0, -1): print(c, flush=True); time.sleep(1)
        print("GO", flush=True)
        ts, kps, scs, t0 = [], [], [], time.time()
        while time.time() - t0 < a.secs:
            f = latest["f"]; t = time.time()
            kp, sc = est(f); kp, sc = app.select_people(kp, sc)
            k2 = np.zeros((2, 17, 2), np.float32); s2 = np.zeros((2, 17), np.float32); n = min(len(kp), 2)
            k2[:n], s2[:n] = kp[:n], sc[:n]
            ts.append(t - t0); kps.append(k2); scs.append(s2)
        print("STOP", flush=True)
        np.savez(analysis.REC_DIR / f"{analysis.rec_name(act)}.npz", ts=np.array(ts, np.float32), kp=np.array(kps), sc=np.array(scs),
                 wh=np.array(f.shape[1::-1]), action=act, time=time.time())
        print(f"saved: {len(ts)} poses, {len(ts) / a.secs:.1f} per second")
    latest["run"] = False; th.join(2); cap.release()


if not a.analyse_only: record()
print("\n" + "=" * 100)
for act in ACTS:
    r = analysis.load_recording(act)
    if r is None: print(f"{act}: no recording"); continue
    info = analysis.score_and_store(app, act, *r)
    print(f"\n{act}:")
    for k, v in info.items():
        if k not in ("action", "model", "time"): print(f"   {k:18s} {v}")
print(f"\nsaved {analysis.RESULTS}")
sys.stdout.flush(); os._exit(0)   # skip interpreter teardown: OpenCV's stream thread can abort it
