"""SafeSense live demo: phone camera -> RTMPose (17 COCO joints) -> SafeSense-Pose -> browser dashboard.

Run:  python app.py --source http://<phone-ip>:8080/video      (Android "IP Webcam" app)
      python app.py --source 0                                 (laptop webcam)
      python app.py --source backup.mp4                        (recorded video, loops)
      python app.py --mock                                     (no camera, synthetic skeleton, for testing)
Then open http://localhost:8000 in the laptop browser.
"""
import argparse, threading, time, collections, json
from pathlib import Path
import numpy as np, cv2, torch, uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from safesense_pose import SafeSensePose, pack, COCO_EDGES, T_STEPS

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--source", default="0", help="IP Webcam URL, webcam index, or video file")
ap.add_argument("--checkpoint", default=str(HERE / "safesense_pose_ntu120.pt"))
ap.add_argument("--pose", default="lightweight", choices=["lightweight", "balanced", "performance"], help="RTMPose size: lightweight is fastest on CPU")
ap.add_argument("--window", type=float, default=3.0, help="seconds of motion the model looks at")
ap.add_argument("--every", type=float, default=0.25, help="seconds between action predictions")
ap.add_argument("--threshold", type=float, default=0.45, help="minimum smoothed confidence to show an action")
ap.add_argument("--still", type=float, default=0.015, help="motion level below which the person counts as still")
ap.add_argument("--actions", default="", help="comma-separated demo actions (default: the list saved in the checkpoint)")
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--mock", action="store_true", help="synthetic skeleton instead of camera and pose model")
args = ap.parse_args()

# ---------------------------------------------------------------- model
ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
CLASSES = ck["classes"]; model = SafeSensePose(ck["num_classes"]); model.load_state_dict(ck["model"]); model.eval()
torch.set_num_threads(2)
ACTIONS = [a.strip() for a in args.actions.split(",") if a.strip()] or list(ck["demo_actions"])
unknown = [a for a in ACTIONS if a not in CLASSES]
if unknown: raise SystemExit(f"not NTU-120 class names: {unknown}")
ACT_IDX = [CLASSES.index(a) for a in ACTIONS]
ALERTS = {"falling", "staggering"}
print(f"model: {sum(p.numel() for p in model.parameters()):,} params, cross-subject top-1 {ck['cross_subject']['top1']:.3f}; demo actions: {len(ACTIONS)}")

# ---------------------------------------------------------------- shared state
lock = threading.Lock()
S = dict(frame=None, frame_t=0.0, cam_fps=0.0, pose_fps=0.0, pose_ms=0.0, act_ms=0.0, persons=0, label="starting", conf=0.0,
         top=[], alert=None, alert_t=0.0, events=collections.deque(maxlen=20), status="starting", jpeg=None, motion=0.0)
buf = collections.deque()  # (t, kp [P,17,2], score [P,17])


# ---------------------------------------------------------------- camera
def camera_loop():
    src = int(args.source) if args.source.isdigit() else args.source
    is_file = isinstance(src, str) and Path(src).exists()
    n, t_fps = 0, time.time()
    while True:
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            with lock: S["status"] = f"cannot open camera source {args.source}, retrying"
            time.sleep(2); continue
        delay = 1.0 / (cap.get(cv2.CAP_PROP_FPS) or 30) if is_file else 0
        with lock: S["status"] = "camera connected"
        while True:
            ok, frame = cap.read()
            if not ok:
                if is_file: cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue   # loop the backup video
                break                                                         # stream dropped: reconnect
            with lock: S["frame"], S["frame_t"] = frame, time.time()
            n += 1
            if time.time() - t_fps > 1: 
                with lock: S["cam_fps"] = n / (time.time() - t_fps)
                n, t_fps = 0, time.time()
            if delay: time.sleep(delay)
        cap.release()
        with lock: S["status"] = "camera stream lost, reconnecting"
        time.sleep(1)


class MockPose:
    """A person squatting, then waving, then falling over, in image pixels; lets the whole pipeline run without a camera."""
    base = np.array([[0, -170], [-8, -178], [8, -178], [-18, -172], [18, -172], [-45, -130], [45, -130], [-55, -70], [55, -70],
                     [-60, -15], [60, -15], [-30, 0], [30, 0], [-32, 90], [32, 90], [-34, 180], [34, 180]], np.float32)
    def __call__(self, frame):
        t = time.time(); kp = self.base.copy(); phase, u = int(t / 6) % 3, (t % 6)
        if phase == 0:                                               # squat: hips and upper body go down and up
            d = 60 * (0.5 - 0.5 * np.cos(2 * np.pi * t / 2.0)); kp[:13, 1] += d; kp[13:15, 1] += d / 2
        elif phase == 1:                                             # wave with the right hand
            kp[10] = kp[6] + [40 + 30 * np.sin(2 * np.pi * t * 1.5), -110]; kp[8] = kp[6] + [35, -55]
        else:                                                        # stand 2 s, fall sideways over 1 s, lie still
            a = np.clip((u - 2.0) / 1.0, 0, 1) ** 2 * (np.pi / 2)
            c, s_ = np.cos(a), np.sin(a); foot = np.array([0, 180], np.float32)
            kp = (kp - foot) @ np.array([[c, s_], [-s_, c]], np.float32) + foot
        kp = kp + [320, 260] + np.random.normal(0, 1.5, kp.shape)
        return kp[None].astype(np.float32), np.full((1, 17), 0.9, np.float32)


def pose_loop():
    if args.mock: est = MockPose()
    else:
        from rtmlib import Body   # downloads the ONNX models on first run
        est = Body(mode=args.pose, backend="onnxruntime", device="cpu")
    last_t, n, t_fps = 0.0, 0, time.time()
    while True:
        with lock: frame, ft = S["frame"], S["frame_t"]
        if args.mock:
            frame = np.full((520, 640, 3), 235, np.uint8); ft = time.time(); time.sleep(1 / 20)
        if frame is None or ft == last_t: time.sleep(0.005); continue
        last_t = ft; t0 = time.time()
        kp, sc = est(frame)
        kp = np.asarray(kp, np.float32).reshape(-1, 17, 2); sc = np.clip(np.asarray(sc, np.float32).reshape(-1, 17), 0, 1)
        keep = sc.mean(1) > 0.3
        kp, sc = kp[keep], sc[keep]
        if len(kp):  # the two biggest people in view, biggest first
            area = np.ptp(kp[..., 0], 1) * np.ptp(kp[..., 1], 1); o = np.argsort(-area)[:2]; kp, sc = kp[o], sc[o]
        ms = (time.time() - t0) * 1000
        with lock:
            buf.append((ft, kp, sc))
            while buf and buf[0][0] < ft - args.window: buf.popleft()
            S["persons"], S["pose_ms"] = len(kp), ms
        annotate(frame, kp, sc)
        n += 1
        if time.time() - t_fps > 1:
            with lock: S["pose_fps"] = n / (time.time() - t_fps)
            n, t_fps = 0, time.time()


def annotate(frame, kp, sc):
    img = frame.copy()
    for p in range(len(kp)):
        col = (0, 170, 255) if p == 0 else (255, 160, 0)
        for i, j in COCO_EDGES:
            if sc[p, i] > 0.3 and sc[p, j] > 0.3:
                cv2.line(img, tuple(int(v) for v in kp[p, i]), tuple(int(v) for v in kp[p, j]), col, 3, cv2.LINE_AA)
        for i in range(17):
            if sc[p, i] > 0.3: cv2.circle(img, tuple(int(v) for v in kp[p, i]), 4, (255, 255, 255), -1, cv2.LINE_AA)
    h, w = img.shape[:2]
    if w > 960: img = cv2.resize(img, (960, int(h * 960 / w)))
    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
    if ok:
        with lock: S["jpeg"] = jpg.tobytes()


# ---------------------------------------------------------------- action recognition
def window_tensor():
    """Last `window` seconds of poses -> one model input [1,2,48,17,5], sampled evenly in time."""
    with lock: items = list(buf)
    items = [it for it in items if len(it[1])]
    if len(items) < 8 or items[-1][0] - items[0][0] < 1.0: return None, 0.0
    ts = np.array([it[0] for it in items]); n = len(items)
    kp = np.zeros((2, n, 17, 2), np.float32); sc = np.zeros((2, n, 17), np.float32)
    for f, (_, k, s) in enumerate(items):
        kp[:len(k), f], sc[:len(s), f] = k[:2], s[:2]
    grid = np.linspace(ts[0], ts[-1], T_STEPS)
    idx = np.clip(np.searchsorted(ts, grid), 0, n - 1)
    x = pack(kp, sc, idx)
    motion = float(np.abs(x[0, :, :, 3:5]).mean())   # mean joint speed of the main person, in torso lengths per step
    return torch.from_numpy(x)[None], motion


def action_loop():
    smooth, stable, prev = None, None, None
    while True:
        time.sleep(args.every); t0 = time.time()
        x, motion = window_tensor()
        with lock: recent = bool(buf) and time.time() - buf[-1][0] < 1.0 and S["persons"] > 0
        if x is None or not recent:
            with lock: S.update(label="no person in view" if not recent else "watching...", conf=0.0, top=[], motion=0.0)
            smooth = None; continue
        with torch.no_grad(): z = model(x)[0]
        p_all = torch.softmax(z, 0)                    # over all 120 NTU actions
        p = torch.softmax(z[ACT_IDX], 0).numpy()        # restricted to the demo actions
        smooth = p if smooth is None else 0.5 * smooth + 0.5 * p
        k = int(np.argmax(smooth)); conf = float(smooth[k]); name = ACTIONS[k]
        if motion < args.still and name not in ("cross arms", "hands up (both hands)"): label = "standing still"
        elif conf < args.threshold or float(p_all[ACT_IDX[k]]) < 0.15: label = "no clear action"
        else: label = name
        top = sorted(zip(ACTIONS, smooth.tolist()), key=lambda t: -t[1])[:5]
        now = time.time()
        with lock:
            S.update(label=label, conf=conf if label == name else 0.0, top=[[a, round(v, 3)] for a, v in top],
                     act_ms=(now - t0) * 1000, motion=round(motion, 4))
            if label == prev and label != stable:      # held for two predictions in a row: log it
                stable = label
                last = S["events"][0] if S["events"] else None
                repeat = last and last["label"] == label and now - last["ts"] < 8
                if label not in ("no clear action", "standing still") and not repeat:
                    S["events"].appendleft(dict(t=time.strftime("%H:%M:%S"), ts=now, label=label, conf=round(conf, 2)))
                elif repeat: last["ts"] = now
                if label in ALERTS: S["alert"], S["alert_t"] = label, now
            if S["alert"] and now - S["alert_t"] > 6: S["alert"] = None
        prev = label


# ---------------------------------------------------------------- web dashboard
app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def index(): return (HERE / "dashboard.html").read_text(encoding="utf-8")

@app.get("/state")
def state():
    with lock:
        d = {k: S[k] for k in ("cam_fps", "pose_fps", "pose_ms", "act_ms", "persons", "label", "conf", "top", "alert", "status", "motion")}
        d["events"] = list(S["events"])
    d.update(actions=ACTIONS, source="synthetic test skeleton" if args.mock else args.source, model_top1=ck["cross_subject"]["top1"])
    return JSONResponse(d)

@app.get("/stream.mjpg")
def stream():
    def gen():
        last = None
        while True:
            with lock: j = S["jpeg"]
            if j is None or j is last: time.sleep(0.01); continue
            last = j
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + j + b"\r\n"
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/clear")
def clear():
    with lock: S["events"].clear(); S["alert"] = None
    return {"ok": True}


if __name__ == "__main__":
    if not args.mock: threading.Thread(target=camera_loop, daemon=True).start()
    else: S["status"] = "running on a synthetic test skeleton"
    threading.Thread(target=pose_loop, daemon=True).start()
    threading.Thread(target=action_loop, daemon=True).start()
    print(f"dashboard: http://localhost:{args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")
