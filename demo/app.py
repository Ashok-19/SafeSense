"""SafeSense live demo: phone camera -> RTMPose (17 COCO joints) -> SafeSense (original architecture, NTU-120) -> browser dashboard.

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
DEFAULT_CK = next((HERE / f for f in ("safesense_orig_ntu120.pt", "safesense_pose_ntu120.pt") if (HERE / f).exists()), HERE / "safesense_orig_ntu120.pt")
ap.add_argument("--checkpoint", default=str(DEFAULT_CK), help="model file (default: the original SafeSense, safesense_orig_ntu120.pt)")
ap.add_argument("--pose", default="lightweight", choices=["lightweight", "balanced", "performance"], help="RTMPose size: lightweight is fastest on CPU")
ap.add_argument("--window", type=float, default=3.0, help="seconds of motion the model looks at")
ap.add_argument("--every", type=float, default=0.25, help="seconds between action predictions")
ap.add_argument("--threshold", type=float, default=0.40, help="minimum smoothed confidence to show an action")
ap.add_argument("--still", type=float, default=0.18, help="movement range (torso lengths) below which the person counts as still; 0 turns it off")
ap.add_argument("--debug", action="store_true", help="print motion and the top prediction for every window")
ap.add_argument("--actions", default="", help='comma-separated NTU action names, "all" for all 120 (default: the single-person actions recognised best)')
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--stgcnpp", action="store_true", help="use the pretrained PYSKL ST-GCN++ (stgcnpp_j.pth / stgcnpp_b.pth next to app.py)")
ap.add_argument("--mock", action="store_true", help="synthetic skeleton instead of camera and pose model")
args = ap.parse_args()

# ---------------------------------------------------------------- model
# Single-person NTU-120 actions that SafeSense-Pose recognises at 90%+ on unseen people (results_ntu120.json),
# plus a few everyday ones. Two-person actions are left out: the demo shows one person.
STRONG = ["jump up", "arm circles", "squat down", "cross arms", "arm swings", "pickup", "staggering", "sitting down",
          "standing up", "falling", "bounce ball", "nod head/bow", "salute", "cross hands in front (say stop)", "shake head",
          "kicking something", "wear jacket", "take off jacket", "check time (from watch)", "butt kicks (kick backward)",
          "hopping (one foot jumping)", "touch back (backache)", "running on the spot", "hands up (both hands)",
          "put the palms together", "thumb down", "side kick", "touch chest (stomachache/heart pain)", "cross toe touch",
          "touch neck (neckache)", "touch head (headache)", "make a phone call/answer phone", "hand waving", "clapping",
          "drink water", "throw", "stretch oneself"]
DEMO_DEFAULT = ["drink water", "sitting down", "standing up", "clapping", "hand waving", "jump up", "make a phone call/answer phone",
                "falling", "touch head (headache)", "touch chest (stomachache/heart pain)", "staggering", "squat down",
                "hands up (both hands)", "arm circles", "running on the spot", "kicking something", "pickup", "cross arms"]
PYSKL = args.stgcnpp
if PYSKL:   # pretrained PYSKL ST-GCN++ on NTU-120 xsub (HRNet 2D), joint and/or bone stream
    from pyskl_stgcnpp import load_pyskl, pyskl_input, CLIP_LEN, FPS
    from safesense_pose import NTU120_CLASSES
    streams = [(f, HERE / f"stgcnpp_{f}.pth") for f in ("j", "b") if (HERE / f"stgcnpp_{f}.pth").exists()]
    if not streams: raise SystemExit(f"no ST-GCN++ weights in {HERE}: run  bash get_stgcnpp.sh  first")
    PY_MODELS = [(f, load_pyskl(pth)) for f, pth in streams]
    args.window = max(args.window, CLIP_LEN / FPS + 0.05)
    ORIG = False; CLASSES = list(NTU120_CLASSES)
    top1 = 0.864 if len(PY_MODELS) == 2 else {"j": 0.844, "b": 0.848}[PY_MODELS[0][0]]
    ck = dict(cross_subject=dict(top1=top1), demo_actions=DEMO_DEFAULT)
    model = PY_MODELS[0][1]
    MODEL_NAME = "PYSKL ST-GCN++ (" + " + ".join({"j": "joint", "b": "bone"}[f] for f, _ in PY_MODELS) + ")"
elif not Path(args.checkpoint).exists():
    raise SystemExit(f"model file not found: {args.checkpoint}\nCopy safesense_orig_ntu120.pt (Kaggle output of notebook 06) into {HERE}")
else:
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    CLASSES = ck["classes"]; ORIG = ck.get("arch") == "safesense_orig"
    MODEL_NAME = "SafeSense (original)" if ORIG else "SafeSense-Pose"
if PYSKL: pass
elif ORIG:   # original SafeSense architecture (06_train_safesense_original_ntu120)
    from safesense_orig import SafeSenseCOCO, body_window
    model = SafeSenseCOCO(ck["num_classes"])
else:      # SafeSense-Pose (04 / 05 notebooks)
    model = SafeSensePose(ck["num_classes"], widths=tuple(ck.get("widths", (64, 64, 64, 128, 128, 128))))
if not PYSKL: model.load_state_dict(ck["model"]); model.eval()
torch.set_num_threads(4 if PYSKL else 2)
ACTIONS = list(CLASSES) if args.actions.strip() == "all" else ([a.strip() for a in args.actions.split(",") if a.strip()] or STRONG)
unknown = [a for a in ACTIONS if a not in CLASSES]
if unknown: raise SystemExit(f"not NTU-120 class names: {unknown}")
ACT_IDX = [CLASSES.index(a) for a in ACTIONS]
ALERTS = {"falling", "staggering"}
print(f"model: {MODEL_NAME}, {sum(p.numel() for p in model.parameters()):,} params, cross-subject top-1 {ck['cross_subject']['top1']:.3f}; demo actions: {len(ACTIONS)}")

# ---------------------------------------------------------------- shared state
lock = threading.Lock()
S = dict(frame=None, frame_t=0.0, cam_fps=0.0, pose_fps=0.0, pose_ms=0.0, act_ms=0.0, persons=0, label="starting", conf=0.0,
         frame_wh=(640, 480), top=[], alert=None, alert_t=0.0, events=collections.deque(maxlen=20), status="starting", jpeg=None, motion=0.0)
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


def select_people(kp, sc):
    """Pose estimator output -> the two biggest confidently detected people, biggest first: [P,17,2], [P,17]."""
    kp = np.asarray(kp, np.float32).reshape(-1, 17, 2); sc = np.clip(np.asarray(sc, np.float32).reshape(-1, 17), 0, 1)
    keep = sc.mean(1) > 0.3
    kp, sc = kp[keep], sc[keep]
    if len(kp):
        area = np.ptp(kp[..., 0], 1) * np.ptp(kp[..., 1], 1); o = np.argsort(-area)[:2]; kp, sc = kp[o], sc[o]
    return kp, sc


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
        kp, sc = select_people(kp, sc)
        ms = (time.time() - t0) * 1000
        with lock:
            S["frame_wh"] = (frame.shape[1], frame.shape[0])
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
def resample(a, sc, ts, grid, score=False):
    """Linear interpolation of a [M, n, ...] array onto time points `grid`; a joint missing at either end is not blended."""
    j = np.clip(np.searchsorted(ts, grid), 1, len(ts) - 1); i = j - 1
    w = np.clip((grid - ts[i]) / np.maximum(ts[j] - ts[i], 1e-6), 0, 1)
    if score: return np.where(np.minimum(sc[:, i], sc[:, j]) > 0, (1 - w)[None, :, None] * sc[:, i] + w[None, :, None] * sc[:, j], 0).astype(np.float32)
    wb = w.reshape((1, -1) + (1,) * (a.ndim - 2))
    both = (np.minimum(sc[:, i], sc[:, j]) > 0)[..., None]
    near = np.where((w < 0.5).reshape((1, -1) + (1,) * (a.ndim - 2)), a[:, i], a[:, j])
    return np.where(both, (1 - wb) * a[:, i] + wb * a[:, j], near).astype(np.float32)


def movement(body):
    """How far the main person's limbs travel in the window: largest per-joint range (5th to 95th percentile, so single
    jittery detections do not count) of the normalised x/y, in torso lengths. About 0.1 standing still, 0.3+ when moving."""
    ok = body[..., 2] > 0.3
    rng = [np.ptp(np.percentile(body[ok[:, v], v, :2], [5, 95], axis=0), 0).max() if ok[:, v].sum() > 3 else 0.0
           for v in range(body.shape[1])]
    return float(max(rng))


def window_tensor():
    """Last `window` seconds of poses -> one model input [1,2,48,17,5], sampled evenly in time."""
    with lock: items = list(buf)
    items = [it for it in items if len(it[1])]
    if len(items) < 8 or items[-1][0] - items[0][0] < 1.0: return None, 0.0
    ts = np.array([it[0] for it in items]); n = len(items)
    kp = np.zeros((2, n, 17, 2), np.float32); sc = np.zeros((2, n, 17), np.float32)
    for f, (_, k, s) in enumerate(items):
        kp[:len(k), f], sc[:len(s), f] = k[:2], s[:2]
    # Interpolate onto an even time grid. The camera and pose rate (often 8 to 15 fps on a laptop CPU) is much lower
    # than NTU's 30 fps, and picking the nearest frame repeated frames, which zeroes the motion between them.
    grid = np.linspace(ts[0], ts[-1], T_STEPS)
    kpg, scg = resample(kp, sc, ts, grid), resample(sc, sc, ts, grid, score=True)
    idx = np.arange(T_STEPS)
    x = pack(kpg, scg, idx)
    motion = movement(x[0])
    if PYSKL:   # PYSKL clips are real-time 30 fps: resample the last 100/30 s onto a 30 fps grid
        n30 = int(min(CLIP_LEN, max(8, round((ts[-1] - ts[0]) * FPS))))
        g30 = np.linspace(ts[-1] - (n30 - 1) / FPS, ts[-1], n30)
        k30, s30 = resample(kp, sc, ts, g30), resample(sc, sc, ts, g30, score=True)
        with lock: wh = S["frame_wh"]
        return tuple(pyskl_input(k30, s30, wh, f) for f, _ in PY_MODELS), motion
    if ORIG:
        raw, mk = body_window(kpg, scg, idx)
        return (torch.from_numpy(raw)[None], torch.from_numpy(mk)[None]), motion
    return (torch.from_numpy(x)[None],), motion


def action_loop():
    smooth, stable, prev = None, None, None
    while True:
        time.sleep(args.every); t0 = time.time()
        x, motion = window_tensor()
        with lock: recent = bool(buf) and time.time() - buf[-1][0] < 1.0 and S["persons"] > 0
        if x is None or not recent:
            with lock: S.update(label="no person in view" if not recent else "watching...", conf=0.0, top=[], motion=0.0)
            smooth = None; continue
        with torch.no_grad():
            if PYSKL:   # average the streams' probabilities, as PYSKL's joint + bone ensemble does
                z = torch.log(torch.stack([torch.softmax(m(xi)[0], 0) for (_, m), xi in zip(PY_MODELS, x)]).mean(0) + 1e-8)
            else:
                z = model(*x)
                z = (z[0] if ORIG else z)[0]
        p_all = torch.softmax(z, 0)                    # over all 120 NTU actions
        p = torch.softmax(z[ACT_IDX], 0).numpy()        # restricted to the demo actions
        smooth = p if smooth is None else 0.5 * smooth + 0.5 * p
        k = int(np.argmax(smooth)); conf = float(smooth[k]); name = ACTIONS[k]
        if args.debug: print(f"motion {motion:.3f} | top {name} {conf:.2f} | all-class p {float(p_all[ACT_IDX[k]]):.2f}")
        low = conf < args.threshold
        if motion < args.still and low: label = "standing still"   # never hides a confident prediction
        elif low: label = "no clear action"
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
    d.update(actions=ACTIONS, source="synthetic test skeleton" if args.mock else args.source, model_top1=ck["cross_subject"]["top1"], model=MODEL_NAME)
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
