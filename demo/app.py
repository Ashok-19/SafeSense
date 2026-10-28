"""SafeSense live demo and control dashboard: phone camera -> RTMPose (17 COCO joints) -> action model -> browser.

Start:  bash start.sh            (opens http://localhost:8765; everything else is set from the dashboard)
   or:  python app.py [--source http://<phone-ip>:8080/video] [--checkpoint safesense_pose_ntu120.pt] [--mock]

The Demo page shows the camera, what SafeSense recognises, how good the input is right now, and buttons that degrade
the input on purpose (hidden limbs, noise, lost frames, low frame rate) so the audience sees the robust pipeline cope.
Setup finds the phone on the network by itself. Record and check, and Tests, are for preparing the demo. Settings
are kept in settings.json, so the next start uses them.
"""
import argparse, threading, time, collections, json, os, socket, subprocess, sys
from pathlib import Path
from urllib.parse import urlparse
import numpy as np, cv2, torch, uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from safesense_pose import SafeSensePose, pack, COCO_EDGES, T_STEPS, NTU120_CLASSES
import robust

HERE = Path(__file__).resolve().parent
SETTINGS = HERE / "settings.json"
REC_DIR = HERE / "recordings"
ap = argparse.ArgumentParser()
ap.add_argument("--source", default=None, help="IP Webcam URL, webcam index, video file, or mock")
ap.add_argument("--checkpoint", default=None, help="model file (default: the last one used, else safesense_pose_ntu120.pt)")
ap.add_argument("--pose", default=None, choices=["lightweight", "balanced", "performance"], help="RTMPose size: lightweight is fastest on CPU")
ap.add_argument("--window", type=float, default=None, help="seconds of motion the model looks at")
ap.add_argument("--every", type=float, default=0.25, help="seconds between action predictions")
ap.add_argument("--threshold", type=float, default=None, help="minimum smoothed confidence to show an action")
ap.add_argument("--still", type=float, default=None, help="movement range (torso lengths) below which the person counts as still; 0 turns it off")
ap.add_argument("--debug", action="store_true", help="print motion and the top prediction for every window")
ap.add_argument("--actions", default=None, help='comma-separated NTU action names, "all" for all 120 (default: the single-person actions recognised best)')
ap.add_argument("--port", type=int, default=8765)
ap.add_argument("--stgcnpp", action="store_true", help="use the pretrained PYSKL ST-GCN++ (stgcnpp_j.pth / stgcnpp_b.pth next to app.py)")
ap.add_argument("--mock", action="store_true", help="synthetic skeleton instead of camera and pose model")
args = ap.parse_args()

# Settings: command line, else settings.json (only when running the dashboard, not when a test script imports this), else defaults.
saved = {}
if __name__ == "__main__" and SETTINGS.exists():
    try: saved = json.loads(SETTINGS.read_text())
    except Exception: saved = {}
DEFAULTS = dict(source="", pose="lightweight", window=3.0, threshold=0.40, still=0.18, actions="",
                sampling=os.environ.get("SAFESENSE_SAMPLING", "interp"), robust=True, compare="pipeline")
for k, v in DEFAULTS.items():
    if getattr(args, k, None) is None: setattr(args, k, saved.get(k, v))
if args.mock: args.source = "mock"
if args.stgcnpp: MODEL_CHOICE = "stgcnpp"
elif args.checkpoint: MODEL_CHOICE = args.checkpoint
else:
    MODEL_CHOICE = saved.get("model") or next((f for f in ("safesense_pose_ntu120.pt", "safesense_orig_ntu120.pt") if (HERE / f).exists()),
                                              "safesense_pose_ntu120.pt")

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
ALERTS = {"falling", "staggering"}
model_lock = threading.RLock()     # held while the model is swapped or run
PYSKL = ORIG = False; PY_MODELS = []; model = None; ck = {}; CLASSES = list(NTU120_CLASSES); MODEL_NAME = "-"
ACTIONS, ACT_IDX = [], []


def set_actions(spec):
    global ACTIONS, ACT_IDX
    acts = list(CLASSES) if spec.strip() == "all" else ([a.strip() for a in spec.split(",") if a.strip()] or STRONG)
    unknown = [a for a in acts if a not in CLASSES]
    if unknown: raise ValueError(f"not NTU-120 action names: {unknown}")
    ACTIONS, ACT_IDX = acts, [CLASSES.index(a) for a in acts]


def build_model(choice):
    """choice: a checkpoint file name/path, or "stgcnpp" -> a model slot (dict) the pipeline can run."""
    if choice == "stgcnpp":   # pretrained PYSKL ST-GCN++ on NTU-120 xsub (HRNet 2D), joint and/or bone stream
        from pyskl_stgcnpp import load_pyskl
        streams = [(f, HERE / f"stgcnpp_{f}.pth") for f in ("j", "b") if (HERE / f"stgcnpp_{f}.pth").exists()]
        if not streams: raise FileNotFoundError(f"no ST-GCN++ weights in {HERE}: run  bash get_stgcnpp.sh  first")
        pm = [(f, load_pyskl(pth)) for f, pth in streams]
        top1 = 0.864 if len(pm) == 2 else {"j": 0.844, "b": 0.848}[pm[0][0]]
        return dict(choice=choice, kind="pyskl", PYSKL=True, ORIG=False, PY_MODELS=pm, model=pm[0][1],
                    ck=dict(cross_subject=dict(top1=top1)), CLASSES=list(NTU120_CLASSES),
                    name="ST-GCN++ (pretrained, " + " + ".join({"j": "joint", "b": "bone"}[f] for f, _ in pm) + ")")
    path = Path(choice) if Path(choice).is_absolute() else HERE / choice
    if not path.exists(): path = Path(choice)
    if not path.exists(): raise FileNotFoundError(f"model file not found: {choice} (put the .pt file next to app.py)")
    c = torch.load(path, map_location="cpu", weights_only=False)
    orig = c.get("arch") == "safesense_orig"
    if orig:   # original SafeSense architecture, corruption-aware training (06_train_safesense_original_ntu120)
        from safesense_orig import SafeSenseCOCO
        m = SafeSenseCOCO(c["num_classes"])
    else:      # SafeSense-Pose (04 / 05 notebooks)
        m = SafeSensePose(c["num_classes"], widths=tuple(c.get("widths", (64, 64, 64, 128, 128, 128))))
    m.load_state_dict(c["model"]); m.eval()
    return dict(choice=choice, kind="orig" if orig else "pose", PYSKL=False, ORIG=orig, PY_MODELS=[], model=m, ck=c,
                CLASSES=c["classes"], name=("SafeSense (original)" if orig else "SafeSense-Pose") + f" [{path.name}]")


COMPARE = None    # optional second model run on the same input, for side by side comparison


def load_model(choice):
    """Make `choice` the main model. Sets the module globals the pipeline and the test scripts read."""
    global PYSKL, ORIG, PY_MODELS, model, ck, CLASSES, MODEL_NAME, MODEL_CHOICE, MAIN
    slot = build_model(choice)
    with model_lock:
        MAIN = slot
        PYSKL, ORIG, PY_MODELS, model, ck, CLASSES = (slot[k] for k in ("PYSKL", "ORIG", "PY_MODELS", "model", "ck", "CLASSES"))
        MODEL_NAME, MODEL_CHOICE = slot["name"], choice
        torch.set_num_threads(4 if PYSKL else 2)
        try: set_actions(args.actions)
        except ValueError as e: print(e, "- using the default action list"); args.actions = ""; set_actions("")
    if PYSKL: args.window = max(args.window, 100 / 30 + 0.05)
    print(f"model: {MODEL_NAME}, {sum(p.numel() for p in model.parameters()):,} params, cross-subject top-1 "
          f"{ck['cross_subject']['top1']:.3f}; demo actions: {len(ACTIONS)}")


def load_compare(choice):
    """The comparison line: "" none, "pipeline" the same model with the robust pipeline switched the other way (the
    default: shows what the pipeline adds), or another model file / "stgcnpp" run on exactly the same input."""
    global COMPARE
    slot = dict(choice="pipeline", kind="pipeline") if choice == "pipeline" else build_model(choice) if choice else None
    with model_lock: COMPARE = slot
    args.compare = choice or ""


MAIN = None
try: load_model(MODEL_CHOICE)
except FileNotFoundError as e:
    if __name__ != "__main__": raise SystemExit(str(e))
    print(e, "\nchoose a model in the dashboard's Setup page")
if __name__ == "__main__" and args.compare:
    try: load_compare(args.compare)
    except Exception as e: print("comparison model not loaded:", e); args.compare = ""
from pyskl_stgcnpp import pyskl_input, CLIP_LEN, FPS
from safesense_orig import body_window

# ---------------------------------------------------------------- shared state
lock = threading.Lock()
S = dict(frame=None, frame_t=0.0, cam_fps=0.0, pose_fps=0.0, pose_ms=0.0, act_ms=0.0, persons=0, label="starting", conf=0.0,
         frame_wh=(640, 480), top=[], top_all=[], alert=None, alert_t=0.0, events=collections.deque(maxlen=20), status="starting",
         jpeg=None, motion=0.0, view=None, rec=None, busy="", quality={}, cmp=None, degrade="", lost=0)
buf = collections.deque()  # (t, kp [P,17,2], score [P,17]) after any injected degradation
DEGRADER = robust.Degrader()


# ---------------------------------------------------------------- camera
def norm_source(s):
    """Phone address as typed -> what OpenCV opens: '172.30.1.5:8080' -> 'http://172.30.1.5:8080/video'; '0' -> webcam 0."""
    s = str(s).strip()
    if s.isdigit() or s in ("", "mock"): return s
    if not s.startswith(("http://", "https://", "rtsp://")) and not Path(s).exists() and ":" in s and "/" not in s.split(":")[0]:
        s = "http://" + s
    if s.startswith("http") and urlparse(s).path in ("", "/"): s = s.rstrip("/") + "/video"
    return s


def local_ips():
    try:
        out = subprocess.run(["ip", "-4", "-o", "addr", "show"], capture_output=True, text=True, timeout=2).stdout
        ips = [ln.split()[3].split("/")[0] for ln in out.splitlines() if len(ln.split()) > 3]
    except Exception:
        try: ips = socket.gethostbyname_ex(socket.gethostname())[2]
        except Exception: ips = []
    try:   # the address of the interface that would carry traffic out (no packet is sent)
        u = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); u.connect(("10.255.255.255", 1)); ips.append(u.getsockname()[0]); u.close()
    except Exception: pass
    return list(dict.fromkeys(ip for ip in ips if not ip.startswith(("127.", "169.254.", "172.17."))))


def camera_check(src):
    """Plain-language diagnosis of a camera address: reachable? serving? gives frames?"""
    src = norm_source(src); ips = local_ips(); res = dict(source=src, laptop_ips=ips, ok=False)
    if src == "mock": return dict(res, ok=True, msg="Synthetic test skeleton: no camera needed.")
    if src == "": return dict(res, msg="No camera address set.")
    if src.startswith("http"):
        u = urlparse(src); host, port = u.hostname, u.port or (443 if u.scheme == "https" else 80)
        same = any(ip.rsplit(".", 1)[0] == host.rsplit(".", 1)[0] for ip in ips)
        net = "" if same or not ips else (f" Your laptop's address is {', '.join(ips) or 'unknown'}, which is on a different network from "
                               f"{host}: connect the laptop to the phone's Wi-Fi network or hotspot.")
        try:
            socket.create_connection((host, port), timeout=3).close()
        except socket.timeout:
            return dict(res, msg=f"No answer from {host}:{port}. The phone can't be reached from this laptop." + net +
                        " If they are on the same Wi-Fi, that network may block devices from talking to each other "
                        "(common on college and public Wi-Fi): use the phone's hotspot instead.")
        except ConnectionRefusedError:
            return dict(res, msg=f"{host} answered, but nothing is serving on port {port}. Open the camera app on the phone (SafeSense Cam starts serving by itself) "
                        "and use the address it shows at the bottom of the screen.")
        except socket.gaierror:
            return dict(res, msg=f"'{host}' is not a valid address.")
        except OSError as e:
            return dict(res, msg=f"Could not reach {host}:{port} ({e.strerror or e})." + net)
    cap = open_capture(src)
    ok, frame = cap.read() if cap.isOpened() else (False, None)
    cap.release()
    if not ok: return dict(res, msg=f"Connected, but no video frames came from {src}. Check the camera app is showing its picture.")
    return dict(res, ok=True, msg=f"Camera OK: {frame.shape[1]}x{frame.shape[0]} frames from {src}.")


def open_capture(src):
    src = int(src) if str(src).isdigit() else src
    if isinstance(src, str) and src.startswith(("http", "rtsp")):
        try: return cv2.VideoCapture(src, cv2.CAP_FFMPEG, [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 6000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 6000])
        except Exception: pass
    return cv2.VideoCapture(src)


def camera_loop():
    n, t_fps = 0, time.time()
    while True:
        src = norm_source(args.source)
        if src in ("", "mock"):
            with lock: S["status"] = "running on a synthetic test skeleton" if src == "mock" else "no camera set"
            time.sleep(0.5); continue
        is_file = Path(str(src)).exists()
        with lock: S["status"] = f"connecting to {src}"
        cap = open_capture(src)
        if not cap.isOpened():
            with lock: S["status"] = f"cannot open camera {src}"
            cap.release(); time.sleep(2); continue
        delay = 1.0 / (cap.get(cv2.CAP_PROP_FPS) or 30) if is_file else 0
        with lock: S["status"] = "camera connected"
        while norm_source(args.source) == src:
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
        with lock: S["status"] = "camera stream lost, reconnecting"; S["frame"] = None
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
    est, est_kind = None, None
    last_t, n, t_fps = 0.0, 0, time.time()
    while True:
        mock = norm_source(args.source) == "mock"
        kind = "mock" if mock else args.pose
        if kind != est_kind:
            if mock: est = MockPose()
            else:
                with lock: S["busy"] = f"loading the {args.pose} pose model (downloads once)"
                try:
                    from rtmlib import Body   # downloads the ONNX models on first run
                    est = Body(mode=args.pose, backend="onnxruntime", device="cpu")
                except Exception as e:
                    with lock: S["busy"] = f"pose model failed to load: {e}"
                    time.sleep(3); continue
                with lock: S["busy"] = ""
            est_kind = kind
        with lock: frame, ft = S["frame"], S["frame_t"]
        if mock:
            frame = np.full((520, 640, 3), 235, np.uint8); ft = time.time(); time.sleep(1 / 20)
        if frame is None or ft == last_t: time.sleep(0.005); continue
        last_t = ft; t0 = time.time()
        kp, sc = est(frame)
        kp, sc = select_people(kp, sc)
        raw_kp, raw_sc = kp, sc
        d = DEGRADER(kp, sc, ft) if DEGRADER.active() else (kp, sc)   # injected degradation, when the presenter turns it on
        ms = (time.time() - t0) * 1000
        with lock:
            S["frame_wh"] = (frame.shape[1], frame.shape[0])
            if d is None: S["lost"] += 1
            else:
                kp, sc = d; buf.append((ft, kp, sc))
            while buf and buf[0][0] < ft - args.window: buf.popleft()
            S["persons"], S["pose_ms"] = len(kp), ms
            rec = S["rec"]
            if rec and rec["phase"] in ("countdown", "recording") and rec["t_go"] <= ft <= rec["t_stop"]:
                k2 = np.zeros((2, 17, 2), np.float32); s2 = np.zeros((2, 17), np.float32); m = min(len(kp), 2)
                k2[:m], s2[:m] = raw_kp[:m], raw_sc[:m]
                rec["data"].append((ft - rec["t_go"], k2, s2)); rec["wh"] = S["frame_wh"]
        annotate(frame, kp, sc, lost=d is None, note=DEGRADER.describe() if DEGRADER.active() else "")
        n += 1
        if time.time() - t_fps > 1:
            with lock: S["pose_fps"] = n / (time.time() - t_fps)
            n, t_fps = 0, time.time()


def annotate(frame, kp, sc, lost=False, note=""):
    if lost: return      # a lost frame never reaches the model, so the video freezes too
    img = frame.copy()
    for p in range(len(kp)):
        col = (0, 170, 255) if p == 0 else (255, 160, 0)
        for i, j in COCO_EDGES:
            if sc[p, i] > 0.3 and sc[p, j] > 0.3:
                cv2.line(img, tuple(int(v) for v in kp[p, i]), tuple(int(v) for v in kp[p, j]), col, 3, cv2.LINE_AA)
        for i in range(17):
            if sc[p, i] > 0.3: cv2.circle(img, tuple(int(v) for v in kp[p, i]), 4, (255, 255, 255), -1, cv2.LINE_AA)
    h, w = img.shape[:2]
    if w > 960: img = cv2.resize(img, (960, int(h * 960 / w))); h, w = img.shape[:2]
    if note:   # show the audience what is being injected
        cv2.rectangle(img, (0, 0), (w, 34), (40, 40, 200), -1)
        cv2.putText(img, "Injected: " + note, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
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


SAMPLING = args.sampling   # kept for the test scripts; window_tensor reads args.sampling


def track(items):
    """Per-frame detections -> [2, n, 17, 2] / [2, n, 17] with each person kept in the same slot across frames
    (nearest body centre to the previous frame), like NTU's per-person tracks. Sorting by size frame by frame swaps
    people whenever two are similar in size."""
    n = len(items); kp = np.zeros((2, n, 17, 2), np.float32); sc = np.zeros((2, n, 17), np.float32)
    prev = [None, None]
    for f, (_, k, s) in enumerate(items):
        k, s = k[:2], s[:2]
        if not len(k): continue
        cen = [(k[p] * s[p, :, None]).sum(0) / max(s[p].sum(), 1e-6) for p in range(len(k))]
        if all(c is None for c in prev): slots = list(range(len(k)))
        elif len(k) == 1:
            d = [np.inf if c is None else np.linalg.norm(cen[0] - c) for c in prev]; slots = [int(np.argmin(d))]
        else:
            d = lambda a, b: 1e9 if prev[b] is None else np.linalg.norm(cen[a] - prev[b])
            slots = [0, 1] if d(0, 0) + d(1, 1) <= d(0, 1) + d(1, 0) else [1, 0]
        for p, sl in enumerate(slots):
            kp[sl, f], sc[sl, f] = k[p], s[p]; prev[sl] = cen[p]
    return kp, sc


def window_inputs(items=None, slots=None, view=None, robust_on=None):
    """Poses (default: the live buffer, last `window` seconds) -> one model input per slot (default: the main model),
    sampled evenly in time, and the movement. With the robust pipeline on (args.robust), unreliable joints are marked
    missing, glitches rejected and short gaps filled first (robust.clean_window). `view`, if a list, gets the
    normalised [2, 48, 17, 5] window the model sees."""
    if items is None:
        with lock: items = list(buf)
    slots = slots or [MAIN]
    items = [it for it in items if len(it[1])]
    if len(items) < 8 or items[-1][0] - items[0][0] < 1.0: return None, 0.0
    ts = np.array([it[0] for it in items]); n = len(items)
    kp, sc = track(items)
    if args.robust if robust_on is None else robust_on: kp, sc = robust.clean_window(kp, sc, ts)
    # Training takes the frame at the centre of each of 48 equal segments of the clip (sample_frames). Do the same on
    # the time axis: each pose sample stands for one frame period dt, so the window spans n * dt.
    dt = float(np.median(np.diff(ts))); span = ts[-1] - ts[0] + dt
    centres = ts[0] + (np.arange(T_STEPS) + 0.5) / T_STEPS * span
    if args.sampling == "nearest":   # exactly training's sampling; right when pose runs near 30 fps
        g = np.clip(np.searchsorted(ts, centres + 1e-6, "right") - 1, 0, n - 1)
        kpg, scg = kp[:, g], sc[:, g]
    else:   # low pose rates: picking frames would repeat them and zero the motion between, so interpolate in time
        grid = np.clip(centres - dt / 2, ts[0], ts[-1])
        kpg, scg = resample(kp, sc, ts, grid), resample(sc, sc, ts, grid, score=True)
    idx = np.arange(T_STEPS)
    x = pack(kpg, scg, idx)
    if view is not None: view.append(x)
    motion = movement(x[0])
    out = []
    for sl in slots:
        if sl["PYSKL"]:   # PYSKL clips are real-time 30 fps: resample the last 100/30 s onto a 30 fps grid
            n30 = int(min(CLIP_LEN, max(8, round((ts[-1] - ts[0]) * FPS))))
            g30 = np.linspace(ts[-1] - (n30 - 1) / FPS, ts[-1], n30)
            k30, s30 = resample(kp, sc, ts, g30), resample(sc, sc, ts, g30, score=True)
            with lock: wh = S["frame_wh"]
            out.append(tuple(pyskl_input(k30, s30, wh, f) for f, _ in sl["PY_MODELS"]))
        elif sl["ORIG"]:
            raw, mk = body_window(kpg, scg, idx)
            out.append((torch.from_numpy(raw)[None], torch.from_numpy(mk)[None]))
        else: out.append((torch.from_numpy(x)[None],))
    return out, motion


def window_tensor(items=None, view=None, robust_on=None):
    """The main model's input and the movement (used by the test scripts)."""
    out, motion = window_inputs(items, [MAIN], view, robust_on)
    return (None if out is None else out[0]), motion


def logits(x, slot=None):
    """Model input -> logits over all classes [C], for `slot` (default: the main model)."""
    sl = slot or MAIN
    with torch.no_grad():
        if sl["PYSKL"]:   # average the streams' probabilities, as PYSKL's joint + bone ensemble does
            return torch.log(torch.stack([torch.softmax(m(xi)[0], 0) for (_, m), xi in zip(sl["PY_MODELS"], x)]).mean(0) + 1e-8)
        z = sl["model"](*x)
        return (z[0] if sl["ORIG"] else z)[0]


def view_json(x):
    """Normalised window [2,48,17,5] -> main body's x, y, score per frame, rounded, for the dashboard's skeleton player."""
    return np.round(x[0][..., :3], 3).tolist()


class Smoother:
    """Turns per-window probabilities into a steady label: moving average, a confidence threshold, and a change of
    label only once the new one has led twice in a row (hysteresis), so a single odd window does not flicker."""
    def __init__(self): self.p = None; self.shown = None; self.cand = None

    def __call__(self, p):
        if self.p is not None and len(self.p) != len(p): self.reset()
        self.p = p if self.p is None else 0.5 * self.p + 0.5 * p
        k = int(np.argmax(self.p)); conf = float(self.p[k])
        if conf < args.threshold: self.cand = None; return None, conf, k
        if self.shown != k:
            if self.cand == k: self.shown = k
            else: self.cand = k; return (self.shown, float(self.p[self.shown]), k) if self.shown is not None else (None, conf, k)
        self.cand = None
        return k, conf, k

    def reset(self): self.p = None; self.shown = None; self.cand = None


def cmp_name(sl):
    if sl["kind"] != "pipeline": return sl["name"]
    return "Same model without the robust pipeline" if args.robust else "Same model with the robust pipeline"


def action_loop():
    main_s, cmp_s, stable, prev = Smoother(), Smoother(), None, None
    while True:
        time.sleep(args.every); t0 = time.time()
        if MAIN is None: continue
        view = []
        with lock: items = list(buf)
        with model_lock:
            cmp_slot = COMPARE; other = cmp_slot is not None and cmp_slot["kind"] == "pipeline"
            slots = [MAIN] + ([cmp_slot] if cmp_slot and not other else [])
            xs, motion = window_inputs(items, slots, view)
            acts, idx = list(ACTIONS), list(ACT_IDX)
            zs = [logits(x, sl) for x, sl in zip(xs, slots)] if xs is not None else None
            if zs is not None and other:     # same model, same poses, robust pipeline the other way
                x2, _ = window_inputs(items, [MAIN], robust_on=not args.robust)
                zs.append(logits(x2[0], MAIN))
                cmp_slot = dict(cmp_slot, CLASSES=MAIN["CLASSES"],
                                name="Same model without the robust pipeline" if args.robust else "Same model with the robust pipeline")
        quality = robust.measure_quality(items, args.window)
        with lock:
            recent = bool(buf) and time.time() - buf[-1][0] < 1.5 and S["persons"] > 0
            S["quality"] = quality; S["degrade"] = DEGRADER.describe() if DEGRADER.active() else ""
        if zs is None or not recent:
            with lock: S.update(label="no person in view" if not recent else "watching...", conf=0.0, top=[], top_all=[], motion=0.0,
                                cmp=dict(name=cmp_name(cmp_slot), label="-", conf=0.0) if cmp_slot else None)
            main_s.reset(); cmp_s.reset(); continue
        # the main model
        z = zs[0]; p_all = torch.softmax(z, 0); p = torch.softmax(z[idx], 0).numpy()
        k, conf, lead = main_s(p)
        if args.debug: print(f"motion {motion:.3f} | quality {quality['score']} | top {acts[lead]} {float(main_s.p[lead]):.2f}")
        if quality["visible"] < 0.35 or quality["fps"] < 2: label, k = "input too degraded", None   # honest: says so
        elif k is not None: label = acts[k]
        elif args.still and motion < args.still: label = "standing still"
        else: label = "no clear action"
        top = sorted(zip(acts, main_s.p.tolist()), key=lambda t: -t[1])[:5]
        pa = p_all.numpy(); top_all = [[CLASSES[i], round(float(pa[i]), 3)] for i in np.argsort(-pa)[:5]]
        cmp = None
        if cmp_slot is not None and len(zs) > 1:   # the comparison model on exactly the same input
            pc = torch.softmax(zs[1][[cmp_slot["CLASSES"].index(a) for a in acts]], 0).numpy()
            kc, cc, _ = cmp_s(pc)
            cmp = dict(name=cmp_slot["name"], label=acts[kc] if kc is not None else "no clear action", conf=round(cc if kc is not None else 0.0, 3))
        now = time.time(); name = label if k is not None else None
        with lock:
            S.update(label=label, conf=conf if name else 0.0, top=[[a, round(v, 3)] for a, v in top], top_all=top_all,
                     act_ms=(now - t0) * 1000, motion=round(motion, 4), view=view_json(view[0]), cmp=cmp)
            if label == prev and label != stable:      # held for two predictions in a row: log it
                stable = label
                last = S["events"][0] if S["events"] else None
                repeat = last and last["label"] == label and now - last["ts"] < 8
                if name and not repeat:
                    S["events"].appendleft(dict(t=time.strftime("%H:%M:%S"), ts=now, label=label, conf=round(conf, 2),
                                                note=S["degrade"]))
                elif repeat: last["ts"] = now
                if label in ALERTS: S["alert"], S["alert_t"] = label, now
            if S["alert"] and now - S["alert_t"] > 6: S["alert"] = None
        prev = label


# ---------------------------------------------------------------- recordings
def rec_name(act): return act.split(" (")[0].replace("/", "_").replace(" ", "_")


def rec_loop():
    """Finishes a guided recording: saves it to recordings/<action>.npz and scores it."""
    import analysis
    while True:
        time.sleep(0.1)
        with lock:
            rec = S["rec"]
            if not rec: continue
            now = time.time()
            if rec["phase"] == "countdown" and now >= rec["t_go"]: rec["phase"] = "recording"
            if rec["phase"] != "recording" or now < rec["t_stop"] + 0.3: continue
            rec["phase"] = "scoring"; data, wh, act = list(rec["data"]), rec.get("wh", S["frame_wh"]), rec["action"]
        REC_DIR.mkdir(exist_ok=True)
        ts = np.array([d[0] for d in data], np.float32)
        kp = np.array([d[1] for d in data], np.float32).reshape(-1, 2, 17, 2); sc = np.array([d[2] for d in data], np.float32).reshape(-1, 2, 17)
        np.savez(REC_DIR / f"{rec_name(act)}.npz", ts=ts, kp=kp, sc=sc, wh=np.array(wh), action=act, time=time.time())
        try: info = analysis.score_and_store(sys.modules[__name__], act, ts, kp, sc, wh)
        except Exception as e: info = dict(error=str(e))
        with lock: rec["phase"] = "done"; rec["result"] = info


# ---------------------------------------------------------------- background jobs (NTU tests)
JOB = dict(name=None, proc=None, log=collections.deque(maxlen=3000), started=0.0, rc=None)


def run_job(name, cmd):
    if JOB["proc"] and JOB["proc"].poll() is None: raise RuntimeError(f"'{JOB['name']}' is still running")
    JOB.update(name=name, started=time.time(), rc=None); JOB["log"].clear(); JOB["log"].append("$ " + " ".join(cmd))
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    p = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, bufsize=1)
    JOB["proc"] = p
    def pump():
        for line in p.stdout: JOB["log"].append(line.rstrip())
        JOB["rc"] = p.wait(); JOB["log"].append(f"[finished, exit code {JOB['rc']}]")
    threading.Thread(target=pump, daemon=True).start()


# ---------------------------------------------------------------- web dashboard
app = FastAPI()


def model_files():
    out = sorted(p.name for p in HERE.glob("*.pt"))
    if (HERE / "stgcnpp_j.pth").exists() or (HERE / "stgcnpp_b.pth").exists(): out.append("stgcnpp")
    return out


def settings_dict():
    return dict(source=args.source, model=MODEL_CHOICE, pose=args.pose, window=args.window, threshold=args.threshold,
                still=args.still, actions=args.actions, sampling=args.sampling, robust=bool(args.robust), compare=args.compare)


@app.get("/", response_class=HTMLResponse)
def index(): return (HERE / "dashboard.html").read_text(encoding="utf-8")


@app.get("/state")
def state():
    with lock:
        d = {k: S[k] for k in ("cam_fps", "pose_fps", "pose_ms", "act_ms", "persons", "label", "conf", "top", "top_all", "alert",
                               "status", "motion", "busy", "quality", "cmp", "degrade", "lost")}
        d["stress"] = dict(STRESS, phases=[dict(p) for p in STRESS["phases"]]) if STRESS["phases"] else None
        d["events"] = list(S["events"])
        rec = S["rec"]
        if rec:
            now = time.time()
            d["rec"] = dict(action=rec["action"], phase=rec["phase"], to_go=round(rec["t_go"] - now, 2), to_stop=round(rec["t_stop"] - now, 2),
                            poses=len(rec["data"]), result=rec.get("result"))
    d.update(source=norm_source(args.source) or "none", model=MODEL_NAME, model_top1=ck.get("cross_subject", {}).get("top1"),
             n_actions=len(ACTIONS), robust=bool(args.robust), conditions=list(robust.CONDITIONS), job=dict(name=JOB["name"], running=bool(JOB["proc"] and JOB["proc"].poll() is None), rc=JOB["rc"]))
    return JSONResponse(d)


@app.get("/api/view")
def api_view():
    with lock: return JSONResponse(dict(view=S["view"]))


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


@app.get("/api/settings")
def get_settings():
    return dict(settings=settings_dict(), models=model_files(), classes=CLASSES, strong=STRONG, laptop_ips=local_ips())


@app.post("/api/settings")
async def post_settings(req: Request):
    new = await req.json(); msgs = []
    try:
        if "actions" in new:
            old = args.actions; args.actions = str(new["actions"])
            try:
                with model_lock: set_actions(args.actions)
            except ValueError: args.actions = old; raise
        for k, typ in (("window", float), ("threshold", float), ("still", float)):
            if k in new and new[k] not in ("", None): setattr(args, k, typ(new[k]))
        if new.get("pose") in ("lightweight", "balanced", "performance"): args.pose = new["pose"]
        if new.get("sampling") in ("interp", "nearest"): args.sampling = new["sampling"]
        if "source" in new: args.source = norm_source(new["source"])
        if "robust" in new: args.robust = bool(new["robust"])
        if "compare" in new and new["compare"] != args.compare:
            with lock: S["busy"] = "loading the comparison"
            try: load_compare(new["compare"]); msgs.append("comparison: " + (cmp_name(COMPARE) if COMPARE else "off"))
            finally:
                with lock: S["busy"] = ""
        if new.get("model") and new["model"] != MODEL_CHOICE:
            with lock: S["busy"] = f"loading {new['model']}"
            try: load_model(new["model"]); msgs.append(f"model loaded: {MODEL_NAME}")
            finally:
                with lock: S["busy"] = ""
    except Exception as e:
        return JSONResponse(dict(ok=False, msg=str(e)), status_code=400)
    SETTINGS.write_text(json.dumps(settings_dict(), indent=1))
    return dict(ok=True, msg="; ".join(msgs) or "saved", settings=settings_dict())


@app.get("/api/camera_test")
def api_camera_test(source: str):
    return camera_check(source)


@app.post("/api/record")
async def api_record(req: Request):
    r = await req.json(); act = r.get("action")
    if act not in CLASSES: return JSONResponse(dict(ok=False, msg=f"unknown action {act}"), status_code=400)
    now = time.time(); cd = float(r.get("countdown", 4)); secs = float(r.get("secs", 6))
    with lock:
        if S["rec"] and S["rec"]["phase"] in ("countdown", "recording", "scoring"): return JSONResponse(dict(ok=False, msg="already recording"), status_code=409)
        S["rec"] = dict(action=act, phase="countdown", t_go=now + cd, t_stop=now + cd + secs, data=[])
    return dict(ok=True)


@app.post("/api/record/cancel")
def api_record_cancel():
    with lock: S["rec"] = None
    return dict(ok=True)


@app.get("/api/recordings")
def api_recordings():
    import analysis
    return dict(results=analysis.load_results(), reference=analysis.reference_info())


@app.get("/api/recording_view")
def api_recording_view(action: str):
    import analysis
    return analysis.compare_view(sys.modules[__name__], action)


@app.post("/api/rescore")
def api_rescore():
    import analysis
    with lock: S["busy"] = "re-scoring all recordings"
    try: n = analysis.rescore_all(sys.modules[__name__])
    finally:
        with lock: S["busy"] = ""
    return dict(ok=True, msg=f"re-scored {n} recordings with {MODEL_NAME}")


@app.post("/api/jobs/run")
async def api_job_run(req: Request):
    r = await req.json(); name = r.get("name"); pkl = r.get("pkl") or "data/ntu120_hrnet.pkl"
    ckpt = MODEL_CHOICE if MODEL_CHOICE != "stgcnpp" else "safesense_pose_ntu120.pt"
    if not (HERE / pkl).exists() and not Path(pkl).exists():
        return JSONResponse(dict(ok=False, msg=f"NTU file not found: {pkl}"), status_code=400)
    py = sys.executable
    if name == "prepare": cmd = [py, "prepare_ntu.py", "--pkl", pkl, "--checkpoint", ckpt]
    elif name == "replay":
        cmd = [py, "replay_test.py", "--pkl", pkl, "--checkpoint", ckpt, "--fps", str(r.get("fps", 10)), "--window", str(r.get("window", 3)),
               "--n", str(r.get("n", 1000)), "--jitter", str(r.get("jitter", 0))] + (["--stgcnpp"] if MODEL_CHOICE == "stgcnpp" else [])
    elif name == "ablate": cmd = [py, "ablate.py", "--pkl", pkl, "--checkpoint", ckpt, "--n", str(r.get("n", 1000))]
    elif name == "robustness":
        cmd = [py, "robustness_test.py", "--pkl", pkl, "--checkpoint", MODEL_CHOICE, "--n", str(r.get("n", 600)),
               "--fps", str(r.get("fps", 10)), "--window", str(args.window)]
        if COMPARE and COMPARE["kind"] != "pipeline": cmd += ["--compare", COMPARE["choice"]]
    else: return JSONResponse(dict(ok=False, msg=f"unknown job {name}"), status_code=400)
    try: run_job(name, cmd)
    except RuntimeError as e: return JSONResponse(dict(ok=False, msg=str(e)), status_code=409)
    if name == "replay": JOB["log"].append(f"(sampling: {args.sampling}; set SAFESENSE_SAMPLING to change it for this test)")
    return dict(ok=True)


@app.get("/api/robustness")
def api_robustness():
    f = REC_DIR / "robustness.json"
    return json.loads(f.read_text()) if f.exists() else dict(rows=[])


@app.post("/api/jobs/stop")
def api_job_stop():
    if JOB["proc"] and JOB["proc"].poll() is None: JOB["proc"].terminate()
    return dict(ok=True)


@app.get("/api/jobs")
def api_jobs(since: int = 0):
    lines = list(JOB["log"])
    return dict(name=JOB["name"], running=bool(JOB["proc"] and JOB["proc"].poll() is None), rc=JOB["rc"], log=lines[-400:])


# ---------------------------------------------------------------- degradation and the live stress test
@app.post("/api/degrade")
async def api_degrade(req: Request):
    """{"condition": name from robust.CONDITIONS} or {"cfg": {...}}: degrade the live pose stream from now on."""
    r = await req.json()
    cfg = robust.CONDITIONS.get(r["condition"]) if r.get("condition") else r.get("cfg") or {}
    if cfg is None: return JSONResponse(dict(ok=False, msg=f"unknown condition {r['condition']}"), status_code=400)
    DEGRADER.set(cfg)
    with lock: S["degrade"] = DEGRADER.describe(); S["lost"] = 0
    return dict(ok=True, degrade=S["degrade"])


STRESS = dict(running=False, phase=-1, phases=[], expected="", msg="")


def stress_loop(names, settle, measure):
    """Cycles the injected degradations while the person keeps doing one action, and logs what each line said."""
    try:
        for i, name in enumerate(names):
            if not STRESS["running"]: break
            DEGRADER.set(robust.CONDITIONS[name])
            with lock: S["degrade"] = DEGRADER.describe(); STRESS["phase"] = i; STRESS["phases"][i]["state"] = "settling"
            t_end = time.time() + settle
            while STRESS["running"] and time.time() < t_end: time.sleep(0.1)
            with lock: STRESS["phases"][i]["state"] = "measuring"
            t_end = time.time() + measure
            while STRESS["running"] and time.time() < t_end:
                with lock:
                    ph = STRESS["phases"][i]
                    ph["main"].append(S["label"]); ph["cmp"].append(S["cmp"]["label"] if S["cmp"] else None)
                    ph["quality"].append(S["quality"].get("score", 0))
                time.sleep(args.every)
            with lock: STRESS["phases"][i]["state"] = "done"; score_stress()
    finally:
        DEGRADER.set({})
        with lock: S["degrade"] = ""; STRESS["running"] = False; STRESS["phase"] = -1; score_stress()


def score_stress():
    """Per phase: the most common answer, and how often each line gave the target (the expected action if one was
    chosen, else what the main line said most under clean input)."""
    def majority(v):
        v = [x for x in v if x in ACTIONS]       # an action, not "no clear action" or "input too degraded"
        return max(set(v), key=v.count) if v else None
    phases = STRESS["phases"]
    target = STRESS["expected"] or (majority(phases[0]["main"]) if phases and phases[0]["main"] else None)
    STRESS["target"] = target
    for ph in phases:
        for k in ("main", "cmp"):
            v = ph[k]; ph[k + "_top"] = majority(v)
            ph[k + "_hit"] = round(sum(x == target for x in v) / len(v), 2) if v and target else None
        ph["q"] = round(float(np.mean(ph["quality"]))) if ph["quality"] else None


@app.post("/api/stress")
async def api_stress(req: Request):
    r = await req.json()
    if r.get("stop"): STRESS["running"] = False; return dict(ok=True)
    if STRESS["running"]: return JSONResponse(dict(ok=False, msg="a stress test is already running"), status_code=409)
    names = [n for n in r.get("conditions") or list(robust.CONDITIONS) if n in robust.CONDITIONS]
    if names[0] != "clean": names = ["clean"] + names
    settle = float(r.get("settle", max(2.0, 0.7 * args.window))); measure = float(r.get("measure", 3.0))
    with lock:
        STRESS.update(running=True, phase=0, expected=r.get("expected") or "", settle=settle, measure=measure,
                      phases=[dict(name=n, state="waiting", main=[], cmp=[], quality=[]) for n in names])
    threading.Thread(target=stress_loop, args=(names, settle, measure), daemon=True).start()
    return dict(ok=True, seconds=round(len(names) * (settle + measure)))


# ---------------------------------------------------------------- finding the phone
def probe(ip, port, timeout=0.4):
    try:
        with socket.create_connection((ip, port), timeout=timeout): pass
    except OSError: return None
    import urllib.request
    info = dict(ip=ip, port=port, address=f"{ip}:{port}", app="")
    try:
        with urllib.request.urlopen(f"http://{ip}:{port}/status.json", timeout=1.5) as r:
            j = json.loads(r.read().decode("utf-8", "replace") or "{}")
            info["app"] = "SafeSense Cam" if "viewers" in j or "fps" in j else "camera app"; info["status"] = j
    except Exception:
        try:
            with urllib.request.urlopen(f"http://{ip}:{port}/shot.jpg", timeout=1.5) as r:
                if r.headers.get("Content-Type", "").startswith("image"): info["app"] = "IP camera"
        except Exception: pass
    return info if info["app"] else None


@app.get("/api/find_camera")
def api_find_camera(port: int = 8080):
    """Scan this laptop's local networks (/24 around each address) for a phone camera app on `port`."""
    from concurrent.futures import ThreadPoolExecutor
    ips = [i for i in local_ips() if not i.startswith("127.")]
    cands = []
    for ip in ips:
        base = ".".join(ip.split(".")[:3])
        cands += [f"{base}.{k}" for k in range(1, 255) if f"{base}.{k}" != ip]
    with ThreadPoolExecutor(64) as ex: found = [f for f in ex.map(lambda a: probe(a, port), cands) if f]
    found.sort(key=lambda f: f["app"] != "SafeSense Cam")
    return dict(found=found, laptop_ips=ips, scanned=len(cands))


@app.get("/api/ping")
def ping(): return dict(ok=True, app="safesense")


if __name__ == "__main__":
    SETTINGS.write_text(json.dumps(settings_dict(), indent=1))
    for f in (camera_loop, pose_loop, action_loop, rec_loop): threading.Thread(target=f, daemon=True).start()
    print(f"dashboard: http://localhost:{args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")
