"""Scoring of recordings (you doing an action in front of the camera) against the model and the NTU-120 people.

Used by the dashboard (app.py) and record_actions.py. `app` is the app module, so the exact live code is used:
app.window_tensor for the model input and app.logits for the model. NTU comparisons come from ntu_reference.npz,
made once by prepare_ntu.py (dashboard: Tests > Prepare NTU reference).
"""
import json, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REC_DIR = HERE / "recordings"
RESULTS = REC_DIR / "results.json"
REF_FILE = HERE / "ntu_reference.npz"
JOINTS = ["nose", "left eye", "right eye", "left ear", "right ear", "left shoulder", "right shoulder", "left elbow", "right elbow",
          "left wrist", "right wrist", "left hip", "right hip", "left knee", "right knee", "left ankle", "right ankle"]
WIN, STEP = 3.0, 0.5
_ref = {"mtime": None, "data": None}


def rec_name(act): return act.split(" (")[0].replace("/", "_").replace(" ", "_")


def load_ref():
    """ntu_reference.npz -> dict, cached; None if it has not been made yet."""
    if not REF_FILE.exists(): return None
    m = REF_FILE.stat().st_mtime
    if _ref["mtime"] != m:
        d = np.load(REF_FILE, allow_pickle=False)
        _ref["data"] = {k: d[k] for k in d.files}; _ref["mtime"] = m
    return _ref["data"]


def reference_info():
    r = load_ref()
    if r is None: return dict(ready=False)
    return dict(ready=True, model=str(r["model"]), made=time.strftime("%Y-%m-%d %H:%M", time.localtime(float(r["time"]))),
                clips_per_action=int(np.median(r["clips"])))


def windows(app, ts, kp, sc, win=WIN, step=STEP):
    """A recording [n] / [n,2,17,2] / [n,2,17] through the live window code: a `win`-second window every `step` seconds.
    -> list of dict(end, p=softmax over all classes, motion, x=normalised window [2,48,17,5])."""
    out = []
    ends = np.arange(min(win, ts[-1]), ts[-1] + 1e-6, step) if len(ts) else []
    for end in ends:
        items = [(t, k[s.mean(1) > 0], s[s.mean(1) > 0]) for t, k, s in zip(ts, kp, sc) if end - win <= t <= end]
        view = []
        with app.model_lock:
            x, motion = app.window_tensor(items, view=view)
            if x is None: continue
            z = app.logits(x).numpy()
        p = np.exp(z - z.max()); p /= p.sum()
        out.append(dict(end=float(end), p=p, motion=motion, x=view[0]))
    return out


def score(app, act, ts, kp, sc, wh):
    """Everything the dashboard shows for one recording, with plain-language hints on what differs from NTU."""
    y = app.CLASSES.index(act); W = windows(app, ts, kp, sc)
    seen_any = (sc[:, 0].mean(1) > 0) if len(sc) else np.zeros(0, bool)
    info = dict(action=act, model=app.MODEL_NAME, time=time.time(), poses=int(len(ts)),
                pose_per_s=round(len(ts) / max(float(ts[-1]) if len(ts) else 0, 1e-6), 1) if len(ts) else 0.0,
                in_view=round(float(seen_any.mean()), 2) if len(seen_any) else 0.0, windows=len(W), hints=[])
    if not W:
        info.update(recognised=False, verdict="no person", hints=["You were not detected for long enough. Stand where the "
                    "camera sees your whole body before GO, and check the skeleton is drawn on you in the Live tab."])
        return info
    P = np.array([w["p"] for w in W]); mean = P.mean(0)
    ranks = [int((q > q[y]).sum()) + 1 for q in P]
    demo = act in app.ACTIONS
    if demo:
        idx = np.array(app.ACT_IDX); pd = P[:, idx]; demo_top1 = float(np.mean(idx[pd.argmax(1)] == y))
    else: demo_top1 = None
    s0 = sc[seen_any, 0] if seen_any.any() else sc[:, 0]; seen = s0 > 0.3
    k0 = kp[seen_any, 0] if seen_any.any() else kp[:, 0]
    h = np.array([np.ptp(k[s > 0.3, 1]) if (s > 0.3).sum() > 3 else 0 for k, s in zip(k0, s0)])
    motion = float(np.median([w["motion"] for w in W]))
    info.update(top1_share=round(float(np.mean([r == 1 for r in ranks])), 2), median_rank=int(np.median(ranks)),
                p_right=round(float(mean[y]), 3), demo_list=demo, demo_top1_share=None if demo_top1 is None else round(demo_top1, 2),
                top3=[[app.CLASSES[i], round(float(mean[i]), 3)] for i in np.argsort(-mean)[:3]],
                joints_seen=round(float(seen.sum(1).mean()), 1), mean_score=round(float(s0[s0 > 0].mean()) if (s0 > 0).any() else 0.0, 2),
                body_height=round(float(np.median(h) / max(wh[1], 1)), 2), motion=round(motion, 3),
                missing=[JOINTS[v] for v in range(17) if seen[:, v].mean() < 0.5])
    share = demo_top1 if demo_top1 is not None else info["top1_share"]
    info["recognised"] = share >= 0.5
    info["verdict"] = "recognised" if share >= 0.5 else ("sometimes" if share > 0 else "missed")
    ref = load_ref(); hints = info["hints"]
    if ref is not None and act in list(ref["classes"]) and ref["clips"][list(ref["classes"]).index(act)] > 0:
        c = list(ref["classes"]).index(act); nm = float(ref["motion"][c])
        info.update(ntu_motion=round(nm, 3), ntu_acc=round(float(ref["acc"][c]), 2))
        if nm > 0 and motion < 0.5 * nm:
            hints.append(f"You moved much less than the NTU people do for this action ({motion:.2f} vs {nm:.2f} torso lengths). "
                         "Watch the NTU example and do it as big as they do.")
        elif nm > 0 and motion > 2 * nm:
            hints.append(f"You moved much more than the NTU people ({motion:.2f} vs {nm:.2f} torso lengths). Compare with the NTU example.")
        if ref["acc"][c] < 0.6:
            hints.append(f"The model gets this action right only {ref['acc'][c]:.0%} of the time even on NTU clips, so misses are expected.")
    if info["in_view"] < 0.8: hints.append(f"You were detected in only {info['in_view']:.0%} of the recording. Be in place before GO.")
    if info["missing"]: hints.append("Joints the pose model mostly missed: " + ", ".join(info["missing"]) + ". Keep your whole body in view.")
    if info["body_height"] > 0.9: hints.append("You fill almost the whole frame height. Step back so there is space above your head and below your feet.")
    elif 0 < info["body_height"] < 0.3: hints.append("You are small in the frame. Move closer to the camera.")
    if info["pose_per_s"] and info["pose_per_s"] < 7: hints.append(f"Poses came at only {info['pose_per_s']} per second; fast actions get blurred.")
    if info["mean_score"] and info["mean_score"] < 0.5: hints.append("The pose model is unsure of your joints. Improve the lighting or avoid a busy background.")
    return info


def load_results():
    try: return json.loads(RESULTS.read_text())
    except Exception: return {}


def store(info):
    REC_DIR.mkdir(exist_ok=True); r = load_results(); r[info["action"]] = info
    RESULTS.write_text(json.dumps(r, indent=1))


def load_recording(act):
    p = REC_DIR / f"{rec_name(act)}.npz"
    if not p.exists(): return None
    d = np.load(p)
    return d["ts"], d["kp"], d["sc"], tuple(int(v) for v in d["wh"])


def score_and_store(app, act, ts, kp, sc, wh):
    info = score(app, act, ts, kp, sc, wh); store(info); return info


def rescore_all(app):
    n = 0
    for p in sorted(REC_DIR.glob("*.npz")):
        act = str(np.load(p)["action"])
        if act in app.CLASSES:
            ts, kp, sc, wh = load_recording(act); score_and_store(app, act, ts, kp, sc, wh); n += 1
    return n


def compare_view(app, act):
    """Your recording's middle window and the NTU example, as the model sees them, plus per-window guesses."""
    r = load_recording(act); out = dict(action=act, you=None, ntu=None, windows=[])
    if r is not None:
        ts, kp, sc, wh = r; W = windows(app, ts, kp, sc); y = app.CLASSES.index(act)
        if W:
            mid = W[len(W) // 2]; out["you"] = np.round(mid["x"][0][..., :3], 3).tolist()
            out["windows"] = [dict(end=round(w["end"], 1), right=round(float(w["p"][y]), 3), motion=round(w["motion"], 2),
                                   top=[[app.CLASSES[i], round(float(w["p"][i]), 3)] for i in np.argsort(-w["p"])[:3]]) for w in W]
    ref = load_ref()
    if ref is not None and act in list(ref["classes"]) and ref["clips"][list(ref["classes"]).index(act)] > 0:
        out["ntu"] = np.round(ref["view"][list(ref["classes"]).index(act)], 3).tolist()
    return out
