"""Robustness test: how the live pipeline holds up when the input degrades, measured on real NTU-120 test clips.

Each test clip (people never seen in training) is replayed at a live pose rate through the demo's own code, under each
degradation of the live stress test (robust.CONDITIONS: hidden arm, hidden legs, noise, lost frames, low frame rate,
a composite of them). Every clip is classified with the robust pipeline on and off, by the main model and, if given,
a second model. The table it prints, and saves for the dashboard, is what the live demo's claims rest on.

  python robustness_test.py --pkl data/ntu120_hrnet.pkl --checkpoint safesense_pose_ntu120.pt --n 600
"""
import argparse, sys, pickle, time, json, collections
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--pkl", required=True); ap.add_argument("--checkpoint", default="safesense_pose_ntu120.pt")
ap.add_argument("--compare", default="", help="second model file, or stgcnpp, run on exactly the same degraded input")
ap.add_argument("--fps", type=float, default=10, help="simulated pose rate on the laptop")
ap.add_argument("--n", type=int, default=600, help="test clips (0 = all)")
ap.add_argument("--window", type=float, default=3.0)
ap.add_argument("--conditions", default="", help="comma-separated names from robust.CONDITIONS (default: all)")
ap.add_argument("--out", default="recordings/robustness.json")
a = ap.parse_args()

sys.argv = ["app.py", "--checkpoint", a.checkpoint, "--actions", "all", "--window", str(a.window)] + \
           (["--stgcnpp"] if a.checkpoint == "stgcnpp" else [])
import app, robust                                    # the demo's own code, without starting its threads
from pathlib import Path

conds = [c.strip() for c in a.conditions.split(",") if c.strip()] or list(robust.CONDITIONS)
slots = [("main", app.MAIN)]
if a.compare:
    slots.append(("compare", app.build_model(a.compare)))
data = pickle.load(open(a.pkl, "rb"))
ann = {x["frame_dir"]: x for x in data["annotations"]}
ids = [k for k in data["split"]["xsub_val"] if k in ann]
rng = np.random.default_rng(0)
if a.n: ids = list(rng.choice(ids, min(a.n, len(ids)), replace=False))
C = app.CLASSES
strong = [C.index(x) for x in app.STRONG]
names = {"main": app.MAIN["name"]}
if a.compare: names["compare"] = slots[1][1]["name"]

# hits[(condition, model, robust)] = [all-120 hits, default-list hits, default-list clips, abstained]
hits = collections.defaultdict(lambda: [0, 0, 0, 0]); t0 = time.time()
for k, fid in enumerate(ids):
    x = ann[fid]; kp = np.asarray(x["keypoint"], np.float32); sc = np.asarray(x["keypoint_score"], np.float32)
    if kp.ndim == 3: kp, sc = kp[None], sc[None]
    h, w = x.get("img_shape", (1080, 1920)); app.S["frame_wh"] = (w, h)
    T = kp.shape[1]; y = int(x["label"])
    for ci, cname in enumerate(conds):
        deg = robust.Degrader(seed=k * 100 + ci); deg.set(robust.CONDITIONS[cname])
        items = []
        for t in np.arange(0, T / 30.0, 1 / a.fps):
            f = min(int(round(t * 30)), T - 1)
            d = deg(*app.select_people(kp[:, f], sc[:, f]), t)
            if d is not None: items.append((t, *d))
        items = [it for it in items if it[0] >= items[-1][0] - a.window] if items else items
        q = robust.measure_quality(items, a.window)
        for rob in (True, False):
            xs, _ = app.window_inputs(items, [s for _, s in slots], robust_on=rob)
            for (sname, sl), xi in zip(slots, xs or [None] * len(slots)):
                r = hits[(cname, sname, rob)]
                if xi is None: r[3] += 1; r[2] += y in strong; continue
                z = app.logits(xi, sl)
                abstain = rob and (q["visible"] < 0.35 or q["fps"] < 2)
                r[3] += abstain
                r[0] += int(z.argmax()) == y and not abstain
                if y in strong:
                    r[2] += 1; r[1] += strong[int(z[strong].argmax())] == y and not abstain
    if (k + 1) % 50 == 0:
        print(f"{k + 1}/{len(ids)} clips  {time.time() - t0:.0f}s", flush=True)

n = len(ids); rows = []
print(f"\n{n} NTU-120 test clips replayed at {a.fps:g} fps. Accuracy on the demo's default action list "
      f"({len(strong)} actions); all-120 accuracy in brackets.\n")
head = f"{'condition':20s}" + "".join(f"{nm[:28] + (' +pipeline' if rob else ' raw'):>40s}" for nm in names.values() for rob in (True, False))
print(head)
for cname in conds:
    line = f"{cname:20s}"
    for sname in names:
        for rob in (True, False):
            r = hits[(cname, sname, rob)]
            acc, acc120 = float(r[1] / max(r[2], 1)), float(r[0] / n)
            line += f"{acc:>30.3f} ({acc120:.3f})  "
            rows.append(dict(condition=cname, model=names[sname], slot=sname, robust=rob, acc=round(acc, 4),
                             acc120=round(acc120, 4), clips=int(r[2]), abstained=int(r[3])))
    print(line)
out = Path(app.HERE / a.out); out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(dict(when=time.strftime("%Y-%m-%d %H:%M"), clips=n, fps=a.fps, window=a.window, pkl=a.pkl,
                               actions=len(strong), models=names, rows=rows), indent=1))
print(f"\nsaved {out} (the dashboard shows it under Tests and on the demo screen)")
