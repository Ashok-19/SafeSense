"""Data quality, controlled degradation and the robust input pipeline for the live demo.

SafeSense is about recognising actions when the input is degraded. This module provides the three pieces the demo
needs to show that live:
  measure_quality  how good the pose stream is right now (visible joints, confidence, frame rate, stability)
  Degrader         injects the paper's degradation types into the live pose stream on demand (missing joints,
                   noise, frame loss, low frame rate), so the audience sees what happens under each
  clean_window     the robust pipeline: treats unreliable joints as missing, rejects glitches and fills short gaps
                   in time, so the model gets an honest input with missing parts marked instead of garbage
The same code runs in the dashboard (app.py) and in the offline robustness test (robustness_test.py).
"""
import numpy as np

JOINT_GROUPS = {"right arm": [8, 10], "left arm": [7, 9], "legs": [11, 12, 13, 14, 15, 16], "head": [0, 1, 2, 3, 4]}
CONF_MIN = 0.3          # joints below this pose confidence count as not seen
GAP_FILL_S = 0.6        # fill a missing joint from its neighbours in time when the gap is at most this long
GLITCH_TORSO = 0.8      # a joint more than this many torso lengths from its local median is a pose glitch

# Conditions of the live stress test and the offline robustness test (the paper's degradation categories, adapted to
# one camera: missing joints, sensor noise, frame loss, and a composite of them).
CONDITIONS = {
    "clean": {},
    "right arm hidden": dict(occlude=["right arm"]),
    "legs hidden": dict(occlude=["legs"]),
    "noise": dict(noise=0.04),
    "50% frames lost": dict(drop=0.5),
    "5 fps": dict(fps=5),
    "composite": dict(occlude=["right arm"], noise=0.03, drop=0.3),
}


class Degrader:
    """Degrades a live pose stream frame by frame. cfg: occlude=[group names], noise=fraction of body height,
    drop=probability a frame is lost, fps=frame-rate cap. Hidden joints get a low confidence and a wrong position,
    the way a pose estimator behaves when it cannot see a limb."""

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed); self.cfg = {}; self.last_kept = -1e9

    def set(self, cfg):
        self.cfg = dict(cfg or {}); self.last_kept = -1e9

    def active(self): return any(self.cfg.get(k) for k in ("occlude", "noise", "drop", "fps"))

    def describe(self):
        c, out = self.cfg, []
        if c.get("occlude"): out.append(", ".join(c["occlude"]) + " hidden")
        if c.get("noise"): out.append(f"noise {c['noise'] * 100:.0f}% of body height")
        if c.get("drop"): out.append(f"{c['drop'] * 100:.0f}% of frames lost")
        if c.get("fps"): out.append(f"{c['fps']} fps")
        return "; ".join(out)

    def __call__(self, kp, sc, t):
        """kp [P,17,2] pixels, sc [P,17] at time t -> degraded copies, or None when the frame is lost."""
        c = self.cfg
        if c.get("fps"):
            if t - self.last_kept < 1.0 / c["fps"] - 1e-3: return None
        if c.get("drop") and self.rng.random() < c["drop"]: return None
        self.last_kept = t
        if not len(kp): return kp, sc
        kp = kp.copy(); sc = sc.copy()
        for p in range(len(kp)):
            ok = sc[p] > CONF_MIN
            pts = kp[p][ok] if ok.sum() >= 2 else kp[p]
            h = max(float(np.ptp(pts[:, 1])), float(np.ptp(pts[:, 0])), 1.0)
            if c.get("noise"): kp[p] += self.rng.normal(0, c["noise"] * h, kp[p].shape)
            for g in c.get("occlude", []):
                idx = JOINT_GROUPS[g]
                lo, hi = pts.min(0), pts.max(0)
                kp[p, idx] = lo + self.rng.random((len(idx), 2)) * (hi - lo)   # where the estimator guesses
                sc[p, idx] = self.rng.uniform(0.05, 0.25, len(idx))
        return kp.astype(np.float32), sc.astype(np.float32)


def torso_length(kp, sc):
    """Median shoulder to hip distance (centres of whichever shoulders and hips are seen), else a third of the body
    height, else None. kp [n,17,2], sc [n,17]."""
    seen = sc > CONF_MIN
    s_ok, h_ok = seen[:, [5, 6]], seen[:, [11, 12]]
    ok = s_ok.any(1) & h_ok.any(1)
    if ok.any():
        sho = (kp[ok][:, [5, 6]] * s_ok[ok][..., None]).sum(1) / s_ok[ok].sum(1, keepdims=True)
        hip = (kp[ok][:, [11, 12]] * h_ok[ok][..., None]).sum(1) / h_ok[ok].sum(1, keepdims=True)
        return max(float(np.median(np.linalg.norm(sho - hip, axis=-1))), 1.0)
    if seen.sum() >= 2: return max(float(np.ptp(kp[seen][:, 1])) / 3, 1.0)
    return None


def clean_window(kp, sc, ts):
    """Robust pipeline on a tracked window: kp [M,n,17,2], sc [M,n,17], ts [n] -> cleaned copies.
    1. joints the pose model is unsure of become missing (score 0), so the model's missing-joint handling takes over
       instead of it trusting a guessed position;
    2. single-frame glitches (a joint far from its local median track) become missing;
    3. short gaps are filled by linear interpolation in time, with a reduced confidence."""
    kp = kp.copy(); sc = sc.copy()
    sc[sc < CONF_MIN] = 0
    n = len(ts)
    for m in range(kp.shape[0]):
        if not (sc[m] > 0).any(): continue
        tl = torso_length(kp[m], sc[m]) or max(float(np.ptp(kp[m][sc[m] > 0][:, 1])) / 3, 1.0)
        for v in range(17):
            seen = np.where(sc[m, :, v] > 0)[0]
            if len(seen) >= 5:          # glitch rejection against the median of the neighbouring seen frames
                p = kp[m, seen, v]
                pad = np.concatenate([np.repeat(p[:1], 2, 0), p, np.repeat(p[-1:], 2, 0)])
                med = np.median(np.lib.stride_tricks.sliding_window_view(pad, 5, axis=0), -1)
                bad = np.linalg.norm(p - med, axis=-1) > GLITCH_TORSO * tl
                sc[m, seen[bad], v] = 0
                seen = seen[~bad]
            if len(seen) < 2 or len(seen) == n: continue
            for a, b in zip(seen[:-1], seen[1:]):   # interior gaps only: nothing is invented before or after
                if b - a > 1 and ts[b] - ts[a] <= GAP_FILL_S:
                    w = ((ts[a + 1:b] - ts[a]) / (ts[b] - ts[a]))[:, None]
                    kp[m, a + 1:b, v] = (1 - w) * kp[m, a, v] + w * kp[m, b, v]
                    sc[m, a + 1:b, v] = 0.8 * min(sc[m, a, v], sc[m, b, v])
    return kp, sc


def measure_quality(items, window):
    """Live input quality over the last `window` seconds of (t, kp [P,17,2], sc [P,17]) items (after any injected
    degradation). Returns plain numbers and an overall 0..100 score with a word."""
    if not items: return dict(score=0, level="no person", visible=0.0, confidence=0.0, fps=0.0, stability=0.0, gap=0.0)
    ts = np.array([it[0] for it in items]); span = max(ts[-1] - ts[0], 1e-6)
    main = [(k[0], s[0]) for _, k, s in items if len(k)]
    if not main: return dict(score=0, level="no person", visible=0.0, confidence=0.0, fps=0.0, stability=0.0, gap=0.0)
    K = np.array([m[0] for m in main]); Sc = np.array([m[1] for m in main])
    seen = Sc > CONF_MIN
    visible = float(seen.mean())
    confidence = float(Sc[seen].mean()) if seen.any() else 0.0
    fps = (len(items) - 1) / span if len(items) > 1 else 0.0
    gap = float(np.diff(ts).max()) if len(ts) > 1 else 0.0
    # stability: frame-to-frame jitter of seen joints beyond smooth motion (second difference), in torso lengths
    tl = torso_length(K, Sc) or 1.0
    jit = []
    for v in range(17):
        idx = np.where(seen[:, v])[0]
        if len(idx) >= 3 and (np.diff(idx) == 1).all():
            d2 = K[idx[2:], v] - 2 * K[idx[1:-1], v] + K[idx[:-2], v]
            jit.append(np.median(np.linalg.norm(d2, axis=-1)) / tl)
    jitter = float(np.median(jit)) if jit else 0.0
    stability = float(np.clip(1 - jitter / 0.25, 0, 1))
    parts = [float(np.clip((visible - 0.5) / 0.45, 0, 1)), min(confidence / 0.7, 1.0), float(np.clip(fps / 12, 0, 1)), stability]
    score = 100 * (0.5 * min(parts) + 0.5 * float(np.mean(parts)))   # the weakest part counts double: one bad part is enough
    if gap > 1.0: score *= 0.7
    level = "good" if score >= 85 else "fair" if score >= 60 else "poor"
    return dict(score=round(score), level=level, visible=round(visible, 3), confidence=round(confidence, 3),
                fps=round(float(fps), 1), stability=round(stability, 3), gap=round(gap, 2))
