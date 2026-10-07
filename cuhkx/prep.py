import pickle, numpy as np, sys
T = int(sys.argv[1]) if len(sys.argv) > 1 else 32
res = pickle.load(open('/home/nnmax/Desktop/final-year-project/cuhk-training/v2/raw.pkl', 'rb'))
DEVS = ['WTLA', 'WTRA', 'WTC', 'WTLL', 'WTRL']
def pick_people(sk):
    # choose one person per frame: the one closest in pose to a running reference (single-person frames seed it)
    singles = [P[0] for _, _, P in sk if len(P) == 1]
    ref = np.median(np.stack(singles), 0) if singles else None
    out = []; nmulti = 0
    for idx, t, P in sk:
        if len(P) == 0: continue
        if len(P) > 1:
            nmulti += 1
            if ref is not None: P = P[np.argsort([np.abs(p - ref).mean() for p in P])]
        out.append((idx, t, P[0])); ref = P[0] if ref is None else 0.7 * ref + 0.3 * P[0]
    return out, nmulti
def interp(t, y, g):
    y = y.reshape(len(t), -1)
    return np.stack([np.interp(g, t, y[:, c]) for c in range(y.shape[1])], 1)
N = len(res); skel = np.zeros((N, T, 17, 3), np.float32); imu = np.zeros((N, T, 5, 16), np.float32)
imask = np.zeros((N, 5), bool); dur = np.zeros(N, np.float32); nfr = np.zeros(N, np.int32); multi = 0
labels, users, trials = [], [], []
for n, ((a, u, tr), sk, im) in enumerate(res):
    labels.append(int(a.split('_')[0])); users.append(int(u[4:])); trials.append(tr)
    fr, m = pick_people(sk); multi += m; nfr[n] = len(fr)
    if fr:
        idx = np.array([f[0] for f in fr], float); ts = np.array([f[1] for f in fr], float)
        if np.isfinite(ts).all() and len(ts) > 1 and ts.max() > ts.min(): tt = ts - ts.min()
        else: tt = (idx - idx.min()) * 0.1
        X = np.stack([f[2] for f in fr]); t0 = ts.min() if np.isfinite(ts).all() else None
        span = max(tt.max(), 1e-3); g = np.linspace(0, span, T); dur[n] = tt.max()
        skel[n] = interp(tt, X, g).reshape(T, 17, 3) if len(tt) > 1 else np.repeat(X, T, 0)
    else: t0 = None
    for k, d in enumerate(DEVS):
        if d not in im or len(im[d][0]) < 2: continue
        t, v = im[d]
        if t0 is not None: tg = t0 + np.linspace(0, dur[n], T)
        else: tg = np.linspace(t.min(), t.max(), T)
        # require overlap; otherwise stretch IMU over the clip
        if t0 is None or (min(tg[-1], t.max()) - max(tg[0], t.min())) < 0.5 * max(dur[n], 0.1): tg = np.linspace(t.min(), t.max(), T)
        imu[n, :, k] = interp(t, v, tg); imask[n, k] = True
print('multi-person frames', multi)
np.savez_compressed(f'/home/nnmax/Desktop/final-year-project/cuhk-training/v2/prep_T{T}.npz', skel=skel, imu=imu, imask=imask, dur=dur, nfr=nfr, label=np.array(labels), user=np.array(users), trial=np.array(trials))
print('saved', N, 'T', T, 'imu coverage', imask.mean(0))
