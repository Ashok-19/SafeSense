import numpy as np, pickle, sys
W = sys.argv[1]
d = np.load(f'{W}/prep_T32.npz'); N, T = d['skel'].shape[:2]
res = pickle.load(open(f'{W}/raw.pkl', 'rb')); acts = {}
for (a, u, tr), _, _ in res: acts[int(a.split('_')[0])] = a
idev = d['imask']; user = np.array([f'user{u}' for u in d['user']])
np.savez_compressed(f'{W}/cuhkx_prepared_v2.npz', skel=d['skel'].astype(np.float16), skel_mask=np.repeat((d['nfr'] > 0)[:, None], T, 1),
    imu=d['imu'][..., :6].astype(np.float32), imu_mask=np.repeat(idev.any(1)[:, None], T, 1), imu_dev=idev, label=d['label'], user=user,
    trial=d['trial'], split=np.where(np.isin(d['user'], [8, 9, 23, 24]), 'test', 'train'), actions=np.array([acts[i] for i in range(40)]),
    devices=np.array(['WTLA', 'WTRA', 'WTC', 'WTLL', 'WTRL']), dur=d['dur'])
print('saved', N, T)
