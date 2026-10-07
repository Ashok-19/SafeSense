import json, glob, os, re, pickle, numpy as np, pandas as pd
from multiprocessing import Pool
R = '/home/nnmax/Desktop/final-year-project/cuhk-training/cuhkx_skel_imu/cuhk-data-extract/cuhkx_skel_imu/HAR/data'
EPOCH = pd.Timestamp('1970-01-01')
def ts(s): return (pd.Timestamp(s) - EPOCH).total_seconds()
def load_sk(d):
    out = []
    for f in sorted(glob.glob(d + '/predictions/*.json')):
        m = re.match(r'Color_(\d{4}-\d\d-\d\d)_(\d\d)-(\d\d)-(\d\d\.\d+)_(\d+)\.json', os.path.basename(f))
        if m: t = ts(f'{m[1]} {m[2]}:{m[3]}:{m[4]}'); idx = int(m[5])
        else: t = np.nan; idx = int(re.findall(r'(\d+)\.json', f)[0])
        j = json.load(open(f))
        P = np.array([p['keypoints'] for p in j], np.float32) if j else np.zeros((0, 17, 3), np.float32)
        out.append((idx, t, P))
    out.sort(key=lambda r: r[0])
    return out
COLS = ['加速度X(g)','加速度Y(g)','加速度Z(g)','角速度X(°/s)','角速度Y(°/s)','角速度Z(°/s)','角度X(°)','角度Y(°)','角度Z(°)','磁场X(uT)','磁场Y(uT)','磁场Z(uT)','四元数0()','四元数1()','四元数2()','四元数3()']
def load_imu(d):
    res = {}
    for f in glob.glob(d + '/*.csv'):
        df = pd.read_csv(f)
        df['dev'] = df['设备名称'].str.extract(r'^(WT\w+?)\(')[0]
        df['t'] = (pd.to_datetime(df['时间']) - EPOCH).dt.total_seconds()
        for dev, g in df.groupby('dev'):
            g = g.sort_values('t').drop_duplicates('t')
            res[dev] = (g['t'].values, g[COLS].values.astype(np.float32))
    return res
def work(key):
    a, u, tr = key
    sd, idir = f'{R}/Skeleton/{a}/{u}/{tr}', f'{R}/IMU/{a}/{u}/{tr}'
    return key, (load_sk(sd) if os.path.isdir(sd) else []), (load_imu(idir) if os.path.isdir(idir) else {})
if __name__ == '__main__':
    keys = set()
    for mod in ('Skeleton', 'IMU'):
        for d in glob.glob(f'{R}/{mod}/*/*/*'): keys.add(tuple(d.split('/')[-3:]))
    keys = sorted(keys); print(len(keys))
    with Pool(4) as p: res = p.map(work, keys, chunksize=16)
    pickle.dump(res, open('/home/nnmax/Desktop/final-year-project/cuhk-training/v2/raw.pkl', 'wb'))
    devs = {}; 
    for k, s, i in res:
        for d in i: devs[d] = devs.get(d, 0) + 1
    print(devs)
