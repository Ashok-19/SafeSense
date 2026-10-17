"""PYSKL ST-GCN++ (Duan et al., "PYSKL: Towards Good Practices for Skeleton Action Recognition", ACM MM 2022),
re-implemented in plain PyTorch so the official NTU-120 weights load without mmcv / mmaction.

Model: configs/stgcn++/stgcn++_ntu120_xsub_hrnet/{j,b}.py (COCO-17 2D HRNet keypoints, 120 classes).
Weights (Apache-2.0, from the PYSKL model zoo):
  joint  http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu120_xsub_hrnet/j.pth  (84.4% xsub)
  bone   http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu120_xsub_hrnet/b.pth  (84.8% xsub)
  joint + bone ensemble 86.4% xsub (PYSKL README, 10-clip testing).
Preprocessing follows the PYSKL pipeline: PreNormalize2D (image centre, 1920x1080 scale), GenSkeFeat (j or b),
100-frame clips at the video frame rate (30 fps on NTU), 2 people, zero padding.
"""
import copy as cp
import numpy as np, torch, torch.nn as nn

CLIP_LEN, FPS, NUM_PERSON, EPS = 100, 30.0, 2, 1e-4
COCO_INWARD = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 5), (12, 6), (9, 7), (7, 5), (10, 8), (8, 6), (5, 0), (6, 0),
               (1, 0), (3, 1), (2, 0), (4, 2)]
BONE_PAIRS = ((0, 0), (1, 0), (2, 0), (3, 1), (4, 2), (5, 0), (6, 0), (7, 5), (8, 6), (9, 7), (10, 8),
              (11, 0), (12, 0), (13, 11), (14, 12), (15, 13), (16, 14))


def _edge2mat(link, n):
    A = np.zeros((n, n))
    for i, j in link: A[j, i] = 1
    return A

def _normalize_digraph(A, dim=0):
    Dl = A.sum(dim); Dn = np.zeros_like(A)
    for i in range(A.shape[1]):
        if Dl[i] > 0: Dn[i, i] = Dl[i] ** -1
    return A @ Dn

def coco_spatial_graph():
    n = 17
    I = _edge2mat([(i, i) for i in range(n)], n)
    In = _normalize_digraph(_edge2mat(COCO_INWARD, n))
    Out = _normalize_digraph(_edge2mat([(j, i) for i, j in COCO_INWARD], n))
    return np.stack((I, In, Out))


class unit_gcn(nn.Module):
    def __init__(s, cin, cout, A, adaptive="init", with_res=True):
        super().__init__()
        s.num_subsets = A.size(0); s.adaptive = adaptive; s.with_res = with_res
        s.bn = nn.BatchNorm2d(cout); s.act = nn.ReLU()
        if adaptive == "init": s.A = nn.Parameter(A.clone())
        else: s.register_buffer("A", A)
        s.conv = nn.Conv2d(cin, cout * A.size(0), 1)
        if with_res:
            s.down = nn.Sequential(nn.Conv2d(cin, cout, 1), nn.BatchNorm2d(cout)) if cin != cout else (lambda x: x)
    def forward(s, x):
        n, c, t, v = x.shape
        res = s.down(x) if s.with_res else 0
        x = s.conv(x).view(n, s.num_subsets, -1, t, v)
        x = torch.einsum("nkctv,kvw->nctw", x, s.A).contiguous()
        return s.act(s.bn(x) + res)


class unit_tcn(nn.Module):
    def __init__(s, cin, cout, kernel_size=9, stride=1, dilation=1, norm=True, dropout=0):
        super().__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        s.conv = nn.Conv2d(cin, cout, (kernel_size, 1), padding=(pad, 0), stride=(stride, 1), dilation=(dilation, 1))
        s.bn = nn.BatchNorm2d(cout) if norm else nn.Identity(); s.drop = nn.Dropout(dropout)
    def forward(s, x): return s.drop(s.bn(s.conv(x)))


class mstcn(nn.Module):
    def __init__(s, cin, cout, stride=1, dropout=0., ms_cfg=((3, 1), (3, 2), (3, 3), (3, 4), ("max", 3), "1x1")):
        super().__init__()
        nb = len(ms_cfg); mid = cout // nb; rem = cout - mid * (nb - 1); act = nn.ReLU()
        br = []
        for i, cfg in enumerate(ms_cfg):
            bc = rem if i == 0 else mid
            if cfg == "1x1": br.append(nn.Conv2d(cin, bc, 1, stride=(stride, 1))); continue
            if cfg[0] == "max":
                br.append(nn.Sequential(nn.Conv2d(cin, bc, 1), nn.BatchNorm2d(bc), act, nn.MaxPool2d((cfg[1], 1), stride=(stride, 1), padding=(1, 0)))); continue
            br.append(nn.Sequential(nn.Conv2d(cin, bc, 1), nn.BatchNorm2d(bc), act, unit_tcn(bc, bc, cfg[0], stride, cfg[1], norm=False)))
        s.branches = nn.ModuleList(br); tin = mid * (nb - 1) + rem
        s.transform = nn.Sequential(nn.BatchNorm2d(tin), act, nn.Conv2d(tin, cout, 1))
        s.bn = nn.BatchNorm2d(cout); s.drop = nn.Dropout(dropout)
    def forward(s, x):
        return s.drop(s.bn(s.transform(torch.cat([b(x) for b in s.branches], 1))))


class STGCNBlock(nn.Module):
    def __init__(s, cin, cout, A, stride=1, residual=True):
        super().__init__()
        s.gcn = unit_gcn(cin, cout, A); s.tcn = mstcn(cout, cout, stride=stride); s.relu = nn.ReLU()
        if not residual: s.residual = lambda x: 0
        elif cin == cout and stride == 1: s.residual = lambda x: x
        else: s.residual = unit_tcn(cin, cout, kernel_size=1, stride=stride)
    def forward(s, x): return s.relu(s.tcn(s.gcn(x)) + s.residual(x))


class STGCN(nn.Module):
    def __init__(s, in_channels=3, base_channels=64, ch_ratio=2, num_stages=10, inflate_stages=(5, 8), down_stages=(5, 8)):
        super().__init__()
        A = torch.tensor(coco_spatial_graph(), dtype=torch.float32)
        s.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        mods = [STGCNBlock(in_channels, base_channels, A.clone(), 1, residual=False)]
        inflate = 0
        for i in range(2, num_stages + 1):
            stride = 1 + (i in down_stages); cin = base_channels
            if i in inflate_stages: inflate += 1
            cout = int(64 * ch_ratio ** inflate + EPS); base_channels = cout
            mods.append(STGCNBlock(cin, cout, A.clone(), stride))
        s.gcn = nn.ModuleList(mods)
    def forward(s, x):                       # x [N, M, T, V, C]
        N, M, T, V, C = x.size()
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = s.data_bn(x.view(N * M, V * C, T))
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        for g in s.gcn: x = g(x)
        return x.reshape((N, M) + x.shape[1:])


class GCNHead(nn.Module):
    def __init__(s, num_classes=120, in_channels=256):
        super().__init__(); s.fc_cls = nn.Linear(in_channels, num_classes)
    def forward(s, x):                       # [N, M, C, T, V] -> [N, classes]
        N, M, C, T, V = x.shape
        return s.fc_cls(x.reshape(N, M, C, -1).mean(-1).mean(1))


class STGCNpp(nn.Module):
    """RecognizerGCN(backbone=STGCN(gcn_adaptive='init', gcn_with_res=True, tcn_type='mstcn'), cls_head=GCNHead)."""
    def __init__(s, num_classes=120):
        super().__init__(); s.backbone = STGCN(); s.cls_head = GCNHead(num_classes)
    def forward(s, x): return s.cls_head(s.backbone(x))


def load_pyskl(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck)
    m = STGCNpp(sd["cls_head.fc_cls.weight"].shape[0])
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"{path}: weights do not match ST-GCN++ (missing {missing[:5]}, unexpected {unexpected[:5]})")
    return m.eval()


def pyskl_input(kp, score, img_wh, feat="j"):
    """kp [M,T,17,2] pixels, score [M,T,17], frames already at ~30 fps -> [1, 2, 100, 17, 3] in PYSKL's format.

    PreNormalize2D on NTU uses the 1920x1080 frame: x and y are centred on the image and divided by half its
    width / height. A live frame of another aspect ratio is mapped as if it were a 16:9 frame of the same height,
    so a body keeps its NTU proportions.
    """
    w, h = img_wh
    kp = kp.astype(np.float32).copy(); score = score.astype(np.float32)
    kp[..., 0] = (kp[..., 0] - w / 2) / (h * 16 / 9 / 2)
    kp[..., 1] = (kp[..., 1] - h / 2) / (h / 2)
    out = (score <= 0.01)
    kp[out] = 0
    x = np.concatenate([kp, score[..., None]], -1)                          # j: x, y, score
    if feat == "b":
        b = np.zeros_like(x)
        for v1, v2 in BONE_PAIRS:
            b[..., v1, :2] = x[..., v1, :2] - x[..., v2, :2]; b[..., v1, 2] = (x[..., v1, 2] + x[..., v2, 2]) / 2
        x = b
    M, T = x.shape[:2]
    idx = np.mod(np.arange(CLIP_LEN), T)                                     # PYSKL test sampling of a short clip: loop it
    x = x[:, idx]
    if M < NUM_PERSON: x = np.concatenate([x, np.zeros((NUM_PERSON - M,) + x.shape[1:], np.float32)])
    return torch.from_numpy(np.ascontiguousarray(x[:NUM_PERSON]))[None]
