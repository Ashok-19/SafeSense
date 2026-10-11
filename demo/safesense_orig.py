"""Original SafeSense architecture (skeleton + IMU, factorized temporal/modality attention), on 2D COCO-17 skeletons.

Identical to the CUHK-X / UTD-MHAD SafeSense model; only the skeleton input changes: each of the 17 joints carries
(x, y, confidence) instead of (x, y, z), from the same normalisation the live demo uses (safesense_pose.normalise).
The IMU branch is kept with the CUHK-X shape (5 sensors x 6 channels) and is marked missing when there is no IMU,
so the same weights can later be fine-tuned with skeleton + IMU on CUHK-X.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from safesense_pose import COCO_EDGES, T_STEPS, pack, sample_frames, NTU120_CLASSES

J, EDGES, IMU_DEV, IMU_CH = 17, COCO_EDGES, 5, 6
SK_IN = 3 * J * 2 + 3 * len(EDGES)      # positions + first differences + bone vectors = 156
IM_IN = IMU_DEV * IMU_CH * 2              # 60


def body_window(kp, score, idx):
    """kp [M,n,17,2] pixels, score [M,n,17] -> main person's [T,17,3] (x, y, conf) and frame mask [T]."""
    x = pack(kp, score, idx)[0]                        # most confident body, normalised
    raw = np.ascontiguousarray(x[..., :3]); mask = (x[..., 2] > 0).any(-1)
    return raw.astype(np.float32), mask


def _diff(x, m):
    d = torch.zeros_like(x); d[:, 1:] = x[:, 1:] - x[:, :-1]
    ok = torch.zeros_like(m); ok[:, 1:] = m[:, 1:] & m[:, :-1]
    return d * ok.unsqueeze(-1).to(x.dtype)


def skel_features(s, m):
    """s [B,T,J,3], m [B,T] bool -> [B,T,SK_IN] (same feature recipe as the CUHK-X / UTD notebooks)."""
    B, T = s.shape[:2]; pos = s.reshape(B, T, -1)
    c, p = zip(*EDGES); bones = (s[:, :, list(c)] - s[:, :, list(p)]).reshape(B, T, -1)
    return torch.cat((pos, _diff(pos, m), bones), -1) * m.unsqueeze(-1).to(s.dtype)


def pool(seq, mask):
    valid = mask.unsqueeze(1).bool(); cnt = valid.sum(-1).clamp_min(1)
    mean = seq.masked_fill(~valid, 0).sum(-1) / cnt
    mx = seq.masked_fill(~valid, -torch.inf).amax(-1); mx = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))
    return torch.cat((mean, mx), 1)

def groups_for(c):
    g = max(1, min(16, c))
    while c % g: g -= 1
    return g

class TCNBlock(nn.Module):
    def __init__(s, c, d, p):
        super().__init__(); s.dw = nn.Conv1d(c, c, 5, padding=2 * d, dilation=d, groups=c); s.pw = nn.Conv1d(c, c, 1)
        s.n = nn.GroupNorm(groups_for(c), c); s.dp = nn.Dropout(p)
    def forward(s, x): return x + s.dp(F.gelu(s.n(s.pw(s.dw(x)))))

class TCN(nn.Module):
    def __init__(s, i, w, p=0.15, dil=(1, 2, 4)):
        super().__init__(); s.inp = nn.Conv1d(i, w, 1); s.b = nn.Sequential(*[TCNBlock(w, d, p) for d in dil])
    def forward(s, x): return s.b(s.inp(x))

class GraphBlock(nn.Module):
    def __init__(s, c, p=0.1):
        super().__init__(); s.a = nn.Linear(c, c); s.b = nn.Linear(c, c, bias=False); s.n = nn.LayerNorm(c); s.dp = nn.Dropout(p)
    def forward(s, x, A): return x + s.dp(F.gelu(s.n(s.a(x) + s.b(torch.einsum("ij,btjc->btic", A, x)))))

class SkeletonGraph(nn.Module):
    def __init__(s, J, edges):
        super().__init__(); A = torch.eye(J)
        for c, p in edges: A[c, p] = A[p, c] = 1
        d = A.sum(1).rsqrt(); s.register_buffer("A", d[:, None] * A * d[None]); s.J = J
        s.inp = nn.Sequential(nn.Linear(6, 96), nn.LayerNorm(96), nn.GELU())
        s.blocks = nn.ModuleList([GraphBlock(96) for _ in range(3)]); s.att = nn.Linear(96, 1); s.t = TCN(96, 96)
    def forward(s, x, m):
        B, T_, _ = x.shape; J = s.J
        pos = x[:, :, :3 * J].reshape(B, T_, J, 3); mot = x[:, :, 3 * J:6 * J].reshape(B, T_, J, 3)
        n = s.inp(torch.cat((pos, mot), -1))
        for b in s.blocks: n = b(n, s.A)
        w = torch.softmax(s.att(n).squeeze(-1), 2)
        return s.t(((n * w.unsqueeze(-1)).sum(2) * m.unsqueeze(-1)).transpose(1, 2))

class Reducer(nn.Module):
    def __init__(s, c):
        super().__init__(); s.steps = nn.ModuleList([nn.Sequential(nn.Conv1d(c, c, 5, stride=2, padding=2, groups=c), nn.Conv1d(c, c, 1), nn.GroupNorm(groups_for(c), c), nn.GELU()) for _ in range(2)])
    def forward(s, x, m):
        v = m.float()
        for st in s.steps:
            x = st(x); v = (F.max_pool1d(v.unsqueeze(1), 3, stride=2, padding=1).squeeze(1) > 0).float(); x = x * v.unsqueeze(1)
        return x, v

class FactorizedBlock(nn.Module):
    def __init__(s, w=96, h=4, p=0.1):
        super().__init__()
        s.tn = nn.LayerNorm(w); s.ta = nn.MultiheadAttention(w, h, dropout=p, batch_first=True)
        s.mn = nn.LayerNorm(w); s.ma = nn.MultiheadAttention(w, h, dropout=p, batch_first=True)
        s.fn = nn.LayerNorm(w); s.ff = nn.Sequential(nn.Linear(w, 2 * w), nn.GELU(), nn.Dropout(p), nn.Linear(2 * w, w), nn.Dropout(p)); s.dp = nn.Dropout(p)
    @staticmethod
    def safe(v):
        v = v.clone(); miss = ~v.any(1); v[miss, 0] = True; return v
    def forward(s, tok, valid):
        B, M, S, W = tok.shape
        t = tok.reshape(B * M, S, W); tv = valid.reshape(B * M, S); n = s.tn(t)
        u, _ = s.ta(n, n, n, key_padding_mask=~s.safe(tv), need_weights=False); t = (t + s.dp(u)) * tv.unsqueeze(-1)
        tok = t.reshape(B, M, S, W)
        m = tok.permute(0, 2, 1, 3).reshape(B * S, M, W); mv = valid.permute(0, 2, 1).reshape(B * S, M); n = s.mn(m)
        u, _ = s.ma(n, n, n, key_padding_mask=~s.safe(mv), need_weights=False); m = (m + s.dp(u)) * mv.unsqueeze(-1)
        tok = m.reshape(B, S, M, W).permute(0, 2, 1, 3)
        return (tok + s.ff(s.fn(tok))) * valid.unsqueeze(-1)

class SafeSense(nn.Module):
    """The original SafeSense network (unchanged from the CUHK-X notebook's SafeSenseCUHK)."""
    def __init__(s, sk_in, im_in, J, edges, nc, width=96, blocks=2):
        super().__init__()
        s.sk_flat = TCN(sk_in, 64); s.sk_graph = SkeletonGraph(J, edges); s.sk_fuse = nn.Conv1d(160, width, 1)
        s.imu = TCN(im_in, 64); s.imu_proj = nn.Conv1d(64, width, 1)
        s.sk_red = Reducer(width); s.im_red = Reducer(width)
        s.blocks = nn.ModuleList([FactorizedBlock(width) for _ in range(blocks)])
        s.cls = nn.Sequential(nn.LayerNorm(2 * width), nn.Linear(2 * width, nc))
        s.sk_exp = nn.Sequential(nn.LayerNorm(2 * width), nn.Linear(2 * width, nc)); s.im_exp = nn.Sequential(nn.LayerNorm(2 * width), nn.Linear(2 * width, nc))
    def forward(s, sk, im, sm, imk):
        avail = torch.stack((sm.any(1), imk.any(1)), 1).float()
        sk = sk * sm.unsqueeze(-1); im = im * imk.unsqueeze(-1)
        a = s.sk_fuse(torch.cat((s.sk_flat(sk.transpose(1, 2)), s.sk_graph(sk, sm.float())), 1))
        b = s.imu_proj(s.imu(im.transpose(1, 2)))
        a, sv = s.sk_red(a, sm); b, iv = s.im_red(b, imk)
        sv = sv * avail[:, :1]; iv = iv * avail[:, 1:]
        tok = torch.stack((a.transpose(1, 2), b.transpose(1, 2)), 1); valid = torch.stack((sv.bool(), iv.bool()), 1)
        for bl in s.blocks: tok = bl(tok, valid)
        se, ie = pool(tok[:, 0].transpose(1, 2), sv), pool(tok[:, 1].transpose(1, 2), iv)
        cnt = valid.float().sum(1).clamp_min(1).unsqueeze(-1)
        fused = (tok * valid.unsqueeze(-1)).sum(1) / cnt
        logits = s.cls(pool(fused.transpose(1, 2), valid.any(1).float()))
        extras = dict(sk_logits=s.sk_exp(se) * avail[:, :1], im_logits=s.im_exp(ie) * avail[:, 1:],
                      sk_emb=se[:, :se.shape[1] // 2], im_emb=ie[:, :ie.shape[1] // 2], avail=avail)
        return logits, extras


class SafeSenseCOCO(nn.Module):
    """SafeSense plus its feature normalisation, taking raw windows: raw [B,T,17,3], mask [B,T] (and optional IMU)."""
    def __init__(s, nc):
        super().__init__(); s.net = SafeSense(SK_IN, IM_IN, J, EDGES, nc)
        s.register_buffer("s_mu", torch.zeros(SK_IN)); s.register_buffer("s_sd", torch.ones(SK_IN))
    def forward(s, raw, mask, imu=None, imask=None):
        B, T = raw.shape[:2]
        sk = (skel_features(raw, mask) - s.s_mu) / s.s_sd * mask.unsqueeze(-1).to(raw.dtype)
        if imu is None:
            imu = raw.new_zeros(B, T, IM_IN); imask = torch.zeros(B, T, dtype=torch.bool, device=raw.device)
        return s.net(sk, imu, mask, imask)
