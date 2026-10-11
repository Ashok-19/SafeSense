"""SafeSense-Pose: skeleton-only action model for 2D COCO-17 keypoints.

Shared by the training notebook and the live demo, so both use exactly the same
preprocessing and network. Input to the model: [B, M, T, V, C] with M=2 bodies,
T=48 frames, V=17 COCO joints, C=5 channels (x, y, score, dx, dy).
"""
import numpy as np, torch, torch.nn as nn

T_STEPS, MAX_BODIES, V = 48, 2, 17
# COCO-17: 0 nose, 1-2 eyes, 3-4 ears, 5-6 shoulders, 7-8 elbows, 9-10 wrists, 11-12 hips, 13-14 knees, 15-16 ankles
COCO_EDGES = [(0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 7), (7, 9), (6, 8), (8, 10),
              (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16), (5, 6), (11, 12)]


def sample_frames(n, t=T_STEPS, rng=None):
    """Indices of t frames spread evenly over n frames; random offset per segment when rng is given."""
    if n <= 0: return np.zeros(t, int)
    edges = np.linspace(0, n, t + 1)
    if rng is None: idx = (edges[:-1] + edges[1:]) / 2
    else: idx = edges[:-1] + rng.random(t) * np.maximum(edges[1:] - edges[:-1], 1e-6)
    return np.clip(np.floor(idx).astype(int), 0, n - 1)


def normalise(kp, score):
    """kp [M,T,V,2] pixels, score [M,T,V] -> [M,T,V,5] centred on the main body's hips and scaled by its torso.

    The same function runs on NTU clips and on live camera windows, so camera distance, framing and
    image size stop mattering. Missing joints (score 0) stay at zero.
    """
    kp = kp.astype(np.float32); score = score.astype(np.float32)
    ok = score > 0.05
    hip = (kp[0, :, 11] + kp[0, :, 12]) / 2; sho = (kp[0, :, 5] + kp[0, :, 6]) / 2
    good = ok[0, :, [5, 6, 11, 12]].all(0)
    if good.any():
        centre = hip[good].mean(0); scale = np.linalg.norm(sho[good] - hip[good], axis=-1).mean()
    else:  # fall back to the bounding box of whatever joints were seen
        pts = kp[ok] if ok.any() else kp.reshape(-1, 2)
        centre = pts.mean(0); scale = max(np.ptp(pts[:, 1]) / 3, 1.0)
    xy = (kp - centre) / max(scale, 1e-3) * ok[..., None]
    vel = np.diff(xy, axis=1, prepend=xy[:, :1]) * (ok & np.roll(ok, 1, axis=1))[..., None]
    return np.concatenate([xy, score[..., None] * ok[..., None], vel], -1)


def pack(kp, score, idx):
    """Pick frames, keep the 2 most confident bodies, pad to MAX_BODIES, normalise -> [M,T,V,5] float32."""
    kp, score = kp[:, idx], score[:, idx]
    order = np.argsort(-score.mean((1, 2)))[:MAX_BODIES]; kp, score = kp[order], score[order]
    if kp.shape[0] < MAX_BODIES:
        pad = MAX_BODIES - kp.shape[0]
        kp = np.concatenate([kp, np.zeros((pad,) + kp.shape[1:], kp.dtype)])
        score = np.concatenate([score, np.zeros((pad,) + score.shape[1:], score.dtype)])
    return normalise(kp, score)


def adjacency():
    A = np.eye(V, dtype=np.float32)
    for i, j in COCO_EDGES: A[i, j] = A[j, i] = 1
    d = A.sum(1) ** -0.5
    return torch.tensor(d[:, None] * A * d[None])


class GraphBlock(nn.Module):
    """Spatial graph convolution over the body (fixed skeleton + learned links) followed by a temporal convolution."""
    def __init__(s, cin, cout, stride=1, drop=0.1):
        super().__init__()
        s.register_buffer("A", adjacency()); s.B = nn.Parameter(torch.zeros(V, V))
        s.gcn = nn.Conv2d(cin, cout, 1); s.self_ = nn.Conv2d(cin, cout, 1); s.bn1 = nn.BatchNorm2d(cout)
        s.tcn = nn.Sequential(nn.Conv2d(cout, cout, (9, 1), (stride, 1), (4, 0)), nn.BatchNorm2d(cout), nn.Dropout(drop))
        s.res = nn.Identity() if (cin == cout and stride == 1) else nn.Sequential(nn.Conv2d(cin, cout, 1, (stride, 1)), nn.BatchNorm2d(cout))
        s.act = nn.ReLU(inplace=True)
    def forward(s, x):  # x [N,C,T,V]
        h = torch.einsum("nctv,vw->nctw", s.gcn(x), s.A + s.B) + s.self_(x)
        return s.act(s.tcn(s.act(s.bn1(h))) + s.res(x))


class SafeSensePose(nn.Module):
    def __init__(s, num_classes, cin=5, widths=(64, 64, 64, 128, 128, 128), drop=0.1):
        super().__init__()
        s.data_bn = nn.BatchNorm1d(cin * V)
        layers, c = [], cin
        for k, w in enumerate(widths):
            layers.append(GraphBlock(c, w, stride=2 if k == 3 else 1, drop=drop)); c = w
        s.blocks = nn.Sequential(*layers); s.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(c, num_classes))
    def forward(s, x):  # x [B,M,T,V,C]
        B, M, Tt, Vv, C = x.shape
        present = (x[..., 2].abs().sum((2, 3)) > 0).float()                       # [B,M] body seen at all
        h = x.permute(0, 1, 4, 3, 2).reshape(B * M, C * Vv, Tt)
        h = s.data_bn(h).reshape(B * M, C, Vv, Tt).permute(0, 1, 3, 2)          # [BM,C,T,V]
        h = s.blocks(h).mean((2, 3)).reshape(B, M, -1)
        w = present / present.sum(1, keepdim=True).clamp_min(1)
        return s.head((h * w[..., None]).sum(1))


NTU120_CLASSES = [
    "drink water", "eat meal/snack", "brushing teeth", "brushing hair", "drop", "pickup", "throw", "sitting down",
    "standing up", "clapping", "reading", "writing", "tear up paper", "wear jacket", "take off jacket", "wear a shoe",
    "take off a shoe", "wear on glasses", "take off glasses", "put on a hat/cap", "take off a hat/cap", "cheer up",
    "hand waving", "kicking something", "reach into pocket", "hopping (one foot jumping)", "jump up",
    "make a phone call/answer phone", "playing with phone/tablet", "typing on a keyboard", "pointing to something with finger",
    "taking a selfie", "check time (from watch)", "rub two hands together", "nod head/bow", "shake head", "wipe face",
    "salute", "put the palms together", "cross hands in front (say stop)", "sneeze/cough", "staggering", "falling",
    "touch head (headache)", "touch chest (stomachache/heart pain)", "touch back (backache)", "touch neck (neckache)",
    "nausea or vomiting condition", "use a fan (with hand or paper)/feeling warm", "punching/slapping other person",
    "kicking other person", "pushing other person", "pat on back of other person", "point finger at the other person",
    "hugging other person", "giving something to other person", "touch other person's pocket", "handshaking",
    "walking towards each other", "walking apart from each other", "put on headphone", "take off headphone",
    "shoot at the basket", "bounce ball", "tennis bat swing", "juggling table tennis balls", "hush (quite)", "flick hair",
    "thumb up", "thumb down", "make ok sign", "make victory sign", "staple book", "counting money", "cutting nails",
    "cutting paper (using scissors)", "snapping fingers", "open bottle", "sniff (smell)", "squat down", "toss a coin",
    "fold paper", "ball up paper", "play magic cube", "apply cream on face", "apply cream on hand back", "put on bag",
    "take off bag", "put something into a bag", "take something out of a bag", "open a box", "move heavy objects",
    "shake fist", "throw up cap/hat", "hands up (both hands)", "cross arms", "arm circles", "arm swings",
    "running on the spot", "butt kicks (kick backward)", "cross toe touch", "side kick", "yawn", "stretch oneself",
    "blow nose", "hit other person with something", "wield knife towards other person",
    "knock over other person (hit with body)", "grab other person's stuff", "shoot at other person with a gun",
    "step on foot", "high-five", "cheers and drink", "carry something with other person", "take a photo of other person",
    "follow other person", "whisper in other person's ear", "exchange things with other person",
    "support somebody with hand", "finger-guessing game (playing rock-paper-scissors)"]
assert len(NTU120_CLASSES) == 120
