# Vendored from the lab's MouseWorld-Track research code (fig8/track_wu/agent_ext7/core_base.py, sha256 fb2649179b35316c).
# Removed: the lab's internal window lists and cluster-specific frame access. scene(), safe_frames(), load_det(), det_mask() are untouched.
"""Shared, model-free logic of the agent test (numpy only). No reference / ground truth is used anywhere in this file."""
import numpy as np
CLIPS = []          # the lab's internal test-window lists were removed from the packaged copy; core.py supplies the clips
SPARSE = 10          # detector consulted every SPARSE frames (1 Hz at 10 fps) unless the agent escalates
SEED_SCORE = 0.40


def load_det(p):
    z = np.load(p, allow_pickle=True)
    return dict(t=z['t'], box=z['box'], score=z['score'], c=z['c'], area=z['area'], T=int(z['T']), W=int(z['W']), H=int(z['H']),
                mbuf=z['mbuf'], moff=z['moff'])


def det_mask(D, i):
    x0, y0, x1, y1 = D['box'][i].astype(int); h, w = y1 - y0, x1 - x0
    m = np.zeros((D['H'], D['W']), bool)
    if h <= 0 or w <= 0: return m
    bits = np.unpackbits(D['mbuf'][D['moff'][i]:D['moff'][i + 1]])[:h * w].reshape(h, w).astype(bool)
    m[y0:y1, x0:x1] = bits; return m


def scene(D):
    """-> N animals (mode of confident detections on sparse frames), body length estimate, anchor frames (all N apart)."""
    T = D['T']; sp = list(range(0, T, SPARSE)); cnt = {}
    for t in sp:
        cnt[t] = int(((D['t'] == t) & (D['score'] >= SEED_SCORE)).sum())
    BL = 2.0 * float(np.sqrt(np.median(D['area'][D['score'] >= SEED_SCORE])))
    # N = the largest count seen WELL SEPARATED (all >= 0.8 body length apart) on >= 10 % of frames: mice in a pile are under-counted
    # (the mode fails when two mice touch most of the time), duplicates / shadows are never far apart for long
    ap = {}
    for t in range(T):
        idx = np.flatnonzero((D['t'] == t) & (D['score'] >= SEED_SCORE)); n = len(idx)
        if n == 0: continue
        if n > 1:
            c = D['c'][idx]; d = np.hypot(*(c[:, None] - c[None]).transpose(2, 0, 1)); d[np.eye(n, dtype=bool)] = 1e9
            if d.min() <= 0.8 * BL: continue
        ap[n] = ap.get(n, 0) + 1
    ok = [n for n, k in ap.items() if k >= 0.10 * T]; N = max(ok) if ok else max(ap, key=ap.get)
    anchors = []
    for t in sp:
        if cnt[t] != N: continue
        c = D['c'][(D['t'] == t) & (D['score'] >= SEED_SCORE)]
        d = np.hypot(*(c[:, None] - c[None]).transpose(2, 0, 1)); d[np.eye(len(c), dtype=bool)] = 1e9
        if N == 1 or d.min() > 0.8 * BL: anchors.append(t)
    return N, BL, anchors, cnt


# frame_path / dataset / native_frames of the lab copy (cluster-specific paths) were removed; core.py defines them.


def safe_frames(D, N, BL):
    """frames where the detector sees exactly N confident mice, all >= 0.8 body length apart (identity cannot be confused there)"""
    out = []
    for t in range(D['T']):
        idx = np.flatnonzero((D['t'] == t) & (D['score'] >= SEED_SCORE))
        if len(idx) != N: continue
        c = D['c'][idx]; d = np.hypot(*(c[:, None] - c[None]).transpose(2, 0, 1)); d[np.eye(N, dtype=bool)] = 1e9
        if N == 1 or d.min() > 0.8 * BL: out.append(t)
    return out
