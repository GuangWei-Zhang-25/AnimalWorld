# Vendored from the lab's MouseWorld-Track research code (fig8/track_wu/agent_ext7/core.py, sha256 7c99039ed6193045).
# Only paths were changed (work folder, clip list and checkpoints now come from MWT_* environment variables); the logic is untouched.
"""Agent core for the EXTERNAL clips: the frozen agent logic (core_base.py = agent/core_frozen_20261112.py, unchanged) with only the clip
list and the frame access replaced. No reference / ground truth is read here."""
import json
from pathlib import Path
from core_base import *          # noqa: F401,F403  (scene, safe_frames, load_det, det_mask, SEED_SCORE, SPARSE)
import os as _os
X = Path(_os.environ.get('MWT_ROOT') or Path(__file__).resolve().parent.parent)
_C = {f"{c['dataset']}__{c['clip']}": c for c in json.load(open(X / 'clips.json'))['clips']}
CLIPS = [(n, 0, c['frames'] - 1) for n, c in _C.items()]
DEV = HELDOUT = TWO_MOUSE = PILE = CLIPS


def _dir(clip): c = _C[clip]; return X / 'data' / c['dataset'] / c['clip']
def frame_path(clip, t0, t1, t): return str(_dir(clip) / 'frames' / f'{t:06d}.jpg')
def dataset(clip): return _C[clip]['dataset']


def native_frames(clip, t0, t1, tmp):
    """camera-rate frames of the clip when the source is faster than the analysis rate (step > 1), else the analysis frames (step 1)"""
    c = _C[clip]; d = _dir(clip)
    if c['step'] > 1 and (d / 'native').exists(): return [str(d / 'native' / f'{i:06d}.jpg') for i in range(c['step'] * (c['frames'] - 1) + 1)], c['step']
    return [frame_path(clip, t0, t1, t) for t in range(t0, t1 + 1)], 1


# ---------------- agent v6 (12 Nov 2026, after the first external run): counting and starting in groups that are always huddled ----------
import numpy as _np
_scene5, _safe5 = scene, safe_frames


def _sust(c, frac): return max([n for n in range(1, 16) if (c >= n).mean() >= frac] + [0])


def scene(D):
    """v6 count: N = max(v5 rule, the largest count the detector reaches on >= 25 % of frames). v5 needs the animals seen apart, which a
    continuously huddled group never is; the second rule needs only that they are DETECTED. Anchors = safe frames (v6) on the sparse grid."""
    try: N5, BL, an5, cnt = _scene5(D)
    except ValueError: BL = 2.0 * float(_np.sqrt(_np.median(D['area'][D['score'] >= SEED_SCORE]))); N5, cnt = 0, {}
    T = D['T']
    c = _np.array([int(((D['t'] == t) & (D['score'] >= SEED_SCORE)).sum()) for t in range(T)]); N = max(N5, _sust(c, 0.25))
    an = [t for t in safe_frames(D, N, BL) if t % SPARSE == 0] or safe_frames(D, N, BL)[:1]
    return N, BL, an, cnt


def safe_frames(D, N, BL):
    """v6: the v5 safe frames (exactly N confident detections, all >= 0.8 body length apart). If they cover < 5 % of the window (a huddled
    group), RELAXED safe frames are used instead: exactly N confident detections whose masks are disjoint (shared pixels < 5 % of the smaller
    mask) and centroids >= 0.4 body length apart: touching animals that the detector still segments as N separate bodies."""
    s = _safe5(D, N, BL)
    if len(s) >= 0.05 * D['T'] or N < 2: return s
    out = []
    for t in range(D['T']):
        idx = _np.flatnonzero((D['t'] == t) & (D['score'] >= SEED_SCORE))
        if len(idx) != N: continue
        c = D['c'][idx]; d = _np.hypot(*(c[:, None] - c[None]).transpose(2, 0, 1)); d[_np.eye(N, dtype=bool)] = 1e9
        if d.min() < 0.4 * BL: continue
        M = [det_mask(D, i) for i in idx]; ok = True
        for a in range(N):
            for b in range(a + 1, N):
                if d[a, b] < 1.5 * BL and (M[a] & M[b]).sum() > 0.05 * min(M[a].sum(), M[b].sum()): ok = False; break
            if not ok: break
        if ok: out.append(t)
    return out


# ---------------- fig8 PREREG §5 (15 Nov 2026): "N = 4 given to the agent". The count is fixed at 4; body length and anchors are
# computed exactly as above (v6 rules) but with N = 4. Nothing else changes.
_scene6 = scene


def scene(D):
    _N, BL, _an, cnt = _scene6(D); N = int(next(iter(_C.values()))["animals"])   # Figure 7 Wu copy: N from clips.json (= 1), was fixed at 4 for the Hu windows
    s = safe_frames(D, N, BL); an = [t for t in s if t % SPARSE == 0] or s[:1]
    return N, BL, an, cnt
