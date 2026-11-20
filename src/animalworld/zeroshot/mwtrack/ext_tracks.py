# Vendored from the lab's MouseWorld-Track research code (fig8/track_wu/agent_ext7/ext_tracks.py, sha256 cbcd3ff670dd7e8f).
# Only paths were changed (work folder, clip list and checkpoints now come from MWT_* environment variables); the logic is untouched.
"""Stored tracks of the external comparison in the layout supervise5 expects from trk_events.load (runs_clips/<model>/<clip>/<key>.npz)."""
from pathlib import Path
import numpy as np
import os as _os
X = Path(_os.environ.get('MWT_ROOT') or Path(__file__).resolve().parent.parent)


def load(m, clip, key, t0, T, NG):
    p = X / 'runs_clips' / m / clip / f'{key}.npz'
    if not p.exists(): return None
    q = np.load(p); tr = {}
    for f, k, x, y in zip(q['pt_frame'], q['pt_track'], q['pt_x'], q['pt_y']):
        if 0 <= int(f) - t0 < T: tr.setdefault(int(k), np.full((T, 2), np.nan))[int(f) - t0] = (x, y)
    return tr
