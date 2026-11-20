"""NEW in animalworld (not part of the frozen agent): move the detector's confidence cut without touching the frozen scripts.

The frozen rules use fixed cuts on the detector score (0.40 for a "confident" detection, 0.30 / 0.25 elsewhere). This step
multiplies the stored scores by 0.40 / CUT (capped at 1), so that a raw score of CUT is what the rules see as 0.40. The
other cuts move by the same factor. The raw file is kept as det_raw.npz (used for the output boxes).
usage: rescale_scores.py dense|local      env: MWT_AGENT, MWT_ROOT, MWT_SEED_CUT (raw score that should count as 0.40)"""
import json
import os
import sys
from pathlib import Path

import numpy as np

A = Path(os.environ['MWT_AGENT']); X = Path(os.environ['MWT_ROOT']); cut = float(os.environ['MWT_SEED_CUT']); k = 0.40 / cut
for c in json.load(open(X / 'clips.json'))['clips']:
    o = A / 'work' / f"{c['dataset']}__{c['clip']}_000000_{c['frames'] - 1:06d}"
    if sys.argv[1] == 'dense':
        if not (o / 'det.npz').exists() or (o / 'det_raw.npz').exists(): continue
        z = dict(np.load(o / 'det.npz', allow_pickle=True)); os.replace(o / 'det.npz', o / 'det_raw.npz')
        z['score'] = np.minimum(1.0, z['score'] * k); np.savez_compressed(o / 'det.npz', **z)
    else:
        if not (o / 'dlocal.npz').exists() or (o / 'dlocal_raw.npz').exists(): continue
        z = dict(np.load(o / 'dlocal.npz', allow_pickle=True)); os.replace(o / 'dlocal.npz', o / 'dlocal_raw.npz')
        for key in z:
            if key.startswith('d_') and z[key].ndim == 2 and z[key].shape[1] == 5: z[key] = z[key].copy(); z[key][:, 3] = np.minimum(1.0, z[key][:, 3] * k)
        np.savez_compressed(o / 'dlocal.npz', **z)
    print('rescaled', sys.argv[1], o.name, 'factor', round(k, 4), flush=True)
