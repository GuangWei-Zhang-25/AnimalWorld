# Vendored from the lab's MouseWorld-Track research code (fig8/track_wu/agent_ext7/detect_local.py, sha256 d4cea395c2e25101).
# Only paths were changed (work folder, clip list and checkpoints now come from MWT_* environment variables); the logic is untouched.
"""Agent tool, escalation step 3: the DETECTOR at the camera's native frame rate through every contact of >= 1 s (RF-DETR-Seg-L, epoch-5
checkpoint, containment NMS). -> work/<win>/dlocal.npz: d_<u>_<v> = rows [native index, x, y, score, area]; step. No reference input.
usage: detect_local.py [LIST]"""
import sys, tempfile, shutil
from pathlib import Path
import numpy as np
from PIL import Image
import os as _os; _HERE = Path(__file__).resolve().parent; H = Path(_os.environ.get('MWT_AGENT') or _HERE); sys.path.insert(0, str(_HERE)); import core
from rfdetr.detr import RFDETR
net = RFDETR.from_checkpoint(_os.environ['MWT_DET_CKPT'], trust_checkpoint=True)
for clip, t0, t1 in (getattr(core, sys.argv[1]) if len(sys.argv) > 1 else core.CLIPS):
    o = H / 'work' / f'{clip}_{t0:06d}_{t1:06d}'
    if (o / 'dlocal.npz').exists() or not (o / 'tracks.npz').exists(): continue
    D = core.load_det(o / 'det.npz'); Z = np.load(o / 'tracks.npz'); N, BL = int(Z['N']), float(Z['BL'])
    safe = core.safe_frames(D, N, BL); spans = [(u, v) for u, v in zip(safe[:-1], safe[1:]) if v - u >= 10]
    if N < 2 or not spans: np.savez_compressed(o / 'dlocal.npz', step=1); continue
    root = Path(tempfile.mkdtemp(prefix='agentdl_'))
    try: NF, step = core.native_frames(clip, t0, t1, str(root / 'nat'))
    except Exception as e: print(clip, t0, 'NATIVE FRAMES FAILED', e, flush=True); shutil.rmtree(root, ignore_errors=True); continue
    out = dict(step=step)
    for u, v in spans:
        rows = []
        for n, p in enumerate(NF[step * u: step * v + 1]):
            d = net.predict(Image.open(p).convert('RGB'), threshold=0.2); B = np.asarray(d.xyxy).reshape(-1, 4); S = np.asarray(d.confidence).reshape(-1); keep = []
            for i in np.argsort(-S):
                bi = B[i]; ai = (bi[2] - bi[0]) * (bi[3] - bi[1])
                if all(max(0, min(bi[2], B[k][2]) - max(bi[0], B[k][0])) * max(0, min(bi[3], B[k][3]) - max(bi[1], B[k][1])) / max(1e-9, min(ai, (B[k][2] - B[k][0]) * (B[k][3] - B[k][1]))) <= 0.8 for k in keep): keep.append(i)
            for i in keep:
                ys, xs = np.nonzero(np.asarray(d.mask[i], bool))
                if len(xs) >= 30: rows.append([n, xs.mean(), ys.mean(), float(S[i]), len(xs)])
        out[f'd_{u}_{v}'] = np.array(rows, np.float32).reshape(-1, 5)
    np.savez_compressed(o / 'dlocal.npz', **out); shutil.rmtree(root, ignore_errors=True); print(clip, t0, 'spans', len(spans), flush=True)
