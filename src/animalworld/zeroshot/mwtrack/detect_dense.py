# Vendored from the lab's MouseWorld-Track research code (fig8/track_wu/agent_ext7/detect_dense.py, sha256 31e0b2737937803c).
# Only paths were changed (work folder, clip list and checkpoints now come from MWT_* environment variables), and a window with no detection
# at all now writes an empty det.npz (box (0, 4), c (0, 2): the shapes the readers index) instead of stopping with an error; the logic is
# otherwise untouched.
"""Agent test, perception: interim RF-DETR-Seg-L on every frame of each test window (the agent later CONSULTS only sparse
frames + escalation stretches; dense is computed up-front because one Slurm job cannot answer requests interactively)."""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import os as _os; _HERE = Path(__file__).resolve().parent; H = Path(_os.environ.get('MWT_AGENT') or _HERE); sys.path.insert(0, str(_HERE)); import core
from rfdetr.detr import RFDETR
net = RFDETR.from_checkpoint(_os.environ['MWT_DET_CKPT'], trust_checkpoint=True)
for clip, t0, t1 in core.CLIPS:
    o = H / 'work' / f'{clip}_{t0:06d}_{t1:06d}'; o.mkdir(parents=True, exist_ok=True)
    if (o / 'det.npz').exists(): continue
    R = dict(t=[], box=[], score=[], c=[], area=[]); buf = []; off = [0]
    for j, t in enumerate(range(t0, t1 + 1)):
        im = Image.open(core.frame_path(clip, t0, t1, t)).convert('RGB'); W, Hh = im.size
        d = net.predict(im, threshold=0.25); B = np.asarray(d.xyxy).reshape(-1, 4); S = np.asarray(d.confidence).reshape(-1); keep = []
        for i in np.argsort(-S):                                   # containment NMS (body vs body+tail duplicates)
            bi = B[i]; ai = (bi[2] - bi[0]) * (bi[3] - bi[1])
            if all(max(0, min(bi[2], B[k][2]) - max(bi[0], B[k][0])) * max(0, min(bi[3], B[k][3]) - max(bi[1], B[k][1])) / max(1e-9, min(ai, (B[k][2] - B[k][0]) * (B[k][3] - B[k][1]))) <= 0.8 for k in keep): keep.append(i)
        for i in keep:
            m = np.asarray(d.mask[i], bool); ys, xs = np.nonzero(m)
            if len(xs) < 30: continue
            x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
            R['t'].append(j); R['box'].append([x0, y0, x1, y1]); R['score'].append(float(S[i])); R['c'].append([xs.mean(), ys.mean()]); R['area'].append(len(xs))
            b = np.packbits(m[y0:y1, x0:x1]); buf.append(b); off.append(off[-1] + len(b))
    np.savez_compressed(o / 'det.npz', t=np.array(R['t'], int), box=np.array(R['box'], float).reshape(-1, 4), score=np.array(R['score'], float),
                        c=np.array(R['c'], float).reshape(-1, 2), area=np.array(R['area'], int), mbuf=(np.concatenate(buf) if buf else np.zeros(0, np.uint8)),
                        moff=np.array(off), T=t1 - t0 + 1, W=W, H=Hh)
    print(clip, t0, 'detections', len(R['t']), flush=True)
