# Vendored from the lab's MouseWorld-Track research code (fig8/track_wu/agent_ext7/track_fb.py, sha256 17c7d3ea110997c5).
# Only paths were changed (work folder, clip list and checkpoints now come from MWT_* environment variables); the logic is untouched.
"""Agent test, tracking tool: SAM2.1-L (released weights) seeded by DETECTOR masks (no reference input) at the first anchor
(run F) and the last anchor (run B); each run propagates both ways over the whole window. Saves per-run centroids/areas and the
per-frame F-vs-B mask IoU matrix (the bidirectional-agreement signal)."""
import sys, os, tempfile
from pathlib import Path
import numpy as np, torch
import os as _os; _HERE = Path(__file__).resolve().parent; H = Path(_os.environ.get('MWT_AGENT') or _HERE); sys.path.insert(0, str(_HERE)); import core, long_baselines as LB
LB.setup()
from sam2.build_sam import build_sam2_video_predictor
cfg, ck = LB.CK['sam2_large']; P = build_sam2_video_predictor(cfg, str(ck), device='cuda')
dt = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
for clip, t0, t1 in core.CLIPS:
    o = H / 'work' / f'{clip}_{t0:06d}_{t1:06d}'
    if (o / 'tracks.npz').exists() or not (o / 'det.npz').exists(): continue
    D = core.load_det(o / 'det.npz'); N, BL, anchors, cnt = core.scene(D); T = D['T']
    if not anchors: print(clip, t0, 'NO ANCHOR'); continue
    a0, a1 = anchors[0], anchors[-1]
    wd = Path(tempfile.mkdtemp(prefix='agent_'))
    for j, t in enumerate(range(t0, t1 + 1)): os.symlink(core.frame_path(clip, t0, t1, t), wd / f'{j:05d}.jpg')
    runs = {}
    for name, a in (('F', a0), ('B', a1)):
        idx = np.flatnonzero((D['t'] == a) & (D['score'] >= core.SEED_SCORE)); idx = idx[np.argsort(D['c'][idx, 0])]     # ids by x at the seed
        M = {}
        with torch.inference_mode(), torch.autocast('cuda', dtype=dt):
            st = P.init_state(str(wd), offload_video_to_cpu=True, offload_state_to_cpu=True)
            for k, i in enumerate(idx): P.add_new_mask(st, frame_idx=int(a), obj_id=k + 1, mask=core.det_mask(D, i))
            for rev in (False, True):
                for fl, ids, lg in P.propagate_in_video(st, start_frame_idx=int(a), reverse=rev):
                    L = lg[:, 0].float(); best = L.argmax(0)
                    M[int(fl)] = np.stack([((best == j) & (L[j] > 0)).cpu().numpy() for j in range(len(ids))])
            P.reset_state(st)
        runs[name] = M; print(clip, t0, name, 'seed frame', a, 'frames', len(M), flush=True)
    C = {r: np.full((T, N, 2), np.nan) for r in runs}; A = {r: np.zeros((T, N)) for r in runs}; FB = np.zeros((T, N, N))
    for t in range(T):
        for r in runs:
            if t not in runs[r]: continue
            for k in range(N):
                ys, xs = np.nonzero(runs[r][t][k]); A[r][t, k] = len(xs)
                if len(xs) >= 30: C[r][t, k] = (xs.mean(), ys.mean())
        if t in runs['F'] and t in runs['B']:
            f = runs['F'][t].reshape(N, -1); b = runs['B'][t].reshape(N, -1)
            inter = f.astype(np.float32) @ b.astype(np.float32).T; un = f.sum(1)[:, None] + b.sum(1)[None] - inter; FB[t] = inter / np.maximum(un, 1)
    np.savez_compressed(o / 'tracks.npz', CF=C['F'], CB=C['B'], AF=A['F'], AB=A['B'], FB=FB, a0=a0, a1=a1, N=N, BL=BL)
