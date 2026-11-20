# Vendored from the lab's MouseWorld-Track research code (fig8/track_wu/agent_ext7/track_local.py, sha256 8f71b6d54c3cc0e9).
# Only paths were changed (work folder, clip list and checkpoints now come from MWT_* environment variables); the logic is untouched.
"""Agent tool: LOCAL TWO-ENDED RE-TRACK of every contact span (the recipe that repaired the reviewer-flagged stretches, now triggered
by the agent itself). For each contact span (u, v) between two safe frames: SAM2.1-L seeded with the DETECTOR masks at u and run
forward to v ('Lf'), and seeded at v and run backward to u ('Lb'), at the camera's native frame rate (e.g. 30 fps; 3x the analysis
rate). Short propagation from the last clean frame, both directions, independent seeds. No reference input.
-> work/<win>/local.npz: f_<u>_<v>, b_<u>_<v> = centroids (native frames, N, 2); step.   usage: track_local.py [LIST]"""
import sys, os, tempfile, shutil
from pathlib import Path
import numpy as np, torch
import os as _os; _HERE = Path(__file__).resolve().parent; H = Path(_os.environ.get('MWT_AGENT') or _HERE); sys.path.insert(0, str(_HERE)); import core, long_baselines as LB
LB.setup()
from sam2.build_sam import build_sam2_video_predictor
cfg, ck = LB.CK['sam2_large']; P = build_sam2_video_predictor(cfg, str(ck), device='cuda')
dt = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
for clip, t0, t1 in (getattr(core, sys.argv[1]) if len(sys.argv) > 1 else core.CLIPS):
    o = H / 'work' / f'{clip}_{t0:06d}_{t1:06d}'
    if (o / 'local.npz').exists() or not (o / 'tracks.npz').exists(): continue
    D = core.load_det(o / 'det.npz'); Z = np.load(o / 'tracks.npz'); N, BL = int(Z['N']), float(Z['BL'])
    if (o / 'scene2.npz').exists(): s2 = np.load(o / 'scene2.npz'); N, BL = int(s2['N']), float(s2['BL'])
    safe = core.safe_frames(D, N, BL); spans = [(u, v) for u, v in zip(safe[:-1], safe[1:]) if v - u > 1]
    if N < 2 or not spans: np.savez_compressed(o / 'local.npz', step=1, N=N); continue
    root = Path(tempfile.mkdtemp(prefix='agentloc_'))
    try: NF, step = core.native_frames(clip, t0, t1, str(root / 'nat'))
    except Exception as e: print(clip, t0, 'NATIVE FRAMES FAILED', e, flush=True); shutil.rmtree(root, ignore_errors=True); continue
    out = dict(step=step, N=N)
    for u, v in spans:
        wd = root / f's{u}'; wd.mkdir(); fr = NF[step * u: step * v + 1]
        for j, p in enumerate(fr): os.symlink(p, wd / f'{j:05d}.jpg')
        for name, a, tt, rev in (('f', 0, u, False), ('b', len(fr) - 1, v, True)):
            idx = np.flatnonzero((D['t'] == tt) & (D['score'] >= core.SEED_SCORE)); C = np.full((len(fr), N, 2), np.nan, np.float32)
            with torch.inference_mode(), torch.autocast('cuda', dtype=dt):
                st = P.init_state(str(wd), offload_video_to_cpu=True, offload_state_to_cpu=True)
                for k, i in enumerate(idx): P.add_new_mask(st, frame_idx=int(a), obj_id=k + 1, mask=core.det_mask(D, i))
                for fl, ids, lg in P.propagate_in_video(st, start_frame_idx=int(a), reverse=rev):
                    Lg = lg[:, 0].float(); best = Lg.argmax(0)
                    for j in range(len(ids)):
                        ys, xs = torch.nonzero((best == j) & (Lg[j] > 0), as_tuple=True)
                        if len(xs) >= 30: C[int(fl), j] = (float(xs.float().mean()), float(ys.float().mean()))
                P.reset_state(st)
            out[f'{name}_{u}_{v}'] = C
        shutil.rmtree(wd, ignore_errors=True)
    np.savez_compressed(o / 'local.npz', **out); shutil.rmtree(root, ignore_errors=True)
    print(clip, t0, 'spans', len(spans), 'native step', step, flush=True)
