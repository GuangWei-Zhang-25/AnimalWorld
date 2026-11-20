"""SAM 3.1 witness of MouseWorld-Track: text prompt only, no reference (the `sam31` branch of the lab's
fig8/track_wu/run_ext.py). One session per clip. -> <MWT_ROOT>/runs_clips/sam31/<dataset>__<clip>/<key>.npz (centroids).

MWT_ROOT          work root holding clips.json and data/<dataset>/<clip>/frames/%06d.jpg
MWT_SAM31_CKPT    path of sam3.1_multiplex.pt
MWT_PROMPT        text prompt (default "mouse")
"""
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

X = Path(os.environ['MWT_ROOT'])
CLIPS = json.load(open(X / 'clips.json'))['clips']
PROMPT = os.environ.get('MWT_PROMPT') or 'mouse'


def name(c): return f"{c['dataset']}__{c['clip']}"
def out(c): return X / 'runs_clips' / 'sam31' / name(c) / f"000000_{c['frames'] - 1:06d}.npz"


def cent(m):
    ys, xs = np.nonzero(m)
    return (float(xs.mean()), float(ys.mean())) if len(xs) >= 30 else None


clips = [c for c in CLIPS if not out(c).exists()]
if not clips:
    raise SystemExit(0)
import torch
if not torch.cuda.is_available():
    sys.exit('sam31_witness: no CUDA device (SAM 3.1 needs a GPU)')
from sam3.model_builder import build_sam3_multiplex_video_predictor
P = build_sam3_multiplex_video_predictor(checkpoint_path=os.environ['MWT_SAM31_CKPT'], default_output_prob_thresh=.5, use_fa3=False)
tmp = Path(tempfile.mkdtemp(prefix='sam31_'))
for c in clips:
    wd = tmp / name(c); wd.mkdir(exist_ok=True)
    for j in range(c['frames']):
        os.symlink((X / 'data' / c['dataset'] / c['clip'] / 'frames' / f'{j:06d}.jpg').resolve(), wd / f'{j:05d}.jpg')
    rows = []; t0 = time.time(); note = ''
    sid = P.handle_request(request=dict(type='start_session', resource_path=str(wd), offload_video_to_cpu=True))['session_id']
    try:
        P.handle_request(request=dict(type='add_prompt', session_id=sid, frame_index=0, text=PROMPT))
        try:
            for r in P.handle_stream_request(request=dict(type='propagate_in_video', session_id=sid, propagation_direction='forward')):
                o = r['outputs']; ids = np.asarray(o['out_obj_ids']).reshape(-1); ms = np.asarray(o['out_binary_masks'])
                if ms.ndim == 2: ms = ms[None]
                for i, m in zip(ids, ms):
                    p = cent(m)
                    if p: rows.append((int(r['frame_index']), int(i), *p))
        except RuntimeError as ex:
            if 'No points are provided' not in str(ex): raise
            note = 'no text detections'
    finally:
        P.handle_request(request=dict(type='close_session', session_id=sid, run_gc_collect=True))
    o = out(c); o.parent.mkdir(parents=True, exist_ok=True); r = np.array(rows, float).reshape(-1, 4)
    np.savez_compressed(o, pt_frame=r[:, 0].astype(np.int32), pt_track=r[:, 1].astype(np.int32), pt_x=r[:, 2], pt_y=r[:, 3], W=c['W'], H=c['H'], N=c['frames'])
    o.with_suffix('.info.json').write_text(json.dumps(dict(seconds=round(time.time() - t0), note=note)))
    print('sam31', name(c), len(rows), note, flush=True)
    shutil.rmtree(wd, ignore_errors=True)
shutil.rmtree(tmp, ignore_errors=True)
