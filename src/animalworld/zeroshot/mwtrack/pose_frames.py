"""MouseWorld-Pose, top-down: one square crop per box (side = 1.3 x the longer box side, min 16 px) -> 256 px ->
crop-pose model (conf floor 0.01) -> the instance nearest the crop centre -> keypoints mapped back to the frame.
Recipe and options of the lab's pose_topdown/improve/r3_infer.py with the selected headline options
(mult=1.3, min=16, pre=dark_gamma0.5, no TTA).   usage: pose_frames.py REQUEST.json OUT.npz
REQUEST.json: {"weights": path, "pre": "dark_gamma0.5", "items": [{"image": path, "boxes": [[x0, y0, x1, y1], ...]}, ...]}
OUT.npz: kp (n_boxes_total, K, 2), conf (n_boxes_total, K), ok (n_boxes_total,) in the order of the request."""
import json
import os
import sys

os.environ.setdefault('YOLO_AUTOINSTALL', 'false'); os.environ.setdefault('ORT_DISABLE_TELEMETRY', '1')
import cv2
import numpy as np
from ultralytics import YOLO

req = json.load(open(sys.argv[1])); M = YOLO(req['weights']); S = 256; MULT, MIN, PRE = 1.3, 16.0, req.get('pre') or 'none'
import torch
DEV = 0 if torch.cuda.is_available() else 'cpu'


def pre(c, mode):
    if mode == 'none': return c
    if mode.startswith('dark_'):
        gy = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY); lo, hi = np.percentile(gy, [50, 99])
        return pre(c, mode[5:]) if (lo < 30 and hi < 128) else c
    if mode.startswith('gamma'):
        gm = float(mode[5:]); return cv2.LUT(c, np.clip(255 * (np.arange(256) / 255.0) ** gm, 0, 255).astype(np.uint8))
    raise ValueError(mode)


KP, CF, OK = [], [], []; K = None
for n_it, it in enumerate(req['items']):
    img = cv2.imread(it['image']); crops, aff = [], []
    for x0, y0, x1, y1 in it['boxes']:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; side = max(max(x1 - x0, y1 - y0) * MULT, MIN); X0, Y0 = cx - side / 2, cy - side / 2
        A_ = cv2.getAffineTransform(np.float32([[X0, Y0], [X0 + side, Y0], [X0, Y0 + side]]), np.float32([[0, 0], [S, 0], [0, S]]))
        c = cv2.warpAffine(img, A_, (S, S), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114))
        crops.append(pre(c, PRE)); aff.append((X0, Y0, side))
    if not crops: continue
    R = M.predict(crops, imgsz=S, conf=0.01, verbose=False, device=DEV)
    for r, (X0, Y0, side) in zip(R, aff):
        if r.keypoints is None or not len(r.boxes):
            KP.append(None); CF.append(None); OK.append(False); continue
        b = r.boxes.xywh.cpu().numpy(); c = r.boxes.conf.cpu().numpy(); kx = r.keypoints.xy.cpu().numpy()
        kc = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else np.ones(kx.shape[:2])
        near = np.hypot(b[:, 0] - S / 2, b[:, 1] - S / 2) <= 0.25 * S
        j = int(np.argmax(np.where(near, c, -1))) if near.any() else int(np.argmax(c))
        K = kx.shape[1]; KP.append(kx[j] * side / S + [X0, Y0]); CF.append(kc[j]); OK.append(True)
    if n_it % 100 == 0: print('pose', n_it, '/', len(req['items']), flush=True)
K = K or 7
kp = np.array([k if k is not None else np.full((K, 2), np.nan) for k in KP], float).reshape(-1, K, 2)
cf = np.array([c if c is not None else np.zeros(K) for c in CF], float).reshape(-1, K)
np.savez_compressed(sys.argv[2], kp=kp, conf=cf, ok=np.array(OK, bool))
print('pose done', len(OK), 'boxes', int(np.sum(OK)), 'with keypoints', flush=True)
