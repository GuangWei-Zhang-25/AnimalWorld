"""Native-framework predictors: Ultralytics (.pt) and RF-DETR checkpoints.

RF-DETR call adapted from TrackPose_demonstration/long_runs/pose_topdown/detect_topdown.py:
    RFDETR.from_checkpoint(ckpt, trust_checkpoint=True).predict(PIL image, threshold=...)
The RF-DETR predictor was NOT executed in this package's tests.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List

from ...formats import labels as L
from ...registry import BackendError, need
from .base import Predictor


class UltralyticsPredictor(Predictor):
    name = "ultralytics"
    tested = False

    def __init__(self, options: Dict[str, Any] = None, **kw: Any):
        super().__init__(options, **kw)
        os.environ.setdefault("YOLO_AUTOINSTALL", "false")
        ul = need("ultralytics", "train", "the Ultralytics predictor")
        w = self.opt.get("weights")
        if not w or not os.path.exists(str(w)):
            raise BackendError(f"ultralytics predictor: weights file not found: {w}")
        self.model = ul.YOLO(str(w))

    def predict(self, image) -> List[dict]:
        kw = {"conf": self.conf, "verbose": False, "device": self.opt.get("device", "cpu")}
        if self.opt.get("imgsz"):
            kw["imgsz"] = int(self.opt["imgsz"])
        r = self.model.predict(image, **kw)[0]
        names = r.names or {0: self.category}
        inst = []
        n = 0 if r.boxes is None else len(r.boxes)
        for k in range(n):
            box = [float(v) for v in r.boxes.xyxy[k].tolist()]
            poly = None
            if r.masks is not None and k < len(r.masks.xy) and len(r.masks.xy[k]) >= 3:
                poly = [[float(x), float(y)] for x, y in r.masks.xy[k]]
            kps = None
            if r.keypoints is not None and r.keypoints.data is not None and k < len(r.keypoints.data):
                d = r.keypoints.data[k].tolist()
                nm = self.kp_names if len(self.kp_names) == len(d) else [f"kp{j}" for j in range(len(d))]
                kps = {a: [float(p[0]), float(p[1]), 2 if (len(p) < 3 or p[2] >= float(self.opt.get("kp_conf", 0.5))) else 1]
                       for a, p in zip(nm, d)}
            inst.append(L.instance(str(names.get(int(r.boxes.cls[k]), self.category)), box, poly, kps,
                                   score=float(r.boxes.conf[k])))
        return inst


class RFDETRPredictor(Predictor):
    name = "rfdetr"
    tested = False

    def __init__(self, options: Dict[str, Any] = None, **kw: Any):
        super().__init__(options, **kw)
        rfdetr = need("rfdetr", "train", "the RF-DETR predictor")
        w = self.opt.get("weights")
        if not w or not os.path.exists(str(w)):
            raise BackendError(f"rfdetr predictor: checkpoint not found: {w}")
        cls = getattr(rfdetr, "RFDETR", None)
        if cls is None:                                   # some rfdetr versions export it from rfdetr.detr only
            from rfdetr.detr import RFDETR as cls
        self.model = cls.from_checkpoint(str(w), trust_checkpoint=True)

    def predict(self, image) -> List[dict]:
        import numpy as np
        det = self.model.predict(image, threshold=self.conf)
        inst = []
        for k in range(len(det.xyxy)):
            poly = None
            if getattr(det, "mask", None) is not None:
                from ..proposers.base import mask_to_polygon
                try:
                    poly, _, _ = mask_to_polygon(np.asarray(det.mask[k]).astype(bool))
                except BackendError:
                    poly = None
            inst.append(L.instance(self.category, [float(v) for v in det.xyxy[k]], poly, None, score=float(det.confidence[k])))
        return inst
