"""Predictors: ``predict(PIL image) -> list of instances`` (formats.labels.instance dicts)."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


class Predictor:
    name = "base"
    tested = False

    def __init__(self, options: Dict[str, Any] = None, **_: Any):
        self.opt = options or {}
        self.category = self.opt.get("category", "animal")
        self.kp_names = list(self.opt.get("keypoints") or [])
        self.conf = float(self.opt.get("conf", 0.25))

    def predict(self, image) -> List[dict]:
        raise NotImplementedError


def nms(boxes: np.ndarray, scores: np.ndarray, thr: float = 0.7) -> List[int]:
    """Greedy non-maximum suppression on xyxy boxes (numpy)."""
    if len(boxes) == 0:
        return []
    x0, y0, x1, y1 = boxes.T
    area = np.maximum(0, x1 - x0) * np.maximum(0, y1 - y0)
    order = np.argsort(-scores)
    keep = []
    while len(order):
        i = int(order[0])
        keep.append(i)
        if len(order) == 1:
            break
        r = order[1:]
        iw = np.maximum(0, np.minimum(x1[i], x1[r]) - np.maximum(x0[i], x0[r]))
        ih = np.maximum(0, np.minimum(y1[i], y1[r]) - np.maximum(y0[i], y0[r]))
        inter = iw * ih
        order = r[inter / np.maximum(1e-9, area[i] + area[r] - inter) <= thr]
    return keep
