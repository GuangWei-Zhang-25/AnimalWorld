"""Classical baseline detector (numpy only, no weights): pixels that differ from the background
level form blobs; each blob is one detection.

It needs a plain, roughly uniform background and well separated animals. It exists so that
`predict` and `track` can be demonstrated and tested without any trained model, and as a sanity
baseline; it is not a substitute for a trained detector.
"""
from __future__ import annotations

from typing import List

import numpy as np

from ...formats import labels as L
from .base import Predictor


def components(mask: np.ndarray) -> List[np.ndarray]:
    """4-connected components of a small boolean mask (iterative flood fill) -> list of (y, x) index arrays."""
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out = []
    for y0, x0 in zip(*np.nonzero(mask)):
        if seen[y0, x0]:
            continue
        stack, pts = [(int(y0), int(x0))], []
        seen[y0, x0] = True
        while stack:
            y, x = stack.pop()
            pts.append((y, x))
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        out.append(np.array(pts))
    return out


class BlobPredictor(Predictor):
    name = "blob"
    tested = True

    def predict(self, image) -> List[dict]:
        work = int(self.opt.get("work_side", 160))
        W, H = image.size
        s = work / max(W, H)
        small = image.convert("L").resize((max(8, int(round(W * s))), max(8, int(round(H * s)))))
        g = np.asarray(small).astype(np.float32)
        diff = np.abs(g - np.median(g))
        mask = diff > float(self.opt.get("threshold", 40))
        min_area = float(self.opt.get("min_area_frac", 0.002)) * mask.size
        max_area = float(self.opt.get("max_area_frac", 0.25)) * mask.size
        sx, sy = W / mask.shape[1], H / mask.shape[0]
        inst = []
        for pts in components(mask):
            if not (min_area <= len(pts) <= max_area):
                continue
            ys, xs = pts[:, 0], pts[:, 1]
            box = [xs.min() * sx, ys.min() * sy, (xs.max() + 1) * sx, (ys.max() + 1) * sy]
            score = float(min(0.99, 0.5 + diff[ys, xs].mean() / 255.0))
            if score >= self.conf:
                inst.append(L.instance(self.category, box, None, None, score=score))
        return inst
