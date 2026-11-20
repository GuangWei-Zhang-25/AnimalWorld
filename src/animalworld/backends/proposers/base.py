"""Proposal models: produce label documents (formats.labels) for images.

    class MyProposer(Proposer):
        name = "my_proposer"
        def plan(self, rows): -> dict                      # must not load a model
        def propose(self, rows):                           # rows: dicts with id, path, W, H, unit, frame
            yield labels.label(row["id"], row["file"], W, H, [labels.instance(...), ...], source=self.name)
"""
from __future__ import annotations

from typing import Any, Dict, Iterator, List


class Proposer:
    name = "base"
    needs_gpu = False
    tested = False

    def __init__(self, options: Dict[str, Any] = None, project=None, **_: Any):
        self.opt = options or {}
        self.project = project
        self.category = project.categories()[0] if project is not None else self.opt.get("category", "animal")
        self.kp_names = project.keypoints() if project is not None else list(self.opt.get("keypoints") or [])

    def plan(self, rows: List[dict]) -> dict:
        return {"backend": self.name, "images": len(rows), "needs_gpu": self.needs_gpu}

    def propose(self, rows: List[dict]) -> Iterator[dict]:
        raise NotImplementedError


def mask_to_polygon(mask, eps: float = 1.0, max_points: int = 120):
    """Largest connected component of a boolean mask -> (polygon, component_fraction, border_fraction).

    Same procedure as poly_of() in mw_det_syn/gap_targeted/scripts/label_img.py (needs OpenCV).
    """
    import numpy as np
    from ...registry import need
    cv2 = need("cv2", "annotate", "turning masks into polygons")
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return None, 0.0, 1.0
    big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    frac = st[big, cv2.CC_STAT_AREA] / max(1, int(mask.sum()))
    cs, _ = cv2.findContours((lab == big).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea).reshape(-1, 2)
    H, W = mask.shape
    edge = float(((c[:, 0] <= 0) | (c[:, 1] <= 0) | (c[:, 0] >= W - 1) | (c[:, 1] >= H - 1)).mean())
    c = cv2.approxPolyDP(c.reshape(-1, 1, 2), eps, True).reshape(-1, 2)
    if len(c) > max_points:
        c = c[np.linspace(0, len(c) - 1, max_points).astype(int)]
    if len(c) < 3:
        return None, float(frac), edge
    return [[float(x), float(y)] for x, y in c], float(frac), edge
