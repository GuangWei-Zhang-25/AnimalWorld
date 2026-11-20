"""Use a trained model as the proposer (the "refine" loop: train small -> propose -> human corrects -> retrain).

options.weights: a .pt (Ultralytics) or .onnx file; default = the newest weights under <project>/models.
options.predictor: ultralytics | onnx | blob (default: by file extension).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List

from ... import registry
from ...formats import labels as L
from .base import Proposer


def newest_weights(project) -> str:
    cands = sorted(list(project.path("models").rglob("*.onnx")) + list(project.path("models").rglob("best.pt")),
                   key=lambda p: p.stat().st_mtime)
    return str(cands[-1]) if cands else ""


class YOLOProposer(Proposer):
    name = "yolo"
    tested = False

    def __init__(self, options: Dict[str, Any] = None, project=None, **kw: Any):
        super().__init__(options, project, **kw)
        self.weights = self.opt.get("weights") or (newest_weights(project) if project is not None else "")
        self.pred_name = self.opt.get("predictor") or ("onnx" if str(self.weights).endswith(".onnx") else "ultralytics")

    def plan(self, rows: List[dict]) -> dict:
        return dict(super().plan(rows), weights=self.weights or None, predictor=self.pred_name,
                    weights_exist=bool(self.weights and Path(self.weights).exists()), conf=self.opt.get("conf", 0.25))

    def propose(self, rows: List[dict]) -> Iterator[dict]:
        from PIL import Image
        if not self.weights and self.pred_name != "blob":
            raise registry.BackendError("yolo proposer: no weights (train a model first or set propose.options.weights)")
        pred = registry.create("predictors", self.pred_name,
                               dict(self.opt, weights=self.weights, category=self.category, keypoints=self.kp_names))
        for r in rows:
            im = Image.open(r["path"]).convert("RGB")
            inst = pred.predict(im)
            yield L.label(r["id"], r["file"], im.size[0], im.size[1], inst, source=f"model:{Path(str(self.weights)).name}")
