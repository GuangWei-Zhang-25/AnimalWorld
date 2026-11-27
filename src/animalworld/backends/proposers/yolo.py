"""Use a trained model as the proposer (the "refine" loop: train small -> propose -> human corrects -> retrain).

Registered as `model` and, for older projects, as `yolo`.

options.weights: a .pt (Ultralytics), .onnx, or RF-DETR .pth / .ckpt file; default = the newest weights under
                 <project>/models (Ultralytics best.pt, an exported .onnx, or an RF-DETR checkpoint_best*.pth).
                 `animalworld propose PROJECT --backend model --weights FILE` sets it for one run, so a released
                 checkpoint (e.g. mouseworld-det.ckpt) can label a new project's images.
options.predictor: ultralytics | onnx | rfdetr | blob (default: by file extension; .pth and .ckpt -> rfdetr).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List

from ... import registry
from ...formats import labels as L
from .base import Proposer


PATTERNS = ("*.onnx", "best.pt", "checkpoint_best*.pth", "*.ckpt")


def newest_weights(project) -> str:
    """The newest trained weights under <project>/models: Ultralytics best.pt, an exported .onnx, or an RF-DETR
    checkpoint (checkpoint_best*.pth, *.ckpt)."""
    cands = sorted({p for pat in PATTERNS for p in project.path("models").rglob(pat)}, key=lambda p: p.stat().st_mtime)
    return str(cands[-1]) if cands else ""


def predictor_for(weights: str) -> str:
    w = str(weights).lower()
    return "onnx" if w.endswith(".onnx") else ("rfdetr" if w.endswith((".pth", ".ckpt")) else "ultralytics")


class YOLOProposer(Proposer):
    name = "yolo"
    tested = False

    def __init__(self, options: Dict[str, Any] = None, project=None, **kw: Any):
        super().__init__(options, project, **kw)
        self.weights = self.opt.get("weights") or (newest_weights(project) if project is not None else "")
        self.pred_name = self.opt.get("predictor") or predictor_for(self.weights)

    def plan(self, rows: List[dict]) -> dict:
        return dict(super().plan(rows), weights=self.weights or None, predictor=self.pred_name,
                    weights_exist=bool(self.weights and Path(self.weights).exists()), conf=self.opt.get("conf", 0.25))

    def propose(self, rows: List[dict]) -> Iterator[dict]:
        from PIL import Image
        if not self.weights and self.pred_name != "blob":
            raise registry.BackendError("model proposer: no weights (train a model first, set propose.options.weights, "
                                        "or pass --weights FILE)")
        pred = registry.create("predictors", self.pred_name,
                               dict(self.opt, weights=self.weights, category=self.category, keypoints=self.kp_names))
        for r in rows:
            im = Image.open(r["path"]).convert("RGB")
            inst = pred.predict(im)
            yield L.label(r["id"], r["file"], im.size[0], im.size[1], inst, source=f"model:{Path(str(self.weights)).name}")
