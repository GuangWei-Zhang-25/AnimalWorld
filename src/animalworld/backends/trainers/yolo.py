"""Ultralytics YOLO trainer (detect / segment / pose). Ultralytics is AGPL-3.0.

Recipes adapted from TrackPose_demonstration/long_runs/synth_setup/yolo_arm.py (detection /
segmentation: imgsz 768, plots off, periodic checkpoints, resume from last.pt) and
pose_topdown/train.py (crop pose model: imgsz 256, mosaic 0, degrees 180, fliplr 0.5, scale 0.25,
translate 0.05, patience 15). Options in project.yaml (train.options) override any of these.

Set YOLO_AUTOINSTALL=false in shared environments so Ultralytics never pip-installs by itself.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from ...registry import BackendError, need
from .base import Trainer

MODELS = {"detect": "yolo11n.pt", "segment": "yolo11n-seg.pt", "pose": "yolo11n-pose.pt"}
BASE = {"epochs": 100, "imgsz": 768, "batch": 16, "workers": 2, "seed": 0, "patience": 20, "plots": False, "save_period": -1}
CROP_POSE = {"imgsz": 256, "epochs": 60, "batch": 128, "patience": 15, "mosaic": 0.0, "degrees": 180, "fliplr": 0.5,
             "scale": 0.25, "translate": 0.05, "close_mosaic": 0}
RESERVED = ("model", "pose_crops", "resume", "export")


class YOLOTrainer(Trainer):
    name = "yolo"
    tested = False

    def _kwargs(self, dataset_dir: Path, out_dir: Path) -> Dict[str, Any]:
        crops = bool(self.opt.get("pose_crops")) and self.task == "pose"
        data = Path(dataset_dir) / ("pose_crops/data.yaml" if crops else "data.yaml")
        kw = dict(BASE)
        if crops:
            kw.update(CROP_POSE)
        kw.update({k: v for k, v in self.opt.items() if k not in RESERVED})
        kw.update(data=str(data), project=str(Path(out_dir)), name="train", exist_ok=True)
        return kw

    def _model(self) -> str:
        return str(self.opt.get("model") or MODELS[self.task])

    def plan(self, dataset_dir: Path, out_dir: Path) -> dict:
        kw = self._kwargs(dataset_dir, out_dir)
        last = Path(out_dir) / "train" / "weights" / "last.pt"
        resume = bool(self.opt.get("resume", False)) and last.exists()   # opt-in: an interrupted run only
        call = (f"YOLO({str(last)!r}).train(resume=True)" if resume else
                f"YOLO({self._model()!r}).train(" + ", ".join(f"{k}={v!r}" for k, v in kw.items()) + ")")
        return {"backend": self.name, "task": self.task, "framework": "ultralytics (AGPL-3.0)", "model": self._model(),
                "python": "from ultralytics import YOLO; " + call, "kwargs": kw, "resume": resume,
                "weights_out": str(Path(out_dir) / "train" / "weights" / "best.pt"),
                "env": {"YOLO_AUTOINSTALL": "false"}}

    def train(self, dataset_dir: Path, out_dir: Path) -> dict:
        os.environ.setdefault("YOLO_AUTOINSTALL", "false")
        ul = need("ultralytics", "train", "the YOLO trainer")
        p = self.plan(dataset_dir, out_dir)
        if p["resume"]:
            ul.YOLO(str(Path(out_dir) / "train" / "weights" / "last.pt")).train(resume=True)
        else:
            ul.YOLO(self._model()).train(**p["kwargs"])
        wdir = Path(out_dir) / "train" / "weights"
        best = wdir / "best.pt" if (wdir / "best.pt").exists() else wdir / "last.pt"
        if not best.exists():
            raise BackendError(f"yolo trainer: no weights were written under {wdir}")
        return {"weights": str(best), "run_dir": str(Path(out_dir) / "train")}

    def export_plan(self, weights, fmt: str) -> dict:
        kw = {"format": fmt, "imgsz": self.opt.get("export_imgsz", self.opt.get("imgsz", BASE["imgsz"])),
              "dynamic": False, "simplify": False}
        kw.update(self.opt.get("export") or {})
        return {"backend": self.name, "python": f"from ultralytics import YOLO; YOLO({str(weights)!r}).export("
                + ", ".join(f"{k}={v!r}" for k, v in kw.items()) + ")", "kwargs": kw,
                "needs": "the `onnx` package for format=onnx", "env": {"YOLO_AUTOINSTALL": "false"}}

    def export(self, weights, fmt: str) -> dict:
        os.environ.setdefault("YOLO_AUTOINSTALL", "false")
        ul = need("ultralytics", "train", "the YOLO exporter")
        if fmt == "onnx":
            need("onnx", "train", "ONNX export (pip install onnx)")
        out = ul.YOLO(str(weights)).export(**self.export_plan(weights, fmt)["kwargs"])
        return {"file": str(out)}
