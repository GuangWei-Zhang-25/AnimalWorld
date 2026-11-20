"""RF-DETR trainer (detection / segmentation).

Adapted from mw_det_syn/finetune_v2/train_e.py:

    from rfdetr import RFDETRSegLarge
    m = RFDETRSegLarge(resolution=768)                       # COCO-pretrained weights
    m.train(dataset_dir=..., dataset_file="yolo", epochs=40, batch_size=4, grad_accum_steps=4,
            early_stopping=True, early_stopping_patience=3, early_stopping_min_delta=0.002,
            checkpoint_interval=1, eval_interval=1, lr=1e-4, output_dir=..., devices=1,
            num_workers=2, resume=<last.ckpt or None>)

`dataset_dir` is the YOLO-layout dataset written by `animalworld dataset` (data.yaml, train/images,
train/labels, val/...). The original script's walltime handling (mid-epoch checkpoint + exit code 75
for self-resubmitting SLURM chunks) is cluster specific and is not ported; resuming from the newest
checkpoint in the output folder is.

NOT EXECUTED in this package's tests (needs the rfdetr package, COCO weights and a GPU).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from ...registry import BackendError, need
from .base import Trainer

DEFAULTS = {"model_class": None, "resolution": 768, "epochs": 40, "batch_size": 4, "grad_accum_steps": 4,
            "early_stopping": True, "early_stopping_patience": 3, "early_stopping_min_delta": 0.002,
            "checkpoint_interval": 1, "eval_interval": 1, "lr": 1e-4, "devices": 1, "num_workers": 2}
CLASS_BY_TASK = {"detect": "RFDETRBase", "segment": "RFDETRSegLarge"}


class RFDETRTrainer(Trainer):
    name = "rfdetr"
    tested = False

    def __init__(self, options: Dict[str, Any] = None, project=None, task: str = "segment", **kw: Any):
        super().__init__({**DEFAULTS, **(options or {})}, project, task, **kw)
        if task not in CLASS_BY_TASK:
            raise BackendError("rfdetr trainer: task must be detect or segment (use the yolo trainer for pose)")

    def _resume(self, out_dir: Path):
        cands = [p for p in (Path(out_dir) / "last.ckpt", Path(out_dir) / "mid.ckpt") if p.exists()]
        return str(max(cands, key=lambda p: p.stat().st_mtime)) if cands else None

    def _kwargs(self, dataset_dir: Path, out_dir: Path) -> Dict[str, Any]:
        kw = {k: v for k, v in self.opt.items() if k not in ("model_class", "resolution", "export")}
        kw.update(dataset_dir=str(Path(dataset_dir)), dataset_file="yolo", output_dir=str(Path(out_dir)),
                  resume=self._resume(out_dir))
        return kw

    def plan(self, dataset_dir: Path, out_dir: Path) -> dict:
        cls = self.opt["model_class"] or CLASS_BY_TASK[self.task]
        kw = self._kwargs(dataset_dir, out_dir)
        return {"backend": self.name, "task": self.task, "framework": "rfdetr", "model": cls,
                "python": f"from rfdetr import {cls}; {cls}(resolution={self.opt['resolution']}).train("
                          + ", ".join(f"{k}={v!r}" for k, v in kw.items()) + ")",
                "kwargs": kw, "weights_out": str(Path(out_dir) / "checkpoint_best_total.pth") + " (or last.ckpt)",
                "needs_gpu": True}

    def train(self, dataset_dir: Path, out_dir: Path) -> dict:
        rfdetr = need("rfdetr", "train", "the RF-DETR trainer")
        cls = self.opt["model_class"] or CLASS_BY_TASK[self.task]
        Model = getattr(rfdetr, cls, None)
        if Model is None:
            raise BackendError(f"rfdetr has no class {cls}; set train.options.model_class")
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        Model(resolution=int(self.opt["resolution"])).train(**self._kwargs(dataset_dir, out_dir))
        cands = sorted([p for pat in ("checkpoint_best*.pth", "*.ckpt", "*.pth") for p in Path(out_dir).glob(pat)],
                       key=lambda p: p.stat().st_mtime)
        if not cands:
            raise BackendError(f"rfdetr trainer: no checkpoint found in {out_dir}")
        best = [p for p in cands if "best" in p.name]
        return {"weights": str((best or cands)[-1]), "run_dir": str(out_dir)}

    def export_plan(self, weights, fmt: str) -> dict:
        return {"backend": self.name, "python": f"from rfdetr import RFDETR; RFDETR.from_checkpoint({str(weights)!r}, "
                f"trust_checkpoint=True).export(output_dir=...)", "format": "onnx",
                "note": "untested; RF-DETR checkpoints can also be used directly by the `rfdetr` predictor"}

    def export(self, weights, fmt: str) -> dict:
        if fmt != "onnx":
            raise BackendError("rfdetr export supports onnx only")
        rfdetr = need("rfdetr", "train", "the RF-DETR exporter")
        out = Path(weights).parent / "onnx"
        rfdetr.RFDETR.from_checkpoint(str(weights), trust_checkpoint=True).export(output_dir=str(out))
        files = sorted(out.glob("*.onnx"))
        if not files:
            raise BackendError("rfdetr export wrote no .onnx file")
        return {"file": str(files[-1])}
