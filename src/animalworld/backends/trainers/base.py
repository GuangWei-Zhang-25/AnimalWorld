"""Trainer backends.

    class MyTrainer(Trainer):
        name = "my_trainer"
        def plan(self, dataset_dir, out_dir): -> dict          # the exact call; must not import the framework
        def train(self, dataset_dir, out_dir): -> {"weights": path, ...}
        def export_plan(self, weights, fmt): -> dict
        def export(self, weights, fmt): -> {"file": path}
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


class Trainer:
    name = "base"
    tested = False

    def __init__(self, options: Dict[str, Any] = None, project=None, task: str = "detect", **_: Any):
        self.opt = options or {}
        self.project = project
        self.task = task

    def plan(self, dataset_dir: Path, out_dir: Path) -> dict:
        raise NotImplementedError

    def train(self, dataset_dir: Path, out_dir: Path) -> dict:
        raise NotImplementedError

    def export_plan(self, weights, fmt: str) -> dict:
        return {"backend": self.name, "note": "this trainer has no export step"}

    def export(self, weights, fmt: str) -> dict:
        raise NotImplementedError(f"{self.name}: export is not available")
