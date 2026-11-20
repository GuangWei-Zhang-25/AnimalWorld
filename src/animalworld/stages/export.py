"""Stage 5a: export trained weights for deployment (ONNX by default)."""
from __future__ import annotations

from pathlib import Path

from .. import registry
from ..config import Project
from ..utils import read_json, write_json


def run(project: Project, dry_run: bool = False, weights: str = None, fmt: str = None, **_) -> dict:
    cfg = project.cfg["export"]
    latest = read_json(project.path("models", "latest.json"), {}) or {}
    weights = weights or latest.get("weights")
    fmt = fmt or cfg["format"]
    name = latest.get("backend") or project.cfg["train"]["backend"]
    task = latest.get("task") or project.cfg["train"]["task"]
    opts = project.options("train", name)
    opts.update(latest.get("train_options") or {})       # e.g. the image size the model was trained at
    opts.update(cfg.get("options") or {})
    be = registry.create("trainers", name, opts,
                         project=project, task=task)
    plan = dict(be.export_plan(weights, fmt), stage="export", weights=weights, format=fmt)
    if dry_run:
        return dict(plan, mode="dry-run")
    if not weights or not Path(weights).exists():
        return dict(plan, mode="waiting", next="no trained weights yet: run `animalworld train` (or pass --weights)")
    res = be.export(weights, fmt)
    latest = dict(latest, exported=res.get("file"), export_format=fmt)
    write_json(project.path("models", "latest.json"), latest)
    return dict(plan, mode="executed", **res)
