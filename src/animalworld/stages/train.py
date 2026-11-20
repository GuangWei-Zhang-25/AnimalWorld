"""Stage 4: model training, gated on the number of accepted images.

Start small: below `train.min_accepted_images` the stage reports how many more accepted images are
needed (keep reviewing; use a quickly trained model as proposer with `animalworld propose
--backend yolo`). Above the threshold the scalable trainer runs. `--dry-run` prints the exact call.
"""
from __future__ import annotations

from .. import registry
from ..config import Project
from ..utils import now_utc, read_json, write_json
from . import review as R


def run(project: Project, dry_run: bool = False, force: bool = False, backend: str = None, task: str = None,
        overrides: dict = None, **_) -> dict:
    cfg = project.cfg["train"]
    name = backend or cfg["backend"]
    task = task or cfg["task"]
    opts = project.options("train", name)
    opts.update(overrides or {})
    be = registry.create("trainers", name, opts, project=project, task=task)
    acc = R.counts(project)["accepted"]
    need_n = int(cfg["min_accepted_images"])
    out_dir = project.path("models", f"{name}_{task}")
    plan = dict(be.plan(project.path("dataset"), out_dir), stage="train", accepted_images=acc,
                min_accepted_images=need_n, gate_passed=acc >= need_n)
    if dry_run:
        return dict(plan, mode="dry-run")
    if acc < need_n and not force:
        return dict(plan, mode="waiting",
                    next=f"{acc} accepted images < train.min_accepted_images = {need_n}: review {need_n - acc} more "
                         "(or lower the threshold, or pass --force to train a small model for the refine loop)")
    if not project.path("dataset", "data.yaml").exists():
        raise RuntimeError("no dataset: run `animalworld dataset` first")
    res = be.train(project.path("dataset"), out_dir)
    rec = dict(res, backend=name, task=task, utc=now_utc(), accepted_images=acc, train_options=opts)
    hist = read_json(project.path("models", "history.json"), []) or []
    hist.append(rec)
    write_json(project.path("models", "history.json"), hist)
    write_json(project.path("models", "latest.json"), rec)
    return dict(plan, mode="executed", **res)
