"""Run the stages in order (`animalworld pipeline`) and report where a project stands (`animalworld status`)."""
from __future__ import annotations

import importlib
from typing import Any, Dict, List

from . import STAGES
from .config import Project
from .utils import read_json, read_jsonl


def stage_fn(name: str):
    return importlib.import_module(f"animalworld.stages.{name}").run


def run_stage(project: Project, stage: str, /, **kw: Any) -> dict:
    res = stage_fn(stage)(project, **kw)
    project.record(stage, res.get("mode", "executed"), {k: v for k, v in res.items() if k not in ("kwargs", "python")})
    return res


def run_all(project: Project, dry_run: bool = False, auto_accept: bool = None, train_dry_run: bool = False,
            force_train: bool = False, until: str = None, start: str = None, echo=None) -> List[dict]:
    out = []
    names = list(STAGES)
    if start:
        names = names[names.index(start):]
    if until:
        names = names[:names.index(until) + 1]
    for name in names:
        kw: Dict[str, Any] = {"dry_run": dry_run}
        if name == "review":
            kw["auto"] = auto_accept
        if name in ("train", "export") and train_dry_run:
            kw["dry_run"] = True
        if name == "train":
            kw["force"] = force_train
        res = run_stage(project, name, **kw)
        out.append(res)
        if echo:
            echo(name, res)
    return out


def status(project: Project) -> dict:
    from .stages import review as R
    p = project.path
    man = read_jsonl(p("prompts", "manifest.jsonl"))
    rep = read_json(p("prompts", "report.json"), {}) or {}
    tax = read_json(p("taxonomy", "taxonomy.json"), {}) or {}
    c = R.counts(project)
    units_done = {r["unit"] for r in read_jsonl(p("generated", "index.jsonl"))}
    latest = read_json(p("models", "latest.json"), {}) or {}
    ds = {s: len(list(p("dataset", s, "images").glob("*"))) if p("dataset", s, "images").is_dir() else 0 for s in ("train", "val")}
    need = int(project.cfg["train"]["min_accepted_images"])
    preds = sorted(d.name for d in p("predictions").iterdir()) if p("predictions").is_dir() else []
    tracks = sorted(d.name for d in p("tracks").iterdir() if (d / "tracks.csv").exists()) if p("tracks").is_dir() else []
    st = project.state()
    return {
        "project": str(project.root), "species": project.species,
        "literature": {"records": len(read_jsonl(p("literature", "records.jsonl"))), "backend": project.cfg["literature"]["backend"]},
        "taxonomy": {"rules": tax.get("rules", 0), "slots_with_evidence": len(tax.get("slots", {}))},
        "prompts": {"items": len(man), "distinct_fraction": rep.get("distinct_fraction_non_empty"), "passed": rep.get("passed")},
        "generate": {"units_done": len(units_done), "units_total": len(man), "images": c["images"],
                     "backend": project.cfg["generate"]["backend"]},
        "propose": {"proposed": c["proposed"], "passed_gates": c["passed_gates"]},
        "review": {"accepted": c["accepted"], "accepted_by_human": c["accepted_by_human"], "rejected": c["rejected"], "pending": c["pending"]},
        "dataset": ds,
        "train": {"accepted_images": c["accepted"], "min_accepted_images": need, "gate_passed": c["accepted"] >= need,
                  "weights": latest.get("weights")},
        "export": {"file": latest.get("exported")},
        "predict": {"runs": preds},
        "track": {"runs": tracks},
        "last_modes": {k: v.get("mode") for k, v in st.items()},
    }
