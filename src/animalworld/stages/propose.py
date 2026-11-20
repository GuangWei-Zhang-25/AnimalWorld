"""Stage 3a: proposal models label the generated images -> labels/proposed/<id>.json (with gate results)."""
from __future__ import annotations

from .. import registry
from ..config import Project
from ..formats.labels import apply_gates
from ..utils import read_jsonl, write_json
from .generate import index_rows


def run(project: Project, dry_run: bool = False, force: bool = False, backend: str = None, limit: int = None,
        weights: str = None, only_unreviewed: bool = False, include_reviewed: bool = False, **_) -> dict:
    cfg = project.cfg["propose"]
    name = backend or cfg["backend"]
    opts = project.options("propose", name)
    if weights:
        opts["weights"] = weights
    rows = index_rows(project)
    out = project.path("labels", "proposed")
    rev = project.path("labels", "reviewed")
    todo = [r for r in rows if force or not (out / f"{r['id']}.json").exists()]
    if (only_unreviewed or force) and not include_reviewed:
        todo = [r for r in todo if not (rev / f"{r['id']}.json").exists()]     # never overwrite a human decision
    if limit:
        todo = todo[:int(limit)]
    be = registry.create("proposers", name, opts, project=project)
    plan = dict(be.plan(todo), stage="propose", images=len(rows), todo=len(todo), gates=cfg["gates"])
    if dry_run:
        return dict(plan, mode="dry-run")
    n_by_unit = {m["id"]: m["n"] for m in read_jsonl(project.path("prompts", "manifest.jsonl"))}
    by_id = {r["id"]: r for r in todo}
    kept = rejected = 0
    reasons = {}
    for doc in be.propose([dict(r, path=str(project.path(r["file"]))) for r in todo]):
        r = by_id[doc["id"]]
        doc["file"] = r["file"]
        doc["unit"] = r["unit"]
        doc = apply_gates(doc, n_by_unit.get(r["unit"]), cfg["gates"])
        write_json(out / f"{doc['id']}.json", doc)
        if doc["kept"]:
            kept += 1
        else:
            rejected += 1
            key = doc["reason"].split(" ")[0]
            reasons[key] = reasons.get(key, 0) + 1
    return dict(plan, mode="executed", proposed=kept + rejected, passed_gates=kept, failed_gates=rejected,
                fail_reasons=reasons)
