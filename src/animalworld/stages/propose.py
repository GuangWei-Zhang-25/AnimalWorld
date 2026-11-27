"""Stage 3a: proposal models label the generated images -> labels/proposed/<id>.json (with gate results).

For units with several frames (a generated clip) the stage then links the proposed boxes over the frames and writes
a proposed identity (`track_id`) per instance: see `link_identities`. propose.link_identities: false turns it off."""
from __future__ import annotations

from .. import registry
from ..config import Project
from ..formats.labels import apply_gates
from ..utils import read_json, read_jsonl, write_json
from .generate import index_rows


def link_identities(project: Project, units: set, n_by_unit: dict, options: dict = None) -> dict:
    """Give every proposed instance a `track_id` that is the same animal on all frames of one generation unit
    (a generated clip, or the frames of one mock unit).

    Method: the package's box tracker (`trackers.iou`: Hungarian assignment on centre distance and box overlap,
    seeded on the first frame where all animals are apart), with the prompted animal count of the unit. It is a
    PROPOSAL like the boxes: it does not use appearance or a video model, so identities can swap when animals
    cross; the reviewer checks and repairs them in the annotator (keys 0-9). Only labels/proposed is
    written, never a review record. A unit with a single proposed frame is left as its proposer wrote it.
    """
    rows = [r for r in index_rows(project) if r["unit"] in units]
    by_unit: dict = {}
    for r in rows:
        by_unit.setdefault(r["unit"], []).append(r)
    out_dir = project.path("labels", "proposed")
    linked = frames = 0
    for unit, rs in by_unit.items():
        rs = sorted(rs, key=lambda r: r.get("frame", 0))
        docs = [read_json(out_dir / f"{r['id']}.json") for r in rs]
        have = [(r, d) for r, d in zip(rs, docs) if d]
        if len(have) < 2:
            continue
        n = n_by_unit.get(unit)
        tr = registry.create("trackers", "iou", dict(options or {}, n_animals=n or None))
        ids = tr.track([d["instances"] for _, d in have])
        for (r, d), row_ids in zip(have, ids):
            for inst, tid in zip(d["instances"], row_ids):
                inst["track_id"] = int(tid) if tid >= 0 else None
            d["identity_source"] = "iou_link"
            write_json(out_dir / f"{r['id']}.json", d)
            frames += 1
        linked += 1
    return {"units_linked": linked, "frames_linked": frames}


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
    link = {}
    if cfg.get("link_identities", True):
        link = link_identities(project, {r["unit"] for r in todo}, n_by_unit, cfg.get("link_options"))
    return dict(plan, mode="executed", proposed=kept + rejected, passed_gates=kept, failed_gates=rejected,
                fail_reasons=reasons, **link)
