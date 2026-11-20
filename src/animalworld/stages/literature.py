"""Stage 1a: literature search for the species -> literature/records.jsonl (cached, resumable)."""
from __future__ import annotations

from ..config import Project
from .. import registry
from ..utils import read_jsonl, write_json, write_jsonl


def run(project: Project, dry_run: bool = False, force: bool = False, backend: str = None, limit: int = None,
        query: str = None, **_) -> dict:
    cfg = project.cfg["literature"]
    name = backend or cfg["backend"]
    limit = int(limit or cfg["limit"])
    opts = {k: v for k, v in cfg.items() if k not in ("backend", "query", "limit")}
    if opts.get("source"):
        opts["source"] = str(project.resolve(opts["source"]))
    be = registry.create("literature", name, opts, cache_dir=project.path("literature", "cache"))
    an = project.slots().get("animal") or {}
    q = query or cfg.get("query") or be.default_query(project.species, an.get("synonyms") or [], project.cfg["description"])
    out = project.path("literature", "records.jsonl")
    plan = {"stage": "literature", "backend": name, "query": q, "limit": limit, "output": str(out),
            "network": bool(be.needs_network)}
    if dry_run:
        return dict(plan, mode="dry-run")
    have = read_jsonl(out)
    meta = project.path("literature", "report.json")
    if have and not force and (len(have) >= limit or name == "offline"):
        return dict(plan, mode="skipped", reason="records.jsonl already present (use --force to search again)",
                    records=len(have))
    recs = be.search(q, limit)
    seen, uniq = set(), []
    for r in recs:
        if r["id"] not in seen:
            seen.add(r["id"]); uniq.append(r)
    write_jsonl(out, uniq)
    rep = dict(plan, mode="executed", records=len(uniq), with_abstract=sum(1 for r in uniq if len(r["abstract"]) >= 60),
               requests_made=be.requests_made)
    write_json(meta, rep)
    return rep
