"""Stage 1c: diversified text prompts -> prompts/manifest.jsonl (one row per image or clip to generate)."""
from __future__ import annotations

import yaml

from .. import registry
from ..backends.writers.template import distinct_report
from ..config import Project
from ..utils import atomic_write_text, read_json, read_jsonl, write_json, write_jsonl


class DistinctError(RuntimeError):
    pass


def run(project: Project, dry_run: bool = False, force: bool = False, n: int = None, writer: str = None, **_) -> dict:
    cfg = project.cfg["prompts"]
    n = int(n or cfg["n"])
    wname = writer or cfg["writer"]
    out = project.path("prompts", "manifest.jsonl")
    plan = {"stage": "prompts", "writer": wname, "n": n, "seed": cfg["seed"], "min_distinct": cfg["min_distinct"],
            "output": str(out)}
    if dry_run:
        return dict(plan, mode="dry-run")
    have = read_jsonl(out)
    if len(have) >= n and not force:
        return dict(plan, mode="skipped", reason="manifest already has enough rows", items=len(have))
    tax = read_json(project.path("taxonomy", "taxonomy.json"), {}) or {}
    w = registry.create("writers", wname, cfg.get("writer_options") or {}, candidates=tax.get("candidates") or {})
    expanded_file = project.path("prompts", "slots_expanded.yaml")
    if wname != "template" and expanded_file.exists() and not force:
        slots = yaml.safe_load(expanded_file.read_text(encoding="utf-8"))
    else:
        slots = w.expand_slots(project.slots(), project.species, project.cfg["description"])
        if wname != "template":
            atomic_write_text(expanded_file, yaml.safe_dump(slots, sort_keys=False, allow_unicode=True, width=120))
    start = 0 if force else len(have)                 # append: earlier rows (and their images) stay valid
    items = ([] if force else have) + [
        w.item(i, slots, project.species, seed=cfg["seed"], empty_fraction=cfg["empty_fraction"],
               val_fraction=cfg["val_fraction"], megapixels=cfg["megapixels"], size_multiple=cfg["size_multiple"],
               weights=tax.get("weights")) for i in range(start, n)]
    rep = distinct_report(items)
    rep["min_distinct"] = cfg["min_distinct"]
    rep["passed"] = rep["distinct_fraction_non_empty"] >= float(cfg["min_distinct"])
    rep["literature_weights_used"] = bool(tax.get("weights"))
    write_json(project.path("prompts", "report.json"), rep)
    if not rep["passed"]:
        raise DistinctError(
            f"only {100 * rep['distinct_fraction_non_empty']:.1f} % of the prompts are distinct (need "
            f">= {100 * float(cfg['min_distinct']):.0f} %): add values to slots.yaml or lower prompts.n "
            f"(report: {project.path('prompts', 'report.json')})")
    write_jsonl(out, items)
    return dict(plan, mode="executed", items=len(items), new=len(items) - (0 if force else len(have)),
                empties=rep["empties"], distinct_fraction=rep["distinct_fraction_non_empty"], passed=True)
