"""Stage 5c: identity tracking over predictions -> tracks/<name>/tracks.jsonl + tracks.csv."""
from __future__ import annotations

import csv
from pathlib import Path

from .. import registry
from ..config import Project
from ..utils import read_json, read_jsonl, write_json, write_jsonl


def run(project: Project, dry_run: bool = False, name: str = None, backend: str = None, n_animals: int = None, **_) -> dict:
    cfg = project.cfg["track"]
    bname = backend or cfg["backend"]
    pdir = project.path("predictions")
    if not name and pdir.is_dir():
        cands = sorted((p for p in pdir.iterdir() if (p / "frames.jsonl").exists()), key=lambda p: p.stat().st_mtime)
        name = cands[-1].name if cands else None
    n_animals = n_animals or cfg.get("n_animals")
    plan = {"stage": "track", "backend": bname, "predictions": name, "n_animals": n_animals or "estimated from the detections"}
    if dry_run:
        return dict(plan, mode="dry-run")
    if not name:
        return dict(plan, mode="waiting", next="no predictions yet: run `animalworld predict --source ...`")
    rows = read_jsonl(pdir / name / "frames.jsonl")
    info = read_json(pdir / name / "info.json", {}) or {}
    tr = registry.create("trackers", bname, dict(project.options("track", bname), n_animals=n_animals), project=project,
                         info=info, work_dir=project.path("tracks", name))
    ids = tr.track([r["instances"] for r in rows])
    out = project.path("tracks", name)
    kp = project.keypoints()
    seen, table = set(), []
    for r, row_ids in zip(rows, ids):
        for inst, tid in zip(r["instances"], row_ids):
            inst["track_id"] = int(tid)
            if tid >= 0:
                seen.add(int(tid))
                x0, y0, x1, y1 = inst["box"]
                rec = {"frame": r["frame"], "id": int(tid), "x0": x0, "y0": y0, "x1": x1, "y1": y1, "score": inst.get("score", 1.0)}
                for k in kp:
                    v = (inst.get("keypoints") or {}).get(k)
                    rec[f"{k}_x"], rec[f"{k}_y"] = (v[0], v[1]) if v and v[2] > 0 else ("", "")
                table.append(rec)
    write_jsonl(out / "tracks.jsonl", rows)
    with open(out / "tracks.csv", "w", newline="", encoding="utf-8") as f:
        cols = ["frame", "id", "x0", "y0", "x1", "y1", "score"] + [f"{k}_{a}" for k in kp for a in "xy"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(table)
    per_frame = [sum(1 for t in row_ids if t >= 0) for row_ids in ids]
    summ = dict(plan, mode="executed", frames=len(rows), identities=len(seen), n_animals=getattr(tr, "n", n_animals),
                frames_with_all_ids=sum(1 for c in per_frame if c == getattr(tr, "n", -1)),
                unassigned_detections=sum(1 for row_ids in ids for t in row_ids if t < 0),
                output=str(Path(out / "tracks.csv")))
    write_json(out / "summary.json", summ)
    return summ
