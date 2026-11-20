"""Stage 3c: accepted labels -> a species dataset in standard formats.

    dataset/
      train/images/*.jpg  train/labels/*.txt     YOLO (detect / segment / pose; Ultralytics and RF-DETR 'yolo' layout)
      val/images/*.jpg    val/labels/*.txt
      data.yaml
      coco/instances_train.json, instances_val.json   (file_name is relative to dataset/)
      pose_crops/...                              optional top-down crop pose dataset
      manifest.csv                                one row per image: split, unit, counts, look

The train / val split is by generation UNIT (one prompt or one clip), never by frame, so frames of
one clip cannot leak across the split (as in mw_det_syn/gap_targeted/scripts/build.py).
"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

from ..config import Project
from ..formats import coco, yolo
from ..utils import atomic_write_text, md5_unit, read_jsonl, seed_from
from . import review as R


def run(project: Project, dry_run: bool = False, task: str = None, **_) -> dict:
    cfg = project.cfg["dataset"]
    task = task or cfg["task"]
    docs = R.accepted_docs(project)
    D = project.path("dataset")
    slots = project.slots()
    cats, kp = project.categories(), project.keypoints()
    plan = {"stage": "dataset", "task": task, "formats": cfg["formats"], "accepted_images": len(docs), "output": str(D),
            "apply_looks": bool(cfg.get("apply_looks")), "pose_crops": bool(cfg.get("pose_crops"))}
    if dry_run:
        return dict(plan, mode="dry-run")
    if not docs:
        return dict(plan, mode="waiting", next="no accepted labels yet: run `animalworld review`")
    if task == "pose" and not kp:
        raise RuntimeError("dataset.task is pose but slots.yaml lists no keypoints")
    man = {m["id"]: m for m in read_jsonl(project.path("prompts", "manifest.jsonl"))}
    val_fraction = float(project.cfg["prompts"]["val_fraction"])
    split_of = {}
    for d in docs:
        u = d.get("unit", d["id"])
        split_of[d["id"]] = man[u]["split"] if u in man else ("val" if md5_unit(u) < val_fraction else "train")
    units = sorted({d.get("unit", d["id"]) for d in docs}, key=md5_unit)
    if len(units) >= 2:                                   # trainers need both splits: move whole units, never frames
        for want in ("val", "train"):
            if want not in split_of.values():
                u = units[0] if want == "val" else units[-1]
                for d in docs:
                    if d.get("unit", d["id"]) == u:
                        split_of[d["id"]] = want
    # validation needs at least one image WITH animals (found in the first real run: the only val image was the
    # empty scene, so every validation metric was 0); move one whole non-empty unit if necessary
    n_inst = {}
    for d in docs:
        n_inst[d.get("unit", d["id"])] = n_inst.get(d.get("unit", d["id"]), 0) + len(d["instances"])
    unit_split = {d.get("unit", d["id"]): split_of[d["id"]] for d in docs}
    full = [u for u in units if n_inst[u] > 0]
    if len(full) >= 2 and not any(unit_split[u] == "val" for u in full):
        for d in docs:
            if d.get("unit", d["id"]) == full[0]:
                split_of[d["id"]] = "val"
    from PIL import Image
    from ..backends.generators import looks as LK
    for s in ("train", "val"):
        for k in ("images", "labels"):
            (D / s / k).mkdir(parents=True, exist_ok=True)
    keep = {"train": set(), "val": set()}
    rows, by_split = [], {"train": [], "val": []}
    for d in docs:
        s = split_of[d["id"]]
        src = project.path(d["file"])
        name = d["id"] + ".jpg"
        dst = D / s / "images" / name
        look = (man.get(d.get("unit", d["id"])) or {}).get("look", "none") if cfg.get("apply_looks") else "none"
        if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
            if look == "none" and src.suffix.lower() in (".jpg", ".jpeg"):
                shutil.copy(src, dst)
            else:
                P = LK.params(look, d.get("unit", d["id"]))
                with Image.open(src) as im:
                    LK.apply(im.convert("RGB"), P, seed_from(d["id"])).save(dst, quality=P["q"])
        (D / s / "labels" / f"{d['id']}.txt").write_text(yolo.to_lines(d, cats, task, kp), encoding="utf-8")
        keep[s].add(d["id"])
        by_split[s].append(dict(d, file=f"{s}/images/{name}"))
        rows.append({"file": f"{s}/images/{name}", "id": d["id"], "unit": d.get("unit", d["id"]), "split": s,
                     "instances": len(d["instances"]), "origin": d.get("origin", ""), "source": d.get("source", ""), "look": look})
    removed = 0
    for s in ("train", "val"):                            # images whose label was un-accepted since the last build
        for p in list((D / s / "images").iterdir()) + list((D / s / "labels").iterdir()):
            if p.stem not in keep[s]:
                p.unlink()
                removed += 1
    atomic_write_text(D / "data.yaml", yolo.data_yaml(D, cats, task, kp, slots.get("flip_idx")))
    if "coco" in cfg["formats"]:
        (D / "coco").mkdir(exist_ok=True)
        for s in ("train", "val"):
            atomic_write_text(D / "coco" / f"instances_{s}.json", json.dumps(coco.to_coco(by_split[s], cats, kp, slots.get("skeleton"))))
    with open(D / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    crops = None
    if cfg.get("pose_crops") and kp:
        crops = {"train": 0, "val": 0}
        PC = D / "pose_crops"
        for s in ("train", "val"):
            for k in ("images", "labels"):
                shutil.rmtree(PC / s / k, ignore_errors=True)
                (PC / s / k).mkdir(parents=True)
            for d in by_split[s]:
                with Image.open(D / d["file"]) as im:
                    im = im.convert("RGB")
                    for j, inst in enumerate(d["instances"]):
                        res = yolo.pose_crop(im, inst, kp)
                        if res:
                            res[0].save(PC / s / "images" / f"{d['id']}_{j}.jpg", quality=92)
                            (PC / s / "labels" / f"{d['id']}_{j}.txt").write_text(res[1], encoding="utf-8")
                            crops[s] += 1
        atomic_write_text(PC / "data.yaml", yolo.data_yaml(PC, cats, "pose", kp, slots.get("flip_idx")))
    return dict(plan, mode="executed", train=len(keep["train"]), val=len(keep["val"]),
                instances=sum(r["instances"] for r in rows), empty_images=sum(1 for r in rows if not r["instances"]),
                stale_files_removed=removed, pose_crops=crops, data_yaml=str(D / "data.yaml"))
