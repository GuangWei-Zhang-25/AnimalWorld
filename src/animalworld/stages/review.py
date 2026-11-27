"""Stage 3b: human-in-the-loop review. Proposals are accepted, corrected or rejected; the result is
labels/reviewed/<id>.json. Also: import / export of labels (COCO, YOLO, DeepLabCut, SLEAP) and
registration of real images for purely manual annotation."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import List, Optional

from ..config import Project
from ..formats import labels as L
from ..utils import IMAGE_EXT, append_jsonl, now_utc, read_json, write_json
from .generate import index_rows


def proposed(project: Project, iid: str) -> Optional[dict]:
    return read_json(project.path("labels", "proposed", f"{iid}.json"))


def reviewed(project: Project, iid: str) -> Optional[dict]:
    return read_json(project.path("labels", "reviewed", f"{iid}.json"))


def _geometry(instances: List[dict]) -> list:
    """What a reviewer can change, rounded to 0.1 px, in a form that compares equal for an untouched proposal."""
    r = lambda v: round(float(v), 1)                                     # noqa: E731
    out = []
    for i in instances or []:
        kp = {k: [r(v[0]), r(v[1]), int(v[2])] for k, v in (i.get("keypoints") or {}).items() if v and int(v[2]) > 0}
        out.append(([r(v) for v in i["box"]], [[r(x), r(y)] for x, y in (i.get("poly") or [])], sorted(kp.items()),
                    i.get("track_id")))
    return sorted(out, key=repr)


def save_review(project: Project, row: dict, status: str, instances: List[dict], origin: str, source: str = "") -> dict:
    """Write the review record of one image.

    origin: who decided: "human" (annotator window), "auto" (automatic acceptance) or "import".
    The record also keeps `proposal_source` (the proposal model the reviewer started from, or None when there was
    no proposal) and `edited` (True when the stored boxes / outlines / keypoints / identities differ from that
    proposal by more than 0.1 px, False when they do not, None when there was no proposal to compare with), so
    accepted-as-proposed and corrected-then-accepted can be counted apart.
    """
    if status not in ("accepted", "rejected"):
        raise ValueError("status must be accepted or rejected")
    pr = proposed(project, row["id"])
    doc = L.label(row["id"], row["file"], row["W"], row["H"], instances, source=source or origin, status=status,
                  unit=row.get("unit", row["id"]), origin=origin, reviewed_utc=now_utc(),
                  proposal_source=pr.get("source") if pr else None,
                  edited=(_geometry(instances) != _geometry(pr["instances"])) if pr else None)
    write_json(project.path("labels", "reviewed", f"{row['id']}.json"), doc)
    return doc


def counts(project: Project) -> dict:
    rows = index_rows(project)
    c = {"images": len(rows), "proposed": 0, "passed_gates": 0, "accepted": 0, "rejected": 0, "pending": 0,
         "accepted_by_human": 0, "accepted_after_edit": 0}
    for r in rows:
        rv, pr = reviewed(project, r["id"]), proposed(project, r["id"])
        if pr:
            c["proposed"] += 1
            c["passed_gates"] += int(bool(pr.get("kept")))
        if rv:
            c[rv["status"]] += 1
            c["accepted_by_human"] += int(rv["status"] == "accepted" and rv.get("origin") == "human")
            c["accepted_after_edit"] += int(rv["status"] == "accepted" and rv.get("origin") == "human" and bool(rv.get("edited")))
        else:
            c["pending"] += 1
    return c


def accepted_docs(project: Project) -> List[dict]:
    out = []
    for r in index_rows(project):
        rv = reviewed(project, r["id"])
        if rv and rv["status"] == "accepted":
            out.append(rv)
    return out


def auto_accept(project: Project) -> dict:
    """Accept every un-reviewed proposal that passed the gates; reject those that failed (origin: auto)."""
    acc = rej = 0
    for r in index_rows(project):
        if reviewed(project, r["id"]):
            continue
        pr = proposed(project, r["id"])
        if not pr:
            continue
        if pr.get("kept"):
            save_review(project, r, "accepted", pr["instances"], "auto", pr.get("source", ""))
            acc += 1
        else:
            d = save_review(project, r, "rejected", pr["instances"], "auto", pr.get("source", ""))
            d["reason"] = pr.get("reason", "")
            write_json(project.path("labels", "reviewed", f"{r['id']}.json"), d)
            rej += 1
    return {"auto_accepted": acc, "auto_rejected": rej}


def add_images(project: Project, folder: Path, copy: bool = True) -> dict:
    """Register real images (for manual annotation or label import). They are copied to images/real/."""
    from PIL import Image
    have = {r["id"] for r in index_rows(project)}
    dst = project.path("images", "real")
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(Path(folder).iterdir()):
        if p.suffix.lower() not in IMAGE_EXT or p.stem in have:
            continue
        q = dst / p.name
        if copy and not q.exists():
            shutil.copy(p, q)
        with Image.open(q) as im:
            W, H = im.size
        append_jsonl(project.path("generated", "index.jsonl"),
                     {"id": p.stem, "unit": p.stem, "file": str(q.relative_to(project.root)), "frame": -1, "W": W, "H": H,
                      "real": True})
        have.add(p.stem)
        n += 1
    return {"images_added": n, "folder": str(dst)}


def import_labels(project: Project, fmt: str, path: Path, images: Path = None, status: str = "accepted") -> dict:
    """Import labels made elsewhere. Images are matched by file stem; unknown images are registered first
    from `images` (or from the folder of the label file).

    status "accepted" (default): the labels are trusted as they are and written to labels/reviewed with
    origin "import", so they enter the dataset without passing the annotator.
    status "proposed": they are written to labels/proposed (source "import:<format>") and wait for review like
    the output of a proposal model; images that already have a review record keep it.
    """
    if status not in ("accepted", "proposed"):
        raise ValueError("status must be accepted or proposed")
    path = Path(path)
    cat = project.categories()[0]
    kp = project.keypoints()
    if fmt == "coco":
        from ..formats import coco
        docs = coco.from_coco(json.loads(path.read_text(encoding="utf-8")))
    elif fmt == "dlc":
        from ..formats import dlc
        docs = dlc.import_dlc(path, category=cat)["docs"]
    elif fmt == "sleap":
        from ..formats import sleap
        docs = sleap.import_sleap(path, category=cat)["docs"]
    elif fmt == "yolo":
        from PIL import Image
        from ..formats import yolo
        task = project.cfg["dataset"]["task"]
        img_dir = Path(images) if images else path.parent / "images"
        docs = []
        for t in sorted(path.glob("*.txt")):
            img = next((img_dir / (t.stem + e) for e in IMAGE_EXT if (img_dir / (t.stem + e)).exists()), None)
            if img is None:
                continue
            with Image.open(img) as im:
                W, H = im.size
            docs.append(L.label(t.stem, img.name, W, H, yolo.from_lines(t.read_text(), W, H, project.categories(), task, kp), "yolo"))
    else:
        raise ValueError("format must be coco, yolo, dlc or sleap")
    if fmt != "yolo":
        img_dir = Path(images) if images else (path if path.is_dir() else path.parent)
    if img_dir.is_dir():
        add_images(project, img_dir)
    rows = {r["id"]: r for r in index_rows(project)}
    n = miss = 0
    for d in docs:
        r = rows.get(d["id"])
        if not r:
            miss += 1
            continue
        for i in d["instances"]:
            i["category"] = cat if len(project.categories()) == 1 else i["category"]
        if status == "proposed":
            write_json(project.path("labels", "proposed", f"{r['id']}.json"),
                       L.label(r["id"], r["file"], r["W"], r["H"], d["instances"], source=f"import:{fmt}",
                               unit=r.get("unit", r["id"])))
        else:
            save_review(project, r, status, d["instances"], "import", f"import:{fmt}")
        n += 1
    return {"format": fmt, "imported": n, "images_not_found": miss, "status": status,
            "note": ("written to labels/proposed: review them with `animalworld review --gui`" if status == "proposed" else
                     "written to labels/reviewed as accepted with origin 'import' (not reviewed in this package)")}


def export_labels(project: Project, fmt: str, out: Path = None, task: str = None) -> dict:
    docs = accepted_docs(project)
    out = Path(out) if out else project.path("exchange", fmt)
    out.mkdir(parents=True, exist_ok=True)
    slots = project.slots()
    kp, skel, cats = project.keypoints(), slots.get("skeleton") or [], project.categories()
    if fmt == "coco":
        from ..formats import coco
        (out / "annotations.json").write_text(json.dumps(coco.to_coco(docs, cats, kp, skel)), encoding="utf-8")
        return {"format": fmt, "images": len(docs), "file": str(out / "annotations.json"), "file_names_relative_to": str(project.root)}
    if fmt == "dlc":
        from ..formats import dlc
        return dict(dlc.export_dlc(docs, kp, out, project.root, video=project.species), format=fmt)
    if fmt == "sleap":
        from ..formats import sleap
        return dict(sleap.export_sleap(docs, cats, kp, skel, out, project.root), format=fmt)
    if fmt == "yolo":
        # Label files only, without the train / val split and the image copies that `animalworld dataset` makes.
        from ..formats import yolo
        task = task or project.cfg["dataset"]["task"]
        (out / "labels").mkdir(exist_ok=True)
        for d in docs:
            (out / "labels" / f"{d['id']}.txt").write_text(yolo.to_lines(d, cats, task, kp), encoding="utf-8")
        (out / "classes.txt").write_text("".join(c + "\n" for c in cats), encoding="utf-8")
        (out / "images.txt").write_text("".join(str((project.root / d["file"]).resolve()) + "\n" for d in docs), encoding="utf-8")
        return {"format": fmt, "task": task, "images": len(docs), "labels": str(out / "labels"), "classes": str(out / "classes.txt"),
                "image_list": str(out / "images.txt"),
                "note": "label files only; `animalworld dataset` writes the complete YOLO dataset (images, split, data.yaml)"}
    raise ValueError("format must be coco, yolo, dlc or sleap")


def run(project: Project, dry_run: bool = False, auto: bool = None, **_) -> dict:
    """Counts, and optionally the automatic acceptance. Reviewing by a person happens in the annotator window
    (`animalworld review PROJECT --gui`, or the Review row of Animal World Studio)."""
    cfg = project.cfg["review"]
    auto = cfg["auto_accept"] if auto is None else auto
    c = counts(project)
    plan = {"stage": "review", "auto_accept": bool(auto), **c}
    if dry_run:
        return dict(plan, mode="dry-run")
    res = {}
    if auto:
        res = auto_accept(project)
    c = counts(project)
    out = dict(plan, mode="executed" if auto else "waiting", **res, **c)
    if c["pending"]:
        out["next"] = (f"{c['pending']} image(s) wait for review: run `animalworld review {project.root} --gui` "
                       "(or --auto to accept every proposal that passed the gates)")
    return out
