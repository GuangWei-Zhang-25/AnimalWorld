"""Stage 2: generation. Walks prompts/manifest.jsonl as a work queue; resumable; several workers may share it."""
from __future__ import annotations

from pathlib import Path

from .. import registry
from ..config import Project
from ..utils import Claims, append_jsonl, now_utc, read_jsonl, write_jsonl


def index_rows(project: Project) -> list:
    return read_jsonl(project.path("generated", "index.jsonl"))


def _extract_frames(video: Path, out_dir: Path, unit: str, stride: int) -> list:
    cv2 = registry.need("cv2", "annotate", "extracting frames from generated clips")
    cap = cv2.VideoCapture(str(video))
    files, f = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if f % max(1, stride) == 0:
            iid = f"{unit}_f{f:04d}"
            p = out_dir / f"{iid}.jpg"
            cv2.imwrite(str(p), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            files.append({"id": iid, "file": str(p), "frame": f})
        f += 1
    cap.release()
    return files


def run(project: Project, dry_run: bool = False, limit: int = None, backend: str = None, **_) -> dict:
    cfg = project.cfg["generate"]
    name = backend or cfg["backend"]
    items = read_jsonl(project.path("prompts", "manifest.jsonl"))
    be = registry.create("generators", name, project.options("generate", name), project=project)
    img_dir = project.path("generated", "images")
    vid_dir = project.path("generated", "videos")
    done_units = {r["unit"] for r in index_rows(project)}
    todo = [m for m in items if m["id"] not in done_units]
    if limit:
        todo = todo[:int(limit)]
    plan = dict(be.plan(todo), stage="generate", manifest_items=len(items), already_done=len(done_units), todo=len(todo))
    if dry_run:
        return dict(plan, mode="dry-run")
    if not items:
        raise RuntimeError("no prompts: run `animalworld prompts` first")
    claims = Claims(project.path("generated", ".claims"))
    claims.reap(lambda u: u in done_units)
    mine = [m for m in todo if claims.claim(m["id"])]
    made = images = 0
    try:
        out_dir = vid_dir if be.kind == "video" else img_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        for res in be.generate(mine, out_dir):
            unit = res["id"]
            if be.kind == "video":
                img_dir.mkdir(parents=True, exist_ok=True)
                files = _extract_frames(Path(res["video"]), img_dir, unit, int(cfg.get("frame_stride", 8)))
            else:
                files = res.pop("files")
            for f in files:
                from PIL import Image
                with Image.open(f["file"]) as im:
                    W, H = im.size
                append_jsonl(project.path("generated", "index.jsonl"),
                             {"id": f["id"], "unit": unit, "file": str(Path(f["file"]).relative_to(project.root)),
                              "frame": f.get("frame", -1), "W": W, "H": H})
                images += 1
            meta = {k: v for k, v in res.items() if k not in ("files",)}
            if "video" in meta:
                meta["video"] = str(Path(meta["video"]).relative_to(project.root))
            append_jsonl(project.path("generated", "meta.jsonl"), dict(meta, utc=now_utc(), images=len(files)))
            claims.release(unit)
            made += 1
    finally:
        for m in mine:
            claims.release(m["id"])
    return dict(plan, mode="executed", units_generated=made, images_written=images,
                images_total=len(index_rows(project)))
