"""Contact sheets: thumbnails with boxes, outlines, keypoints and identities drawn (PIL only).

Used to LOOK at proposals, labels and predictions before trusting any number:
    animalworld sheet PROJECT --what proposals      labels/proposed  (gate result in the caption)
    animalworld sheet PROJECT --what reviewed       labels/reviewed
    animalworld sheet PROJECT --what predictions --name NAME [--tracks]
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image, ImageDraw

from .config import Project
from .utils import read_json, read_jsonl

COL = [(230, 25, 75), (60, 180, 75), (67, 99, 216), (245, 130, 49), (145, 30, 180), (66, 212, 244), (240, 50, 230),
       (191, 239, 69), (250, 190, 212), (70, 153, 144), (154, 99, 36), (128, 0, 0)]


def draw_instances(im: Image.Image, instances: List[dict], scale: float = 1.0, by_track: bool = False) -> Image.Image:
    d = ImageDraw.Draw(im)
    lw = max(2, round(im.width / 220))
    for k, i in enumerate(instances):
        tid = i.get("track_id")
        c = COL[(tid if (by_track and tid is not None and tid >= 0) else k) % len(COL)]
        if by_track and tid is not None and tid < 0:
            c = (150, 150, 150)
        if i.get("poly"):
            pts = [(x * scale, y * scale) for x, y in i["poly"]]
            d.line(pts + pts[:1], fill=c, width=lw)
        x0, y0, x1, y1 = [v * scale for v in i["box"]]
        d.rectangle([x0, y0, x1, y1], outline=c, width=max(1, lw - 1))
        tag = (f"id {tid}" if by_track and tid is not None else str(k + 1)) + (f" {i['score']:.2f}" if i.get("score", 1) < 1 else "")
        d.text((x0 + 2, max(0, y0 - 11)), tag, fill=c)
        for v in (i.get("keypoints") or {}).values():
            if v and v[2] > 0:
                d.ellipse([v[0] * scale - 2, v[1] * scale - 2, v[0] * scale + 2, v[1] * scale + 2], fill=c)
    return im


def contact_sheet(tiles: List[Tuple[Image.Image, List[dict], str]], out: Path, cols: int = 6, tile_w: int = 360,
                  by_track: bool = False) -> Path:
    """tiles: (image, instances, caption). Images are scaled to tile_w; captions go under each tile."""
    cells = []
    for im, inst, cap in tiles:
        s = tile_w / im.width
        t = im.convert("RGB").resize((tile_w, max(1, round(im.height * s))))
        cells.append((draw_instances(t, inst, s, by_track), cap))
    th = max(c[0].height for c in cells) + 30
    rows = (len(cells) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tile_w, rows * th), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    for k, (t, cap) in enumerate(cells):
        x, y = (k % cols) * tile_w, (k // cols) * th
        sheet.paste(t, (x, y))
        d.text((x + 4, y + th - 28), cap[:tile_w // 6], fill=(0, 0, 0))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=90)
    return out


def _video_frames(path: Path, wanted: List[int]) -> dict:
    from .registry import need
    cv2 = need("cv2", "deploy", "reading video frames for a sheet")
    cap = cv2.VideoCapture(str(path))
    out, f, want = {}, 0, set(wanted)
    while want:
        ok, fr = cap.read()
        if not ok:
            break
        if f in want:
            out[f] = Image.fromarray(fr[:, :, ::-1].copy())
            want.discard(f)
        f += 1
    cap.release()
    return out


def make(project: Project, what: str, name: Optional[str] = None, out: Optional[Path] = None, limit: int = 48,
         cols: int = 6, tracks: bool = False, every: int = 1) -> dict:
    tiles = []
    if what in ("proposals", "reviewed"):
        sub = "proposed" if what == "proposals" else "reviewed"
        for r in read_jsonl(project.path("generated", "index.jsonl")):
            d = read_json(project.path("labels", sub, f"{r['id']}.json"))
            if not d:
                continue
            cap = f"{r['id']}  n={len(d['instances'])}" + (f" want {d['expected_n']}" if d.get("expected_n") is not None else "")
            cap += ("  PASS" if d.get("kept") else f"  FAIL: {d.get('reason', '')}") if what == "proposals" else f"  {d['status']} ({d.get('origin', '')})"
            tiles.append((Image.open(project.path(r["file"])), d["instances"], cap))
            if len(tiles) >= limit:
                break
        out = out or project.path("evidence", f"{what}_sheet.jpg")
    elif what == "predictions":
        if not name:                              # default: the newest run (as `track` does)
            root = project.path("tracks" if tracks else "predictions")
            cands = sorted((d for d in root.iterdir() if d.is_dir()), key=lambda d: d.stat().st_mtime) if root.is_dir() else []
            if not cands:
                raise RuntimeError("no predictions yet: run `animalworld predict --source ...`" + (" and `animalworld track`" if tracks else ""))
            name = cands[-1].name
        base = project.path("tracks" if tracks else "predictions", name)
        rows = read_jsonl(base / ("tracks.jsonl" if tracks else "frames.jsonl"))[::max(1, every)][:limit]
        info = read_json(project.path("predictions", name, "info.json"), {}) or {}
        src = Path(info.get("source", ""))
        vid = _video_frames(src, [r["frame"] for r in rows]) if src.is_file() and src.suffix.lower() != ".jpg" and not src.is_dir() else {}
        for r in rows:
            im = vid.get(r["frame"]) if vid else (Image.open(src / r["name"]) if src.is_dir() else Image.open(src))
            if im is None:
                continue
            tiles.append((im, r["instances"], f"frame {r['frame']}  n={len(r['instances'])}"))
        out = out or project.path("evidence", f"{'tracks' if tracks else 'predictions'}_{name}_sheet.jpg")
    else:
        raise ValueError("--what must be proposals, reviewed or predictions")
    if not tiles:
        raise RuntimeError(f"nothing to draw for {what}")
    p = contact_sheet(tiles, out, cols=cols, by_track=tracks)
    return {"sheet": str(p), "tiles": len(tiles)}
