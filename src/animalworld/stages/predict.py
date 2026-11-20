"""Stage 5b: run a model on real recordings (image folder or video) -> predictions/<name>/frames.jsonl."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Tuple

from .. import registry
from ..config import Project
from ..utils import IMAGE_EXT, VIDEO_EXT, read_json, write_json, write_jsonl


def iter_frames(source: Path, stride: int = 1, limit: int = 0) -> Iterator[Tuple[int, str, "object"]]:
    """Yield (frame index, name, PIL image) from an image folder, one image, or a video file."""
    from PIL import Image
    source = Path(source)
    if source.is_dir():
        files = sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXT)
        for k, p in enumerate(files[::max(1, stride)]):
            if limit and k >= limit:
                return
            yield k * max(1, stride), p.name, Image.open(p).convert("RGB")
    elif source.suffix.lower() in IMAGE_EXT:
        yield 0, source.name, Image.open(source).convert("RGB")
    elif source.suffix.lower() in VIDEO_EXT:
        cv2 = registry.need("cv2", "deploy", "reading video files")
        cap = cv2.VideoCapture(str(source))
        f = n = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if f % max(1, stride) == 0:
                yield f, f"{source.stem}_f{f:06d}", Image.fromarray(frame[:, :, ::-1].copy())
                n += 1
                if limit and n >= limit:
                    break
            f += 1
        cap.release()
    else:
        raise FileNotFoundError(f"predict: {source} is not an image folder, image or video")


def appearance(im, box) -> float:
    """Mean luminance inside the box (0..1): a cheap identity cue (dark vs light coat) for the tracker."""
    import numpy as np
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    w, h = x1 - x0, y1 - y0
    crop = im.crop((x0 + w // 4, y0 + h // 4, max(x0 + w // 4 + 1, x1 - w // 4), max(y0 + h // 4 + 1, y1 - h // 4))).convert("L")
    return round(float(np.asarray(crop).mean()) / 255.0, 4)


def run(project: Project, dry_run: bool = False, source: str = None, weights: str = None, backend: str = None,
        name: str = None, stride: int = 1, limit: int = 0, conf: float = None, **_) -> dict:
    cfg = project.cfg["predict"]
    latest = read_json(project.path("models", "latest.json"), {}) or {}
    weights = weights or cfg.get("weights") or latest.get("exported") or latest.get("weights")
    bname = backend or (cfg["backend"] if (cfg.get("weights") or backend) else None)
    if not bname:
        bname = "onnx" if str(weights).endswith(".onnx") else ("ultralytics" if str(weights).endswith(".pt") else cfg["backend"])
    source = source or cfg.get("source")
    opts = dict(cfg.get("options") or {}, weights=weights, conf=conf if conf is not None else cfg["conf"],
                category=project.categories()[0], keypoints=project.keypoints())
    plan = {"stage": "predict", "backend": bname, "weights": weights, "source": str(source) if source else None,
            "conf": opts["conf"], "stride": stride}
    if dry_run:
        return dict(plan, mode="dry-run")
    if not source:
        return dict(plan, mode="waiting", next="no recording given: pass --source VIDEO_OR_FOLDER (or set predict.source)")
    src = Path(str(source)).expanduser()
    if not src.is_absolute():            # relative: the current folder first, then the project folder
        src = src.resolve() if src.exists() else project.resolve(source)
    name = name or src.stem
    out = project.path("predictions", name)
    pred = registry.create("predictors", bname, opts)
    rows, n_inst, size = [], 0, None
    for f, fname, im in iter_frames(src, stride, limit):
        inst = pred.predict(im)
        for i in inst:
            i["appearance"] = appearance(im, i["box"])
        size = im.size
        rows.append({"frame": f, "name": fname, "instances": inst})
        n_inst += len(inst)
    write_jsonl(out / "frames.jsonl", rows)
    info = dict(plan, name=name, frames=len(rows), instances=n_inst, W=size[0] if size else 0, H=size[1] if size else 0,
                source=str(src), output=str(out / "frames.jsonl"))
    write_json(out / "info.json", info)
    return dict(info, mode="executed")
