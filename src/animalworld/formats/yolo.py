"""YOLO txt labels (detect / segment / pose, Ultralytics conventions) <-> internal label documents.

Also builds the top-down pose crop dataset used by the crop pose model (adapted from
TrackPose_demonstration/long_runs/pose_topdown/build_crops.py).
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import yaml

from . import labels as L

TASKS = ("detect", "segment", "pose")


def _c(v: float) -> float:
    return min(1.0, max(0.0, v))


def to_lines(doc: dict, categories: List[str], task: str = "segment", kp_names: List[str] = None) -> str:
    W, H = float(doc["W"]), float(doc["H"])
    out = []
    for i in doc["instances"]:
        c = categories.index(i["category"]) if i.get("category") in categories else 0
        x0, y0, x1, y1 = i["box"]
        box = f"{_c((x0 + x1) / 2 / W):.6f} {_c((y0 + y1) / 2 / H):.6f} {_c((x1 - x0) / W):.6f} {_c((y1 - y0) / H):.6f}"
        if task == "detect":
            out.append(f"{c} {box}")
        elif task == "segment":
            poly = i.get("poly") or [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
            out.append(f"{c} " + " ".join(f"{_c(x / W):.6f} {_c(y / H):.6f}" for x, y in poly))
        elif task == "pose":
            ks = L.kp_array(i, kp_names or [])
            out.append(f"{c} {box} " + " ".join(
                (f"{_c(x / W):.6f} {_c(y / H):.6f} {int(v)}" if v > 0 else "0 0 0") for x, y, v in ks))
        else:
            raise ValueError(f"task must be one of {TASKS}")
    return "\n".join(out) + ("\n" if out else "")


def from_lines(text: str, W: int, H: int, categories: List[str], task: str = "segment", kp_names: List[str] = None) -> List[dict]:
    inst = []
    for line in text.splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        cat = categories[int(p[0])] if int(p[0]) < len(categories) else str(p[0])
        v = [float(x) for x in p[1:]]
        if task == "segment":
            poly = [[v[k] * W, v[k + 1] * H] for k in range(0, len(v) - 1, 2)]
            inst.append(L.instance(cat, L.box_of_poly(poly), poly))
            continue
        cx, cy, w, h = v[0] * W, v[1] * H, v[2] * W, v[3] * H
        box = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
        kps = None
        if task == "pose":
            kps = {nm: [v[4 + 3 * k] * W, v[5 + 3 * k] * H, int(v[6 + 3 * k])] for k, nm in enumerate(kp_names or [])
                   if 6 + 3 * k < len(v)}
        inst.append(L.instance(cat, box, None, kps))
    return inst


def data_yaml(root: Path, categories: List[str], task: str = "segment", kp_names: List[str] = None,
              flip_idx: List[int] = None) -> str:
    d = {"path": str(Path(root).resolve()), "train": "train/images", "val": "val/images",
         "names": {k: c for k, c in enumerate(categories)}}
    if task == "pose":
        d["kpt_shape"] = [len(kp_names or []), 3]
        d["flip_idx"] = list(flip_idx or range(len(kp_names or [])))
    return yaml.safe_dump(d, sort_keys=False)


def pose_crop(im, inst: dict, kp_names: List[str], pad: float = 0.15, size: int = 256):
    """One instance -> (crop image, YOLO-pose label line) or None. Square crop, side = max(w, h) * (1 + 2 * pad)."""
    from ..backends.proposers.pose import crop_square
    crop, X0, Y0, side = crop_square(im, inst["box"], 1.0 + 2 * pad, size)
    f = lambda x, y: ((x - X0) / side, (y - Y0) / side)      # noqa: E731
    x0, y0 = f(inst["box"][0], inst["box"][1])
    x1, y1 = f(inst["box"][2], inst["box"][3])
    x0, y0, x1, y1 = _c(x0), _c(y0), _c(x1), _c(y1)
    parts = [f"0 {(x0 + x1) / 2:.6f} {(y0 + y1) / 2:.6f} {x1 - x0:.6f} {y1 - y0:.6f}"]
    nvis = 0
    for x, y, v in L.kp_array(inst, kp_names):
        u, w = f(x, y)
        if v > 0 and 0 <= u <= 1 and 0 <= w <= 1:
            parts.append(f"{u:.6f} {w:.6f} 2")
            nvis += 1
        else:
            parts.append("0 0 0")
    if nvis == 0:
        return None
    return crop, " ".join(parts) + "\n"
