"""Top-down pose proposer: crops every labelled animal, runs a crop pose model, maps keypoints back.

Adapted from TrackPose_demonstration/long_runs/pose_topdown/pose_topdown.py:
  * square crop around the box centre, side = max(box side * 1.3, 16), warped to 256 px, grey (114) padding;
  * crop pose model at imgsz 256, conf 0.01; take the highest-confidence instance whose box centre is
    within 0.25 * S of the crop centre, otherwise the highest overall;
  * decode back to the frame: kp = k * side / S + [X0, Y0].
Boxes come from labels/reviewed (preferred) or labels/proposed. The crop model is any predictor
backend that returns keypoints (Ultralytics .pt or ONNX YOLO-pose).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

from PIL import Image

from ... import registry
from ...formats import labels as L
from ...utils import read_json
from .base import Proposer

S = 256
PAD = (114, 114, 114)


def crop_square(im: Image.Image, box: List[float], scale: float = 1.3, size: int = S):
    """-> (crop image size x size, X0, Y0, side): the square window in frame pixels."""
    x0, y0, x1, y1 = box
    side = max(max(x1 - x0, y1 - y0) * scale, 16.0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    X0, Y0 = cx - side / 2, cy - side / 2
    crop = im.transform((size, size), Image.AFFINE, (side / size, 0, X0, 0, side / size, Y0),
                        resample=Image.BILINEAR, fillcolor=PAD)
    return crop, X0, Y0, side


def decode(kps: Dict[str, List[float]], X0: float, Y0: float, side: float, size: int = S) -> Dict[str, List[float]]:
    return {k: [v[0] * side / size + X0, v[1] * side / size + Y0, v[2] if len(v) > 2 else 2] for k, v in kps.items()}


def pick_central(instances: List[dict], size: int = S) -> Optional[dict]:
    if not instances:
        return None
    c = size / 2
    near = [i for i in instances if abs((i["box"][0] + i["box"][2]) / 2 - c) <= 0.25 * size
            and abs((i["box"][1] + i["box"][3]) / 2 - c) <= 0.25 * size]
    return max(near or instances, key=lambda i: i.get("score", 0))


def pose_for_boxes(im: Image.Image, instances: List[dict], predict_crop: Callable[[Image.Image], List[dict]],
                   scale: float = 1.3) -> List[dict]:
    out = []
    for i in instances:
        crop, X0, Y0, side = crop_square(im, i["box"], scale)
        best = pick_central(predict_crop(crop))
        j = dict(i)
        if best and best.get("keypoints"):
            j["keypoints"] = {k: [round(v[0], 2), round(v[1], 2), int(v[2])] for k, v in decode(best["keypoints"], X0, Y0, side).items()}
        out.append(j)
    return out


class TopDownPoseProposer(Proposer):
    name = "pose"
    tested = False       # the crop / decode maths is unit-tested; no real pose model was run

    def plan(self, rows: List[dict]) -> dict:
        w = self.opt.get("weights")
        return dict(super().plan(rows), weights=w, weights_exist=bool(w and Path(w).exists()), crop=S, box_scale=1.3,
                    predictor=self.opt.get("predictor") or ("onnx" if str(w).endswith(".onnx") else "ultralytics"))

    def propose(self, rows: List[dict]) -> Iterator[dict]:
        w = self.opt.get("weights")
        if not w:
            raise registry.BackendError("pose proposer: set propose.options.weights to a crop pose model")
        name = self.opt.get("predictor") or ("onnx" if str(w).endswith(".onnx") else "ultralytics")
        pred = registry.create("predictors", name, dict(self.opt, weights=w, conf=self.opt.get("conf", 0.01),
                                                        imgsz=S, category=self.category, keypoints=self.kp_names))
        root = self.project.root
        for r in rows:
            prev = read_json(root / "labels" / "reviewed" / f"{r['id']}.json") or read_json(root / "labels" / "proposed" / f"{r['id']}.json")
            if not prev:
                continue
            im = Image.open(r["path"]).convert("RGB")
            inst = pose_for_boxes(im, prev["instances"], pred.predict, float(self.opt.get("box_scale", 1.3)))
            yield L.label(r["id"], r["file"], im.size[0], im.size[1], inst, source="pose_topdown")
