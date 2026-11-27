"""Internal label document: one JSON per image.

    {"id": "mouse_000012", "file": "generated/images/mouse_000012.jpg", "W": 320, "H": 240,
     "source": "sam", "status": "proposed" | "accepted" | "rejected",
     "kept": true, "reason": "",                       # result of the automatic gates
     "instances": [{"category": "mouse", "box": [x0, y0, x1, y1], "poly": [[x, y], ...] | null,
                    "keypoints": {"nose": [x, y, v], ...} | null, "score": 0.93, "track_id": null}]}

Review records (labels/reviewed) add: "unit", "origin" ("human" | "auto" | "import"), "reviewed_utc",
"proposal_source" (the proposal model the reviewer started from, or null) and "edited" (true / false: the stored
labels differ / do not differ from that proposal; null when there was none). Proposals of multi-frame units carry
"identity_source": "iou_link" and a proposed "track_id" per instance (see stages/propose.py).

Boxes are absolute pixels, x1/y1 exclusive-ish floats. Keypoint visibility v: 0 not labelled,
1 labelled but hidden, 2 visible (COCO convention).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def poly_area(poly: List[List[float]]) -> float:
    a = 0.0
    for (x0, y0), (x1, y1) in zip(poly, poly[1:] + poly[:1]):
        a += x0 * y1 - x1 * y0
    return abs(a) / 2.0


def box_of_poly(poly: List[List[float]]) -> List[float]:
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    return [min(xs), min(ys), max(xs), max(ys)]


def box_area(b: List[float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def iou(a: List[float], b: List[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / max(1e-9, box_area(a) + box_area(b) - inter)


def instance(category: str, box: List[float], poly=None, keypoints=None, score: float = 1.0, track_id=None, **extra) -> dict:
    d = {"category": category, "box": [round(float(v), 2) for v in box],
         "poly": [[round(float(x), 2), round(float(y), 2)] for x, y in poly] if poly else None,
         "keypoints": {k: [round(float(v[0]), 2), round(float(v[1]), 2), int(v[2]) if len(v) > 2 else 2]
                       for k, v in keypoints.items()} if keypoints else None,
         "score": round(float(score), 4), "track_id": track_id}
    d.update(extra)
    return d


def label(image_id: str, file: str, W: int, H: int, instances: List[dict], source: str, status: str = "proposed",
          **extra: Any) -> dict:
    d = {"id": image_id, "file": file, "W": int(W), "H": int(H), "source": source, "status": status,
         "kept": True, "reason": "", "instances": instances}
    d.update(extra)
    return d


def apply_gates(doc: dict, expected_n: Optional[int], gates: Dict[str, Any]) -> dict:
    """Automatic acceptance gates, adapted from mw_det_syn/gap_targeted/scripts/label_img.py.

    Instances below `min_score` or outside the area band are dropped; the IMAGE is then rejected when
    the count differs from the prompted count (count gate), when an empty scene has a detection, or
    when an outline lies largely on the image border (for an instance without an outline: when that
    share of its box perimeter lies on the border, e.g. 0.5 for a box in a corner).
    """
    W, H = doc["W"], doc["H"]
    area_img = float(W * H)
    kept_inst, n_raw = [], len(doc["instances"])
    for i in doc["instances"]:
        if i.get("score", 1.0) < float(gates.get("min_score", 0.5)):
            continue
        a = poly_area(i["poly"]) if i.get("poly") else box_area(i["box"])
        if not (float(gates.get("min_area_frac", 0)) * area_img <= a <= float(gates.get("max_area_frac", 1)) * area_img):
            continue
        kept_inst.append(i)
    reason = ""
    if expected_n is not None and gates.get("count_gate", True):
        if expected_n == 0 and n_raw > 0:
            reason = f"empty scene with {n_raw} detection(s)"
        elif len(kept_inst) != expected_n:
            reason = f"count {len(kept_inst)} != {expected_n}"
    if not reason:
        mb = float(gates.get("max_border_frac", 1.0))
        for i in kept_inst:
            pts = i.get("poly") or []
            if pts:
                on = sum(1 for x, y in pts if x <= 0.5 or y <= 0.5 or x >= W - 1.5 or y >= H - 1.5) / len(pts)
            else:                                   # no outline (a detector or VLM that returns boxes): the share
                x0, y0, x1, y1 = i["box"]           # of the BOX perimeter that lies on the image border
                w, h = max(0.0, x1 - x0), max(0.0, y1 - y0)
                on = ((h if x0 <= 0.5 else 0.0) + (h if x1 >= W - 1.5 else 0.0) + (w if y0 <= 0.5 else 0.0)
                      + (w if y1 >= H - 1.5 else 0.0)) / max(1e-9, 2 * (w + h))
            if on >= mb:
                reason = "border"
                break
    out = dict(doc)
    out.update(instances=kept_inst, kept=not reason, reason=reason, n_raw=n_raw, expected_n=expected_n)
    return out


def kp_array(inst: dict, names: List[str]) -> List[Tuple[float, float, int]]:
    k = inst.get("keypoints") or {}
    return [tuple(k[n]) if n in k and k[n] is not None else (0.0, 0.0, 0) for n in names]
