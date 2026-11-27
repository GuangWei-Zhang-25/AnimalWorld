"""Display-free logic of the desktop annotator: the view transform (image pixels <-> canvas pixels) and the
editable label document (boxes, outlines, keypoints, undo). `tkapp.py` only draws and forwards mouse / key events,
so everything here is unit-tested without a display.

Instances use the package's label format (formats/labels.py): {"category", "box": [x0, y0, x1, y1],
"poly": [[x, y], ...] | None, "keypoints": {name: [x, y, v]} | None, "score", "track_id"}.
"""
from __future__ import annotations

from ..formats import labels as L

import copy
from typing import Dict, List, Optional, Tuple

HANDLES = ("nw", "ne", "se", "sw")
MIN_BOX = 3.0            # image pixels; smaller drags are treated as a click, not a new box


class View:
    """canvas = image * scale + offset."""

    def __init__(self, scale: float = 1.0, ox: float = 0.0, oy: float = 0.0):
        self.scale, self.ox, self.oy = float(scale), float(ox), float(oy)

    def to_canvas(self, x: float, y: float) -> Tuple[float, float]:
        return x * self.scale + self.ox, y * self.scale + self.oy

    def to_image(self, cx: float, cy: float) -> Tuple[float, float]:
        return (cx - self.ox) / self.scale, (cy - self.oy) / self.scale

    def fit(self, W: int, H: int, cw: int, ch: int, margin: int = 8) -> None:
        """Show the whole image, centred."""
        cw, ch = max(cw, 2 * margin + 1), max(ch, 2 * margin + 1)
        self.scale = max(0.02, min((cw - 2 * margin) / max(1, W), (ch - 2 * margin) / max(1, H)))
        self.ox = (cw - W * self.scale) / 2.0
        self.oy = (ch - H * self.scale) / 2.0

    def zoom_at(self, cx: float, cy: float, factor: float, lo: float = 0.05, hi: float = 40.0) -> None:
        """Zoom keeping the image point under the canvas point (cx, cy) fixed."""
        x, y = self.to_image(cx, cy)
        self.scale = max(lo, min(hi, self.scale * factor))
        self.ox, self.oy = cx - x * self.scale, cy - y * self.scale

    def pan(self, dx: float, dy: float) -> None:
        self.ox += dx
        self.oy += dy

    def visible(self, W: int, H: int, cw: int, ch: int) -> Optional[Tuple[int, int, int, int]]:
        """The image rectangle (x0, y0, x1, y1, integers) that falls inside a cw x ch canvas, or None."""
        x0, y0 = self.to_image(0, 0)
        x1, y1 = self.to_image(cw, ch)
        x0, y0, x1, y1 = max(0, int(x0)), max(0, int(y0)), min(W, int(x1) + 1), min(H, int(y1) + 1)
        return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def _norm_box(b: List[float]) -> List[float]:
    return [min(b[0], b[2]), min(b[1], b[3]), max(b[0], b[2]), max(b[1], b[3])]


def _box_of(pts: List[List[float]]) -> List[float]:
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return [min(xs), min(ys), max(xs), max(ys)]


class Doc:
    """The labels of one image while they are edited. Every change goes through a method that records an undo step."""

    def __init__(self, instances: List[dict], W: int, H: int, category: str, kp_names: List[str]):
        self.instances: List[dict] = copy.deepcopy(instances or [])
        self.W, self.H, self.category, self.kp_names = int(W), int(H), category, list(kp_names or [])
        self._undo: List[List[dict]] = []
        self.selected: Optional[int] = 0 if self.instances else None
        self.dirty = False

    # ---- undo ----
    def checkpoint(self) -> None:
        self._undo.append(copy.deepcopy(self.instances))
        del self._undo[:-100]
        self.dirty = True

    def undo(self) -> bool:
        if not self._undo:
            return False
        self.instances = self._undo.pop()
        if self.selected is not None and self.selected >= len(self.instances):
            self.selected = len(self.instances) - 1 if self.instances else None
        return True

    def _clamp(self, x: float, y: float) -> Tuple[float, float]:
        return min(max(0.0, float(x)), float(self.W)), min(max(0.0, float(y)), float(self.H))

    # ---- creating ----
    def add_box(self, x0, y0, x1, y1) -> Optional[int]:
        b = _norm_box([*self._clamp(x0, y0), *self._clamp(x1, y1)])
        if b[2] - b[0] < MIN_BOX or b[3] - b[1] < MIN_BOX:
            return None
        self.checkpoint()
        self.instances.append({"category": self.category, "box": b, "poly": None, "keypoints": None, "score": 1.0,
                               "track_id": None})
        self.selected = len(self.instances) - 1
        return self.selected

    def add_poly(self, pts: List[Tuple[float, float]]) -> Optional[int]:
        pts = [list(self._clamp(x, y)) for x, y in pts]
        if len(pts) < 3:
            return None
        b = _box_of(pts)
        if b[2] - b[0] < MIN_BOX or b[3] - b[1] < MIN_BOX:
            return None
        self.checkpoint()
        self.instances.append({"category": self.category, "box": b, "poly": pts, "keypoints": None, "score": 1.0,
                               "track_id": None})
        self.selected = len(self.instances) - 1
        return self.selected

    def set_track_id(self, i: int, track_id: Optional[int]) -> None:
        """Give animal i the identity `track_id` (None clears it). The identity is the same animal's number on every
        frame of one clip; if another animal of this image already has that number the two are swapped, so a
        reviewer repairs an identity swap with one key."""
        old = self.instances[i].get("track_id")
        if old == track_id:
            return
        self.checkpoint()
        if track_id is not None:
            for j, other in enumerate(self.instances):
                if j != i and other.get("track_id") == track_id:
                    other["track_id"] = old
        self.instances[i]["track_id"] = track_id

    def next_keypoint(self, i: int) -> Optional[str]:
        """The first keypoint name the instance does not have yet (keypoints are placed in the species' order)."""
        have = self.instances[i].get("keypoints") or {}
        return next((n for n in self.kp_names if n not in have or not have[n] or have[n][2] <= 0), None)

    def set_keypoint(self, i: int, name: str, x: float, y: float, v: int = 2, record: bool = True) -> None:
        if name not in self.kp_names:
            raise KeyError(name)
        if record:
            self.checkpoint()
        k = self.instances[i].get("keypoints") or {}
        k[name] = [*self._clamp(x, y), int(v)]
        self.instances[i]["keypoints"] = k

    # ---- editing (call checkpoint() once at the start of a drag, then the *_to methods without recording) ----
    def move_instance(self, i: int, dx: float, dy: float) -> None:
        """Move box, outline and keypoints together; the box is kept inside the image."""
        b = self.instances[i]["box"]
        dx = min(max(dx, -b[0]), self.W - b[2])
        dy = min(max(dy, -b[1]), self.H - b[3])
        ins = self.instances[i]
        ins["box"] = [b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy]
        if ins.get("poly"):
            ins["poly"] = [[x + dx, y + dy] for x, y in ins["poly"]]
        if ins.get("keypoints"):
            ins["keypoints"] = {n: [v[0] + dx, v[1] + dy, v[2]] for n, v in ins["keypoints"].items()}

    def resize_box(self, i: int, handle: str, x: float, y: float) -> None:
        x, y = self._clamp(x, y)
        b = list(self.instances[i]["box"])
        if "w" in handle:
            b[0] = min(x, b[2] - MIN_BOX)
        if "e" in handle:
            b[2] = max(x, b[0] + MIN_BOX)
        if "n" in handle:
            b[1] = min(y, b[3] - MIN_BOX)
        if "s" in handle:
            b[3] = max(y, b[1] + MIN_BOX)
        self.instances[i]["box"] = b

    def move_keypoint(self, i: int, name: str, x: float, y: float) -> None:
        v = self.instances[i]["keypoints"][name]
        self.instances[i]["keypoints"][name] = [*self._clamp(x, y), v[2]]

    def move_vertex(self, i: int, j: int, x: float, y: float) -> None:
        ins = self.instances[i]
        ins["poly"][j] = list(self._clamp(x, y))
        ins["box"] = _box_of(ins["poly"])

    def toggle_visibility(self, i: int, name: str) -> None:
        """visible (2) <-> labelled but hidden (1)."""
        self.checkpoint()
        v = self.instances[i]["keypoints"][name]
        v[2] = 1 if v[2] == 2 else 2

    def delete_keypoint(self, i: int, name: str) -> None:
        self.checkpoint()
        k = self.instances[i].get("keypoints") or {}
        k.pop(name, None)
        self.instances[i]["keypoints"] = k or None

    def delete_instance(self, i: int) -> None:
        self.checkpoint()
        del self.instances[i]
        self.selected = min(i, len(self.instances) - 1) if self.instances else None

    def set_poly(self, i: int, poly: Optional[List[List[float]]]) -> None:
        self.checkpoint()
        self.instances[i]["poly"] = [list(self._clamp(x, y)) for x, y in poly] if poly else None

    # ---- hit testing (image coordinates; tol = pick radius in image pixels) ----
    def handle_points(self, i: int) -> Dict[str, Tuple[float, float]]:
        x0, y0, x1, y1 = self.instances[i]["box"]
        return {"nw": (x0, y0), "ne": (x1, y0), "se": (x1, y1), "sw": (x0, y1)}

    def hit(self, x: float, y: float, tol: float):
        """What is under (x, y): ("kp", i, name) | ("vertex", i, j) | ("handle", i, h) | ("box", i) | None.
        The selected instance is tested first; points win over handles, handles over the box body."""
        order = list(range(len(self.instances)))
        if self.selected is not None and self.selected < len(order):
            order.remove(self.selected)
            order.insert(0, self.selected)
        for i in order:
            for n, v in (self.instances[i].get("keypoints") or {}).items():
                if v and v[2] > 0 and abs(v[0] - x) <= tol and abs(v[1] - y) <= tol:
                    return ("kp", i, n)
        if self.selected is not None and self.selected < len(self.instances):
            i = self.selected
            for h, (hx, hy) in self.handle_points(i).items():
                if abs(hx - x) <= tol and abs(hy - y) <= tol:
                    return ("handle", i, h)
            for j, (px, py) in enumerate(self.instances[i].get("poly") or []):
                if abs(px - x) <= tol and abs(py - y) <= tol:
                    return ("vertex", i, j)
        inside = [i for i in order if self.instances[i]["box"][0] <= x <= self.instances[i]["box"][2]
                  and self.instances[i]["box"][1] <= y <= self.instances[i]["box"][3]]
        if inside:                               # the smallest box under the pointer (nested boxes stay reachable)
            return ("box", min(inside, key=lambda i: (self.instances[i]["box"][2] - self.instances[i]["box"][0])
                               * (self.instances[i]["box"][3] - self.instances[i]["box"][1])))
        return None


def clean_instances(raw, categories, kp_names):
    """Validate what the browser sends; anything malformed is dropped rather than stored."""
    out = []
    for i in raw or []:
        try:
            box = [float(v) for v in i["box"]][:4]
            if len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
                continue
            poly = [[float(p[0]), float(p[1])] for p in i["poly"]] if i.get("poly") and len(i["poly"]) >= 3 else None
            kps = None
            if i.get("keypoints"):
                kps = {k: [float(v[0]), float(v[1]), int(v[2])] for k, v in i["keypoints"].items()
                       if k in kp_names and v is not None and int(v[2]) > 0} or None
            cat = i.get("category") if i.get("category") in categories else categories[0]
            out.append(L.instance(cat, box, poly, kps, score=float(i.get("score", 1.0)), track_id=i.get("track_id")))
        except (KeyError, TypeError, ValueError, IndexError):
            continue
    return out
