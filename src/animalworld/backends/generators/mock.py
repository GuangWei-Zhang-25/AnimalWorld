"""Mock generator (CPU, PIL only): draws simple animal-like shapes so the whole pipeline runs anywhere.

It is NOT a substitute for a generative model; it exists so that install tests, tutorials and
continuous integration can exercise every stage in seconds. Because the geometry is known, the
exact boxes, outlines and keypoints are written next to the images (generated/truth/), and the
`mock` proposal backend reads them back with a little noise.
"""
from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Dict, Iterator, List

import numpy as np
from PIL import Image, ImageDraw

from ...utils import seed_from, write_json
from .base import Generator

COLOURS = {"black": (28, 28, 30), "white": (238, 238, 232), "brown": (112, 78, 52), "grey": (105, 105, 110),
           "pink": (232, 178, 168), "dark": (66, 44, 32), "spotted": (225, 185, 175)}


def colour_for(appearance: str) -> tuple:
    for k, c in COLOURS.items():
        if k in str(appearance):
            return c
    r = random.Random(seed_from("colour", appearance))
    return (30, 30, 32) if r.random() < 0.5 else (236, 236, 230)


def keypoint_positions(names: List[str], a: float, b: float) -> List[tuple]:
    """Body-frame (forward, left) position for each keypoint name; generic heuristics on the name."""
    out = []
    k = max(1, len(names) - 1)
    for i, nm in enumerate(names):
        n = nm.lower()
        lat = 0.55 * b if "left" in n else (-0.55 * b if "right" in n else 0.0)
        if any(w in n for w in ("nose", "snout", "beak", "head")):
            f = 0.95 * a
        elif "ear" in n or "eye" in n:
            f = 0.6 * a
        elif any(w in n for w in ("neck", "withers", "shoulder")):
            f = 0.35 * a
        elif "tail" in n:
            f = -0.95 * a
        elif any(w in n for w in ("hip", "rump", "hind")):
            f = -0.5 * a
        else:
            f = a * (0.9 - 1.8 * i / k)
        out.append((f, lat))
    return out


def place_animals(n: int, W: int, H: int, rng: random.Random, rel_len: float = 0.16) -> List[dict]:
    out: List[dict] = []
    L = max(14.0, rel_len * min(W, H) * (1.0 if n <= 4 else 0.75))
    for _ in range(n):
        for _try in range(200):
            cx, cy = rng.uniform(0.7 * L, W - 0.7 * L), rng.uniform(0.7 * L, H - 0.7 * L)
            if all(math.hypot(cx - o["cx"], cy - o["cy"]) > 1.15 * L for o in out):
                break
        out.append({"cx": cx, "cy": cy, "theta": rng.uniform(0, 2 * math.pi), "L": L * rng.uniform(0.9, 1.1)})
    return out


def geometry(an: dict, kp_names: List[str]) -> dict:
    a, b = 0.5 * an["L"], 0.22 * an["L"]
    c, s = math.cos(an["theta"]), math.sin(an["theta"])
    to = lambda f, l: (an["cx"] + f * c - l * s, an["cy"] + f * s + l * c)     # noqa: E731
    poly = [to(a * math.cos(t), b * math.sin(t) * (1.0 - 0.25 * math.cos(t))) for t in np.linspace(0, 2 * math.pi, 28, endpoint=False)]
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    return {"poly": [[round(x, 1), round(y, 1)] for x, y in poly],
            "box": [round(min(xs), 1), round(min(ys), 1), round(max(xs), 1), round(max(ys), 1)],
            "keypoints": {nm: [round(v, 1) for v in to(f, l)] + [2] for nm, (f, l) in zip(kp_names, keypoint_positions(kp_names, a, b))},
            "ears": [to(0.6 * a, 0.75 * b), to(0.6 * a, -0.75 * b)], "tail": [to(-a, 0), to(-a - 0.7 * an["L"] * 0.5, 0.25 * b)],
            "ear_r": 0.16 * an["L"] * 0.5 + 1}


def render(W: int, H: int, animals: List[dict], appearances: List[str], kp_names: List[str], bg_seed: int,
           noise_seed: int = 0):
    """-> (PIL image, list of instance dicts with box / poly / keypoints)."""
    r = random.Random(bg_seed)
    base = np.array([r.randint(118, 160), r.randint(118, 160), r.randint(112, 150)], dtype=np.float32)
    rng = np.random.default_rng(bg_seed + 7919 * noise_seed)
    yy, xx = np.mgrid[0:H, 0:W]
    grad = 10.0 * ((xx / max(1, W - 1)) - 0.5) * r.uniform(-1, 1) + 10.0 * ((yy / max(1, H - 1)) - 0.5) * r.uniform(-1, 1)
    arr = base[None, None, :] + grad[..., None] + rng.normal(0, 3.0, (H, W, 1))
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    inst = []
    for an, app in zip(animals, appearances):
        g = geometry(an, kp_names)
        col = colour_for(app)
        d.line([tuple(g["tail"][0]), tuple(g["tail"][1])], fill=col, width=max(1, int(an["L"] * 0.05)))
        d.polygon([tuple(p) for p in g["poly"]], fill=col)
        for ex, ey in g["ears"]:
            er = g["ear_r"]
            d.ellipse([ex - er, ey - er, ex + er, ey + er], fill=col)
        inst.append({"box": g["box"], "poly": g["poly"], "keypoints": g["keypoints"], "appearance": app})
    return img, inst


def simulate(n: int, frames: int, W: int, H: int, seed: int, rel_len: float = 0.16) -> List[List[dict]]:
    """Smooth random walks for `n` animals over `frames` frames (used for mock clips and the demo recording)."""
    rng = random.Random(seed)
    cur = place_animals(n, W, H, rng, rel_len)
    out = []
    for _ in range(frames):
        out.append([dict(a) for a in cur])
        for a in cur:
            a["theta"] += rng.uniform(-0.25, 0.25)
            step = 0.12 * a["L"]
            nx, ny = a["cx"] + step * math.cos(a["theta"]), a["cy"] + step * math.sin(a["theta"])
            if not (0.7 * a["L"] < nx < W - 0.7 * a["L"] and 0.7 * a["L"] < ny < H - 0.7 * a["L"]):
                a["theta"] += math.pi * 0.6
                continue
            if all(math.hypot(nx - o["cx"], ny - o["cy"]) > 1.1 * a["L"] for o in cur if o is not a):
                a["cx"], a["cy"] = nx, ny
            else:
                a["theta"] += 0.8
    return out


class MockGenerator(Generator):
    name = "mock"
    kind = "image"
    needs_gpu = False
    tested = True

    def __init__(self, options: Dict[str, Any] = None, project=None, **kw: Any):
        super().__init__(options, project, **kw)
        self.frames = int(self.opt.get("frames", 1))
        if self.frames > 1:
            self.kind = "frames"
        self.kp = project.keypoints() if project is not None else list(self.opt.get("keypoints") or [])

    def plan(self, items):
        return dict(super().plan(items), max_side=int(self.opt.get("max_side", 320)), frames_per_item=self.frames,
                    note="CPU shape renderer for tests; not a generative model")

    def _size(self, m):
        ms = int(self.opt.get("max_side", 320))
        s = ms / max(m["W"], m["H"])
        return max(32, int(round(m["W"] * s))), max(32, int(round(m["H"] * s)))

    def generate(self, items: List[dict], out_dir: Path) -> Iterator[dict]:
        out_dir = Path(out_dir)
        truth = out_dir.parent / "truth"
        for m in items:
            W, H = self._size(m)
            seq = simulate(m["n"], self.frames, W, H, m["seed"])
            files = []
            for f, animals in enumerate(seq):
                img, inst = render(W, H, animals, m.get("appearance") or ["black"] * m["n"], self.kp, m["seed"], f)
                iid = m["id"] if self.frames == 1 else f"{m['id']}_f{f:04d}"
                p = out_dir / f"{iid}.jpg"
                p.parent.mkdir(parents=True, exist_ok=True)
                tmp = p.with_suffix(".tmp.jpg")
                img.save(tmp, quality=92)
                tmp.replace(p)
                for k, i in enumerate(inst):
                    i["track_id"] = k
                write_json(truth / f"{iid}.json", {"id": iid, "unit": m["id"], "frame": f, "W": W, "H": H, "instances": inst})
                files.append({"id": iid, "file": str(p), "frame": f})
            yield {"id": m["id"], "files": files, "generator": "mock"}
