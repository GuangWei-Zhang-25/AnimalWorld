"""Automatic zoom for videos in which the animals are small in the frame.

The detectors resize every frame to a fixed square input (504 x 504 for MouseWorld-DET), so what matters is the animal's
share of the image: a mouse that is 40 px long in a 1280 px wide frame is 16 px at the model input, where detection has
collapsed. This module looks at the whole clip first (the scout), estimates the animal size and where the animals are,
and, only when the animals are too small or the frame is two cameras side by side, analyses each region as its own
stream (the unchanged detector / tracker / keypoint steps on a cropped and, for small crops, enlarged copy of the
frames) and maps everything back to full-frame pixel coordinates.

Nothing here is set per video. The numbers below were fixed from the lab's own data before the feature was run on any
collaborator clip (see the pre-registration note quoted in STATUS.md):

    SIZE_MIN_PX   the animal's longer side at the 504 px model input below which the whole frame is not used
    SIZE_TARGET   what a region is sized towards
    SIZE_MAX_PX   a region is never so tight that the animal becomes larger than this
    ZOOM_MAX      a crop is never enlarged by more than this factor

When the estimated size is at or above SIZE_MIN_PX and no camera seam is found, NOTHING is done: the whole-frame path
runs exactly as before.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..utils import write_json

MODEL_INPUT = 504.0          # side of the square the RF-DETR-Seg-L detectors resize every image to
SIZE_MIN_PX = 64.0           # below this (animal's longer side at the model input) the whole frame is not used
SIZE_TARGET_PX = 112.0       # regions are sized towards this
SIZE_MAX_PX = 192.0          # ... and never tighter than this
ZOOM_MAX = 4.0               # largest enlargement of a crop
CROP_SIDE_PX = 1024.0        # crops are enlarged towards this longer side (SAM 2.1 works at 1024 px), within ZOOM_MAX
MARGIN_BL = 1.0              # margin around the activity of a region, in animal lengths
LINK_BL = 3.0                # detections closer than this (animal lengths) belong to one activity area
GROW_BL = 2.0                # a region whose animal touches its edge is grown by this much and analysed again (once)
MAX_REGIONS = 8
TILE_GRIDS = ((2, 2), (3, 3))  # the scout's tiles: the frame cut into 2 x 2 and 3 x 3 overlapping pieces
TILE_OVERLAP = 0.25
SEAM_RATIO = 4.0             # a camera seam: the step across the middle column / row is this many times the usual step
CONF = 0.4                   # "confident" detection, as everywhere in the tracker
SEAM_HALF_ASPECT_MAX = 2.0   # deviation 3: a camera seam leaves two halves that are each an ordinary camera image (at most 2 : 1)


def _cv2():
    from . import pipeline as P
    return P._cv2()


# ------------------------------------------------------------------ the scout ---------------------------------------
def find_seams(paths: List[Path]) -> dict:
    """Two cameras recorded side by side (or one above the other) give a hard step exactly in the middle of the frame, on
    every frame. -> {"x": column or None, "y": row or None}"""
    cv2 = _cv2()
    rx, ry = [], []
    for p in paths:
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        g = g.astype(np.float32); H, W = g.shape
        dx = np.abs(np.diff(g, axis=1)).mean(0); dy = np.abs(np.diff(g, axis=0)).mean(1)
        if W >= 64 and W % 2 == 0:
            rx.append(dx[W // 2 - 1] / max(0.5, float(np.median(dx))))
        if H >= 64 and H % 2 == 0:
            ry.append(dy[H // 2 - 1] / max(0.5, float(np.median(dy))))
    out = {"x": None, "y": None}
    if not rx and not ry:
        return out
    H, W = cv2.imread(str(paths[0])).shape[:2]
    # DEVIATION 3 (see DEVIATIONS_AUTOZOOM.md): each half must itself be shaped like a camera image (sides at most
    # SEAM_HALF_ASPECT_MAX : 1). A straight edge of the cage exactly at the middle row of one 16 : 9 camera (two halves of
    # 3.6 : 1) is not two cameras; two 4 : 3 cameras side by side (halves 1.33 : 1) are.
    if rx and np.mean(np.array(rx) >= SEAM_RATIO) >= 0.8 and seam_halves_ok(W, H, "x"):
        out["x"] = int(W // 2)
    if ry and np.mean(np.array(ry) >= SEAM_RATIO) >= 0.8 and seam_halves_ok(W, H, "y"):
        out["y"] = int(H // 2)
    return out


SEAM_CROSS_SHARE = 0.10    # deviation 4: share of the sample frames on which an animal straddles the seam that rules it out


def crossings(dets: List[dict], at: float, axis: str, T: int) -> int:
    """Sample frames on which a confident, animal-sized detection (longer side at most 1.5 x the median) lies across the line
    `at` with at least 30 % of its extent on each side. Two cameras recorded side by side never show that (a box stretched over
    both cameras is much larger than an animal and is not counted)."""
    if not dets:
        return 0
    L = float(np.median([max(d["box"][2] - d["box"][0], d["box"][3] - d["box"][1]) for d in dets]))
    a, b = (0, 2) if axis == "x" else (1, 3)
    hit = set()
    for d in dets:
        bx = d["box"]; ext = bx[b] - bx[a]
        if max(bx[2] - bx[0], bx[3] - bx[1]) <= 1.5 * L and ext > 0 and bx[a] <= at - 0.3 * ext and bx[b] >= at + 0.3 * ext:
            hit.add(d["t"])
    return len(hit)


def seam_halves_ok(W: int, H: int, axis: str) -> bool:
    """Each half must be shaped like an ordinary (landscape) camera image: width / height between 1 and SEAM_HALF_ASPECT_MAX.
    Two 640 x 480 cameras side by side (halves 1.33) pass; the middle row of one 16 : 9 frame (halves 3.6), the middle column
    of an 800 x 600 or 2448 x 2048 frame (portrait halves) do not (deviation 3, tightened 21 Nov)."""
    w, h = (W / 2.0, float(H)) if axis == "x" else (float(W), H / 2.0)
    return 1.0 <= w / max(1.0, h) <= SEAM_HALF_ASPECT_MAX


def motion_blobs(paths: List[Path], W: int, H: int) -> List[dict]:
    """Things that move: frames at 480 px width against their median; blobs of plausible area. Works when nothing is detected.
    -> [{t, c: (x, y), w, h}] in full-frame pixels."""
    cv2 = _cv2()
    sw = 480; k = sw / W; sh = max(8, int(round(H * k)))
    small = []
    for p in paths:
        im = cv2.imread(str(p))
        if im is not None:
            small.append(cv2.resize(im, (sw, sh), interpolation=cv2.INTER_AREA))
    if len(small) < 5:
        return []
    S = np.stack(small).astype(np.int16); bg = np.median(S, 0)
    out = []
    ker = np.ones((3, 3), np.uint8)
    for t, fr in enumerate(S):
        d = (np.abs(fr - bg).max(2) > 28).astype(np.uint8)
        d = cv2.morphologyEx(d, cv2.MORPH_OPEN, ker)
        n, lab, st, cen = cv2.connectedComponentsWithStats(d, 8)
        for i in range(1, n):
            a = st[i, cv2.CC_STAT_AREA]
            if 12 <= a <= 0.02 * sw * sh:
                out.append(dict(t=t, c=(float(cen[i][0] / k), float(cen[i][1] / k)), w=float(st[i, cv2.CC_STAT_WIDTH] / k), h=float(st[i, cv2.CC_STAT_HEIGHT] / k)))
    return out


def tiles_of(W: int, H: int) -> List[tuple]:
    out = []
    for gx, gy in TILE_GRIDS:
        tw = W / (gx - (gx - 1) * TILE_OVERLAP); th = H / (gy - (gy - 1) * TILE_OVERLAP)
        for iy in range(gy):
            for ix in range(gx):
                x0 = int(round(ix * tw * (1 - TILE_OVERLAP))); y0 = int(round(iy * th * (1 - TILE_OVERLAP)))
                out.append((x0, y0, min(W, int(round(x0 + tw))), min(H, int(round(y0 + th)))))
    return out


def _tile_detections(job, idx: List[int]) -> List[dict]:
    """The detector on overlapping pieces of the sample frames (2 x 2 and 3 x 3): a piece is a 2 to 3 times zoom.
    Detections cut by an inner edge of a piece are dropped (the neighbouring piece sees that animal whole)."""
    from . import pipeline as P
    from .. import weights as Wt
    cv2 = _cv2()
    m = job.meta; W, H = m["W"], m["H"]
    T = tiles_of(W, H)
    dets: List[dict] = []
    if job.engine["engine"] == "full":
        root = job.work / "scout"
        clips = []
        for ti, (x0, y0, x1, y1) in enumerate(T):
            d = root / "data" / "video" / f"t{ti:02d}" / "frames"; d.mkdir(parents=True, exist_ok=True)
            clips.append(dict(dataset="video", clip=f"t{ti:02d}", frames=len(idx), animals=0, fps=1.0, step=1, W=x1 - x0, H=y1 - y0))
        for j, a in enumerate(idx):
            im = cv2.imread(str(P.frame_path(job, a)))
            for ti, (x0, y0, x1, y1) in enumerate(T):
                f = root / "data" / "video" / f"t{ti:02d}" / "frames" / f"{j:06d}.jpg"
                if not f.exists():
                    cv2.imwrite(str(f), im[y0:y1, x0:x1], [cv2.IMWRITE_JPEG_QUALITY, 93])
        write_json(root / "clips.json", {"rule": "scout tiles, written by animalworld.zeroshot.zoom", "clips": clips})
        try:
            P.run_step(job, "scout: MouseWorld-DET on pieces of the sample frames", "det", "detect_dense.py",
                       env=dict(MWT_ROOT=str(root), MWT_AGENT=str(root / "det"), MWT_DET_CKPT=str(Wt.resolve("mouseworld-det"))))
        except P.NothingFound:
            pass
        from .mwtrack import core_base as CB
        for ti, (x0, y0, x1, y1) in enumerate(T):
            f = root / "det" / "work" / f"video__t{ti:02d}_000000_{len(idx) - 1:06d}" / "det.npz"
            if not f.exists():
                continue
            D = CB.load_det(f)
            for i in range(len(D["t"])):
                b = D["box"][i]
                dets.append(dict(t=int(D["t"][i]), box=[b[0] + x0, b[1] + y0, b[2] + x0, b[3] + y0], score=float(D["score"][i]), area=float(D["area"][i]),
                                 c=(float(D["c"][i][0] + x0), float(D["c"][i][1] + y0)), tile=ti, lb=[b[0], b[1], b[2], b[3]]))
    else:
        from PIL import Image
        pred = P._predictor(job)
        for j, a in enumerate(idx):
            with Image.open(P.frame_path(job, a)) as im:
                im = im.convert("RGB")
                for ti, (x0, y0, x1, y1) in enumerate(T):
                    for i in pred.predict(im.crop((x0, y0, x1, y1))):
                        b = i["box"]
                        dets.append(dict(t=j, box=[b[0] + x0, b[1] + y0, b[2] + x0, b[3] + y0], score=float(i["score"]),
                                         area=float(P._poly_area(i.get("poly")) or 0.6 * (b[2] - b[0]) * (b[3] - b[1])),
                                         c=((b[0] + b[2]) / 2 + x0, (b[1] + b[3]) / 2 + y0), tile=ti, lb=list(b)))
    keep = []
    for d in dets:
        x0, y0, x1, y1 = T[d["tile"]]; b = d["lb"]; tw, th = x1 - x0, y1 - y0
        cut = (b[0] <= 2 and x0 > 0) or (b[1] <= 2 and y0 > 0) or (b[2] >= tw - 2 and x1 < W) or (b[3] >= th - 2 and y1 < H)
        if not cut:
            keep.append(d)
    return keep


def pool(whole: List[dict], tiled: List[dict]) -> List[dict]:
    """One list of detections per sample frame from the whole-frame pass and the pieces: the same animal seen twice is kept
    once (highest score), and a whole-frame box much larger than the animals the pieces found is dropped (a box across two
    animals or two arenas)."""
    big = None
    conf_t = [max(d["box"][2] - d["box"][0], d["box"][3] - d["box"][1]) for d in tiled if d["score"] >= CONF]
    if len(conf_t) >= 5:
        big = 2.5 * float(np.median(conf_t))
    cand = [d for d in whole if big is None or max(d["box"][2] - d["box"][0], d["box"][3] - d["box"][1]) <= big] + tiled
    out: List[dict] = []
    for d in sorted(cand, key=lambda d: -d["score"]):
        dup = False
        for e in out:
            if e["t"] != d["t"]:
                continue
            b = e["box"]
            if (b[0] <= d["c"][0] <= b[2] and b[1] <= d["c"][1] <= b[3]) or (d["box"][0] <= e["c"][0] <= d["box"][2] and d["box"][1] <= e["c"][1] <= d["box"][3]):
                dup = True; break
        if not dup:
            out.append(d)
    return out


def sustained(counts: np.ndarray) -> int:
    """The count the detector reaches on at least a quarter of the sample frames (the second half of the tracker's count rule)."""
    return max([n for n in range(1, 16) if (counts >= n).mean() >= 0.25] + [0])


def _side(c, seams: dict) -> tuple:
    return (seams.get("x") is not None and c[0] >= seams["x"], seams.get("y") is not None and c[1] >= seams["y"])


def find_regions(dets: List[dict], T: int, W: int, H: int, L: float, seams: dict, blobs: List[dict] = ()) -> List[dict]:
    """Regions of interest from the confident detections of the sample frames.
    1. Activity areas: detections closer than LINK_BL animal lengths are linked (never across a camera seam).
    2. An area is split along x or y when the animals never mix across the split on the sample frames: the count on each
       side is the same on (nearly) every frame and no animal's centre lies in the gap (two arenas side by side with one
       animal each -> two regions; five mice in one open field -> one region).
    3. Each region = the boxes of its detections plus MARGIN_BL animal lengths, not beyond a seam or a split, at least large
       enough that the animal is not above SIZE_MAX_PX at the model input.
    -> [{box: [x0, y0, x1, y1], animals: n, limits: sides that may not grow}]"""
    P = [d for d in dets if d["score"] >= CONF]
    if not P or L <= 0:
        return []
    n = len(P); parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    C = np.array([d["c"] for d in P]); sd = [_side(d["c"], seams) for d in P]
    for i in range(n):
        dist = np.hypot(*(C - C[i]).T)
        for j in np.flatnonzero(dist <= LINK_BL * L):
            if j > i and sd[i] == sd[j]:
                parent[find(i)] = find(int(j))
    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    def split(members: List[int], lim: dict) -> List[tuple]:
        ts = sorted({P[i]["t"] for i in members})
        if len(ts) >= 6:
            best = None
            for ax in (0, 1):
                v = np.sort(C[members, ax]); gaps = np.diff(v)
                if len(gaps) and gaps.max() >= 0.5 * L:
                    k = int(np.argmax(gaps)); cut = float((v[k] + v[k + 1]) / 2)
                    a = np.array([sum(1 for i in members if P[i]["t"] == t and C[i, ax] < cut) for t in ts])
                    b = np.array([sum(1 for i in members if P[i]["t"] == t and C[i, ax] >= cut) for t in ts])
                    na, nb = int(np.bincount(a).argmax()), int(np.bincount(b).argmax())
                    if na >= 1 and nb >= 1 and (a == na).mean() >= 0.9 and (b == nb).mean() >= 0.9 and (best is None or gaps.max() > best[0]):
                        best = (float(gaps.max()), ax, cut)
            if best is not None:
                _, ax, cut = best
                lo = [i for i in members if C[i, ax] < cut]; hi = [i for i in members if C[i, ax] >= cut]
                l1 = dict(lim); l2 = dict(lim)
                l1["x1" if ax == 0 else "y1"] = cut; l2["x0" if ax == 0 else "y0"] = cut
                return split(lo, l1) + split(hi, l2)
        return [(members, lim)]
    regions = []
    for members in groups.values():
        s = sd[members[0]]
        lim = dict(x0=float(seams["x"]) if s[0] else 0.0, x1=float(seams["x"]) if (seams.get("x") is not None and not s[0]) else float(W),
                   y0=float(seams["y"]) if s[1] else 0.0, y1=float(seams["y"]) if (seams.get("y") is not None and not s[1]) else float(H))
        for mem, lm in split(members, lim):
            B = np.array([P[i]["box"] for i in mem])
            counts = np.array([sum(1 for i in mem if P[i]["t"] == t) for t in range(T)])
            regions.append(dict(raw=[float(B[:, 0].min()), float(B[:, 1].min()), float(B[:, 2].max()), float(B[:, 3].max())], lim=lm, animals=sustained(counts),
                                share=float((counts >= max(1, sustained(counts))).mean()), counts=counts))
    for bl in blobs:                                           # movement just outside a region's detections extends it
        for r in regions:
            x0, y0, x1, y1 = r["raw"]
            if x0 - 1.5 * L <= bl["c"][0] <= x1 + 1.5 * L and y0 - 1.5 * L <= bl["c"][1] <= y1 + 1.5 * L and \
                    r["lim"]["x0"] <= bl["c"][0] <= r["lim"]["x1"] and r["lim"]["y0"] <= bl["c"][1] <= r["lim"]["y1"]:
                r["raw"] = [min(x0, bl["c"][0] - bl["w"] / 2), min(y0, bl["c"][1] - bl["h"] / 2), max(x1, bl["c"][0] + bl["w"] / 2), max(y1, bl["c"][1] + bl["h"] / 2)]
                break
    regions = [r for r in regions if r["animals"] >= 1]
    for r in regions:
        r["box"] = shape_box(r["raw"], r["lim"], L, MARGIN_BL)
    # regions that overlap after their margins are one region (unless a seam or a split keeps them apart: then they were clipped)
    merged = True
    while merged:
        merged = False
        for i in range(len(regions)):
            for j in range(i + 1, len(regions)):
                a, b = regions[i]["box"], regions[j]["box"]
                if min(a[2], b[2]) - max(a[0], b[0]) > 0 and min(a[3], b[3]) - max(a[1], b[1]) > 0:
                    regions[i] = merge_two(regions[i], regions[j], L); del regions[j]; merged = True; break
            if merged:
                break
    while len(regions) > MAX_REGIONS:                          # never more regions than is sensible: the two nearest become one
        best = None
        for i in range(len(regions)):
            for j in range(i + 1, len(regions)):
                a, b = regions[i]["box"], regions[j]["box"]
                d = math.hypot((a[0] + a[2] - b[0] - b[2]) / 2, (a[1] + a[3] - b[1] - b[3]) / 2)
                if best is None or d < best[0]:
                    best = (d, i, j)
        _, i, j = best
        regions[i] = merge_two(regions[i], regions[j], L); del regions[j]
    regions.sort(key=lambda r: (r["box"][1] // max(1, int(H / 4)), r["box"][0]))
    return regions


def merge_two(a: dict, b: dict, L: float) -> dict:
    raw = [min(a["raw"][0], b["raw"][0]), min(a["raw"][1], b["raw"][1]), max(a["raw"][2], b["raw"][2]), max(a["raw"][3], b["raw"][3])]
    lim = dict(x0=min(a["lim"]["x0"], b["lim"]["x0"]), y0=min(a["lim"]["y0"], b["lim"]["y0"]), x1=max(a["lim"]["x1"], b["lim"]["x1"]), y1=max(a["lim"]["y1"], b["lim"]["y1"]))
    r = dict(raw=raw, lim=lim, animals=a["animals"] + b["animals"], share=min(a.get("share", 1.0), b.get("share", 1.0)))
    if a.get("counts") is not None and b.get("counts") is not None:
        r["counts"] = np.asarray(a["counts"]) + np.asarray(b["counts"])          # per sample frame: the confident detections of both
    r["box"] = shape_box(raw, lim, L, MARGIN_BL)
    return r


def shape_box(raw, lim: dict, L: float, margin_bl: float) -> List[int]:
    """raw activity box -> the region: margin added, at least L * 504 / SIZE_MAX_PX on each side (so the animal is not too
    large), sides at most 2 : 1, all inside the limits (frame, seam, split)."""
    x0, y0, x1, y1 = raw
    x0 -= margin_bl * L; y0 -= margin_bl * L; x1 += margin_bl * L; y1 += margin_bl * L
    need = L * MODEL_INPUT / SIZE_MAX_PX

    def widen(a, b, lo, hi, want):
        if b - a < want:
            c = (a + b) / 2; a, b = c - want / 2, c + want / 2
        if a < lo:
            b += lo - a; a = lo
        if b > hi:
            a -= b - hi; b = hi
        return max(lo, a), min(hi, b)
    x0, x1 = widen(x0, x1, lim["x0"], lim["x1"], need); y0, y1 = widen(y0, y1, lim["y0"], lim["y1"], need)
    if (x1 - x0) > 2 * (y1 - y0):
        y0, y1 = widen(y0, y1, lim["y0"], lim["y1"], (x1 - x0) / 2)
    if (y1 - y0) > 2 * (x1 - x0):
        x0, x1 = widen(x0, x1, lim["x0"], lim["x1"], (y1 - y0) / 2)
    return [int(math.floor(x0)), int(math.floor(y0)), int(math.ceil(x1)), int(math.ceil(y1))]


def zoom_factor(box) -> float:
    side = max(box[2] - box[0], box[3] - box[1])
    return float(min(ZOOM_MAX, max(1.0, CROP_SIDE_PX / max(1.0, side))))


def size_at_input(L: float, w: float, h: float) -> float:
    """The animal's longer side at the model input when an image of w x h is resized to the square input."""
    return max(L / max(1.0, w), L / max(1.0, h)) * MODEL_INPUT


def load_region_file(path, W: int, H: int) -> List[dict]:
    """--zoom REGIONFILE: JSON [{"box": [x0, y0, x1, y1], "animals": n}, ...] in full-frame pixels."""
    data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    out = []
    for k, r in enumerate(data):
        b = [int(v) for v in r["box"]]
        if not (0 <= b[0] < b[2] <= W and 0 <= b[1] < b[3] <= H):
            raise ValueError(f"region {k + 1} of {path}: box {b} is not inside the {W} x {H} frame")
        out.append(dict(box=b, animals=int(r.get("animals", 1)), raw=[float(v) for v in b], lim=dict(x0=b[0], y0=b[1], x1=b[2], y1=b[3]), share=1.0, given=True))
    return out


# ------------------------------------------------------------------ the plan -----------------------------------------
def plan(job, n_whole: int) -> int:
    """Decide between the whole frame and zoomed regions. Sets job.plan and, when regions are used, job.n_auto (the sum of the
    per-region counts). Returns the proposed total count."""
    from . import pipeline as P
    cv2 = _cv2()
    m = job.meta; W, H = m["W"], m["H"]
    mode = job.opt.zoom
    job.plan = dict(zoom=False, mode=mode, text="zoom off (--zoom off): whole frame")
    if mode == "off":
        return n_whole
    cd = getattr(job, "count_det", None) or dict(idx=[], dets=[])
    idx = cd["idx"]; whole = cd["dets"]; T = len(idx)
    if mode != "auto":
        regs = load_region_file(mode, W, H)
        L = float(np.median([max(d["box"][2] - d["box"][0], d["box"][3] - d["box"][1]) for d in whole if d["score"] >= CONF] or [0.0]))
        return _set_plan(job, regs, L, None, dict(x=None, y=None), "regions given by the user (" + str(mode) + ")", cv2)
    paths = [P.frame_path(job, a) for a in idx]
    seams = find_seams(paths[:: max(1, len(paths) // 12)]) if paths else dict(x=None, y=None)
    confw = [d for d in whole if d["score"] >= CONF]
    for ax in ("x", "y"):                       # DEVIATION 4: a seam that animals walk across is not a camera seam
        if seams.get(ax) is not None:
            k = crossings(confw, seams[ax], ax, T)
            if k >= max(2, SEAM_CROSS_SHARE * T):
                job.log(f"zoom: the step at the middle {'column' if ax == 'x' else 'row'} {seams[ax]} is crossed by an animal-sized "
                        f"detection on {k} of {T} sample frames: one camera, not two")
                seams[ax] = None
    f_whole = float(np.median([max((d["box"][2] - d["box"][0]) / W, (d["box"][3] - d["box"][1]) / H) for d in confw])) * MODEL_INPUT if confw else 0.0
    seen = float(np.mean([any(d["t"] == t for d in confw) for t in range(T)])) if T else 0.0
    mosaic = W >= 2 * H or H >= 2 * W          # the shape of two cameras side by side: always worth a closer look (see below)
    if f_whole >= SIZE_MIN_PX and seen >= 0.5 and seams["x"] is None and seams["y"] is None and not mosaic:
        job.plan = dict(zoom=False, mode=mode, size_px_at_input=round(f_whole, 1), seams=seams,
                        text=f"whole frame: the animals are about {f_whole:.0f} px long at the {MODEL_INPUT:.0f} px model input (zoom starts below {SIZE_MIN_PX:.0f} px)")
        job.log("zoom: " + job.plan["text"])
        return n_whole
    # the animals are small, absent or the frame holds two cameras: look closer
    job.set(message="scout: looking for small animals")
    tiled = _tile_detections(job, idx) if T else []
    dets = pool(whole, tiled)
    conf = [d for d in dets if d["score"] >= CONF]
    blobs = motion_blobs(paths, W, H) if paths else []
    if not conf:
        if not blobs:
            job.plan = dict(zoom=False, mode=mode, size_px_at_input=0.0, seams=seams, nothing=True,
                            text="the scout found no animal: nothing detected on the whole frame or on its pieces, and nothing moves")
            job.warnings.append("scout: " + job.plan["text"])
            job.log("zoom: " + job.plan["text"])
            return n_whole
        # nothing detected but something moves: regions from movement alone, the animal size from the moving blobs
        L = 1.3 * float(np.median([max(b["w"], b["h"]) for b in blobs]))
        dets = [dict(t=b["t"] * max(1, T) // max(1, len(paths)), box=[b["c"][0] - b["w"] / 2, b["c"][1] - b["h"] / 2, b["c"][0] + b["w"] / 2, b["c"][1] + b["h"] / 2],
                     score=CONF, area=b["w"] * b["h"] * 0.6, c=b["c"]) for b in blobs]
        why = "nothing was detected; regions come from movement alone"
    else:
        L = float(np.median([max(d["box"][2] - d["box"][0], d["box"][3] - d["box"][1]) for d in conf]))
        why = None
    s0 = size_at_input(L, W, H)
    # DEVIATION 1 from the pre-registration (20 Nov 2026, after the first collaborator clip; see STATUS.md): a frame shaped like two
    # cameras side by side (at least 2 : 1) in which one half shows no animal and no movement on any sample frame is treated
    # as two cameras even when no step is visible at the middle column, so the empty half is left out.
    if mosaic and seams["x"] is None and seams["y"] is None:
        ax = 0 if W >= 2 * H else 1; mid = (W if ax == 0 else H) / 2.0
        lo = sum(1 for d in conf if d["c"][ax] < mid) + sum(1 for b in blobs if b["c"][ax] < mid)
        hi = sum(1 for d in conf if d["c"][ax] >= mid) + sum(1 for b in blobs if b["c"][ax] >= mid)
        if (lo == 0) != (hi == 0):
            seams["x" if ax == 0 else "y"] = int(mid)
            seams["inferred"] = "one half of the frame is empty on every sample frame"
    if s0 >= SIZE_MIN_PX and seams["x"] is None and seams["y"] is None and seen >= 0.5:
        job.plan = dict(zoom=False, mode=mode, size_px_at_input=round(s0, 1), seams=seams,
                        text=f"whole frame: the animals are about {s0:.0f} px long at the model input (zoom starts below {SIZE_MIN_PX:.0f} px)")
        job.log("zoom: " + job.plan["text"])
        return n_whole
    regs = find_regions(dets, T, W, H, L, seams, blobs)
    if not regs:
        job.plan = dict(zoom=False, mode=mode, size_px_at_input=round(s0, 1), seams=seams,
                        text=f"whole frame: the animals look small ({s0:.0f} px at the model input) but no region with a steady animal count was found")
        job.warnings.append("zoom: " + job.plan["text"])
        job.log("zoom: " + job.plan["text"])
        return n_whole
    return _set_plan(job, regs, L, s0, seams, why, cv2)


def _set_plan(job, regs: List[dict], L: float, s0: Optional[float], seams: dict, why: Optional[str], cv2) -> int:
    from . import pipeline as P
    m = job.meta
    out = []
    for k, r in enumerate(regs):
        b = r["box"]; z = zoom_factor(b)
        out.append(dict(region=k + 1, box=[int(v) for v in b], animals=int(r["animals"]), zoom=round(z, 3), lim=r.get("lim"), share=round(float(r.get("share", 1.0)), 2),
                        size_px_at_input=round(size_at_input(L, b[2] - b[0], b[3] - b[1]), 1) if L else None, given=bool(r.get("given")),
                        support=support_of(r.get("counts"))))
    total = sum(r["animals"] for r in out)
    small = [r for r in out if r["size_px_at_input"] is not None and r["size_px_at_input"] < SIZE_MIN_PX]
    txt = (f"zoom: {len(out)} region(s), animals per region {[r['animals'] for r in out]}"
           + (f"; the animals are about {s0:.0f} px long at the model input on the whole frame" if s0 is not None else "")
           + f" and {[r['size_px_at_input'] for r in out]} px inside the regions (enlarged {[r['zoom'] for r in out]} times)"
           + ("; the frame is two cameras side by side: regions never cross the seam" if (seams.get("x") is not None or seams.get("y") is not None) else "")
           + (f"; {why}" if why else ""))
    job.plan = dict(zoom=True, mode=job.opt.zoom, size_px_at_input=None if s0 is None else round(s0, 1), body_length_px=round(L, 1), seams=seams, regions=out, text=txt)
    job.log(txt)
    for r in small:
        job.warnings.append(f"region {r['region']}: the animal is still small after the zoom ({r['size_px_at_input']:.0f} px at the model input; detection "
                            f"is reliable from about {SIZE_MIN_PX:.0f} px): the area the animals use is large compared with the animal")
    write_json(job.work / "zoom.json", job.plan)
    # the proposal as a picture under the count sheet: the middle sample frame with the regions and their counts
    try:
        cd = getattr(job, "count_det", None) or {}
        idx = cd.get("idx") or [0]
        im = cv2.imread(str(P.frame_path(job, idx[len(idx) // 2])))
        lw = max(2, im.shape[1] // 500)
        for r in out:
            x0, y0, x1, y1 = r["box"]
            cv2.rectangle(im, (x0, y0), (x1 - 1, y1 - 1), (0, 140, 255), lw)
            lab = f"region {r['region']}: {r['animals']}"
            cv2.putText(im, lab, (x0 + 6, y0 + 22 * lw // 2 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * lw, (255, 255, 255), 2 * lw, cv2.LINE_AA)
            cv2.putText(im, lab, (x0 + 6, y0 + 22 * lw // 2 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * lw, (0, 100, 220), lw, cv2.LINE_AA)
        if job.count_sheet and Path(job.count_sheet).exists():
            sheet = cv2.imread(str(job.count_sheet)); w = sheet.shape[1]
            im = cv2.resize(im, (w, max(1, int(round(im.shape[0] * w / im.shape[1])))), interpolation=cv2.INTER_AREA)
            bar = np.full((30, w, 3), 255, np.uint8)
            cv2.putText(bar, "Proposed regions and animals per region (each is analysed zoomed in)", (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.imwrite(str(job.count_sheet), np.vstack([sheet, bar, im]), [cv2.IMWRITE_JPEG_QUALITY, 90])
    except Exception as e:                                          # the picture is a help, never a reason to stop
        job.log(f"(could not draw the regions on the count sheet: {type(e).__name__})")
    job.n_auto = int(total)
    return int(total)


# ------------------------------------------------------------------ analysing the regions ----------------------------
def support_of(counts) -> Optional[List[float]]:
    """support[n - 1] = share of the sample frames on which the region shows at least n confident detections (n = 1 .. 16)."""
    if counts is None:
        return None
    c = np.asarray(counts)
    if not len(c):
        return None
    return [round(float((c >= n).mean()), 3) for n in range(1, 17)]


def _supp(r: dict, n: int) -> float:
    s = r.get("support")
    if not s or n < 1:
        return 1.0 if n >= 1 else 0.0
    return float(s[min(n, len(s)) - 1])


def split_count(regs: List[dict], N: int) -> List[int]:
    """The confirmed total N over the regions; the result always sums to N.
    Equal to the proposal: the proposal. More: the extra animals go to the regions with the most room (largest area per
    proposed animal). Fewer (the proposal over-counts: a reflection, a second camera view, a sporadic false detection):
    one animal at a time is taken from the region whose CURRENT count is seen on the smallest share of the sample frames
    (`support`, re-evaluated after every step). The earlier rule ranked the regions once, by the support of the proposed
    count, and so could empty a region that clearly holds an animal (gfp_nor_5boxes_objects: [3, 1, 2, 2] -> [0, 1, 2, 2])."""
    n = [int(r["animals"]) for r in regs]
    if N <= 0 or not regs:
        return [0] * len(regs)
    while sum(n) < N:
        k = max(range(len(regs)), key=lambda i: (regs[i]["box"][2] - regs[i]["box"][0]) * (regs[i]["box"][3] - regs[i]["box"][1]) / (n[i] + 1))
        n[k] += 1
    while sum(n) > N:
        k = min((i for i in range(len(regs)) if n[i] > 0), key=lambda i: (_supp(regs[i], n[i]), regs[i].get("share", 1.0), -n[i], i))
        n[k] -= 1
    return n


def make_region_job(job, k: int, box: List[int], z: float):
    """A job whose frames are the region's crop, enlarged by z: the unchanged engine then runs on it."""
    from . import pipeline as P
    cv2 = _cv2()
    m = job.meta
    sub = P.Job(job.video, job.work / "regions_out", job.opt, name=f"region_{k}")
    sub.work = job.work / f"region_{k}"; sub.dir = sub.work
    x0, y0, x1, y1 = box
    cw, ch = int(round((x1 - x0) * z)) // 2 * 2, int(round((y1 - y0) * z)) // 2 * 2
    nat = sub.work / "native"; nat.mkdir(parents=True, exist_ok=True)
    for g in range(m["native"]):
        f = nat / f"{g:06d}.jpg"
        if f.exists():
            continue
        im = cv2.imread(str(job.work / "native" / f"{g:06d}.jpg"))[y0:y1, x0:x1]
        if (im.shape[1], im.shape[0]) != (cw, ch):
            im = cv2.resize(im, (cw, ch), interpolation=cv2.INTER_CUBIC if z > 1 else cv2.INTER_AREA)
        cv2.imwrite(str(f), im, [cv2.IMWRITE_JPEG_QUALITY, 93])
        if g % 100 == 0:
            job.check()
    fr = sub.work / "frames"; fr.mkdir(exist_ok=True)
    for j, i in enumerate(range(0, m["native"], m["step"])):
        if not (fr / f"{j:06d}.jpg").exists():
            P._link(nat / f"{i:06d}.jpg", fr / f"{j:06d}.jpg")
    sub.meta = dict(m, W=cw, H=ch)
    sub.engine = job.engine; sub.stop = job.stop; sub.timings = job.timings; sub.warnings = []
    sub.on_line = lambda line, k=k: job.log(f"[region {k}] {line}")
    sub.zx, sub.zy = cw / float(x1 - x0), ch / float(y1 - y0)
    return sub


def map_back(rows: List[dict], box: List[int], zx: float, zy: float, region: int, id0: int, cv2) -> List[dict]:
    """Rows of a region (crop pixels) -> full-frame pixels; identities shifted so that they are unique across regions."""
    X0, Y0 = box[0], box[1]
    out = []
    for r in rows:
        q = dict(r)
        q["id"] = r["id"] + id0; q["region"] = region; q["zoom"] = (zx + zy) / 2
        q["x"], q["y"] = r["x"] / zx + X0, r["y"] / zy + Y0
        if r.get("tx") is not None:
            q["tx"], q["ty"] = r["tx"] / zx + X0, r["ty"] / zy + Y0
        if r.get("box"):
            b = r["box"]
            nb = [int(math.floor(b[0] / zx + X0)), int(math.floor(b[1] / zy + Y0)), int(math.ceil(b[2] / zx + X0)), int(math.ceil(b[3] / zy + Y0))]
            nb[2] = max(nb[2], nb[0] + 1); nb[3] = max(nb[3], nb[1] + 1)
            q["box"] = nb
            if r.get("mask") is not None:
                bits, h, w = r["mask"]
                mk = np.unpackbits(bits)[:h * w].reshape(h, w)
                if (zx, zy) != (1.0, 1.0) or (nb[3] - nb[1], nb[2] - nb[0]) != (h, w):
                    mk = cv2.resize(mk, (nb[2] - nb[0], nb[3] - nb[1]), interpolation=cv2.INTER_NEAREST)
                q["mask"] = (np.packbits(mk.astype(bool)), nb[3] - nb[1], nb[2] - nb[0])
        if r.get("det_box"):
            b = r["det_box"]; q["det_box"] = [b[0] / zx + X0, b[1] / zy + Y0, b[2] / zx + X0, b[3] / zy + Y0]
        if r.get("kp") is not None:
            q["kp"] = np.asarray(r["kp"], float) / [zx, zy] + [X0, Y0]
        out.append(q)
    return out


def touches(frames: List[List[dict]], w: int, h: int, lim_free: dict) -> List[str]:
    """Sides of the crop that an animal's box touches on at least 5 frames and that the region may still grow towards."""
    hit = dict(x0=0, y0=0, x1=0, y1=0)
    for rows in frames:
        f = dict(x0=False, y0=False, x1=False, y1=False)
        for r in rows:
            b = r.get("box")
            if b:
                f["x0"] |= b[0] <= 2; f["y0"] |= b[1] <= 2; f["x1"] |= b[2] >= w - 2; f["y1"] |= b[3] >= h - 2
        for k in hit:
            hit[k] += f[k]
    return [k for k in hit if hit[k] >= 5 and lim_free.get(k)]


def run_regions(job, N: int) -> dict:
    from . import pipeline as P
    cv2 = _cv2()
    m = job.meta; W, H = m["W"], m["H"]
    plan_ = job.plan; regs = [dict(r) for r in plan_["regions"]]; L = float(plan_.get("body_length_px") or 0.0)
    counts = split_count(regs, N)
    if counts != [r["animals"] for r in regs]:
        job.log(f"the confirmed number of animals ({N}) differs from the proposal ({sum(r['animals'] for r in regs)}): animals per region now {counts}")
    for r, c in zip(regs, counts):
        r["animals"] = c
    full = job.engine["engine"] == "full"
    results = []; id0 = 0; k = 0; retried = set()
    todo = [r for r in regs if r["animals"] >= 1]
    done_regs = []
    while k < len(todo):
        r = todo[k]; z = zoom_factor(r["box"])
        tag = f"{len(done_regs) + 1}{'b' if id(r) in retried else ''}"
        sub = make_region_job(job, tag, r["box"], z)
        base = 0.20 + 0.60 * k / max(1, len(todo)); span = 0.60 / max(1, len(todo))
        sub.set = lambda status=None, progress=None, message=None, base=base, span=span, tag=tag: job.set(
            "tracking", base + span * max(0.0, ((progress or 0.2) - 0.2) / 0.7), f"region {tag}: {message}" if message else None)
        job.log(f"region {tag}: box {r['box']} enlarged {z:.2f} times, {r['animals']} animal(s)")
        try:
            res = P.track_full(sub, r["animals"]) if full else P.track_simple(sub, r["animals"])
        except (P.StepError, P.NothingFound) as e:
            if str(e).startswith(P.NO_ANIMAL[:40]) or isinstance(e, P.NothingFound):
                job.warnings.append(f"region {tag} (box {r['box']}): no animal was detected in this region on any frame; the region is dropped "
                                    f"({r['animals']} of the {N} animals are not tracked)")
            else:
                job.warnings.append(f"region {tag} (box {r['box']}): nothing tracked: {str(e)[:200]}")
            job.log(f"region {tag}: dropped ({str(e)[:160]})")
            k += 1
            continue
        if not any(q.get("box") for rows in res["frames"] for q in rows):
            job.warnings.append(f"region {tag} (box {r['box']}): no animal was detected in this region on any frame; the region is dropped "
                                f"({r['animals']} of the {N} animals are not tracked)")
            job.log(f"region {tag}: dropped (no box on any frame)")
            k += 1
            continue
        lim = r.get("lim") or dict(x0=0, y0=0, x1=W, y1=H)
        free = dict(x0=r["box"][0] > lim["x0"] + 1, y0=r["box"][1] > lim["y0"] + 1, x1=r["box"][2] < lim["x1"] - 1, y1=r["box"][3] < lim["y1"] - 1)
        sides = touches(res["frames"], sub.meta["W"], sub.meta["H"], free) if not r.get("given") else []
        if sides and id(r) not in retried and L > 0:
            b = list(r["box"]); g = int(round(GROW_BL * L))
            if "x0" in sides: b[0] = max(int(lim["x0"]), b[0] - g)
            if "y0" in sides: b[1] = max(int(lim["y0"]), b[1] - g)
            if "x1" in sides: b[2] = min(int(lim["x1"]), b[2] + g)
            if "y1" in sides: b[3] = min(int(lim["y1"]), b[3] + g)
            nr = dict(r, box=b)
            for other in list(todo[k + 1:]):                     # growing into a neighbour: the two become one region
                o = other["box"]
                if min(b[2], o[2]) - max(b[0], o[0]) > 0 and min(b[3], o[3]) - max(b[1], o[1]) > 0:
                    nr["box"] = [min(b[0], o[0]), min(b[1], o[1]), max(b[2], o[2]), max(b[3], o[3])]; nr["animals"] += other["animals"]; todo.remove(other)
                    b = nr["box"]
            job.log(f"region {tag}: an animal touches the edge of the region ({', '.join(sides)}); the region is grown to {nr['box']} and analysed again")
            retried.add(id(nr)); todo[k] = nr
            continue
        if sides:
            job.warnings.append(f"region {tag}: an animal still touches the edge of the region after it was grown; it may leave the analysed area")
        for w_ in sub.warnings:
            job.warnings.append(f"region {tag}: {w_}")
        results.append((r, sub, res, id0, len(done_regs) + 1)); id0 += r["animals"]; done_regs.append(dict(region=len(done_regs) + 1, box=r["box"], animals=r["animals"], zoom=round(z, 3)))
        k += 1
    if not results:
        raise P.StepError(P.NO_ANIMAL + f" (zoom: {len(todo)} region(s) were analysed)")
    NO = max(len(res["frames"]) for _, _, res, _, _ in results)
    frames: List[List[dict]] = [[] for _ in range(NO)]
    for r, sub, res, i0, num in results:
        for g, rows in enumerate(res["frames"]):
            frames[g] += map_back(rows, r["box"], sub.zx, sub.zy, num, i0, cv2)
    first = results[0][2]
    geo = dict(first.get("geometry") or {})
    for key in ("seeds", "repairs", "checked", "bad", "inconsistent"):
        if any(key in (res.get("geometry") or {}) for _, _, res, _, _ in results):
            geo[key] = sum(int((res.get("geometry") or {}).get(key, 0)) for _, _, res, _, _ in results)
    out = dict(first)
    out.update(frames=frames, body_length_px=float(np.median([res["body_length_px"] / ((sub.zx + sub.zy) / 2) for _, sub, res, _, _ in results])),
               windows=[dict(w_, region=num) for _, _, res, _, num in results for w_ in res.get("windows", [])],
               seams=[dict(s_, region=num) for _, _, res, _, num in results for s_ in res.get("seams", [])],
               fallback_windows=sum(int(res.get("fallback_windows") or 0) for _, _, res, _, _ in results), geometry=geo, regions=done_regs,
               zoom=dict(used=True, text=plan_["text"], size_px_at_input=plan_.get("size_px_at_input"), seams=plan_.get("seams"), regions=done_regs,
                         note="every region was analysed as its own video (cropped, enlarged by its zoom factor) and mapped back to full-frame pixels; "
                              "identities are unique across regions and an animal is confined to its region"))
    return out
