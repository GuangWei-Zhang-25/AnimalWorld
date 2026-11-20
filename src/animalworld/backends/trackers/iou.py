"""Default tracker: fixed number of animals, Hungarian assignment on centre distance + (1 - IoU) +
coat-brightness difference. Numpy / pure Python.

Ideas taken from the lab's light tracker (mouseworld_app/src/mouseworld/track.py, AGPL-3.0), written
fresh and much simpler: the animal count N is the largest count seen on >= 25 % of frames; tracking
starts at the first frame where exactly N detections are all apart, runs forward and then backward
from there; a slot that loses its detection keeps its last position; identities never multiply.
It does not resolve identity through long full occlusions or huddles; use the `rules` adapter (or a
dedicated tracker) for crowded scenes, and check the result by eye.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from ...formats.labels import iou
from .base import Tracker

BIG = 1e6


def hungarian(cost: List[List[float]]) -> List[int]:
    """Minimum-cost assignment for an n x m matrix (n <= m). Returns the column chosen for each row."""
    n, m = len(cost), len(cost[0]) if cost else 0
    if n == 0:
        return []
    assert n <= m
    INF = float("inf")
    u, v = [0.0] * (n + 1), [0.0] * (m + 1)
    p, way = [0] * (m + 1), [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], INF, 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    out = [-1] * n
    for j in range(1, m + 1):
        if p[j]:
            out[p[j] - 1] = j - 1
    return out


def centre(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def estimate_n(frames: List[List[dict]], share: float = 0.25) -> int:
    counts = [len(f) for f in frames]
    if not counts:
        return 0
    for n in range(max(counts), 0, -1):
        if sum(1 for c in counts if c >= n) >= share * len(counts):
            return n
    return 0


def body_length(frames: List[List[dict]]) -> float:
    sides = sorted(math.sqrt(max(1.0, (i["box"][2] - i["box"][0]) * (i["box"][3] - i["box"][1]))) for f in frames for i in f)
    return sides[len(sides) // 2] if sides else 1.0


class IoUTracker(Tracker):
    name = "iou"
    tested = True

    def __init__(self, options: Dict[str, Any] = None, **kw: Any):
        super().__init__(options, **kw)
        self.gate = float(self.opt.get("gate_bl", 1.5))               # max centre jump per frame, in body lengths
        self.w_app = float(self.opt.get("appearance_weight", 3.0))
        self.memory = float(self.opt.get("appearance_memory", 0.3))

    def _assign(self, slots: List[dict], dets: List[dict], bl: float) -> List[int]:
        """-> for each detection the slot index or -1."""
        ids = [-1] * len(dets)
        if not dets or not slots:
            return ids
        n, m = len(slots), len(dets)
        size = max(n, m)
        cost = [[BIG] * size for _ in range(size)]
        for a, s in enumerate(slots):
            for b, d in enumerate(dets):
                cs, cd = centre(s["box"]), centre(d["box"])
                dist = math.hypot(cs[0] - cd[0], cs[1] - cd[1]) / bl
                if dist > self.gate * (1 + min(s["missed"], 5)):
                    continue
                c = dist + (1.0 - iou(s["box"], d["box"]))
                if s.get("app") is not None and d.get("appearance") is not None:
                    c += self.w_app * abs(s["app"] - d["appearance"])
                cost[a][b] = c
        match = hungarian(cost)
        for a in range(n):
            b = match[a]
            if b < m and cost[a][b] < BIG:
                ids[b] = a
        return ids

    def _run(self, frames, order, slots, bl, out):
        for t in order:
            dets = frames[t]
            ids = self._assign(slots, dets, bl)
            got = set()
            for d, a in zip(dets, ids):
                if a >= 0:
                    s = slots[a]
                    s["box"], s["missed"] = d["box"], 0
                    if d.get("appearance") is not None:
                        s["app"] = d["appearance"] if s.get("app") is None else (1 - self.memory) * s["app"] + self.memory * d["appearance"]
                    got.add(a)
            for a, s in enumerate(slots):
                if a not in got:
                    s["missed"] += 1
            out[t] = ids

    def track(self, frames: List[List[dict]]) -> List[List[int]]:
        T = len(frames)
        out: List[List[int]] = [[-1] * len(f) for f in frames]
        n = int(self.n) if self.n else estimate_n(frames)
        self.n = n
        if not n or not T:
            return out
        bl = body_length(frames)
        start: Optional[int] = None
        for t, f in enumerate(frames):
            if len(f) == n and all(math.hypot(*(a - b for a, b in zip(centre(f[i]["box"]), centre(f[j]["box"])))) > 0.5 * bl
                                   for i in range(n) for j in range(i)):
                start = t
                break
        if start is None:
            start = max(range(T), key=lambda t: min(len(frames[t]), n))
        seed = sorted(frames[start][:n], key=lambda d: (centre(d["box"])[0], centre(d["box"])[1]))
        mk = lambda: [{"box": d["box"], "missed": 0, "app": d.get("appearance")} for d in seed]     # noqa: E731
        while len(seed) < n:                     # fewer detections than animals at the best frame
            seed.append({"box": [-1e5, -1e5, -1e5 + 1, -1e5 + 1], "appearance": None})
        self._run(frames, range(start, T), mk(), bl, out)
        self._run(frames, range(start - 1, -1, -1), mk(), bl, out)
        return out
