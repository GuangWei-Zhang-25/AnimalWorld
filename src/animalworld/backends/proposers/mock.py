"""Mock proposer: reads the known geometry written by the mock generator and adds label noise.

Options: jitter_px (default 1.0), miss_rate (default 0.0: probability of dropping an instance, to
exercise the count gate), score (default 0.9).
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Iterator, List

from ...formats import labels as L
from ...utils import read_json, seed_from
from .base import Proposer


class MockProposer(Proposer):
    name = "mock"
    tested = True

    def propose(self, rows: List[dict]) -> Iterator[dict]:
        j = float(self.opt.get("jitter_px", 1.0))
        miss = float(self.opt.get("miss_rate", 0.0))
        for r in rows:
            truth = read_json(Path(r["path"]).parent.parent / "truth" / f"{r['id']}.json")
            if truth is None:
                raise FileNotFoundError(f"mock proposer: no ground truth for {r['id']} (it only works on images "
                                        "from the mock generator; use the sam or yolo proposer for real images)")
            rng = random.Random(seed_from("mockprop", r["id"]))
            inst = []
            for i in truth["instances"]:
                if rng.random() < miss:
                    continue
                dx, dy = rng.uniform(-j, j), rng.uniform(-j, j)
                poly = [[x + dx, y + dy] for x, y in i["poly"]]
                kps = {k: [v[0] + rng.uniform(-j, j), v[1] + rng.uniform(-j, j), v[2]] for k, v in (i.get("keypoints") or {}).items()}
                inst.append(L.instance(self.category, L.box_of_poly(poly), poly, kps or None,
                                       score=float(self.opt.get("score", 0.9)), track_id=i.get("track_id")))
            yield L.label(r["id"], r["file"], truth["W"], truth["H"], inst, source="mock")
