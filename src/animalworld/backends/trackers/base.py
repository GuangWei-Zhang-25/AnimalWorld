"""Trackers: ``track(frames) -> ids`` where frames is a list (one entry per frame) of instance lists and
ids has the same shape, with -1 for a detection that received no identity."""
from __future__ import annotations

from typing import Any, Dict, List


class Tracker:
    name = "base"
    tested = False

    def __init__(self, options: Dict[str, Any] = None, **_: Any):
        self.opt = options or {}
        self.n = self.opt.get("n_animals")

    def track(self, frames: List[List[dict]]) -> List[List[int]]:
        raise NotImplementedError
