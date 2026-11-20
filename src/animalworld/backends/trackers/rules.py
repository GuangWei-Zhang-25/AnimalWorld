"""Adapter for an external rule-based tracker (the lab's MouseWorld-Track "agent" rules).

The rule set (TrackPose_demonstration/long_runs/agent_v7/, frozen zero-shot version v7b) is not a
pure function: `run(clip, t0, t1)` reads a work folder (det.npz from a detector, tracks.npz from
SAM 2 forward / backward propagation, optional extras) and writes fused7.npz. It also depends on
SAM 2 witnesses that this package does not produce. It is therefore NOT bundled. This adapter:

  1. writes the detections of this project in a neutral npz (`det_export.npz`: t, box, score, c, area,
     T, W, H; the field names follow the agent's det.npz, but mask bits are not included);
  2. calls a user-supplied Python callable `module:function` (options.entry) with
     (work_dir, n_animals, options) which must return per-frame identities, one list per frame.

Since 0.2.0 the agent itself IS bundled, but as the zero-shot engine for the pretrained MouseWorld detectors
(`animalworld zeroshot`, animalworld/zeroshot/); this adapter remains the hook for running your own rule code on a
project-trained detector.

Without options.entry it stops with an explanation. UNTESTED against the real agent; only the export
and the calling convention are exercised with a stand-in callable in the test-suite.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from ...registry import BackendError
from .base import Tracker


def export_detections(frames: List[List[dict]], path: Path, W: int = 0, H: int = 0) -> Path:
    t, box, score = [], [], []
    for k, f in enumerate(frames):
        for i in f:
            t.append(k)
            box.append(i["box"])
            score.append(i.get("score", 1.0))
    box = np.array(box, dtype=np.float32).reshape(-1, 4)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, t=np.array(t, dtype=np.int32), box=box, score=np.array(score, dtype=np.float32),
                        c=np.stack([(box[:, 0] + box[:, 2]) / 2, (box[:, 1] + box[:, 3]) / 2], 1) if len(box) else np.zeros((0, 2), np.float32),
                        area=(box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1]) if len(box) else np.zeros((0,), np.float32),
                        T=len(frames), W=W, H=H)
    return path


class RuleTrackerAdapter(Tracker):
    name = "rules"
    tested = False

    def __init__(self, options: Dict[str, Any] = None, info: dict = None, work_dir: Path = None, **kw: Any):
        super().__init__(options, **kw)
        self.info = info or {}
        self.work_dir = Path(work_dir) if work_dir else Path(".")

    def track(self, frames: List[List[dict]]) -> List[List[int]]:
        det = export_detections(frames, self.work_dir / "det_export.npz", self.info.get("W", 0), self.info.get("H", 0))
        entry = self.opt.get("entry")
        if not entry:
            raise BackendError(
                f"rules tracker: detections exported to {det}. Set track.options.entry to 'module:function' (and "
                "track.options.pythonpath to the folder that holds it) to run your rule-based tracker on them. "
                "See docs/BACKENDS.md, section 'Rule-based tracker adapter'.")
        if self.opt.get("pythonpath"):
            sys.path.insert(0, str(self.opt["pythonpath"]))
        mod, fn = entry.split(":")
        ids = getattr(importlib.import_module(mod), fn)(str(self.work_dir), self.n, dict(self.opt))
        if len(ids) != len(frames) or any(len(a) != len(b) for a, b in zip(ids, frames)):
            raise BackendError("rules tracker: the callable must return one identity per detection, per frame")
        if not self.n:
            self.n = len({i for row in ids for i in row if i >= 0})
        return [[int(i) for i in row] for row in ids]
