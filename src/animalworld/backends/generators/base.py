"""Generator backends: turn manifest rows (prompts) into image or video files.

Interface (see docs/BACKENDS.md):

    class MyGenerator(Generator):
        name = "my_generator"; kind = "image"            # or "video"
        def plan(self, items):  -> dict                  # what would run; must not load any model
        def generate(self, items, out_dir):              # yields one dict per finished item
            ... write out_dir/<id><self.ext> atomically ...
            yield {"id": item["id"], "file": path, ...}

The `generate` stage owns everything else: the work queue, claims, resumption, the metadata log,
frame extraction for videos and the image index.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List


class Generator:
    name = "base"
    kind = "image"          # "image" -> one file per item; "video" -> one clip per item; "frames" -> a folder of frames
    ext = ".jpg"
    needs_gpu = False
    tested = False          # True only for backends executed in this package's test-suite

    def __init__(self, options: Dict[str, Any] = None, project=None, **_: Any):
        self.opt = options or {}
        self.project = project

    def plan(self, items: List[dict]) -> dict:
        return {"backend": self.name, "kind": self.kind, "items": len(items), "needs_gpu": self.needs_gpu}

    def generate(self, items: List[dict], out_dir: Path) -> Iterator[dict]:
        raise NotImplementedError
