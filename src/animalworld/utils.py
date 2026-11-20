"""Small stdlib helpers shared by every stage (atomic writes, JSONL, claim locks, hashing)."""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Iterable, Iterator


def slug(text: Any) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", str(text)).strip("_").lower()
    return s or "x"


def md5_unit(text: str) -> float:
    """Deterministic number in [0, 1) from a string (used for train/val splits; no leakage between units)."""
    return int(hashlib.md5(text.encode()).hexdigest(), 16) / 2 ** 128


def seed_from(*parts: Any) -> int:
    return int(hashlib.md5("|".join(str(p) for p in parts).encode()).hexdigest()[:8], 16)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=1, ensure_ascii=False) + "\n")


def read_json(path: Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    atomic_write_text(path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def append_jsonl(path: Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


class Claims:
    """O_EXCL lock files so several workers can share one work queue.

    Adapted from the work-queue in mw_det_syn/gap_targeted/scripts/gen_qwen.py: a worker claims an
    item, stale claims (no output, older than `stale_s`) are reaped, unfinished claims are released.
    """

    def __init__(self, directory: Path, stale_s: float = 3600.0):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.stale_s = stale_s
        self.owner = os.environ.get("SLURM_JOB_ID", str(os.getpid()))

    def _p(self, item_id: str) -> Path:
        return self.dir / f"{item_id}.lock"

    def reap(self, is_done) -> int:
        n, now = 0, time.time()
        for lk in self.dir.glob("*.lock"):
            try:
                if not is_done(lk.stem) and now - lk.stat().st_mtime > self.stale_s:
                    lk.unlink()
                    n += 1
            except OSError:
                pass
        return n

    def claim(self, item_id: str) -> bool:
        try:
            fd = os.open(self._p(item_id), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        os.write(fd, self.owner.encode())
        os.close(fd)
        return True

    def release(self, item_id: str) -> None:
        try:
            self._p(item_id).unlink()
        except OSError:
            pass

    def held(self, item_id: str) -> bool:
        return self._p(item_id).exists()


def chunks(seq: list, n: int) -> Iterator[list]:
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".webm")
