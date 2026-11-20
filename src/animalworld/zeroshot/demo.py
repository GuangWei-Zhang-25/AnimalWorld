"""`animalworld demo`: fetch the small demo clips, check them, and open Animal World Run with them loaded.

    animalworld demo                       window with the demo clips listed; press Start
    animalworld demo --headless --out DIR  the same clips through the engine without a window

The clips are a separate download, like the weights (the files of demo_release/). They are looked for in
$ANIMALWORLD_DEMO (a folder you already have), then in <cache>/demo, then fetched from the demo address, which is
$ANIMALWORLD_DEMO_URL or the constant below (an https address, a file:// address or a plain local folder). Every file is
checked against the SHA-256 recorded here before it is used.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import List

from .. import __version__, weights as W

# Address of the release that holds the demo clips. EDIT after the upload (or set ANIMALWORLD_DEMO_URL). While it still
# contains the placeholder OWNER/REPO (e.g. set through the variable) nothing is downloaded and the command says so.
DEMO_URL = "https://github.com/GuangWei-Zhang-25/AnimalWorld/releases/download/demo-v1/"
_UNSET = "OWN" + "ER/" + "RE" + "PO"

# Filled from demo_release/manifest.json (see that folder's README for sources and licence).
CLIPS: List[dict] = [
    # CLIPS_BEGIN
    dict(file="mouse_x1_cage.mp4", sha256="17eb2e3ed8a62e7a8621854f02ba548bffa54e28f2f5e730dd6b90d87fc5f010", size=9570075, animals=1,
         about="one mouse walking around the cage"),
    dict(file="mouse_x2_cage.mp4", sha256="b414a7cd679bd5a66baee95dba125efbdf5dd93ef73e3b05ecea4b89259ee30f", size=11201994, animals=2,
         about="two mice, contact at the start, then apart"),
    dict(file="mouse_x2_arena_a.mp4", sha256="cf50545000cef90816b9f69058af352c6ba4794d3e46ed313e8e2448cb9d0162", size=5861163, animals=2,
         about="two black mice in a grey arena, one short contact"),
    dict(file="mouse_x2_arena_b.mp4", sha256="5ac37f2eb8b28189a1e8d384a7c3c5ff4b6182268fced565d54f983c599db18f", size=8303477, animals=2,
         about="two black mice in a bright arena, one short contact"),
    dict(file="mouse_x4_crossing.mp4", sha256="eb51850b4299f87ece17da84f3b0ec193534103e1af1515894b6b7460d3c9b8e", size=12137581, animals=4,
         about="four mice with crossings"),
    dict(file="mouse_x5_huddle.mp4", sha256="4c8c41b8ae904e06560db7a4dda12a6afc179485f7bb0c63984f2c2879678d0b", size=11636979, animals=5,
         about="five mice with close contacts"),
    # CLIPS_END
]


def demo_dir() -> Path:
    return W.cache_dir() / "demo"


def demo_url() -> str:
    u = os.environ.get("ANIMALWORLD_DEMO_URL") or DEMO_URL
    return "" if (not u or _UNSET in u) else u


def _local(d: Path, e: dict):
    p = d / e["file"]
    return p if p.is_file() else None


def fetch(clips: List[dict] = None) -> List[Path]:
    """All demo clips as local files (found, copied or downloaded), each with the recorded SHA-256."""
    out = []
    for e in clips if clips is not None else CLIPS:
        own = Path(os.environ["ANIMALWORLD_DEMO"]).expanduser() if os.environ.get("ANIMALWORLD_DEMO") else None
        p = (own and _local(own, e)) or _local(demo_dir(), e)
        if p is None:
            url = demo_url()
            if not url:
                raise W.WeightsError(
                    f"demo clip {e['file']} is not in {demo_dir()} and no download address is set (DEMO_URL in animalworld/zeroshot/demo.py "
                    "is still the placeholder, ANIMALWORLD_DEMO_URL is empty). Put the demo clips in that folder, or set ANIMALWORLD_DEMO "
                    "to the folder that holds them, or set ANIMALWORLD_DEMO_URL to the release address or a local folder.")
            dst = demo_dir() / e["file"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            src = Path(url[7:] if url.startswith("file://") else url).expanduser()
            if "://" not in url or url.startswith("file://"):                # a local folder: copy, then check
                if not (src / e["file"]).is_file():
                    raise W.WeightsError(f"demo clip {e['file']} is not in the folder {src}")
                print(f"animalworld: copying demo clip {e['file']} from {src}", flush=True)
                tmp = dst.with_name(dst.name + ".part")
                shutil.copyfile(src / e["file"], tmp)
                got = W.sha256(tmp)
                if got != e["sha256"]:
                    os.replace(tmp, dst.with_name(dst.name + ".badsum"))
                    raise W.WeightsError(f"demo clip {e['file']}: SHA-256 {got[:16]}... does not match the recorded {e['sha256'][:16]}...; "
                                         f"kept as {dst.name}.badsum and NOT used")
                os.replace(tmp, dst)
                p = dst
            else:
                p = W.fetch_file(dict(e, title="demo clip " + e["file"]), url if url.endswith("/") else url + "/", dst)
        elif W.sha256(p) != e["sha256"]:
            raise W.WeightsError(f"demo clip {p}: its SHA-256 does not match the recorded one; remove or replace that file")
        out.append(p)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="animalworld demo", description="Fetch the demo clips and run Animal World Run on them.")
    ap.add_argument("--headless", action="store_true", help="no window: run the engine on the demo clips (needs --out)")
    ap.add_argument("--out", help="results folder (default with a window: ~/animalworld_demo_results)")
    ap.add_argument("--engine", default="auto", choices=("auto", "full", "lite", "mock"))
    ap.add_argument("--tracker-detector", help="see `animalworld zeroshot --help` (default: scored_pair)")
    ap.add_argument("--sam3", action="store_true", help="also use the optional SAM 3.1 witness (gated weights)")
    ap.add_argument("--list", action="store_true", help="only list the demo clips and where they are looked for")
    ap.add_argument("--version", action="version", version=f"animalworld {__version__}")
    a = ap.parse_args(argv)
    if a.list:
        print("looked for in: " + ", ".join(str(x) for x in ([os.environ["ANIMALWORLD_DEMO"]] if os.environ.get("ANIMALWORLD_DEMO") else []) + [demo_dir()]))
        print("download address: " + (demo_url() or "(not set: DEMO_URL is still the placeholder)"))
        for e in CLIPS:
            print(f"  {e['file']:<26} {e['size'] / 1e6:5.1f} MB  {e['animals']} animal(s)  {e['about']}")
        return 0
    if a.headless and not a.out:
        print("animalworld demo: --headless needs --out DIR", file=sys.stderr)
        return 2
    try:
        clips = fetch()
    except W.WeightsError as e:
        print(f"animalworld demo: {e}", file=sys.stderr)
        return 2
    if not clips:
        print("animalworld demo: this build lists no demo clips", file=sys.stderr)
        return 2
    extra = ["--engine", a.engine] + (["--tracker-detector", a.tracker_detector] if a.tracker_detector else []) + (["--sam3"] if a.sam3 else [])
    if a.headless:
        from .cli import main as zmain
        rc = 0
        for p, e in zip(clips, CLIPS):          # the count of each demo clip is known (counted by eye when the bundle was made)
            rc = max(rc, zmain([str(p), "--out", a.out, "--n-animals", str(e["animals"])] + extra))
        return rc
    from .app import main as amain
    return amain([str(p) for p in clips] + ["--out", a.out or str(Path.home() / "animalworld_demo_results")] + extra)


if __name__ == "__main__":
    raise SystemExit(main())
