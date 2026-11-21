"""`animalworld zeroshot VIDEO_OR_FOLDER ... --out DIR`: the engine of Animal World Run without a window."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .. import weights as W
from . import pipeline as P


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="animalworld zeroshot",
                                 description="Detect, track and pose mice in videos with the pretrained MouseWorld models. No training, no project.")
    ap.add_argument("videos", nargs="+", help="video files and / or folders of videos")
    ap.add_argument("--out", required=True, help="output folder (one sub-folder per video)")
    ap.add_argument("--n-animals", type=int, help="number of animals, if you already counted them (same for every video)")
    ap.add_argument("--yes", action="store_true", help="accept the automatic animal count without asking")
    ap.add_argument("--engine", default="auto", choices=("auto", "full", "lite", "mock"),
                    help="auto: full on a suitable GPU, else lite (CPU, weaker tracking). mock: test engine without models")
    ap.add_argument("--max-seconds", type=float, help="analyse only the first N seconds of each video")
    ap.add_argument("--window-seconds", type=float, default=60.0, help="length of one tracking window (default 60)")
    ap.add_argument("--tracker-detector", choices=P.TRACKER_DETECTORS,
                    help="detector behind MouseWorld-Track's witnesses. scored_pair (default): the two checkpoints the published tracking "
                         "numbers were measured with, 20 identity fixes (73 per hour; 21 with --sam3) on the 34 public benchmark clips. strongest: "
                         "MouseWorld-DET alone, about 25 %% less computing, 42 fixes. strongest_two_seeds: MouseWorld-DET plus a second "
                         "seed, 39 fixes")
    ap.add_argument("--sam3", action="store_true",
                    help="add the SAM 3.1 text-prompt witness to the tracker. OFF by default: the SAM 3.1 weights are gated by Meta's licence "
                         "(accept it at huggingface.co/facebook/sam3.1, then fetch sam3.1_multiplex.pt) and 3.5 GB. On the 34 public benchmark "
                         "clips the default needed 20 identity fixes without it and 21 with it")
    ap.add_argument("--analysis-fps", type=float, default=10.0,
                    help="rate at which the identity tracker itself runs (default 10; the rules were written and scored at about 10 per "
                         "second, other values are untested territory)")
    ap.add_argument("--analysis-frames-only", action="store_true",
                    help="write rows only for the frames the tracker analyses (about 10 per second) instead of every camera frame")
    ap.add_argument("--zoom", default="auto", metavar="auto|off|REGIONFILE",
                    help="auto (default): look at the whole clip first; when the animals are small in the frame (below about 64 px at the "
                         "detector's 504 px input) or the frame is two cameras side by side, analyse each region where animals are active zoomed "
                         "in and map the results back to full-frame pixels. off: always the whole frame. REGIONFILE: a JSON list "
                         '[{"box": [x0, y0, x1, y1], "animals": n}, ...] in full-frame pixels')
    ap.add_argument("--fill", default="sam2", choices=("sam2", "detector"),
                    help="where the box and outline of every output frame come from. sam2 (default): one SAM 2.1 propagation at the output "
                         "frame rate, anchored on the resolved identities: one source for every frame, stable from frame to frame. "
                         "detector: MouseWorld-DET's own box on every frame: faster, and the boxes flicker between frames")
    ap.add_argument("--keep-work", action="store_true",
                    help="keep the temporary frames and tracker intermediates (they are always kept when a video fails)")
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    try:
        import cv2  # noqa: F401
    except ImportError:
        print('animalworld zeroshot: OpenCV is missing. Install the run extra:  pip install "animalworld[run]"', file=sys.stderr)
        return 2
    try:
        vids = P.collect(a.videos)
    except FileNotFoundError as e:
        print(f"animalworld zeroshot: {e}", file=sys.stderr)
        return 2
    if not vids:
        print("animalworld zeroshot: no video file found (" + ", ".join(P.VIDEO_EXT) + ")", file=sys.stderr)
        return 2
    if a.n_animals is not None and not 1 <= a.n_animals <= 100:
        print("animalworld zeroshot: --n-animals must be between 1 and 100", file=sys.stderr)
        return 2
    print("checking the computer (GPU, installed model libraries, weight files) ...", flush=True)
    try:
        eng = W.choose_engine(a.engine, a.tracker_detector, a.sam3)
    except W.WeightsError as e:
        print(f"animalworld zeroshot: {e}", file=sys.stderr)
        return 2
    print(f"engine: {eng['engine']} on {eng['device']}", flush=True)
    if eng["engine"] == "lite":
        print("NOTE: no usable GPU setup was found, so the CPU engine runs: MouseWorld-DET-Lite with a simple tracker.\n"
              "      Tracking quality is LOWER than MouseWorld-Track (identities are not reliable when animals touch). Reasons:", flush=True)
        for w in eng["why"]:
            print("      - " + w, flush=True)
    names, failed = set(), 0
    for v in vids:
        name = P.slug(v.stem)
        while name in names:
            name += "_2"
        names.add(name)
        job = P.Job(v, a.out, P.Options(engine=eng["engine"], n_animals=a.n_animals, max_seconds=a.max_seconds, window_s=a.window_seconds,
                                        keep_work=a.keep_work, tracker_detector=a.tracker_detector, sam3=a.sam3 or None,
                                        all_frames=not a.analysis_frames_only, analysis_fps=a.analysis_fps, zoom=a.zoom, fill=a.fill), name=name)
        P.guarded(P.prepare_and_count, job, eng)
        if job.status == "confirm":
            if a.yes:
                job.n = job.n_auto
                job.log(f"automatic count {job.n_auto} accepted without a look (--yes)")
            elif sys.stdin.isatty():
                ans = input(f"{v.name}: {job.n_auto} animal(s) found (look at {job.count_sheet}). Count only the animals that move freely "
                            "(not an animal under a cup or behind a grid). Enter to accept, or type the right number: ").strip()
                job.n = int(ans) if ans.isdigit() and int(ans) > 0 else job.n_auto
                job.confirmed_by_user = True
            else:
                print(f"{v.name}: automatic count {job.n_auto}. Not started: pass --n-animals N or --yes.", file=sys.stderr)
                failed += 1
                continue
            job.status = "queued"
        if job.status == "queued":
            P.guarded(P.analyse, job)
        if job.status == "done":
            print(f"{v.name}: {job.message}\n  results: {job.dir}", flush=True)
            if (job.plan or {}).get("text"):
                print("  " + job.plan["text"], flush=True)
            for w_ in job.warnings[:12]:
                print("  WARNING: " + w_, flush=True)
        else:
            failed += 1
            print(f"{v.name}: {job.status}: {job.message}\n  log: {job.dir / 'run.log'}", file=sys.stderr, flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
