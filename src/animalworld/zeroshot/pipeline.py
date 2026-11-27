"""Zero-shot analysis of mouse videos with the pretrained MouseWorld models (the engine behind `animalworld run`,
`animalworld zeroshot` and the last row of Animal World Studio).

Per video:  frames -> animal count (shown for confirmation) -> detection, identity tracking, keypoints -> files.

Engines
  full   MouseWorld-DET (RF-DETR-Seg-L) + MouseWorld-Track (the frozen v7j rule set with its witnesses: SAM 2.1-L
         forward / backward runs from two detector seeds, local two-ended re-tracks through every contact, the
         detector at camera rate through long contacts, and a SAM 3.1 text-prompt run) + MouseWorld-Pose. Needs a
         data-centre class NVIDIA GPU. Each model step is its own process (see weights.python_for).
  lite   MouseWorld-DET-Lite (ONNX, CPU) + the simple distance / overlap tracker + the ONNX crop keypoint model.
         The fallback for computers without a usable GPU. Identity tracking is clearly worse (no SAM witnesses, no
         contact rules): expect swaps whenever animals touch.
  mock   blob detector on a plain background + the simple tracker. For tests only.

Analysis rate: about 10 frames per second (every `step`-th camera frame; the tracker rules assume 10 fps). Long
videos are cut into windows (default 60 s, 2 s overlap); identities are carried across a seam by matching positions
in the overlap. The seam matching is new in this package and has NOT been scored against reference tracks.

Outputs in <out>/<video name>/:
  tracks.csv        one row per animal per analysed frame: video_frame, t_s, id, x, y, box, det_score, status, keypoints
  masks.npz         outlines as packed bit masks inside each box (see load_masks)
  overlay.mp4       the analysed frames with outlines, boxes, identities, keypoints
  contact_sheet.jpg 12 evenly spaced overlay frames
  summary.html      one page: what ran, counts, warnings, the contact sheet   (+ summary.json, run.log)
"""
from __future__ import annotations

import csv
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from .. import __version__, weights as W
from ..backends.trackers.iou import hungarian
from ..utils import VIDEO_EXT, slug, write_json

KP = ["nose", "left_ear", "right_ear", "neck", "left_hip", "right_hip", "tail_base"]
SKEL = [[0, 3], [1, 3], [2, 3], [3, 4], [3, 5], [4, 6], [5, 6]]
COLOURS = [(53, 75, 230), (136, 84, 60), (60, 160, 0), (220, 190, 40), (0, 150, 255), (170, 60, 170), (90, 220, 230), (150, 150, 150),
           (30, 30, 150), (200, 120, 220)]            # BGR: red, navy, green, cyan, orange, purple, yellow, grey, dark red, pink
SCRIPTS = Path(__file__).resolve().parent / "mwtrack"
BIG = 1e9
# Which detector feeds MouseWorld-Track's witnesses. Measured on the 34 public benchmark clips (20 Nov 2026, animal count
# given, packaged engine): scored_pair 21 identity fixes (76 per hour), strongest 42 (153), strongest_two_seeds 39 (142).
#   scored_pair  DEFAULT (user decision, 20 Nov 2026). The two older checkpoints the tracking numbers of the manuscript were
#                measured with ("epoch5" seeds F / B and the local steps, "mid_1408" seeds F2 / B2). Output boxes, outlines
#                and the animal count still come from MouseWorld-DET.
#   strongest    MouseWorld-DET everywhere: one detector for counts, boxes, outlines AND the tracker; no second seed
#                (witnesses F, B, local, S, D). About 25 % less computing, twice the identity fixes on the benchmark.
#   strongest_two_seeds  MouseWorld-DET seeds F / B and the local steps and gives the output boxes; "mid_1408" is kept only
#                as the second, independent seed (F2 / B2).
#   strongest_recut  EXPERIMENTAL (20 Nov 2026): as strongest, but a raw MouseWorld-DET score of RECUT (default 0.35, or
#                $ANIMALWORLD_SEED_CUT) counts as the rules' "confident" 0.40 (mwtrack/rescale_scores.py; the frozen
#                scripts are untouched). The value was chosen on the 34 benchmark clips, i.e. it is fitted to them.
TRACKER_DETECTORS = ("scored_pair", "strongest", "strongest_two_seeds", "strongest_recut")
RECUT = 0.35
DEFAULT_TRACKER_DETECTOR = "scored_pair"


class Stopped(Exception):
    pass


class NothingFound(RuntimeError):
    """A detector pass returned no detection at all (not an error of the program: a result)."""


class StepError(RuntimeError):
    pass


class Options:
    def __init__(self, engine: str = "auto", n_animals: Optional[int] = None, window_s: float = 60.0, overlap_s: float = 2.0,
                 analysis_fps: float = 10.0, max_seconds: Optional[float] = None, keep_work: bool = False, count_frames: int = 40,
                 supervisor: str = "v7j", prompt: str = "mouse", tracker_detector: str = None, sam3: bool = None,
                 all_frames: bool = True, fill: str = "sam2", zoom: str = "auto"):
        self.engine, self.n_animals, self.window_s, self.overlap_s = engine, n_animals, float(window_s), float(overlap_s)
        self.analysis_fps, self.max_seconds, self.keep_work = float(analysis_fps), max_seconds, bool(keep_work)
        self.count_frames, self.supervisor, self.prompt = int(count_frames), supervisor, prompt
        td = tracker_detector or os.environ.get("ANIMALWORLD_TRACKER_DETECTOR") or DEFAULT_TRACKER_DETECTOR
        if td not in TRACKER_DETECTORS:
            raise ValueError(f"tracker_detector must be one of {', '.join(TRACKER_DETECTORS)}")
        self.tracker_detector = td
        # SAM 3.1 (gated weights, 3.5 GB) is an OPTION: the default tracker runs without its text-prompt witness.
        self.sam3 = bool(sam3) if sam3 is not None else os.environ.get("ANIMALWORLD_SAM3", "").lower() in ("1", "true", "yes", "on")
        # Rows for EVERY camera frame: the tracker runs at its scored rate (about 10 per second); the frames in between are
        # filled from per-frame MouseWorld-DET detections attached to the tracked identities (see fill_rows).
        self.all_frames = bool(all_frames)
        # Where the box / outline of every output row comes from. "sam2" (default): one final SAM 2.1 propagation at the output
        # frame rate, anchored on the resolved identities, so the geometry has ONE source on every frame. "detector": the
        # earlier behaviour (MouseWorld-DET's own box on every frame), faster, and it flickers between frames.
        if fill not in ("sam2", "detector"):
            raise ValueError("fill must be 'sam2' or 'detector'")
        self.fill = fill
        # "auto": look at the whole clip first and analyse zoomed regions when the animals are small in the frame (zoom.py);
        # "off": always the whole frame; otherwise the path of a region file.
        self.zoom = zoom or "auto"


class Job:
    """One video. `status`: added, preparing, counting, confirm, queued, tracking, done, failed, stopped."""

    def __init__(self, video, out_root, options: Options = None, name: str = None):
        self.video = Path(video).expanduser().resolve()
        self.opt = options or Options()
        self.name = name or slug(self.video.stem)
        self.dir = Path(out_root).expanduser().resolve() / self.name
        self.work = self.dir / "_work"
        self.status, self.progress, self.message = "added", 0.0, ""
        self.meta: dict = {}
        self.n_auto: Optional[int] = None
        self.n: Optional[int] = self.opt.n_animals
        self.count_sheet: Optional[Path] = None
        self.results: Dict[str, str] = {}
        self.warnings: List[str] = []
        self.engine: dict = {}
        self.stop = threading.Event()
        self.timings: Dict[str, float] = {}             # seconds per stage, written to summary.json
        self.plan: dict = {}                            # whole frame or zoomed regions (zoom.plan)
        self.count_det: dict = {}
        self.on_change: Callable[["Job"], None] = lambda job: None
        self._log_lock = threading.Lock()

    def log(self, text: str, file_only: bool = False) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        line = text.rstrip("\n")
        with self._log_lock:
            with open(self.dir / "run.log", "a", encoding="utf-8") as f:
                f.write(line + "\n")
        if not file_only:
            self.on_line(line)

    def on_line(self, line: str) -> None:           # replaced by the GUI / CLI
        print(f"[{self.name}] {line}", flush=True)

    def set(self, status: str = None, progress: float = None, message: str = None) -> None:
        if status:
            self.status = status
        if progress is not None:
            self.progress = float(progress)
        if message is not None:
            self.message = message
        self.on_change(self)

    def check(self) -> None:
        if self.stop.is_set():
            raise Stopped()


def collect(paths) -> List[Path]:
    """Video files from a mixed list of files and folders (folders are not searched recursively)."""
    out: List[Path] = []
    for p in paths:
        p = Path(str(p)).expanduser()
        if p.is_dir():
            out += sorted(q for q in p.iterdir() if q.is_file() and q.suffix.lower() in VIDEO_EXT)
        elif p.is_file() and p.suffix.lower() in VIDEO_EXT:
            out.append(p)
        elif not p.exists():
            raise FileNotFoundError(f"{p} does not exist")
    seen, uniq = set(), []
    for p in out:
        r = p.resolve()
        if r not in seen:
            seen.add(r)
            uniq.append(r)
    return uniq


def usable_cpus() -> int:
    """CPU threads this process may really use. On a cluster or in a container os.cpu_count() reports the whole machine (e.g. 128)
    while the job owns a few cores; starting that many threads made the CPU engine about 15 times slower (seen 20 Nov 2026)."""
    try:
        n = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        n = os.cpu_count() or 2
    for var in ("SLURM_CPUS_PER_TASK", "OMP_NUM_THREADS"):
        v = os.environ.get(var, "")
        if v.isdigit() and int(v) > 0:
            n = min(n, int(v))
    return max(1, n)


def _cv2():
    try:
        import cv2
        return cv2
    except ImportError as e:
        raise StepError('reading videos needs OpenCV: pip install "animalworld[run]"') from e


def _link(src: Path, dst: Path) -> None:
    try:
        os.symlink(src, dst)
    except OSError:                                # e.g. Windows without the symlink privilege
        shutil.copy(src, dst)


# ------------------------------------------------------------------ frames ----------------------------------------
def extract(job: Job) -> dict:
    """Decode the video once, in order (no seeking: container seeks drop or repeat frames on some cameras), and write
    every camera frame as a JPEG. Analysis frame j is camera frame j * step."""
    cv2 = _cv2()
    meta_f = job.work / "meta.json"
    if meta_f.exists():
        job.meta = json.loads(meta_f.read_text())
        return job.meta
    nat = job.work / "native"
    nat.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(job.video))
    if not cap.isOpened():
        raise StepError(f"cannot open {job.video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    if not (1.0 <= fps <= 1000.0) or math.isnan(fps):
        job.warnings.append(f"the file reports an implausible frame rate ({fps}); 30 frames/s assumed, so times in seconds may be wrong")
        fps = 30.0
    header = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    limit = int(round(job.opt.max_seconds * fps)) if job.opt.max_seconds else 0
    target = min(header, limit) if (limit and header) else (limit or header)
    Wd = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); Hd = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    need_gb = (target or 0) * Wd * Hd * 0.25 / 1e9
    free_gb = shutil.disk_usage(job.work).free / 1e9
    job.log(f"video: {Wd}x{Hd}, {fps:.3f} frames/s, header says {header} frames; decoding {'the first %.0f s' % job.opt.max_seconds if limit else 'all of it'}"
            f" (temporary frames: roughly {need_gb:.1f} GB, free {free_gb:.0f} GB)")
    if need_gb > 0.9 * free_gb:
        raise StepError(f"not enough free disk space in {job.work} for the temporary frames (about {need_gb:.0f} GB needed, {free_gb:.0f} GB free)")
    n = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        cv2.imwrite(str(nat / f"{n:06d}.jpg"), fr, [cv2.IMWRITE_JPEG_QUALITY, 93])
        n += 1
        if limit and n >= limit:
            break
        if n % 50 == 0:
            job.check()
            job.set(progress=0.10 * (n / target if target else 0.5), message=f"reading frames ({n})")
    cap.release()
    if n == 0:
        raise StepError(f"no frame could be decoded from {job.video} (not a video file, or a codec this OpenCV build cannot read)")
    if n < 5:
        raise StepError(f"{job.video.name} has only {n} frame(s): too short to track")
    if header and not limit and abs(n - header) > max(2, 0.01 * header):
        job.warnings.append(f"the file header announces {header} frames but {n} could be decoded; only the decoded part is analysed")
    step = max(1, int(round(fps / job.opt.analysis_fps)))
    fr_dir = job.work / "frames"
    fr_dir.mkdir(exist_ok=True)
    idx = list(range(0, n, step))
    for j, i in enumerate(idx):
        if not (fr_dir / f"{j:06d}.jpg").exists():
            _link(nat / f"{i:06d}.jpg", fr_dir / f"{j:06d}.jpg")
    job.meta = dict(video=str(job.video), fps=fps, step=step, W=Wd, H=Hd, native=n, frames=len(idx), analysis_fps=fps / step,
                    seconds=n / fps, header_frames=header)
    write_json(meta_f, job.meta)
    return job.meta


def frame_path(job: Job, j: int) -> Path:
    return job.work / "frames" / f"{j:06d}.jpg"


# ------------------------------------------------------------------ running a model step ---------------------------
def run_step(job: Job, label: str, role: str, script: str, args: List[str] = (), env: Dict[str, str] = None) -> None:
    """One model step as its own process (argument list, never a shell string); output goes to the job log line by line."""
    job.check()
    tmp = job.work / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    argv = [W.python_for(role), str(SCRIPTS / script), *[str(a) for a in args]]
    e = W.step_env(dict(env or {}, TMPDIR=str(tmp), OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "2")))
    job.log(f"--- {label}: {Path(argv[0]).name} {script}")
    t0 = time.time()
    kw = {"start_new_session": True} if os.name != "nt" else {}
    pr = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=e, text=True,
                          errors="replace", cwd=str(job.work), **kw)
    tail: List[str] = []

    def pump():
        for line in pr.stdout:
            tail.append(line.rstrip())
            del tail[:-30]
            job.log("    " + line.rstrip())
    th = threading.Thread(target=pump, daemon=True)
    th.start()
    while pr.poll() is None:
        if job.stop.is_set():
            try:
                os.killpg(pr.pid, signal.SIGTERM) if os.name != "nt" else pr.terminate()
            except (OSError, ProcessLookupError):
                pass
            pr.wait()
            raise Stopped()
        time.sleep(0.3)
    th.join(timeout=5)
    job.log(f"--- {label}: exit {pr.returncode} after {time.time() - t0:.0f} s")
    job.timings[label] = round(job.timings.get(label, 0.0) + time.time() - t0, 1)
    if pr.returncode != 0 and script in ("detect_dense.py", "detect_local.py") and any("need at least one array" in x for x in tail):
        raise NothingFound(f"{label}: the detector found no animal on any frame")
    if pr.returncode != 0:
        raise StepError(f"{label} failed (exit {pr.returncode}): " + " | ".join(tail[-4:]))


# ------------------------------------------------------------------ counting ---------------------------------------
def count_rule(D: dict) -> tuple:
    """The agent's count (core.py, v6): N = max(largest count seen WELL SEPARATED on >= 10 % of frames, largest count the
    detector reaches on >= 25 % of frames). -> (N, body length, per-frame confident counts)."""
    from .mwtrack import core_base as CB
    T = int(D["T"])
    conf = D["score"] >= CB.SEED_SCORE
    c = np.array([int(((D["t"] == t) & conf).sum()) for t in range(T)])
    if not conf.any():
        return 0, 0.0, c
    BL = 2.0 * float(np.sqrt(np.median(D["area"][conf])))
    try:
        n5 = int(CB.scene(D)[0])
    except ValueError:
        n5 = 0
    sust = max([n for n in range(1, 16) if (c >= n).mean() >= 0.25] + [0])
    return max(n5, sust), BL, c


def _sample(n_frames: int, k: int) -> List[int]:
    k = min(k, n_frames)
    return sorted({int(round(x)) for x in np.linspace(0, n_frames - 1, k)})


def _make_root(job: Job, root: Path, clips: List[dict], frames_of: Dict[str, List[int]], native: bool) -> None:
    """A work root in the layout the agent scripts expect: clips.json + data/<dataset>/<clip>/{frames,native}/%06d.jpg."""
    m = job.meta
    for c in clips:
        d = root / "data" / c["dataset"] / c["clip"]
        (d / "frames").mkdir(parents=True, exist_ok=True)
        idx = frames_of[c["clip"]]
        for j, a in enumerate(idx):
            if not (d / "frames" / f"{j:06d}.jpg").exists():
                _link(frame_path(job, a).resolve(), d / "frames" / f"{j:06d}.jpg")
        if native and m["step"] > 1:
            (d / "native").mkdir(exist_ok=True)
            n0 = idx[0] * m["step"]
            for i in range(m["step"] * (len(idx) - 1) + 1):
                if not (d / "native" / f"{i:06d}.jpg").exists():
                    _link(job.work / "native" / f"{n0 + i:06d}.jpg", d / "native" / f"{i:06d}.jpg")
    write_json(root / "clips.json", {"rule": "written by animalworld.zeroshot", "clips": clips})


def _clip(job: Job, name: str, n_frames: int, animals: int) -> dict:
    m = job.meta
    return dict(dataset="video", clip=name, frames=n_frames, animals=int(animals), fps=round(m["analysis_fps"], 3), step=m["step"],
                W=m["W"], H=m["H"])


def _predictor(job: Job):
    """In-process detector of the lite / mock engines."""
    from .. import registry
    if job.engine["engine"] == "mock":
        return registry.create("predictors", "blob", {"category": "mouse", "conf": 0.25})
    return registry.create("predictors", "onnx", {"weights": str(W.resolve("mouseworld-det-lite")), "conf": 0.25, "category": "mouse",
                                                  "task": "segment", "threads": usable_cpus()})


def _poly_area(p) -> float:
    from ..formats.labels import poly_area
    return poly_area(p) if p else 0.0


def _simple_detect(job: Job, paths: List[Path], pred, base: float, span: float) -> List[List[dict]]:
    from PIL import Image
    out = []
    for k, a in enumerate(paths):
        if k % 10 == 0:
            job.check()
            job.set(progress=base + span * k / max(1, len(paths)), message=f"detecting ({k}/{len(paths)})")
        with Image.open(a) as im:
            im = im.convert("RGB")
            inst = pred.predict(im)
            for i in inst:
                x0, y0, x1, y1 = [int(round(v)) for v in i["box"]]
                w, h = max(1, x1 - x0), max(1, y1 - y0)
                crop = im.crop((x0 + w // 4, y0 + h // 4, max(x0 + w // 4 + 1, x1 - w // 4), max(y0 + h // 4 + 1, y1 - h // 4))).convert("L")
                i["appearance"] = float(np.asarray(crop).mean()) / 255.0
        out.append(inst)
    return out


def count(job: Job) -> int:
    """Automatic animal count on evenly spaced frames + a sheet to look at. The number is a PROPOSAL: the user confirms it."""
    cv2 = _cv2()
    m = job.meta
    idx = _sample(m["frames"], job.opt.count_frames)
    job.set("counting", 0.10, "counting animals")
    if job.engine["engine"] == "full":
        root = job.work / "count"
        _make_root(job, root, [_clip(job, "count", len(idx), 0)], {"count": idx}, native=False)
        from .mwtrack import core_base as CB
        try:
            run_step(job, "MouseWorld-DET on sample frames", "det", "detect_dense.py",
                     env=dict(MWT_ROOT=str(root), MWT_AGENT=str(root / "det"), MWT_DET_CKPT=str(W.resolve("mouseworld-det"))))
            D = CB.load_det(root / "det" / "work" / f"video__count_000000_{len(idx) - 1:06d}" / "det.npz")
        except NothingFound:
            job.log("the detector found nothing on any sample frame")
            D = dict(t=np.zeros(0, int), score=np.zeros(0), area=np.zeros(0), c=np.zeros((0, 2)), box=np.zeros((0, 4)), T=len(idx))
        boxes = [[(D["box"][i], float(D["score"][i])) for i in np.flatnonzero(D["t"] == t)] for t in range(len(idx))]
    else:
        dets = _simple_detect(job, [frame_path(job, a) for a in idx], _predictor(job), 0.10, 0.08)
        t, sc, ar, cc = [], [], [], []
        for k, f in enumerate(dets):
            for i in f:
                b = i["box"]
                t.append(k); sc.append(i["score"]); cc.append([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
                ar.append(_poly_area(i.get("poly")) or 0.6 * (b[2] - b[0]) * (b[3] - b[1]))
        D = dict(t=np.array(t, int), score=np.array(sc, float), area=np.array(ar, float), c=np.array(cc, float).reshape(-1, 2), T=len(idx))
        boxes = [[(i["box"], i["score"]) for i in f] for f in dets]
    n, bl, c = count_rule(D)
    job.n_auto = int(n)
    # what the whole-frame pass saw on the sample frames, for the scout (zoom.py)
    job.count_det = dict(idx=idx, dets=[dict(t=k, box=[float(v) for v in b], score=float(s_), c=((float(b[0]) + float(b[2])) / 2, (float(b[1]) + float(b[3])) / 2),
                                             area=0.6 * (float(b[2]) - float(b[0])) * (float(b[3]) - float(b[1]))) for k in range(len(idx)) for b, s_ in boxes[k]])
    # the sheet: six sample frames with every detection drawn and the per-frame count
    pick = _sample(len(idx), 6)
    tiles = []
    for k in pick:
        im = cv2.imread(str(frame_path(job, idx[k])))
        for b, s in boxes[k]:
            x0, y0, x1, y1 = [int(round(float(v))) for v in b]
            col = (60, 180, 60) if s >= 0.4 else (0, 165, 255)
            cv2.rectangle(im, (x0, y0), (x1, y1), col, max(2, im.shape[1] // 400))
        sc_ = 520.0 / im.shape[1]
        im = cv2.resize(im, (520, max(1, int(round(im.shape[0] * sc_)))), interpolation=cv2.INTER_AREA)
        nconf = sum(1 for _, s in boxes[k] if s >= 0.4)
        cv2.rectangle(im, (0, 0), (520, 26), (255, 255, 255), -1)
        cv2.putText(im, f"t = {idx[k] * m['step'] / m['fps']:.1f} s    {nconf} found", (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(im)
    while len(tiles) % 3:
        tiles.append(np.full_like(tiles[0], 255))
    sheet = np.vstack([np.hstack(tiles[r:r + 3]) for r in range(0, len(tiles), 3)])
    job.count_sheet = job.dir / "count_sheet.jpg"
    cv2.imwrite(str(job.count_sheet), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    hist = {int(v): int((c == v).sum()) for v in np.unique(c)}
    job.log(f"animal count proposal: {n} (confident detections per sample frame: {hist}); sheet: {job.count_sheet}")
    write_json(job.work / "count.json", {"n_auto": int(n), "per_frame": c.tolist(), "sample_frames": idx, "body_length_px": bl})
    if n == 0:
        job.warnings.append("no animal was detected on the sample frames")
    return int(n)


# ------------------------------------------------------------------ full engine ------------------------------------
def windows(n_frames: int, afps: float, window_s: float, overlap_s: float) -> List[tuple]:
    """[(first analysis frame, number of frames), ...] covering 0 .. n_frames - 1 with `overlap` shared frames."""
    Wn = max(20, int(round(window_s * afps))); O = max(2, min(Wn // 4, int(round(overlap_s * afps))))
    if n_frames <= Wn + Wn // 4:
        return [(0, n_frames)]
    out, a = [], 0
    while True:
        if n_frames - a <= Wn + Wn // 4:
            out.append((a, n_frames - a))
            return out
        out.append((a, Wn))
        a += Wn - O


def _pts(path: Path, T: int, N: int) -> np.ndarray:
    q = np.load(path)
    X = np.full((T, N, 2), np.nan)
    for f, k, x, y in zip(q["pt_frame"], q["pt_track"], q["pt_x"], q["pt_y"]):
        if 0 <= int(f) < T and 0 <= int(k) < N:
            X[int(f), int(k)] = (x, y)
    return X


def stitch(prev: np.ndarray, new: np.ndarray, bl: float) -> tuple:
    """Identity link across a seam. prev, new: (O, N, 2) positions of the same overlap frames in the two windows.
    -> (perm: new index -> global index, worst matched distance in body lengths, margin to the next-best assignment)."""
    N = prev.shape[1]
    cost = [[BIG] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            ok = np.isfinite(prev[:, i, 0]) & np.isfinite(new[:, j, 0])
            if ok.any():
                cost[i][j] = float(np.hypot(*(prev[ok, i] - new[ok, j]).T).mean()) / max(bl, 1e-6)
    m = hungarian(cost)                                # global i -> new j
    perm = {j: i for i, j in enumerate(m)}
    worst = max(cost[i][m[i]] for i in range(N))
    best = sum(cost[i][m[i]] for i in range(N))
    margin = BIG
    if N > 1:
        for a in range(N):
            for b in range(a + 1, N):                 # every single swap
                alt = best - cost[a][m[a]] - cost[b][m[b]] + cost[a][m[b]] + cost[b][m[a]]
                margin = min(margin, alt - best)
    return [perm[j] for j in range(N)], worst, margin


def tracker_steps(root: Path, opt: Options) -> tuple:
    """-> (agent folder A, second-seed folder B or None, folder holding MouseWorld-DET's detections, [steps])."""
    pair = opt.tracker_detector == "scored_pair"
    two = pair or opt.tracker_detector == "strongest_two_seeds"
    A, B = root / "agent_a", (root / "agent_b" if two else None)
    Dm = root / "det_main" if pair else A
    base = dict(MWT_ROOT=str(root), MWT_SAM2_CKPT=str(W.resolve("sam2.1-hiera-large")), MWT_PROMPT=opt.prompt)
    dmain = str(W.resolve("mouseworld-det"))
    da = str(W.resolve("mouseworld-track-det-a")) if pair else dmain
    ver = opt.supervisor
    steps = []
    recut = opt.tracker_detector == "strongest_recut"
    rc = dict(MWT_AGENT=str(A), MWT_SEED_CUT=os.environ.get("ANIMALWORLD_SEED_CUT") or str(RECUT))
    if pair:
        steps.append(("MouseWorld-DET on every analysed frame", "det", "detect_dense.py", [], dict(MWT_AGENT=str(Dm), MWT_DET_CKPT=dmain), 0.12))
    steps += [
        ("tracker detector A on every analysed frame" if pair else "MouseWorld-DET on every analysed frame", "det", "detect_dense.py", [],
         dict(MWT_AGENT=str(A), MWT_DET_CKPT=da), 0.10),
    ] + ([("moving the confidence cut (experimental)", "cpu", "rescale_scores.py", ["dense"], rc, 0.005)] if recut else []) + [
        ("SAM 2.1 forward / backward witnesses" + (" (seed A)" if two else ""), "sam2", "track_fb.py", [], dict(MWT_AGENT=str(A)), 0.14),
        ("SAM 2.1 local two-ended re-tracks through contacts", "sam2", "track_local.py", [], dict(MWT_AGENT=str(A)), 0.10),
        ("detector at camera rate through long contacts", "det", "detect_local.py", [], dict(MWT_AGENT=str(A), MWT_DET_CKPT=da), 0.06),
    ] + ([("moving the confidence cut, camera-rate detections (experimental)", "cpu", "rescale_scores.py", ["local"], rc, 0.005)] if recut else [])
    if two:
        steps += [
            ("tracker detector B on every analysed frame", "det", "detect_dense.py", [],
             dict(MWT_AGENT=str(B), MWT_DET_CKPT=str(W.resolve("mouseworld-track-det-b"))), 0.10),
            ("SAM 2.1 forward / backward witnesses (seed B)", "sam2", "track_fb.py", [], dict(MWT_AGENT=str(B)), 0.12),
        ]
    if opt.sam3:
        steps.append(("SAM 3.1 text-prompt witness (optional)", "sam3", "sam31_witness.py", [], dict(MWT_SAM31_CKPT=str(W.resolve("sam3.1-multiplex"))), 0.12))
    steps += [
        (f"MouseWorld-Track rules ({ver})", "cpu", "run_supervisor.py", [ver],
         dict(MWT_AGENT=str(A), **({"MWT_SEED2": str(B / "work")} if two else {})), 0.04),
    ]
    tot = sum(x[5] for x in steps)
    return A, B, Dm, [(a, b, c, d, dict(base, **e), w / tot) for a, b, c, d, e, w in steps]


def track_full(job: Job, N: int) -> dict:
    m = job.meta
    root = job.work / "track"
    wins = windows(m["frames"], m["analysis_fps"], job.opt.window_s, job.opt.overlap_s)
    clips = [_clip(job, f"w{k:04d}", T, N) for k, (a0, T) in enumerate(wins)]
    _make_root(job, root, clips, {f"w{k:04d}": list(range(a0, a0 + T)) for k, (a0, T) in enumerate(wins)}, native=True)
    A, B, Dm, steps = tracker_steps(root, job.opt)
    ver = job.opt.supervisor
    step = int(m["step"]); fill = job.opt.all_frames and step > 1
    p = 0.20
    if fill:      # MouseWorld-DET on EVERY camera frame: boxes / outlines for all rows, and what the in-between frames are filled from
        froot = job.work / "fill"; fclips = []
        for k, (a0, T) in enumerate(wins):
            n = step * (T - 1) + 1; d = froot / "data" / "video" / f"w{k:04d}" / "frames"; d.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                if not (d / f"{i:06d}.jpg").exists():
                    _link(job.work / "native" / f"{a0 * step + i:06d}.jpg", d / f"{i:06d}.jpg")
            fclips.append(dict(_clip(job, f"w{k:04d}", n, N), step=1))
        write_json(froot / "clips.json", {"rule": "camera-rate frames, written by animalworld.zeroshot", "clips": fclips})
        job.set("tracking", p, "MouseWorld-DET on every camera frame")
        try:
            run_step(job, "MouseWorld-DET on every camera frame", "det", "detect_dense.py",
                     env=dict(MWT_ROOT=str(froot), MWT_AGENT=str(froot / "det"), MWT_DET_CKPT=str(W.resolve("mouseworld-det"))))
        except NothingFound:
            raise StepError(NO_ANIMAL)
        p += 0.08
    try:
        checked = False
        for label, role, script, args, env, w in steps:
            if fill and Dm != A and env.get("MWT_AGENT") == str(Dm):
                continue                      # the camera-rate pass above already holds MouseWorld-DET's detections of the analysed frames
            if role != "det" and not checked:
                checked = True                # the tracker's own detector found nothing at all: the SAM / rule steps have nothing to start from
                if confident_detections(A / "work") == 0:
                    raise NothingFound("the tracker's detector found no animal on any analysed frame")
            job.set("tracking", p, label)
            run_step(job, label, role, script, args, env)
            p += 0.46 * w
    except NothingFound as e:
        job.log(str(e) + "; the remaining tracker steps are skipped")
    # ---- assemble: positions, identities across seams, flags
    sup = {}
    f = root / f"supervisor_summary_{ver}.json"
    if f.exists():
        sup = json.loads(f.read_text()).get(ver, {})
    TA = m["frames"]
    X = np.full((TA, N, 2), np.nan); flagged = np.zeros(TA, bool); bls = []; seams = []; prev_end = -1; have = False; win_info = []; fallback_used = []
    from .mwtrack import core_base as CB
    for k, (a0, T) in enumerate(wins):
        name = f"video__w{k:04d}"; key = f"000000_{T - 1:06d}"
        s = sup.get(name, {})
        pf = root / "runs_clips" / f"agent_{ver}" / name / f"{key}.npz"
        tf = A / "work" / f"{name}_{key}" / "tracks.npz"
        info = dict(window=k, start_s=round(a0 / m["analysis_fps"], 1), seconds=round(T / m["analysis_fps"], 1), status=s.get("status", "NO OUTPUT"),
                    contacts=s.get("contact_spans"), flagged_contacts=len(s.get("flagged_contacts") or []), witnesses=s.get("witnesses"))
        win_info.append(info)
        if not pf.exists():
            why = "no frame with all animals apart was found (the tracker cannot start)" if not tf.exists() else s.get("error", "the rule step gave no output")
            Xw, bl = _fallback_positions(job, k, T, N, Dm, fill, step)
            if Xw is None:
                job.warnings.append(f"{info['start_s']:.0f}-{info['start_s'] + info['seconds']:.0f} s: NOT TRACKED: {why}; and no detection to fall back on")
                info["status"] = "NOT TRACKED"
                have = False
                continue
            job.warnings.append(f"{info['start_s']:.0f}-{info['start_s'] + info['seconds']:.0f} s: MouseWorld-Track could not start ({why}). "
                                "FALLBACK used for this part: detector + simple position tracker; identities are NOT reliable and every row is flagged")
            info["status"] = "FALLBACK (simple tracker)"; fallback_used.append(k)
            flagged[a0: a0 + T] = True
        else:
            Xw = _pts(pf, T, N); bl = float(np.load(tf)["BL"])
        bls.append(bl)
        perm = list(range(N))
        if have and prev_end >= a0:
            ov = slice(a0, prev_end + 1)
            perm, worst, margin = stitch(X[ov], Xw[: prev_end + 1 - a0], bl)
            ok = worst <= 0.5 and margin >= 0.5
            seams.append(dict(at_s=round(a0 / m["analysis_fps"], 1), worst_distance_bl=round(worst, 2), margin_bl=None if margin >= BIG else round(margin, 2), ok=bool(ok)))
            if not ok:
                flagged[max(0, a0 - 5): prev_end + 6] = True
                job.warnings.append(f"seam at {a0 / m['analysis_fps']:.0f} s: identities were linked by position with a small margin; check this moment")
        elif k > 0 and not have and np.isfinite(X).any():
            job.warnings.append(f"{a0 / m['analysis_fps']:.0f} s: identities restart here (the window before was not tracked); numbers before and after are unrelated")
        new0 = prev_end + 1 - a0 if (have and prev_end >= a0) else 0
        for j in range(N):
            X[a0 + new0: a0 + T, perm[j]] = Xw[new0:, j]
        for u, v in s.get("flagged_contacts") or []:
            flagged[a0 + int(u): a0 + int(v) + 1] = True
        if s.get("status") == "NO SAFE FRAME":
            flagged[a0: a0 + T] = True
        prev_end = a0 + T - 1; have = True
    BL = float(np.median(bls)) if bls else 0.0
    # ---- boxes and outlines from MouseWorld-DET, attached to the identities by position (every camera frame when filling)
    def dets_of(k):
        a0, T = wins[k]
        if fill:
            df = job.work / "fill" / "det" / "work" / f"video__w{k:04d}_000000_{step * (T - 1):06d}" / "det.npz"
        else:
            df = Dm / "work" / f"video__w{k:04d}_000000_{T - 1:06d}" / "det.npz"
            if df.with_name("det_raw.npz").exists():
                df = df.with_name("det_raw.npz")          # strongest_recut: output boxes keep the detector's own scores
        if not df.exists():
            return None, {}
        D = CB.load_det(df); by_t: Dict[int, List[int]] = {}
        for i, t in enumerate(D["t"]):
            by_t.setdefault(int(t), []).append(i)
        return D, by_t
    S_ = step if fill else 1                              # output frames per analysed frame
    NO = S_ * (TA - 1) + 1 if TA else 0
    frames: List[List[dict]] = [[] for _ in range(NO)]
    done_until = -1
    for k, (a0, T) in enumerate(wins):
        D, by_t = dets_of(k)
        if D is None:
            continue
        for t in range(S_ * (T - 1) + 1):
            g = a0 * S_ + t
            if g <= done_until:
                continue
            prior, src = prior_positions(X, g, S_)
            ids = [i for i in range(N) if np.isfinite(prior[i, 0])]
            frames[g] = fill_rows(ids, prior, src, [(D["c"][d], d) for d in by_t.get(t, [])], BL,
                                  flagged[g // S_] or (g % S_ != 0 and g // S_ + 1 < TA and flagged[g // S_ + 1]))
            for r in frames[g]:
                d = r.pop("det")
                if d is not None:
                    x0, y0, x1, y1 = [int(v) for v in D["box"][d]]
                    r.update(box=[x0, y0, x1, y1], score=float(D["score"][d]), mask=(D["mbuf"][D["moff"][d]:D["moff"][d + 1]].copy(), y1 - y0, x1 - x0))
        done_until = a0 * S_ + S_ * (T - 1)
    if not any(frames):
        raise StepError(NO_ANIMAL)
    # ---- ONE geometry source for every output frame: SAM 2.1 at the output rate, anchored on the resolved identities
    geom = None
    if job.opt.fill == "sam2" and BL > 0:
        job.set("tracking", 0.74, "SAM 2.1 at the output frame rate (stable boxes and outlines)")
        geom = _stable_geometry(job, root, wins, X, N, BL, S_, fill, Dm)
        # analysed frames on which an identity's stable mask does not contain the tracker's own position (see sam_row_consistent);
        # the frames between two analysed frames are not given the mask either when one of the two analysed frames failed
        tracked_ok: Dict[int, list] = {}
        for g in range(NO):
            for r in frames[g]:
                if r["source"] == "tracked":
                    st = geom["rows"].get((g, r["id"] - 1))
                    tracked_ok.setdefault(r["id"], []).append((g, st is None or not st[3] or sam_row_consistent((r["x"], r["y"]), st[0], BL)))
        for g in range(NO):
            for r in frames[g]:
                r["det_box"] = r["box"]
                st = geom["rows"].get((g, r["id"] - 1))
                if st is not None and st[3] and not bracket_consistent(tracked_ok.get(r["id"], []), g):
                    st = None                                          # the mask is on another animal around here: not used
                    geom["inconsistent"] = geom.get("inconsistent", 0) + 1
                if st is not None and st[3]:
                    box, c, mk, _ = st
                    if r["box"]:
                        r["agree"] = _iou(r["box"], box) >= 0.3
                    r["box"], r["mask"], r["geom"] = [int(v) for v in box], mk, "sam2"
                    if r["source"] == "tracked":
                        r["tx"], r["ty"] = r["x"], r["y"]              # the tracker's own position, kept in track_x / track_y
                    else:
                        r["source"] = "filled"
                    r["x"], r["y"] = float(c[0]), float(c[1])          # ONE source for the position too: the centre of the stable mask
                else:
                    r["geom"] = "detector" if r["box"] else "none"
    n_rej = reject_jumps(frames, BL, m["fps"] / (step if not fill else 1) if not fill else m["fps"])
    if n_rej:
        job.log(f"{n_rej} detector box(es) were rejected: they would have moved an animal further in one frame than an animal can move")
    # ---- keypoints (MouseWorld-Pose, top-down on the attached boxes)
    job.set("tracking", 0.82, "MouseWorld-Pose keypoints")
    fpath = (lambda g: job.work / "native" / f"{g:06d}.jpg") if fill else (lambda g: frame_path(job, g))
    items, where = [], []
    for g in range(NO):
        bx = [(r, r["box"]) for r in frames[g] if r["box"]]
        if bx:
            items.append({"image": str(fpath(g).resolve()), "boxes": [b for _, b in bx]})
            where += [r for r, _ in bx]
    if items:
        req = job.work / "pose_request.json"; outp = job.work / "pose.npz"
        req.write_text(json.dumps({"weights": str(W.resolve("mouseworld-pose")), "pre": "dark_gamma0.5", "items": items}))
        run_step(job, "MouseWorld-Pose on one crop per animal", "pose", "pose_frames.py", [req, outp])
        z = np.load(outp)
        for r, kp, kc, ok in zip(where, z["kp"], z["conf"], z["ok"]):
            if ok:
                r["kp"], r["kc"] = kp, kc
    td = job.opt.tracker_detector
    return dict(frames=frames, body_length_px=BL, windows=win_info, seams=seams, supervisor=ver, camera_rate=fill, fallback_windows=len(fallback_used),
                geometry=dict(source="SAM 2.1 propagation at the output frame rate, anchored on the resolved identities; rows where it disagreed with "
                                     "the resolved position keep the detector's box" if geom else "MouseWorld-DET's own box on every frame (option --fill detector)",
                              **({k: geom.get(k, 0) for k in ("seeds", "repairs", "checked", "bad", "inconsistent")} if geom else {})),
                models=["MouseWorld-DET", "MouseWorld-Track", "MouseWorld-Pose"], tracker_detector=td, sam3=job.opt.sam3,
                tracker=f"MouseWorld-Track, rule set {ver}; detector for the witnesses: "
                        + {"strongest": "MouseWorld-DET alone (option 'strongest': about 25 % less computing; 42 identity fixes on the 34 public benchmark "
                                        "clips where the scored pair needed 21)",
                           "strongest_recut": "MouseWorld-DET alone with a moved confidence cut (option 'strongest_recut': experimental, fitted to the benchmark clips)",
                           "strongest_two_seeds": "MouseWorld-DET with mid_1408 as second seed (option 'strongest_two_seeds': 39 identity fixes on the 34 public "
                                                  "benchmark clips where the scored pair needed 21)",
                           "scored_pair": "the scored pair epoch5 + mid_1408 (default)"}[td]
                        + f"; witnesses: SAM 2.1-L forward/backward{' from two detector seeds' if td in ('scored_pair', 'strongest_two_seeds') else ''}, local re-tracks, "
                        "detector at camera rate"
                        + (f", SAM 3.1 text prompt '{job.opt.prompt}' (option --sam3)" if job.opt.sam3 else
                           "; WITHOUT the SAM 3.1 text-prompt witness (default: its weights are gated; option --sam3 turns it on)")
                        + ". On the 34 public benchmark clips the scored pair needed 20 identity fixes (73 per hour) without SAM 3.1 and 21 (76 per hour) with it")


def confident_detections(work: Path, cut: float = 0.4) -> int:
    """Confident detections (score >= cut, the tracker's SEED_SCORE) in every det.npz under `work`; -1 when there is no det.npz."""
    fs = sorted(Path(work).glob("*/det.npz"))
    if not fs:
        return -1
    n = 0
    for f in fs:
        with np.load(f) as z:
            n += int((np.asarray(z["score"], float) >= cut).sum())
    return n


NO_ANIMAL = ("no animal was found in this video: the detector returned nothing on any frame. If animals are visible, they may be too small, "
             "too dark or too bright for the pretrained models; nothing was tracked")


def _iou(a, b) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - iw * ih
    return iw * ih / u if u > 0 else 0.0


def _window_det(job: Job, k: int, T: int, Dm: Path, fill: bool, step: int) -> Optional[Path]:
    df = (job.work / "fill" / "det" / "work" / f"video__w{k:04d}_000000_{step * (T - 1):06d}" / "det.npz") if fill else \
         (Dm / "work" / f"video__w{k:04d}_000000_{T - 1:06d}" / "det.npz")
    return df if df.exists() else None


def _fallback_positions(job: Job, k: int, T: int, N: int, Dm: Path, fill: bool, step: int) -> tuple:
    """Positions for a window MouseWorld-Track could not start in (no frame with all animals apart): the detector's boxes on the
    analysed frames linked by the simple position tracker. -> ((T, N, 2) or None, body length)."""
    from .. import registry
    from .mwtrack import core_base as CB
    df = _window_det(job, k, T, Dm, fill, step)
    if df is None:
        return None, 0.0
    D = CB.load_det(df); S_ = step if fill else 1
    dets = [[] for _ in range(T)]
    for i, t in enumerate(D["t"]):
        if int(t) % S_ == 0 and int(t) // S_ < T:
            dets[int(t) // S_].append(dict(box=[float(v) for v in D["box"][i]], score=float(D["score"][i]), c=D["c"][i]))
    if not any(dets):
        return None, 0.0
    ids = registry.create("trackers", "iou", {"n_animals": N}).track(dets)
    Xw = np.full((T, N, 2), np.nan)
    for t in range(T):
        for d, tid in zip(dets[t], ids[t]):
            if 0 <= tid < N:
                Xw[t, tid] = d["c"]
    conf = D["score"] >= 0.4
    bl = 2.0 * float(np.sqrt(np.median(D["area"][conf] if conf.any() else D["area"])))
    return (Xw if np.isfinite(Xw).any() else None), bl


def _stable_geometry(job: Job, root: Path, wins: List[tuple], X: np.ndarray, N: int, BL: float, S_: int, fill: bool, Dm: Path) -> dict:
    """Run mwtrack/final_masks.py per window. -> {"rows": {(output frame, identity): (box, centroid, packed mask, ok)}, counts}"""
    rows: Dict[tuple, tuple] = {}; tot = dict(seeds=0, repairs=0, checked=0, bad=0)
    step = int(job.meta["step"])
    for k, (a0, T) in enumerate(wins):
        df = _window_det(job, k, T, Dm, fill, step)
        Xw = X[a0:a0 + T]
        if df is None or not np.isfinite(Xw).any():
            continue
        fdir = (job.work / "fill" if fill else root) / "data" / "video" / f"w{k:04d}" / "frames"
        n = S_ * (T - 1) + 1
        req = job.work / f"final_{k:04d}.json"; outp = job.work / f"final_{k:04d}.npz"
        req.write_text(json.dumps(dict(frames=str(fdir), n=n, step=S_, N=N, BL=BL, det=str(df),
                                       X=[[None if not np.isfinite(p[0]) else [float(p[0]), float(p[1])] for p in row] for row in Xw])))
        run_step(job, "SAM 2.1 at the output frame rate (stable boxes and outlines)", "sam2", "final_masks.py", [req, outp],
                 env=dict(MWT_SAM2_CKPT=str(W.resolve("sam2.1-hiera-large"))))
        z = np.load(outp)
        for key in tot:
            tot[key] += int(z[key])
        for i in range(len(z["frame"])):
            g = a0 * S_ + int(z["frame"][i]); key = (g, int(z["id"][i]))
            if key in rows:
                continue                                         # the overlap of two windows: the earlier window's row stays
            x0, y0, x1, y1 = [int(v) for v in z["box"][i]]
            rows[key] = ((x0, y0, x1, y1), z["c"][i], (z["mbuf"][z["moff"][i]:z["moff"][i + 1]].copy(), y1 - y0, x1 - x0), bool(z["ok"][i]))
    return dict(rows=rows, **tot)


MAX_SPEED_BL_S = 15.0        # body lengths per second: faster than a mouse runs; a larger step is a wrong attachment (a reflection, another animal)


SAM_POS_MARGIN_BL = 0.1    # the identity's own position may lie this far (body lengths) outside the stable mask's box


def sam_row_consistent(pos, box, BL: float) -> bool:
    """The stable SAM 2.1 geometry is used on an analysed frame only if the tracker's resolved position of that identity lies
    inside the mask's box, within SAM_POS_MARGIN_BL body lengths (the frames in between: see bracket_consistent; a filled
    frame's detector box is NOT used as the test, because that box is itself what flickers). A mask that drifted onto the neighbouring animal (two mice touching at the start of a clip)
    fails this test even when its centre is still within the repair distance of final_masks.py."""
    m = SAM_POS_MARGIN_BL * max(0.0, float(BL))
    return box[0] - m <= pos[0] <= box[2] + m and box[1] - m <= pos[1] <= box[3] + m


def bracket_consistent(tracked: List[tuple], g: int) -> bool:
    """tracked: [(analysed output frame, mask consistent with the tracker there)] of one identity, in frame order. A frame
    passes when it is an analysed frame that passed, or lies between two analysed frames (the nearest on each side, where
    they exist) that both passed."""
    import bisect
    gs = [t[0] for t in tracked]
    i = bisect.bisect_left(gs, g)
    if i < len(gs) and gs[i] == g:
        return bool(tracked[i][1])
    lo = tracked[i - 1][1] if i > 0 else True
    hi = tracked[i][1] if i < len(gs) else True
    return bool(lo and hi)


def reject_jumps(frames: List[List[dict]], BL: float, out_fps: float) -> int:
    """A row whose geometry comes from a single detector box (not from the stable propagation) and that would move its identity
    further than MAX_SPEED_BL_S allows since that identity's last accepted position keeps its place in the table but loses the
    box: the conflicting detection is recorded as rejected (geometry 'rejected'), never drawn as the animal."""
    if BL <= 0 or out_fps <= 0:
        return 0
    last: Dict[int, tuple] = {}; n = 0
    for g, rows in enumerate(frames):
        for r in rows:
            p = last.get(r["id"])
            if r.get("box") and r.get("geom") != "sam2" and r.get("source") != "tracked" and p is not None:
                lim = MAX_SPEED_BL_S * BL * (g - p[0]) / out_fps
                if math.hypot(r["x"] - p[1], r["y"] - p[2]) > max(lim, 0.5 * BL):
                    r.update(box=None, mask=None, kp=None, kc=None, score=None, geom="rejected", source="interpolated", status="flagged")
                    r["x"], r["y"] = p[1], p[2]
                    n += 1
                    continue
            if r.get("box"):
                last[r["id"]] = (g, r["x"], r["y"])
    return n


def jitter(frames: List[List[dict]], phase: int = 1) -> dict:
    """Frame-to-frame instability of the geometry, per output-frame transition g -> g + 1 of the same identity.
    edge: mean over the four box edges of |edge motion - centroid motion along that axis| (px): motion of the box that the
    animal's own movement does not explain. keypoint: the same for each keypoint. `by_phase`: the edge value split by
    (g mod phase), phase = output frames per analysed frame; a source that alternates shows up as unequal phases.
    `alternation` = largest / smallest phase mean (1.0 = no pattern)."""
    edge: Dict[int, list] = {}; kpv: list = []; cen: list = []
    last: Dict[int, tuple] = {}
    for g, rows in enumerate(frames):
        cur = {}
        for r in rows:
            if not r.get("box"):
                continue
            cur[r["id"]] = (g, r["x"], r["y"], r["box"], r.get("kp"), r.get("kc"))
            p = last.get(r["id"])
            if p is None or p[0] != g - 1:
                continue
            dx, dy = r["x"] - p[1], r["y"] - p[2]
            e = (abs(r["box"][0] - p[3][0] - dx) + abs(r["box"][2] - p[3][2] - dx) + abs(r["box"][1] - p[3][1] - dy) + abs(r["box"][3] - p[3][3] - dy)) / 4.0
            edge.setdefault((g - 1) % max(1, phase), []).append(e); cen.append(math.hypot(dx, dy))
            if r.get("kp") is not None and p[4] is not None:
                d = np.hypot(*(np.asarray(r["kp"]) - np.asarray(p[4]) - [dx, dy]).T)
                okk = (np.asarray(r["kc"]) >= 0.3) & (np.asarray(p[5]) >= 0.3)
                if okk.any():
                    kpv.append(float(d[okk].mean()))
        last = cur
    allv = [v for vs in edge.values() for v in vs]
    if not allv:
        return dict(transitions=0)
    ph = {int(k): round(float(np.mean(v)), 2) for k, v in sorted(edge.items())}
    return dict(transitions=len(allv), edge_px=round(float(np.mean(allv)), 2), keypoint_px=round(float(np.mean(kpv)), 2) if kpv else None,
                centroid_px=round(float(np.mean(cen)), 2), by_phase=ph,
                alternation=round(max(ph.values()) / max(1e-6, min(ph.values())), 2) if len(ph) > 1 else 1.0)


def prior_positions(X: np.ndarray, g: int, S_: int) -> tuple:
    """Where each identity is expected at output frame g. On an analysed frame: the tracked position. Between two analysed
    frames: the straight line between the two tracked positions (only if both exist). -> ((N, 2) array, "tracked" | "between")"""
    a, r = divmod(g, S_)
    if r == 0:
        return X[a], "tracked"
    if a + 1 >= len(X):
        return np.full_like(X[a], np.nan), "between"
    w = r / S_
    return (1 - w) * X[a] + w * X[a + 1], "between"


def fill_rows(ids: List[int], prior: np.ndarray, src: str, dets: List[tuple], BL: float, flag: bool) -> List[dict]:
    """Rows of one output frame. Each identity takes the detection nearest its expected position (one-to-one, within half a
    body length). `source`: tracked = position from the tracker (analysed frame); filled = a frame between two analysed frames,
    position / box / outline from MouseWorld-DET's detection there; interpolated = between two analysed frames and no detection
    could be attached, so the position is the straight line between the neighbours (no box, no keypoints)."""
    match = {}
    if ids and dets:
        n_, m_ = len(ids), len(dets); size = max(n_, m_)
        cost = [[BIG] * size for _ in range(size)]
        for a, i in enumerate(ids):
            for b, (c, _) in enumerate(dets):
                dist = float(np.hypot(*(c - prior[i])))
                if dist <= 0.5 * BL:
                    cost[a][b] = dist
        h = hungarian(cost)
        match = {ids[a]: dets[h[a]] for a in range(n_) if h[a] < m_ and cost[a][h[a]] < BIG}
    rows = []
    for i in ids:
        got = match.get(i)
        if src == "tracked":
            x, y, source = float(prior[i, 0]), float(prior[i, 1]), "tracked"
        elif got is not None:
            x, y, source = float(got[0][0]), float(got[0][1]), "filled"
        else:
            x, y, source = float(prior[i, 0]), float(prior[i, 1]), "interpolated"
        rows.append(dict(id=i + 1, x=x, y=y, box=None, score=None, mask=None, kp=None, kc=None, status="flagged" if flag else "ok",
                         source=source, det=got[1] if got is not None else None))
    return rows


# ------------------------------------------------------------------ lite / mock engines ----------------------------
def track_simple(job: Job, N: int) -> dict:
    cv2 = _cv2()
    from .. import registry
    from PIL import Image
    m = job.meta; lite = job.engine["engine"] == "lite"
    cam = job.opt.all_frames and m["step"] > 1            # the simple engines can afford to look at every camera frame directly
    paths = [job.work / "native" / f"{g:06d}.jpg" for g in range(m["native"])] if cam else [frame_path(job, g) for g in range(m["frames"])]
    TA = len(paths)
    dets = _simple_detect(job, paths, _predictor(job), 0.20, 0.45)
    job.set("tracking", 0.66, "linking identities (simple tracker)")
    tr = registry.create("trackers", "iou", {"n_animals": N})
    ids = tr.track(dets)
    pose = None
    if lite:
        pose = registry.create("predictors", "onnx", {"weights": str(W.resolve("mouseworld-pose-lite")), "conf": 0.01, "category": "mouse",
                                                      "task": "pose", "keypoints": KP, "imgsz": 256, "threads": usable_cpus()})
    frames: List[List[dict]] = [[] for _ in range(TA)]
    sides = []
    for g in range(TA):
        if g % 10 == 0:
            job.check()
            job.set(progress=0.66 + 0.16 * g / max(1, TA), message=f"keypoints ({g}/{TA})" if lite else "writing")
        img = None
        for inst, tid in zip(dets[g], ids[g]):
            if tid < 0:
                continue
            x0, y0, x1, y1 = [int(round(v)) for v in inst["box"]]
            x0, y0 = max(0, x0), max(0, y0); x1, y1 = min(m["W"], max(x1, x0 + 1)), min(m["H"], max(y1, y0 + 1))
            sides.append(math.sqrt((x1 - x0) * (y1 - y0)))
            r = dict(id=int(tid) + 1, x=(x0 + x1) / 2.0, y=(y0 + y1) / 2.0, box=[x0, y0, x1, y1], score=float(inst.get("score", 1.0)),
                     mask=None, kp=None, kc=None, status="ok", source="tracked")
            if inst.get("poly"):
                mk = np.zeros((y1 - y0, x1 - x0), np.uint8)
                cv2.fillPoly(mk, [np.round(np.array(inst["poly"], float) - [x0, y0]).astype(np.int32)], 1)
                if mk.any():
                    ys, xs = np.nonzero(mk); r["x"], r["y"] = float(xs.mean() + x0), float(ys.mean() + y0)
                    r["mask"] = (np.packbits(mk.astype(bool)), y1 - y0, x1 - x0)
            if pose is not None:
                if img is None:
                    img = cv2.imread(str(paths[g]))
                S = 256; cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; side = max(max(x1 - x0, y1 - y0) * 1.3, 16); X0, Y0 = cx - side / 2, cy - side / 2
                A_ = cv2.getAffineTransform(np.float32([[X0, Y0], [X0 + side, Y0], [X0, Y0 + side]]), np.float32([[0, 0], [S, 0], [0, S]]))
                crop = cv2.warpAffine(img, A_, (S, S), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114))
                pi = pose.predict(Image.fromarray(crop[:, :, ::-1].copy()))
                pi = [q for q in pi if q.get("keypoints")]
                if pi:
                    near = [q for q in pi if math.hypot((q["box"][0] + q["box"][2]) / 2 - S / 2, (q["box"][1] + q["box"][3]) / 2 - S / 2) <= 0.25 * S]
                    q = max(near or pi, key=lambda q: q["score"])          # as MouseWorld-Pose: the most confident instance near the crop centre
                    r["kp"] = np.array([[q["keypoints"][k][0] * side / S + X0, q["keypoints"][k][1] * side / S + Y0] for k in KP])
                    r["kc"] = np.array([0.9 if q["keypoints"][k][2] >= 2 else 0.3 for k in KP])
            frames[g].append(r)
    if lite:
        job.warnings.append("CPU engine: MouseWorld-DET-Lite with a simple position tracker. Identities are NOT reliable when animals touch; "
                            "use a computer with an NVIDIA GPU for MouseWorld-Track.")
    return dict(frames=frames, body_length_px=float(np.median(sides)) if sides else 0.0, windows=[], seams=[], supervisor=None, camera_rate=cam,
                models=["MouseWorld-DET-Lite", "simple tracker", "MouseWorld-Pose (CPU, earlier checkpoint)"] if lite else ["blob detector (test)", "simple tracker"],
                tracker="simple distance / overlap tracker with a fixed animal count (NOT MouseWorld-Track)")


# ------------------------------------------------------------------ outputs ----------------------------------------
def load_masks(path) -> dict:
    """masks.npz -> {(video_frame, id): boolean array of the box size}; `box` gives its place in the frame (x0, y0, x1, y1)."""
    z = np.load(path)
    out = {}
    for k in range(len(z["video_frame"])):
        x0, y0, x1, y1 = z["box"][k]
        h, w = int(y1 - y0), int(x1 - x0)
        out[(int(z["video_frame"][k]), int(z["id"][k]))] = np.unpackbits(z["mbuf"][z["moff"][k]:z["moff"][k + 1]])[:h * w].reshape(h, w).astype(bool)
    return out


def _ffmpeg() -> Optional[str]:
    exe = os.environ.get("ANIMALWORLD_FFMPEG") or shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


class _Video:
    def __init__(self, path: Path, w: int, h: int, fps: float, cv2):
        self.cv2, self.path, self.proc, self.vw = cv2, path, None, None
        exe = _ffmpeg()
        if exe:
            try:
                self.proc = subprocess.Popen([exe, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps:.4f}",
                                              "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p",
                                              "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except OSError:
                self.proc = None
        if self.proc is None:
            self.vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        self.codec = "H.264" if self.proc else "MPEG-4 part 2 (install ffmpeg for H.264)"

    def write(self, im) -> None:
        if self.proc:
            try:
                self.proc.stdin.write(im.tobytes())
            except (BrokenPipeError, OSError):
                pass
        else:
            self.vw.write(im)

    def close(self) -> None:
        if self.proc:
            try:
                self.proc.stdin.close()
            except OSError:
                pass
            self.proc.wait()
        else:
            self.vw.release()


def _draw(cv2, im, rows: List[dict], trails: Dict[int, list], sc: float, masks_only: bool = False):
    """Draw the rows of one frame. Colour, line width and the label's place follow fixed rules (identity -> colour; label at
    the box's top-left corner), so nothing changes from frame to frame except the geometry itself.
    masks_only: the segmentation alone (translucent fill + outline per identity; no box, no skeleton, no trail, no label)."""
    lw = max(1, int(round(im.shape[1] / 640)))
    for r in rows:
        col = COLOURS[(r["id"] - 1) % len(COLOURS)]
        if r["mask"] is not None and r["box"]:
            bits, h, w = r["mask"]; x0, y0, x1, y1 = r["box"]
            mk = np.unpackbits(bits)[:h * w].reshape(h, w).astype(np.uint8)
            if sc != 1.0:
                x0, y0, x1, y1 = [int(round(v * sc)) for v in (x0, y0, x1, y1)]
                if x1 > x0 and y1 > y0:
                    mk = cv2.resize(mk, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
            x1 = min(x1, im.shape[1]); y1 = min(y1, im.shape[0]); mk = mk[: y1 - y0, : x1 - x0]
            if mk.size:
                roi = im[y0:y1, x0:x1]
                a_ = 0.45 if masks_only else 0.35
                roi[mk > 0] = ((1 - a_) * roi[mk > 0] + a_ * np.array(col)).astype(np.uint8)
                cs, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(roi, cs, -1, col, lw)
        if masks_only:
            continue
        tr = trails.setdefault(r["id"], [])
        tr.append((int(r["x"] * sc), int(r["y"] * sc))); del tr[:-25]
        for a, b in zip(tr[:-1], tr[1:]):
            cv2.line(im, a, b, col, lw, cv2.LINE_AA)
        if r["kp"] is not None:
            P = [(int(x * sc), int(y * sc)) if c >= 0.3 else None for (x, y), c in zip(r["kp"], r["kc"])]
            for a, b in SKEL:
                if P[a] and P[b]:
                    cv2.line(im, P[a], P[b], (255, 255, 255), lw, cv2.LINE_AA)
            for p_ in P:
                if p_:
                    cv2.circle(im, p_, 2 * lw + 1, col, -1, cv2.LINE_AA); cv2.circle(im, p_, 2 * lw + 1, (255, 255, 255), 1, cv2.LINE_AA)
        if r["box"]:
            x0, y0, x1, y1 = [int(round(v * sc)) for v in r["box"]]
            cv2.rectangle(im, (x0, y0), (x1, y1), col, lw)
            org = (x0, max(14 * lw, y0 - 4))
        else:
            org = (int(r["x"] * sc) + 6, int(r["y"] * sc) - 6)
            cv2.circle(im, (int(r["x"] * sc), int(r["y"] * sc)), 4 * lw, col, lw)
        label = str(r["id"]) + ("?" if r["status"] == "flagged" else "")
        cv2.putText(im, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * lw, (255, 255, 255), 3 * lw, cv2.LINE_AA)
        cv2.putText(im, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * lw, col, max(1, lw + 1), cv2.LINE_AA)
    return im


def write_outputs(job: Job, res: dict, N: int) -> dict:
    cv2 = _cv2()
    m = job.meta; frames = res["frames"]; TA = len(frames); fps = m["fps"]
    cam = bool(res.get("camera_rate")); step = 1 if cam else m["step"]              # camera frames per output frame
    fpath = (lambda g: job.work / "native" / f"{g:06d}.jpg") if cam else (lambda g: frame_path(job, g))
    job.set("tracking", 0.88, "writing tracks.csv and masks.npz")
    cols = ["video_frame", "t_s", "id", "x", "y", "x0", "y0", "x1", "y1", "det_score", "status", "source"] + [f"{k}_{a}" for k in KP for a in ("x", "y", "conf")] \
        + ["region", "zoom", "geometry", "detector_agrees", "track_x", "track_y"]
    mf, mi, mb, buf, off = [], [], [], [], [0]
    n_rows = n_box = n_kp = n_flag = 0
    n_src = {"tracked": 0, "filled": 0, "interpolated": 0}
    per_id = {i: 0 for i in range(1, N + 1)}
    with open(job.dir / "tracks.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for g in range(TA):
            for r in sorted(frames[g], key=lambda r: r["id"]):
                row = [g * step, f"{g * step / fps:.4f}", r["id"], f"{r['x']:.2f}", f"{r['y']:.2f}"]
                row += list(r["box"]) if r["box"] else ["", "", "", ""]
                row += [f"{r['score']:.3f}" if r["score"] is not None else "", r["status"], r.get("source", "tracked")]
                n_src[r.get("source", "tracked")] = n_src.get(r.get("source", "tracked"), 0) + 1
                if r["kp"] is not None:
                    for (x, y), c in zip(r["kp"], r["kc"]):
                        row += [f"{x:.2f}", f"{y:.2f}", f"{c:.3f}"]
                    n_kp += 1
                else:
                    row += [""] * (3 * len(KP))
                row += [r.get("region", 0), f"{r.get('zoom', 1.0):.3g}", r.get("geom") or ("detector" if r["box"] else "none"),
                        "" if r.get("agree") is None else int(bool(r["agree"])),
                        f"{r['tx']:.2f}" if r.get("tx") is not None else "", f"{r['ty']:.2f}" if r.get("ty") is not None else ""]
                w.writerow(row)
                n_rows += 1; n_box += bool(r["box"]); n_flag += r["status"] == "flagged"; per_id[r["id"]] = per_id.get(r["id"], 0) + 1
                if r["mask"] is not None:
                    mf.append(g * step); mi.append(r["id"]); mb.append(r["box"]); buf.append(r["mask"][0]); off.append(off[-1] + len(r["mask"][0]))
    np.savez_compressed(job.dir / "masks.npz", video_frame=np.array(mf, np.int64), id=np.array(mi, np.int16), box=np.array(mb, np.int32).reshape(-1, 4),
                        mbuf=np.concatenate(buf) if buf else np.zeros(0, np.uint8), moff=np.array(off, np.int64), W=m["W"], H=m["H"],
                        note="mask k: np.unpackbits(mbuf[moff[k]:moff[k+1]])[:h*w].reshape(h, w) with h, w = box height, width; see animalworld.zeroshot.pipeline.load_masks")
    # ---- overlay video + contact sheet
    sc = min(1.0, 1280.0 / m["W"]); ow, oh = int(round(m["W"] * sc)) // 2 * 2, int(round(m["H"] * sc)) // 2 * 2
    vid = _Video(job.dir / "overlay.mp4", ow, oh, fps / step, cv2)
    pick = set(_sample(TA, 12)); tiles = []; trails: Dict[int, list] = {}
    # zoom: a second overlay that shows only the analysed regions, side by side and enlarged (the unused part of the frame is gone)
    regs = res.get("regions") or []; vid2 = None
    if regs:
        Hc = 480.0; tws = [(rg["box"][2] - rg["box"][0]) * Hc / max(1, rg["box"][3] - rg["box"][1]) for rg in regs]
        k2 = min(1.0, 1920.0 / (sum(tws) + 6 * (len(regs) - 1))); Hc = int(Hc * k2) // 2 * 2; tws = [max(2, int(t * k2) // 2 * 2) for t in tws]
        w2 = (sum(tws) + 6 * (len(regs) - 1) + 1) // 2 * 2
        vid2 = _Video(job.dir / "overlay_regions.mp4", w2, Hc, fps / step, cv2); trails2: List[Dict[int, list]] = [dict() for _ in regs]
    vid3 = _Video(job.dir / "overlay_masks.mp4", ow, oh, fps / step, cv2)       # the segmentation alone, to judge the masks
    for g in range(TA):
        if g % 20 == 0:
            job.check()
            job.set(progress=0.90 + 0.08 * g / max(1, TA), message=f"drawing the overlay ({g}/{TA})")
        im = cv2.imread(str(fpath(g)))
        if vid2 is not None:
            canvas = np.full((Hc, w2, 3), 255, np.uint8); xo = 0
            for i_, rg in enumerate(regs):
                x0, y0, x1, y1 = rg["box"]; sx = tws[i_] / float(x1 - x0)
                t_ = cv2.resize(im[y0:y1, x0:x1], (tws[i_], Hc), interpolation=cv2.INTER_CUBIC if sx > 1 else cv2.INTER_AREA)
                rr = [dict(r, x=r["x"] - x0, y=r["y"] - y0, box=[r["box"][0] - x0, r["box"][1] - y0, r["box"][2] - x0, r["box"][3] - y0] if r["box"] else None,
                           kp=(np.asarray(r["kp"]) - [x0, y0]) if r["kp"] is not None else None) for r in frames[g] if r.get("region") == rg["region"]]
                canvas[:, xo:xo + tws[i_]] = _draw(cv2, t_, rr, trails2[i_], sx)
                xo += tws[i_] + 6
            cv2.putText(canvas, f"{g * step / fps:7.2f} s", (8, Hc - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, f"{g * step / fps:7.2f} s", (8, Hc - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            vid2.write(canvas)
        if sc != 1.0 or im.shape[1] != ow or im.shape[0] != oh:
            im = cv2.resize(im, (ow, oh), interpolation=cv2.INTER_AREA)
        for rg in res.get("regions") or []:                    # the analysed regions, drawn thinly
            bx = [int(round(v * sc)) for v in rg["box"]]
            cv2.rectangle(im, (bx[0], bx[1]), (bx[2] - 1, bx[3] - 1), (160, 160, 160), 1)
        vid3.write(_draw(cv2, im.copy(), frames[g], {}, sc, masks_only=True))
        im = _draw(cv2, im, frames[g], trails, sc)
        cv2.putText(im, f"{g * step / fps:7.2f} s", (8, oh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(im, f"{g * step / fps:7.2f} s", (8, oh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        vid.write(im)
        if g in pick:
            tw = 480; t_ = cv2.resize(im, (tw, max(1, int(round(oh * tw / ow)))), interpolation=cv2.INTER_AREA)
            tiles.append(t_)
    vid.close(); vid3.close()
    if vid2 is not None:
        vid2.close()
    used = sum((rg["box"][2] - rg["box"][0]) * (rg["box"][3] - rg["box"][1]) for rg in regs) / float(m["W"] * m["H"]) if regs else 1.0
    if regs and res.get("zoom"):
        res["zoom"]["frame_used_pct"] = round(100.0 * used, 1)
        res["zoom"]["short"] = f"used {100.0 * used:.0f} % of the frame in {len(regs)} region(s), enlarged x{np.mean([rg['zoom'] for rg in regs]):.1f}"
        res["zoom"]["text"] = res["zoom"]["short"] + ". " + (res["zoom"].get("text") or "")
    while len(tiles) % 4:
        tiles.append(np.full_like(tiles[0], 255))
    cv2.imwrite(str(job.dir / "contact_sheet.jpg"), np.vstack([np.hstack(tiles[r:r + 4]) for r in range(0, len(tiles), 4)]), [cv2.IMWRITE_JPEG_QUALITY, 88])
    # ---- one-page summary
    # frames in which EVERY counted animal has a box (a row without a box is a position only: the animal was not found there)
    all_n = sum(1 for g in range(TA) if len({r["id"] for r in frames[g] if r["box"]}) >= N)
    jit = jitter(frames, m["step"] if cam else 1)
    n_geom = {}
    for rows_ in frames:
        for r in rows_:
            kgeom = r.get("geom") or ("detector" if r["box"] else "none"); n_geom[kgeom] = n_geom.get(kgeom, 0) + 1
    fb = int(res.get("fallback_windows") or 0)
    summ = dict(
        video=str(job.video), animalworld=__version__, engine=job.engine["engine"], device=job.engine.get("device"), models=res["models"], tracker=res["tracker"],
        tracker_detector=res.get("tracker_detector"), animals=N, animals_auto_count=job.n_auto, count_confirmed_by="user" if job.opt.n_animals or getattr(job, "confirmed_by_user", False) else "automatic count accepted without a look",
        camera_fps=round(fps, 3), analysed_every_nth_frame=m["step"] if job.engine["engine"] == "full" or not cam else 1,
        analysis_fps=round(m["analysis_fps"] if job.engine["engine"] == "full" or not cam else fps, 3), seconds_analysed=round(m["seconds"], 2),
        output_every_nth_frame=step, rows_tracked=n_src["tracked"], rows_filled=n_src["filled"], rows_interpolated=n_src["interpolated"],
        sam3=bool(res.get("sam3")), frames_analysed=TA, frames_with_all_animals=all_n, frames_with_all_animals_pct=round(100.0 * all_n / max(1, TA), 1),
        rows=n_rows, rows_with_box=n_box, rows_with_keypoints=n_kp, rows_flagged=n_flag, frames_per_id=per_id,
        geometry=res.get("geometry"), rows_by_geometry=n_geom, jitter=jit, zoom=res.get("zoom"), regions=res.get("regions"),
        status=("FALLBACK: MouseWorld-Track could not start; simple tracker used, identities not reliable" if fb else
                ("ok" if n_box else "no animal found")),
        body_length_px=round(res["body_length_px"], 1), windows=res["windows"], seams=res["seams"], warnings=job.warnings,
        overlay_codec=vid.codec,
        files=dict(tracks="tracks.csv", masks="masks.npz", overlay="overlay.mp4", overlay_masks="overlay_masks.mp4", contact_sheet="contact_sheet.jpg",
                   count_sheet="count_sheet.jpg", log="run.log",
                   **({"overlay_regions": "overlay_regions.mp4"} if regs else {})),
        note="Nothing here was checked by a person. Look at overlay.mp4 before using the tracks; rows marked 'flagged' are moments the tracker itself was unsure about.")
    write_json(job.dir / "summary.json", summ)
    esc = lambda s: str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")      # noqa: E731
    facts = [("Video", job.video.name), ("Engine", f"{summ['engine']} on {summ['device']}"), ("Models", ", ".join(summ["models"])), ("Tracker", summ["tracker"]),
             ("Animals", f"{N} (automatic count: {job.n_auto}; {summ['count_confirmed_by']})"),
             ("Analysed", f"{summ['seconds_analysed']} s of video at {summ['camera_fps']} frames/s; identity tracking on every {summ['analysed_every_nth_frame']}. frame; "
                          f"rows written for every {step}. frame ({TA} frames)"),
             ("Row sources", f"tracked {n_src['tracked']}, filled between tracked frames from MouseWorld-DET {n_src['filled']}, interpolated (no detection) {n_src['interpolated']}"),
             ("All animals found", f"{all_n} of {TA} frames ({summ['frames_with_all_animals_pct']} %): frames in which every counted animal has a box"),
             ("Status", summ["status"]),
             ("Zoom", (res.get("zoom") or {}).get("text") or "whole frame"),
             ("Geometry", f"{(res.get('geometry') or {}).get('source', 'n/a')}; rows by source: {n_geom}; frame-to-frame box jitter {jit.get('edge_px')} px "
                          f"(by phase {jit.get('by_phase')}, alternation {jit.get('alternation')})"),
             ("Rows", f"{n_rows} (with box {n_box}, with keypoints {n_kp}, flagged as unsure {n_flag})"),
             ("Windows", "; ".join(f"{w_['start_s']:.0f} s: {w_['status']} ({w_['contacts']} contacts, {w_['flagged_contacts']} flagged)" for w_ in res["windows"]) or "n/a")]
    html = ["<!doctype html><meta charset='utf-8'><title>Animal World Run: " + esc(job.video.name) + "</title>",
            "<style>body{font:14px/1.45 system-ui,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#111}table{border-collapse:collapse}"
            "td{padding:3px 12px 3px 0;vertical-align:top;border-bottom:1px solid #ddd}img{max-width:100%}.w{color:#a00}</style>",
            f"<h2>{esc(job.video.name)}</h2><table>" + "".join(f"<tr><td>{esc(a)}</td><td>{esc(b)}</td></tr>" for a, b in facts) + "</table>"]
    if job.warnings:
        html.append("<h3 class='w'>Warnings</h3><ul class='w'>" + "".join(f"<li>{esc(x)}</li>" for x in job.warnings) + "</ul>")
    html.append("<p>" + esc(summ["note"]) + "</p><h3>12 evenly spaced frames of overlay.mp4</h3><img src='contact_sheet.jpg'>"
                "<h3>Frames used for the animal count</h3><img src='count_sheet.jpg'>"
                "<p>Files: tracks.csv, masks.npz, overlay.mp4, summary.json, run.log</p>")
    (job.dir / "summary.html").write_text("\n".join(html), encoding="utf-8")
    job.results = {k: str(job.dir / v) for k, v in dict(summ["files"], summary="summary.html").items()}
    return summ


# ------------------------------------------------------------------ the two phases ---------------------------------
def prepare_and_count(job: Job, engine: dict) -> int:
    """Phase 1: decode + automatic count. Leaves the job in status 'confirm' (or 'queued' when a count was given)."""
    job.engine = engine
    job.dir.mkdir(parents=True, exist_ok=True)
    job.set("preparing", 0.0, "reading the video")
    job.log(f"animalworld {__version__}  engine {engine['engine']} on {engine.get('device')}  video {job.video}")
    t0 = time.time()
    extract(job)
    job.timings["decoding the video to frames"] = round(time.time() - t0, 1)
    t0 = time.time()
    n = count(job)
    job.timings["animal count (incl. its detector pass)"] = round(time.time() - t0, 1)
    from . import zoom as Z
    t0 = time.time()
    n = Z.plan(job, n)                                   # looks at the whole clip; may propose regions and a per-region count
    job.timings["scout (animal size, regions)"] = round(time.time() - t0, 1)
    if job.opt.n_animals:
        job.n = int(job.opt.n_animals)
        if job.n != n:
            job.log(f"the given number of animals ({job.n}) differs from the automatic count ({n}); the given number is used")
        job.set("queued", 0.20, f"{job.n} animals (given)")
    else:
        job.set("confirm", 0.20, f"automatic count: {n}; confirm or correct it")
    return n


def analyse(job: Job) -> dict:
    """Phase 2: tracking + keypoints + files, with the confirmed number of animals."""
    N = int(job.n or 0)
    if N < 1:
        raise StepError("no animal was found, so there is nothing to track. If animals are visible, give the number yourself "
                        "(--n-animals N, or type it in the window) and start again")
    t0 = time.time()
    job.set("tracking", 0.20, "starting")
    job.log(f"tracking {N} animal(s)")
    from . import zoom as Z
    plan = getattr(job, "plan", None) or {}
    if plan.get("zoom"):
        res = Z.run_regions(job, N)
    else:
        res = track_full(job, N) if job.engine["engine"] == "full" else track_simple(job, N)
        if plan:
            res["zoom"] = dict(used=False, text=plan.get("text"), size_px_at_input=plan.get("size_px_at_input"))
    t1 = time.time()
    summ = write_outputs(job, res, N)
    job.timings["writing tracks, masks, overlay, summary"] = round(time.time() - t1, 1)
    summ["seconds_of_computing"] = round(time.time() - t0)
    summ["timings_s"] = dict(job.timings)
    write_json(job.dir / "summary.json", summ)
    if not job.opt.keep_work:
        shutil.rmtree(job.work, ignore_errors=True)          # temporary frames and intermediates of THIS run only
    job.set("done", 1.0, ("done, but NOT with MouseWorld-Track (simple fallback, identities not reliable): " if summ["status"].startswith("FALLBACK") else "done: ")
            + f"{summ['frames_with_all_animals_pct']} % of frames with all {N} animals" + (f"; {len(job.warnings)} warning(s)" if job.warnings else ""))
    return summ


def guarded(fn, job: Job, *a):
    """Run one phase; turn every failure into a job status instead of a traceback in the user's face."""
    try:
        return fn(job, *a)
    except Stopped:
        job.log("stopped by the user")
        job.set("stopped", message="stopped")
    except (StepError, W.WeightsError, FileNotFoundError, OSError, ValueError, KeyError, RuntimeError) as e:
        import traceback
        job.log("FAILED: " + str(e))
        job.log(traceback.format_exc(), file_only=True)          # the technical detail goes to run.log, not in the user's face
        job.set("failed", message=str(e)[:300])
    return None
