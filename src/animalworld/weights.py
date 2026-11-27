"""Model files and runtimes for the pretrained MouseWorld models.

No model file is inside the package. A name resolves, in this order, to

  1. $ANIMALWORLD_WEIGHTS/<file>                a folder you manage (shared lab storage, an offline computer)
  2. <cache>/weights/<file>                     $ANIMALWORLD_CACHE, else ~/.cache/animalworld
  3. (SAM 3.1 only) the Hugging Face cache, if you already fetched facebook/sam3.1 there
  4. a download of <url>/<file> into the cache, checked against its SHA-256, then moved into place

The MouseWorld weights are a separate download (the files of weights_release/). BASE_URL below is the address of that release; a placeholder address is used until
they are uploaded; until then step 4 does nothing for them unless $ANIMALWORLD_WEIGHTS_URL points at a copy. Third-party weights (SAM 2.1, SAM 3.1) are never
redistributed here: SAM 2.1 is fetched from Meta's own address at first use, SAM 3.1 is gated and must be fetched
by you from its official Hugging Face page.

Runtimes: each GPU step of the tracker runs as its own process. By default that process is THIS Python. A site can
point a step at another interpreter with $ANIMALWORLD_PY_<ROLE> or <config dir>/runtimes.toml (roles: det, sam2,
sam3, pose, cpu), e.g. when SAM 3.1 lives in its own environment.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Dict, List, Optional

from . import __version__

# Where the MouseWorld weight files are downloaded from: the address of the release that holds the files of
# weights_release/ (GitHub: https://github.com/<owner>/<repo>/releases/download/<tag>/). EDIT THIS ONE LINE after the
# upload, or set the environment variable ANIMALWORLD_WEIGHTS_URL (a mirror, a lab server, a local folder). An address
# that still contains the placeholder OWNER/REPO counts as "no address": nothing is downloaded and the app says so.
BASE_URL = "https://github.com/GuangWei-Zhang-25/AnimalWorld/releases/download/weights-v1/"
WEIGHTS_VERSION = "1"
SAM2_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/"     # Meta's own download address
_UNSET = "OWN" + "ER/" + "RE" + "PO"

REGISTRY: Dict[str, dict] = {
    "mouseworld-det": dict(
        file="mouseworld-det.ckpt", size=146037082, own=True, title="MouseWorld-DET",
        sha256="67c7866edf0ee23ce6d0b5fc201afa914ebde0e7c28e74c4bd43d7479498ffb3",
        about="RF-DETR-Seg-L detector + outlines (fine-tune b_full_lr5e5_cos_c04 from the epoch-5 checkpoint, selection F1 0.924); "
              "the program's default detector; inference-only copy"),
    "mouseworld-pose": dict(
        file="mouseworld-pose.pt", size=19756597, own=True, title="MouseWorld-Pose",
        sha256="fc81ae3a7f3f9e82d514f54e849789076a930f0ca5ca73ba8ff11cab096f9868",
        about="7 keypoints per mouse on a 256 px crop (YOLO11s-pose, run crop_pose_r3bottom; r3_b_g050 in the paper)"),
    "mouseworld-det-lite": dict(
        file="mouseworld-det-lite.onnx", size=40753393, own=True, title="MouseWorld-DET-Lite",
        sha256="535cb600eaa33927727e82c967d53625c6791d7b99966624b1a4d77d697aa366",
        about="YOLO11s-seg detector for computers without a GPU (ONNX)"),
    "mouseworld-pose-lite": dict(
        file="mouseworld-pose-lite.onnx", size=40075237, own=True, title="MouseWorld-Pose (CPU, earlier checkpoint)",
        sha256="bc489ef712735b564f5f40ba73bcc4478815820f3d138b24a2bd57678855a8c0",
        about="crop keypoint model as ONNX; an EARLIER checkpoint than MouseWorld-Pose"),
    "mouseworld-track-det-a": dict(
        file="mouseworld-track-det-a.ckpt", size=146041778, own=True, title="MouseWorld-Track detector A",
        sha256="43cfdd42dd5e79ab14b3c3686ef2321be2becdb825fba2c6748fa0ae4cf57a96",
        about="RF-DETR-Seg-L 'epoch5': the checkpoint scored as MouseWorld-DET in the paper's detection figures; first "
              "detector of the default tracker configuration (scored_pair)"),
    "mouseworld-track-det-b": dict(
        file="mouseworld-track-det-b.ckpt", size=146041778, own=True, title="MouseWorld-Track detector B",
        sha256="109fbca8d75611c2edceb39948118b1623796e3e1dc28847c7cfde6af094506c",
        about="RF-DETR-Seg-L 'mid_1408'; second detector seed of the default tracker configuration (scored_pair)"),
    "sam2.1-hiera-large": dict(
        file="sam2.1_hiera_large.pt", size=898083611, own=False, title="SAM 2.1 (Hiera-L)", url=SAM2_URL,
        sha256="2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318",
        about="Meta's released weights (Apache-2.0); forward / backward / local witnesses of MouseWorld-Track"),
    "sam3.1-multiplex": dict(
        file="sam3.1_multiplex.pt", size=3502755717, own=False, title="SAM 3.1 (multiplex)", url=None,
        sha256="0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6",
        about="OPTIONAL (--sam3). Meta's released weights (SAM License, gated); extra text-prompt witness of MouseWorld-Track",
        manual="SAM 3.1 is gated: accept the licence at https://huggingface.co/facebook/sam3.1 , then run "
               "`huggingface-cli download facebook/sam3.1 sam3.1_multiplex.pt` (or copy the file into the weights folder)"),
}
FULL = ("mouseworld-det", "mouseworld-pose", "sam2.1-hiera-large")         # nothing gated: SAM 3.1 is an option (--sam3)
SAM3 = ("sam3.1-multiplex",)
SCORED_PAIR = ("mouseworld-track-det-a", "mouseworld-track-det-b")
LITE = ("mouseworld-det-lite", "mouseworld-pose-lite")
ROLES = ("det", "sam2", "sam3", "pose", "cpu")
ROLE_MODULES = {"det": ("torch", "rfdetr"), "sam2": ("torch", "sam2"), "sam3": ("torch", "sam3"), "pose": ("torch", "ultralytics", "cv2"),
                "cpu": ("numpy", "scipy")}


class WeightsError(RuntimeError):
    pass


# ------------------------------------------------------------------ where files live ------------------------------
def cache_dir() -> Path:
    if os.environ.get("ANIMALWORLD_CACHE"):
        return Path(os.environ["ANIMALWORLD_CACHE"]).expanduser()
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "animalworld"


def search_dirs() -> List[Path]:
    out = []
    if os.environ.get("ANIMALWORLD_WEIGHTS"):
        out.append(Path(os.environ["ANIMALWORLD_WEIGHTS"]).expanduser())
    out.append(cache_dir() / "weights")
    return out


def base_url(entry: dict) -> str:
    u = entry.get("url") if not entry.get("own") else (os.environ.get("ANIMALWORLD_WEIGHTS_URL") or BASE_URL)
    if not u or _UNSET in u:
        return ""                                   # placeholder still in place: no download address yet
    return u if u.endswith("/") else u + "/"


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find(name: str) -> Optional[Path]:
    e = REGISTRY[name]
    for d in search_dirs():
        p = d / e["file"]
        if p.is_file():
            return p
    if name == "sam3.1-multiplex":
        hub = Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface") / "hub" / "models--facebook--sam3.1" / "snapshots"
        for p in sorted(hub.glob("*/sam3.1_multiplex.pt")) if hub.is_dir() else []:
            return p
    return None


def _missing_message(name: str) -> str:
    e = REGISTRY[name]
    where = " or ".join(str(d) for d in search_dirs())
    if e.get("manual"):
        return f"{e['title']}: {e['file']} not found in {where}. {e['manual']}."
    if e.get("own") and not base_url(e):
        return (f"{e['title']}: {e['file']} not found in {where}, and no download address is set (BASE_URL in animalworld/weights.py is "
                f"still the placeholder, ANIMALWORLD_WEIGHTS_URL is empty). Copy the file into one of these folders, set "
                f"ANIMALWORLD_WEIGHTS to the folder that holds it, or set ANIMALWORLD_WEIGHTS_URL to the release address.")
    return f"{e['title']}: {e['file']} not found in {where} and the download did not work."


def download(name: str, progress: Callable[[str, int, int], None] = None, tries: int = 3) -> Path:
    """Fetch one weight file into the cache and check its SHA-256 (see fetch_file)."""
    e = REGISTRY[name]
    url = base_url(e)
    if not url:
        raise WeightsError(_missing_message(name))
    return fetch_file(e, url, cache_dir() / "weights" / e["file"], progress, tries)


def fetch_file(e: dict, url: str, dst: Path, progress: Callable[[str, int, int], None] = None, tries: int = 3) -> Path:
    """Fetch e["file"] from url into dst and check its SHA-256. Plain GET requests with a generic User-Agent, nothing else.

    An interrupted download leaves <file>.part; the next attempt (in this call or a later run) continues it with an HTTP
    Range request when the server allows that, and otherwise starts again. A finished file with the wrong SHA-256 is set
    aside as <file>.badsum, never used, and one fresh download is tried."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    src = url + e["file"]
    print(f"animalworld: downloading {e['title']} ({e['size'] / 1e6:.0f} MB) from {src} (once; kept in {dst.parent})", flush=True)
    last: Optional[Exception] = None
    fresh_after_bad_sum = False
    for attempt in range(1, tries + 1):
        have = tmp.stat().st_size if tmp.exists() else 0
        if have > e["size"]:
            tmp.unlink(); have = 0
        try:
            if have < e["size"]:
                hdr = {"User-Agent": f"animalworld/{__version__}"}
                if have:
                    hdr["Range"] = f"bytes={have}-"
                with urllib.request.urlopen(urllib.request.Request(src, headers=hdr), timeout=60) as r:
                    resumed = have > 0 and getattr(r, "status", 200) == 206
                    if have and not resumed:
                        have = 0                                   # the server sent the whole file again
                    elif resumed:
                        print(f"animalworld: continuing {e['file']} at {have / 1e6:.0f} MB", flush=True)
                    with open(tmp, "ab" if resumed else "wb") as f:
                        done = have
                        while True:
                            chunk = r.read(1 << 20)
                            if not chunk:
                                break
                            f.write(chunk)
                            done += len(chunk)
                            if progress:
                                progress(e["file"], done, e["size"])
            if tmp.stat().st_size < e["size"]:
                raise OSError(f"connection ended after {tmp.stat().st_size} of {e['size']} bytes")
        except (urllib.error.URLError, OSError) as ex:
            last = ex
            if isinstance(ex, urllib.error.HTTPError) and ex.code in (401, 403, 404):
                break                                              # asking again will not help
            if isinstance(ex, urllib.error.HTTPError) and ex.code == 416 and tmp.exists():
                tmp.unlink()                                       # a stale partial file the server cannot continue
            print(f"animalworld: download attempt {attempt} of {tries} did not finish ({type(ex).__name__}: {ex})", flush=True)
            continue
        got = sha256(tmp)
        if e.get("sha256") and got != e["sha256"]:
            bad = dst.with_name(dst.name + ".badsum")
            os.replace(tmp, bad)
            last = WeightsError(f"{e['title']}: the downloaded file has SHA-256 {got[:16]}..., expected {e['sha256'][:16]}... "
                                f"It was kept as {bad} and is NOT used.")
            if fresh_after_bad_sum:
                raise last
            fresh_after_bad_sum = True
            print(f"animalworld: {e['file']} failed its SHA-256 check; one fresh download is tried", flush=True)
            continue
        os.replace(tmp, dst)
        return dst
    if isinstance(last, WeightsError):
        raise last
    raise WeightsError(f"{e['title']}: download from {src} failed ({type(last).__name__}: {last}). A partial file, if any, is kept "
                       f"and continued next time. Or fetch {e['file']} by hand and put it in {dst.parent}.") from last


def resolve(name: str, allow_download: bool = True) -> Path:
    if name not in REGISTRY:
        raise WeightsError(f"unknown model {name!r}; known: {', '.join(REGISTRY)}")
    p = find(name)
    if p:
        return p
    if allow_download and base_url(REGISTRY[name]):
        return download(name)
    raise WeightsError(_missing_message(name))


def status(names=None, verify: bool = False) -> List[dict]:
    out = []
    for n in names or REGISTRY:
        e = REGISTRY[n]
        p = find(n)
        row = {"name": n, "title": e["title"], "file": e["file"], "found": bool(p), "path": str(p) if p else None,
               "downloadable": bool(base_url(e)), "about": e["about"]}
        if p and verify:
            row["sha256_ok"] = sha256(p) == e["sha256"]
        if not p:
            row["how"] = _missing_message(n)
        out.append(row)
    return out


# ------------------------------------------------------------------ runtimes --------------------------------------
def _runtime_file() -> Dict[str, str]:
    from . import credentials
    f = credentials.config_dir() / "runtimes.toml"
    if not f.exists():
        return {}
    try:
        return credentials._parse(f.read_text(encoding="utf-8"))
    except credentials.CredentialError:
        return {}


def python_for(role: str) -> str:
    """The interpreter that runs a tracker step. $ANIMALWORLD_PY_<ROLE>, then runtimes.toml, then this Python."""
    v = os.environ.get(f"ANIMALWORLD_PY_{role.upper()}") or _runtime_file().get(role)
    return str(Path(v).expanduser()) if v else sys.executable


def step_env(extra: Dict[str, str] = None) -> Dict[str, str]:
    env = dict(os.environ, ORT_DISABLE_TELEMETRY="1", YOLO_AUTOINSTALL="false", YOLO_OFFLINE="true", PYTHONUNBUFFERED="1",
               HF_HUB_DISABLE_TELEMETRY="1", DO_NOT_TRACK="1")
    # Ultralytics keeps a per-user settings file (with analytics "sync" on by default). The app uses its OWN copy with sync
    # off, so it neither depends on nor changes the user's Ultralytics settings; YOLO_OFFLINE also switches the reporting off.
    ycfg = cache_dir() / "ultralytics"
    try:
        (ycfg / "Ultralytics").mkdir(parents=True, exist_ok=True)
        sf = ycfg / "Ultralytics" / "settings.json"
        if sf.exists():
            t = sf.read_text(encoding="utf-8")
            if '"sync": true' in t:
                sf.write_text(t.replace('"sync": true', '"sync": false'), encoding="utf-8")
        env["YOLO_CONFIG_DIR"] = str(ycfg)
    except OSError:
        pass
    pre = os.environ.get("ANIMALWORLD_LD_PRELOAD") or _runtime_file().get("ld_preload")
    if pre:
        env["LD_PRELOAD"] = pre
    env.update(extra or {})
    return env


def probe_role(role: str, timeout: float = 1200.0) -> dict:
    """Can the role's interpreter import what the step needs, and does it see a CUDA device? Runs a short subprocess."""
    py = python_for(role)
    mods = ROLE_MODULES[role]
    code = ("import importlib, json\nout = {'missing': []}\n"
            f"for m in {list(mods)!r}:\n"
            "    try: importlib.import_module(m)\n"
            "    except Exception as e: out['missing'].append(m + ' (' + type(e).__name__ + ')')\n"
            "try:\n"
            "    import torch; out['cuda'] = bool(torch.cuda.is_available()); out['torch'] = torch.__version__\n"
            "    if out['cuda']: out['gpu'] = torch.cuda.get_device_name(0); out['gpu_gb'] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1); out['cc'] = torch.cuda.get_device_capability(0)[0]\n"
            "except Exception: out['cuda'] = False\n"
            + ("try:\n    import sam3.model_builder as b; out['sam31'] = hasattr(b, 'build_sam3_multiplex_video_predictor')\nexcept Exception: out['sam31'] = False\n" if role == "sam3" else "")
            + "print('PROBE ' + json.dumps(out))\n")
    try:
        r = subprocess.run([py, "-c", code], capture_output=True, text=True, timeout=timeout, env=step_env(), stdin=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired) as e:
        why = (f"the model libraries did not finish loading within {timeout:.0f} s (a very slow disk?); try again" if isinstance(e, subprocess.TimeoutExpired)
               else f"the interpreter could not be started ({type(e).__name__})")
        return {"role": role, "python": py, "ok": False, "missing": [why], "cuda": False}
    import json
    line = next((ln for ln in r.stdout.splitlines() if ln.startswith("PROBE ")), None)
    d = json.loads(line[6:]) if line else {"missing": ["probe failed: " + (r.stderr.strip().splitlines() or ["?"])[-1][:200]], "cuda": False}
    if role == "sam3" and not d.get("sam31", True) and not d["missing"]:
        d["missing"].append("sam3 without SAM 3.1 video support (install it from github.com/facebookresearch/sam3)")
    d.update(role=role, python=py, ok=not d["missing"] and (d.get("cuda", False) or role == "cpu"))
    return d


def nvidia_smi() -> Optional[str]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        r = subprocess.run([exe, "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
        return r.stdout.strip().splitlines()[0] if r.returncode == 0 and r.stdout.strip() else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def sam3_available() -> tuple:
    """(usable, reason). SAM 3.1 is optional: its weights are gated by Meta's licence and the code is not on PyPI."""
    if not find("sam3.1-multiplex"):
        return False, _missing_message("sam3.1-multiplex")
    p = probe_role("sam3")
    if p["missing"]:
        return False, f"the SAM 3.1 code is not usable in {p['python']}: {', '.join(p['missing'])} (bash install.sh --gpu --sam3)"
    if not p.get("cuda"):
        return False, "no CUDA device visible for SAM 3.1"
    return True, ""


def choose_engine(want: str = "auto", tracker_detector: str = None, sam3: bool = False) -> dict:
    """-> {"engine": "full" | "lite" | "mock", "why": [...], "device": text, "weights": [...]}.

    full  needs a CUDA GPU (Ampere or newer), the runtimes and five weight files, none of them gated (with sam3=True also
          the SAM 3.1 code and its gated weights; if those are missing the request fails with the reason);
    lite  needs onnxruntime + opencv and the two ONNX files; it is the CPU fallback and tracks worse.
    """
    if want == "mock":
        return {"engine": "mock", "why": ["test engine: blob detector on a plain background, no model"], "device": "CPU", "weights": []}
    why: List[str] = []
    if want in ("auto", "full"):
        # one interpreter for several roles (the usual case) is asked once; a cold network disk can need minutes to load torch
        from concurrent.futures import ThreadPoolExecutor
        roles = [r for r in ROLES if r != "sam3" or sam3]
        with ThreadPoolExecutor(len(roles)) as ex:
            probes = list(ex.map(probe_role, roles))
        gpu = next((p for p in probes if p.get("cuda")), None)
        for p in probes:
            if p["missing"]:
                why.append(f"{p['role']} step ({p['python']}): cannot import {', '.join(p['missing'])}")
            elif p["role"] != "cpu" and not p.get("cuda"):
                why.append(f"{p['role']} step: no CUDA device visible")
        if gpu and gpu.get("cc", 8) < 8:
            why.append(f"{gpu.get('gpu')}: older than the Ampere generation; the SAM steps were only run on Ampere or newer cards")
        td = tracker_detector or os.environ.get("ANIMALWORLD_TRACKER_DETECTOR") or "scored_pair"
        need = FULL + (SCORED_PAIR if td == "scored_pair" else SCORED_PAIR[1:] if td == "strongest_two_seeds" else ()) + (SAM3 if sam3 else ())
        miss = [s for s in status(need) if not s["found"] and not s["downloadable"]]
        why += [s["how"] for s in miss]
        if not why:
            return {"engine": "full", "why": [], "device": f"{gpu.get('gpu')} ({gpu.get('gpu_gb')} GB)", "weights": status(need), "probes": probes, "tracker_detector": td, "sam3": bool(sam3)}
        if want == "full":
            raise WeightsError("the full engine cannot run here:\n  - " + "\n  - ".join(why))
    lite_missing = []
    for m in ("onnxruntime", "cv2"):
        try:
            __import__(m)
        except ImportError:
            lite_missing.append(m)
    miss = [s["how"] for s in status(LITE) if not s["found"] and not s["downloadable"]]
    if lite_missing or miss:
        raise WeightsError("no engine can run here:\n  - " + "\n  - ".join(
            why + ([f"CPU engine: pip install \"animalworld[run]\" (missing {', '.join(lite_missing)})"] if lite_missing else []) + miss))
    return {"engine": "lite", "why": why, "device": "CPU", "weights": status(LITE)}
