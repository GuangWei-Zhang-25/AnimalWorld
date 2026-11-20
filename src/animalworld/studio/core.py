"""Animal World Studio, GUI-independent core: form validation, saving project.yaml, building the stage commands,
and the job runner. The tkinter window (studio/app.py) is a thin layer over this module.

  * A Run button starts the SAME command the CLI runs: [sys.executable, "-m", "animalworld", <stage>, <project>, ...]
    as an argument list (never a shell string). Only the stages and flags in `build_commands` can be started and every
    form value is validated first. Output goes to <project>/logs/studio_<stage>_<utc>.log; the window tails that file.
  * API keys are stored with `credentials.set_value` (file mode 0600, outside the project) and never returned, logged
    or written to project.yaml (the config only holds `token_credential: NAME`).
"""
from __future__ import annotations

import copy
import json
import os
import re
import secrets
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .. import __version__, credentials, pipeline, registry
from ..config import CONFIG_NAME, ConfigError, Project, deep_merge

HEADER = ("# animalworld project. Edit freely; every stage re-reads this file.\n"
          "# Never put API tokens here: use `animalworld credentials set NAME`.\n")
TASKS = ("detect", "segment", "pose")
LABEL_FORMATS_IN = ("coco", "yolo", "dlc", "sleap")
LABEL_FORMATS_OUT = ("coco", "dlc", "sleap")
ENGINES = ("auto", "full", "lite", "mock")
_CRED = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")

# Worked example: Qwen-Image 2.1 exactly as run for real on 18 Nov 2026: a local diffusers folder, class
# QwenImage21Pipeline, 30 steps, true CFG 4.0, bfloat16, split path, 1.25 MP, sizes in multiples of 32.
# The folder is looked for in the cache (<cache>/models/Qwen-Image-2.1); if it is not there the backend loads the same
# model by its official Hugging Face id (a download of about 31 GB at first use; that route is untested here).
QWEN_OFFICIAL = "https://huggingface.co/Qwen/Qwen-Image-2.1"
QWEN_LOCAL_CLASS = "QwenImage21Pipeline"


def qwen_default_path() -> str:
    from ..weights import cache_dir
    return str(cache_dir() / "models" / "Qwen-Image-2.1")


QWEN_EXAMPLE = {"model_path": qwen_default_path(), "model_id": "Qwen/Qwen-Image-2.1", "pipeline_class": QWEN_LOCAL_CLASS,
                "num_inference_steps": 30, "true_cfg_scale": 4.0, "dtype": "bfloat16", "split": True}
API_STYLES = {
    "openai_images": {"body": {"prompt": "{prompt}", "size": "{width}x{height}", "n": 1},
                      "response": {"b64_path": "data.0.b64_json"}},
    "generic_url": {"body": {"prompt": "{prompt}", "negative_prompt": "{negative_prompt}", "width": "{width}",
                             "height": "{height}", "seed": "{seed}"},
                    "response": {"url_path": "images.0.url"}},
}


class FormError(ValueError):
    """A form value that was refused; the message is shown next to the form."""


# ---------------------------------------------------------------- validation ----------------------------------------
def _int(v, name, lo, hi, optional=False):
    if v in (None, "") and optional:
        return None
    try:
        if isinstance(v, bool) or (isinstance(v, float) and v != int(v)):
            raise ValueError
        n = int(str(v).strip()) if not isinstance(v, (int, float)) else int(v)
    except (TypeError, ValueError):
        raise FormError(f"{name}: a whole number is needed") from None
    if not lo <= n <= hi:
        raise FormError(f"{name}: must be between {lo} and {hi}")
    return n


def _float(v, name, lo, hi):
    try:
        if isinstance(v, bool):
            raise ValueError
        x = float(v)
    except (TypeError, ValueError):
        raise FormError(f"{name}: a number is needed") from None
    if not (lo <= x <= hi):
        raise FormError(f"{name}: must be between {lo} and {hi}")
    return x


def _choice(v, name, allowed):
    if v not in allowed:
        raise FormError(f"{name}: choose one of {', '.join(map(str, allowed))}")
    return v


def _text(v, name, maxlen=400, optional=True):
    s = "" if v is None else str(v).strip()
    if not s and not optional:
        raise FormError(f"{name}: is empty")
    if len(s) > maxlen or any(ord(c) < 32 for c in s):
        raise FormError(f"{name}: too long or contains control characters")
    return s


def _path(v, name, optional=False, must_exist=False):
    """A path that will be ONE argv item / one YAML string. Never a flag, never multi-line."""
    s = "" if v is None else str(v).strip()
    if not s:
        if optional:
            return ""
        raise FormError(f"{name}: is empty")
    if len(s) > 2000 or any(ord(c) < 32 for c in s):
        raise FormError(f"{name}: not a valid path")
    if s.startswith("-"):
        raise FormError(f"{name}: a path may not start with '-'")
    s = os.path.expanduser(s)
    if must_exist and not os.path.exists(s):
        raise FormError(f"{name}: {s!r} does not exist on this computer")
    return s


def _bool(v):
    return v in (True, 1, "1", "true", "on", "yes")


# ---------------------------------------------------------------- config ---------------------------------------------
def read_user_config(project: Project) -> dict:
    return yaml.safe_load((project.root / CONFIG_NAME).read_text(encoding="utf-8")) or {}


def _set(d: dict, path: str, value, replace=False):
    keys = path.split(".")
    for k in keys[:-1]:
        if not isinstance(d.get(k), dict):
            d[k] = {}
        d = d[k]
    if isinstance(value, dict) and isinstance(d.get(keys[-1]), dict) and not replace:
        d[keys[-1]] = deep_merge(d[keys[-1]], value)
    else:
        d[keys[-1]] = value


def save_config(project: Project, sets: List[Tuple[str, Any, bool]]) -> Project:
    """The one place Studio writes project.yaml: (dotted path, value, replace) triples are merged into the user's YAML.
    Anything that looks like a secret is refused, so a key can never land in the project folder."""
    raw = read_user_config(project)
    new = copy.deepcopy(raw)
    for path, value, replace in sets:
        _set(new, path, value, replace)
    bad = credentials.find_inline_secrets(new)
    if bad:
        raise FormError("refused: project.yaml must not contain secrets (" + ", ".join(bad) + "). "
                        "Use the API key field; only the credential NAME is stored in the project.")
    if new != raw:
        text = HEADER + yaml.safe_dump(new, sort_keys=False, allow_unicode=True)
        tmp = project.root / (CONFIG_NAME + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, project.root / CONFIG_NAME)
    return Project.load(project.root)


def stage_values(stage: str, v: dict, project: Project) -> List[Tuple[str, Any, bool]]:
    """Validated form values of one stage -> config writes. Unknown stages / fields are ignored or refused here."""
    v = v or {}
    if stage == "project":
        return [("description", _text(v.get("description"), "description", 600), False)]
    if stage == "literature":
        return [("literature.backend", _choice(v.get("backend"), "backend", registry.available("literature")), False),
                ("literature.limit", _int(v.get("limit"), "limit", 1, 1000000), False)]
    if stage == "prompts":
        return [("prompts.n", _int(v.get("n"), "target number of images", 1, 10000000), False),
                ("prompts.seed", _int(v.get("seed"), "seed", 0, 2 ** 31 - 1), False),
                ("prompts.megapixels", _float(v.get("megapixels", 1.25), "megapixels", 0.05, 16.0), False),
                ("prompts.size_multiple", _int(v.get("size_multiple", 32), "size multiple", 1, 256), False)]
    if stage == "generate":
        b = _choice(v.get("backend"), "backend", registry.available("generators"))
        opts: Dict[str, Any] = {}
        if b == "qwen_image":
            mp, mid = _path(v.get("model_path"), "weights folder", optional=True), _text(v.get("model_id"), "Hugging Face id", 200)
            if mid and not re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9._-]+$", mid):
                raise FormError("Hugging Face id: expected the form owner/name, e.g. Qwen/Qwen-Image")
            if not mp and not mid:
                raise FormError("open weights: give a local weights folder or a Hugging Face id")
            opts = {"model_path": mp or None, "model_id": mid or None,
                    "pipeline_class": _text(v.get("pipeline_class") or QWEN_EXAMPLE["pipeline_class"], "pipeline class", 80),
                    "num_inference_steps": _int(v.get("num_inference_steps", 30), "steps", 1, 500),
                    "true_cfg_scale": _float(v.get("true_cfg_scale", 4.0), "CFG scale", 0.0, 50.0),
                    "dtype": _choice(v.get("dtype") or "bfloat16", "dtype", ("bfloat16", "float16", "float32")),
                    "split": _bool(v.get("split", True)), "compile": False}
            if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", opts["pipeline_class"]):
                raise FormError("pipeline class: a Python class name is needed")
        elif b == "http_api":
            ep = _text(v.get("endpoint"), "endpoint", 500, optional=False)
            if not re.match(r"^https?://[^\s]+$", ep):
                raise FormError("endpoint: must start with http:// or https://")
            style = _choice(v.get("style") or "openai_images", "request style", tuple(API_STYLES))
            body = copy.deepcopy(API_STYLES[style]["body"])
            model = _text(v.get("model"), "model name", 200)
            if model:
                body = dict({"model": model}, **body)
            cred = _text(v.get("credential"), "credential name", 64)
            if cred and not _CRED.match(cred):
                raise FormError("credential name: letters, digits and underscores, starting with a letter")
            opts = {"endpoint": ep, "token_credential": cred or None, "body": body,
                    "response": copy.deepcopy(API_STYLES[style]["response"]), "style": style}
        elif b == "wan_video":
            opts = {"model_path": _path(v.get("wan_model_path"), "Wan weights folder", optional=True) or None}
        return [("generate.backend", b, False), ("generate.options", opts, True)]
    if stage == "propose":
        b = _choice(v.get("backend"), "backend", [x for x in registry.available("proposers") if x != "pose"])
        opts = {}
        if b == "sam":
            opts = {"variant": _choice(v.get("variant") or "sam2", "SAM variant", ("sam3", "sam2")),
                    "prompt": _text(v.get("prompt"), "text prompt", 80) or None}
        elif b == "yolo":
            opts = {"weights": _path(v.get("weights"), "detector weights", optional=True) or None}
        out = [("propose.backend", b, False), ("propose.options", opts, True)]
        pw = _path(v.get("pose_weights"), "keypoint model weights", optional=True)
        if pw:
            out.append(("propose.options_for.pose", {"weights": pw}, True))
        return out
    if stage == "dataset":
        return [("dataset.task", _choice(v.get("task"), "task", TASKS), False)]
    if stage == "train":
        return [("train.backend", _choice(v.get("backend"), "backend", registry.available("trainers")), False),
                ("train.task", _choice(v.get("task"), "task", TASKS), False),
                ("train.min_accepted_images", _int(v.get("min_accepted_images"), "image threshold", 1, 100000000), False)]
    if stage in ("review", "export", "deploy"):
        return []
    raise FormError(f"unknown stage {stage!r}")


def form_values(project: Project) -> dict:
    """What the forms show: the merged configuration, with the Qwen-Image worked example as the generation default."""
    c = project.cfg
    g = c["generate"]
    go = g.get("options") or {}
    q = dict(QWEN_EXAMPLE)
    if g["backend"] == "qwen_image":
        q.update({k: go[k] for k in q if go.get(k) is not None})
        q["model_path"] = go.get("model_path") or ""
        q["model_id"] = go.get("model_id") or ""
    api = {"endpoint": "", "model": "", "credential": "", "style": "openai_images"}
    if g["backend"] == "http_api":
        api = {"endpoint": go.get("endpoint") or "", "model": (go.get("body") or {}).get("model", ""),
               "credential": go.get("token_credential") or "", "style": go.get("style") or "openai_images"}
    po = c["propose"].get("options") or {}
    return {
        "project": {"species": project.species, "description": c.get("description", "")},
        "literature": {"backend": c["literature"]["backend"], "limit": c["literature"]["limit"]},
        "prompts": {k: c["prompts"][k] for k in ("n", "seed", "megapixels", "size_multiple")},
        "generate": dict(q, backend=g["backend"], **api,
                         wan_model_path=(go.get("model_path") or "") if g["backend"] == "wan_video" else ""),
        "propose": {"backend": c["propose"]["backend"], "variant": po.get("variant") or "sam2",
                    "prompt": po.get("prompt") or "", "weights": po.get("weights") or "",
                    "pose_weights": ((c["propose"].get("options_for") or {}).get("pose") or {}).get("weights") or ""},
        "dataset": {"task": c["dataset"]["task"]},
        "train": {"backend": c["train"]["backend"], "task": c["train"]["task"],
                  "min_accepted_images": c["train"]["min_accepted_images"]},
        "deploy": {"n_animals": c["track"].get("n_animals") or ""},
    }


# ---------------------------------------------------------------- commands -------------------------------------------
def build_commands(action: str, params: dict, project: Project, python: str = None) -> List[List[str]]:
    """The whitelisted commands of one Run button, as argument lists. Values arrive validated or are validated here;
    a form value can only ever become ONE argv item."""
    p = params or {}
    base = [python or sys.executable, "-m", "animalworld"]
    P = str(project.root)

    def cmd(*a):
        return base + [str(x) for x in a]
    if action == "literature":
        return [cmd("literature", P) + (["--force"] if _bool(p.get("force")) else [])]
    if action == "prompts":
        return [cmd("taxonomy", P), cmd("prompts", P) + (["--force"] if _bool(p.get("force")) else [])]
    if action == "generate":
        lim = _int(p.get("limit"), "limit", 1, 10000000, optional=True)
        return [cmd("generate", P) + (["--limit", str(lim)] if lim else [])]
    if action == "propose":
        return [cmd("propose", P) + (["--force"] if _bool(p.get("force")) else [])]
    if action == "propose_keypoints":
        w = _path(p.get("pose_weights"), "keypoint model weights", must_exist=True)
        return [cmd("propose", P, "--backend", "pose", "--weights", w, "--force")]
    if action == "review_auto":
        return [cmd("review", P, "--auto")]
    if action == "import_labels":
        fmt = _choice(p.get("format"), "format", LABEL_FORMATS_IN)
        c = cmd("import-labels", P, "--format", fmt, _path(p.get("path"), "label file or folder", must_exist=True))
        img = _path(p.get("images"), "image folder", optional=True, must_exist=True)
        return [c + (["--images", img] if img else [])]
    if action == "export_labels":
        fmt = _choice(p.get("format"), "format", LABEL_FORMATS_OUT)
        out = _path(p.get("out"), "output folder", optional=True)
        return [cmd("export-labels", P, "--format", fmt) + (["--out", out] if out else [])]
    if action == "dataset":
        return [cmd("dataset", P)]
    if action == "train":
        c = cmd("train", P)
        if _bool(p.get("force")):
            c.append("--force")
        if _bool(p.get("dry_run")):
            c.append("--dry-run")
        for key, lo, hi in (("epochs", 1, 100000), ("imgsz", 32, 4096)):
            n = _int(p.get(key), key, lo, hi, optional=True)
            if n:
                c += ["--set", f"{key}={n}"]
        return [c]
    if action == "export":
        return [cmd("export", P)]
    if action in ("deploy_pretrained", "deploy_project"):
        vids = p.get("videos")
        vids = [vids] if isinstance(vids, str) else list(vids or [])
        vids = [_path(x, "video or folder", must_exist=True) for x in vids if str(x).strip()]
        if not vids:
            raise FormError("video or folder: is empty")
        n = _int(p.get("n_animals"), "number of animals", 1, 100, optional=True)
        if action == "deploy_pretrained":
            out = _path(p.get("out"), "output folder", optional=True) or str(project.path("zeroshot"))
            eng = _choice(p.get("engine") or "auto", "engine", ENGINES)
            return [base + ["zeroshot"] + vids + ["--out", out, "--yes"] + (["--n-animals", str(n)] if n else [])
                    + ["--engine", eng]]
        if len(vids) != 1:
            raise FormError("this project's model runs on one video or one image folder at a time")
        return [cmd("predict", P, "--source", vids[0]), cmd("track", P) + (["--n-animals", str(n)] if n else []),
                cmd("sheet", P, "--what", "predictions", "--tracks", "--every", "30", "--limit", "24")]   # a picture of the result to look at
    raise FormError(f"unknown action {action!r}")


SAVE_BEFORE = {"literature": "literature", "prompts": "prompts", "generate": "generate", "propose": "propose",
               "propose_keypoints": "propose", "dataset": "dataset", "train": "train"}


def job_env() -> dict:
    env = dict(os.environ, ORT_DISABLE_TELEMETRY="1", YOLO_AUTOINSTALL="false", PYTHONUNBUFFERED="1")
    src = str(Path(__file__).resolve().parents[2])                  # so `-m animalworld` is THIS copy (source tree or site-packages)
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def run_commands(cmds: List[List[str]], cwd, out, stopped=lambda: False, on_proc=lambda pr: None) -> int:
    """Run argument lists one after another (never through a shell), writing `$ command`, the output and `[exit N]`
    to the binary stream `out`. Stops at the first failure. Used by the in-window job and by the SLURM wrapper."""
    rc = 0
    for argv in cmds:
        if stopped():
            break
        shown = "animalworld " + " ".join(json.dumps(a) if re.search(r"[^\w./:=,+-]", a) else a for a in argv[3:])
        out.write(credentials.redact(f"$ {shown}\n").encode())
        out.flush()
        kw = {"start_new_session": True} if os.name != "nt" else {}
        pr = subprocess.Popen(argv, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=job_env(),
                              cwd=str(cwd), shell=False, **kw)
        on_proc(pr)
        rc = pr.wait()
        out.write(f"[exit {rc}]\n".encode())
        out.flush()
        if rc != 0:
            break
    return rc


# ---------------------------------------------------------------- jobs -----------------------------------------------
class Job:
    def __init__(self, jid: str, action: str, cmds: List[List[str]], log: Path, cwd: Path):
        self.id, self.action, self.cmds, self.log, self.cwd = jid, action, cmds, log, cwd
        self.status, self.returncode, self.proc, self.started, self.ended = "running", None, None, time.time(), None
        self._stop = False

    def run(self):
        rc = 0
        try:
            with open(self.log, "ab", buffering=0) as f:
                rc = run_commands(self.cmds, self.cwd, f, lambda: self._stop, lambda pr: setattr(self, "proc", pr))
        except OSError as e:
            rc = 127
            with open(self.log, "ab") as f:
                f.write(f"[could not start: {type(e).__name__}: {e}]\n".encode())
        self.returncode, self.ended = rc, time.time()
        self.status = "stopped" if self._stop else ("done" if rc == 0 else "failed")

    def stop(self):
        self._stop = True
        pr = self.proc
        if pr and pr.poll() is None:
            try:
                os.killpg(pr.pid, signal.SIGTERM) if os.name != "nt" else pr.terminate()
            except (OSError, ProcessLookupError):
                pass

    def info(self, offset: int = 0) -> dict:
        text, size = "", 0
        try:
            size = self.log.stat().st_size
            offset = max(0, min(int(offset), size))
            with open(self.log, "rb") as f:
                f.seek(offset)
                text = f.read(200000).decode("utf-8", "replace")
        except OSError:
            offset = 0
        return {"id": self.id, "action": self.action, "status": self.status, "returncode": self.returncode,
                "log": str(self.log), "text": credentials.redact(text), "offset": offset + len(text.encode("utf-8", "replace")),
                "seconds": round((self.ended or time.time()) - self.started, 1)}


# ---------------------------------------------------------------- SLURM (optional) -----------------------------------
SLURM_DEFAULTS = {"partition": "gpu", "gres": "gpu:1", "cpus": 4, "mem": "32G", "time": "02:00:00"}
_SLURM_RE = {"partition": r"^[A-Za-z0-9][A-Za-z0-9_-]*(,[A-Za-z0-9][A-Za-z0-9_-]*)*$", "gres": r"^([A-Za-z0-9_]+(:[A-Za-z0-9_.-]+){0,2})?$",
             "mem": r"^[0-9]{1,6}[MG]$", "time": r"^([0-9]{1,2}-)?[0-9]{1,3}:[0-5][0-9]:[0-5][0-9]$"}
SLURM_WRAPPER = Path(__file__).with_name("slurm_wrapper.sh")


def slurm_options(v: dict) -> dict:
    """Validated sbatch resources. Each value must match a strict pattern; nothing else can reach the sbatch call."""
    v = v or {}
    out = {}
    for k, pat in _SLURM_RE.items():
        s = str(v.get(k, SLURM_DEFAULTS[k]) or "").strip()
        if len(s) > 80 or not re.match(pat, s) or (not s and k != "gres"):
            raise FormError(f"SLURM {k}: {s!r} is not valid (example: {SLURM_DEFAULTS[k]})")
        out[k] = s
    out["cpus"] = _int(v.get("cpus", SLURM_DEFAULTS["cpus"]), "SLURM CPUs", 1, 512)
    return out


def slurm_command(opts: dict, log: Path, job_json: Path, name: str, python: str = None) -> List[str]:
    """The sbatch call as an argument list. The script is a fixed two-line wrapper shipped with the package; the stage
    commands live in job_json as argument lists. No form value is ever placed inside shell text."""
    o = slurm_options(opts)
    argv = ["sbatch", "--parsable", f"--job-name=aw_{re.sub(r'[^A-Za-z0-9_]', '_', name)[:40]}", f"--partition={o['partition']}",
            f"--cpus-per-task={o['cpus']}", f"--mem={o['mem']}", f"--time={o['time']}", f"--output={log}"]
    if o["gres"]:
        argv.append(f"--gres={o['gres']}")
    return argv + [str(SLURM_WRAPPER), python or sys.executable, str(job_json)]


class SlurmJob(Job):
    """The same commands, submitted with sbatch; the batch job writes the same log file, which the window tails.
    UNTESTED on a real queue (the tests check only the sbatch argument list and the wrapper run locally)."""

    def __init__(self, jid, action, cmds, log, cwd, opts: dict):
        super().__init__(jid, action, cmds, log, cwd)
        self.opts, self.slurm_id = slurm_options(opts), None
        self.job_json = Path(log).with_suffix(".job.json")

    def submit_argv(self) -> List[str]:
        return slurm_command(self.opts, self.log, self.job_json, self.id)

    def run(self):
        rc = 1
        try:
            self.job_json.write_text(json.dumps({"cmds": self.cmds, "cwd": str(self.cwd)}), encoding="utf-8")
            p = subprocess.run(self.submit_argv(), capture_output=True, text=True, stdin=subprocess.DEVNULL, shell=False, cwd=str(self.cwd), env=job_env())
            if p.returncode != 0 or not re.match(r"^\d+", p.stdout.strip()):
                raise OSError((p.stderr or p.stdout or "sbatch failed").strip()[:500])
            self.slurm_id = re.match(r"^\d+", p.stdout.strip()).group(0)
            with open(self.log, "ab") as f:
                f.write(f"[submitted SLURM job {self.slurm_id}; waiting for it to start]\n".encode())
            while not self._stop:
                time.sleep(5)
                q = subprocess.run(["squeue", "-h", "-j", self.slurm_id, "-o", "%T"], capture_output=True, text=True,
                                   stdin=subprocess.DEVNULL, shell=False)
                if q.returncode != 0 or not q.stdout.strip():
                    break
            m = re.findall(r"^\[exit (-?\d+)\]$", self.log.read_text(errors="replace"), flags=re.M)
            rc = int(m[-1]) if m else 1
        except OSError as e:
            with open(self.log, "ab") as f:
                f.write(f"[SLURM: {type(e).__name__}: {e}]\n".encode())
        self.returncode, self.ended = rc, time.time()
        self.status = "stopped" if self._stop else ("done" if rc == 0 else "failed")

    def stop(self):
        self._stop = True
        if self.slurm_id:
            try:
                subprocess.run(["scancel", self.slurm_id], stdin=subprocess.DEVNULL, shell=False, capture_output=True)
            except OSError:
                pass


class Studio:
    def __init__(self, project_root=None):
        self.project: Optional[Project] = Project.load(project_root) if project_root else None
        self.jobs: Dict[str, Job] = {}
        self.current: Optional[Job] = None
        self.lock = threading.Lock()

    # ---- project ----
    def open_project(self, body: dict) -> dict:
        path = Path(_path(body.get("path"), "project folder")).resolve()
        if (path / CONFIG_NAME).exists():
            self.project = Project.load(path)
            return {"opened": str(path), "created": False}
        species = _text(body.get("species"), "species", 80, optional=False)
        if path.exists() and not path.is_dir():
            raise FormError("project folder: this is a file")
        credentials._check_location(path / "x")                      # a project may not live where credentials are kept
        self.project = Project.init(path, species, _text(body.get("description"), "description", 600))
        return {"opened": str(path), "created": True}

    def need_project(self) -> Project:
        if not self.project:
            raise FormError("create or open a project first")
        self.project = Project.load(self.project.root)               # every request sees the file as it is now
        return self.project

    # ---- state ----
    def state(self) -> dict:
        out = {"version": __version__, "project": None, "job": self.current.info(1 << 60) if self.current else None,
               "credentials": [r["name"] for r in credentials.names()],
               "backends": {k: registry.available(k) for k in ("literature", "generators", "proposers", "trainers")},
               "defaults": {"species": "mouse", "description": "laboratory mouse, home cage and open field, top and oblique cameras",
                            "qwen": QWEN_EXAMPLE}}
        if self.current:
            out["job"].pop("text", None)
        if self.project:
            try:
                pr = self.need_project()
                out.update(project=str(pr.root), species=pr.species, status=pipeline.status(pr), values=form_values(pr),
                           keypoints=pr.keypoints())
            except (ConfigError, OSError, yaml.YAMLError, KeyError) as e:
                out.update(project=str(self.project.root), error=credentials.redact(str(e)))
        return out

    # ---- actions ----
    def save(self, stage: str, values: dict) -> dict:
        pr = self.need_project()
        self.project = save_config(pr, stage_values(stage, values, pr))
        return {"saved": stage}

    def set_credential(self, body: dict) -> dict:
        name = _text(body.get("name"), "credential name", 64, optional=False)
        if not _CRED.match(name):
            raise FormError("credential name: letters, digits and underscores, starting with a letter")
        value = body.get("value")
        if not isinstance(value, str) or not value.strip() or len(value) > 4000 or "\n" in value.strip():
            raise FormError("API key: is empty or not a single line")
        credentials.set_value(name, value.strip())
        credentials._LOADED_VALUES.add(value.strip())               # redact it from anything shown from now on
        return {"stored": name, "where": "credentials file outside the project (mode 600)"}

    def run(self, action: str, values: dict, params: dict, slurm: dict = None) -> dict:
        pr = self.need_project()
        with self.lock:
            if self.current and self.current.status == "running":
                raise FormError(f"'{self.current.action}' is still running: wait for it or press Stop")
            if action in SAVE_BEFORE and values is not None:
                pr = self.project = save_config(pr, stage_values(SAVE_BEFORE[action], values, pr))
            cmds = build_commands(action, params, pr)
            logs = pr.path("logs")
            logs.mkdir(parents=True, exist_ok=True)
            jid = f"{action}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{secrets.token_hex(2)}"
            log = logs / f"studio_{jid}.log"
            job = SlurmJob(jid, action, cmds, log, pr.root, slurm) if slurm else Job(jid, action, cmds, log, pr.root)
            self.jobs[jid] = self.current = job
            threading.Thread(target=job.run, daemon=True).start()
        return {"job": jid, "commands": len(cmds)}


