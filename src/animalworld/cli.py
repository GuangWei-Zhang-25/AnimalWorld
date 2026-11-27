"""Command line: `animalworld <command> ...` (also `python -m animalworld`)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import STAGES, __version__, credentials, registry
from .config import ConfigError, Project


def _print(obj) -> None:
    print(credentials.redact(json.dumps(obj, indent=1, ensure_ascii=False, default=str)))


def _stage_parser(sub, name: str, help_: str):
    p = sub.add_parser(name, help=help_)
    p.add_argument("project", help="project folder (contains project.yaml)")
    p.add_argument("--dry-run", action="store_true", help="print the plan / the exact call; run nothing")
    return p


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="animalworld",
        description="Species name in, deployable detection / pose / tracking models out. "
                    "Stages: " + " -> ".join(STAGES) + ". Every stage reads and writes the project folder and is resumable.")
    ap.add_argument("--version", action="version", version=f"animalworld {__version__}")
    sub = ap.add_subparsers(dest="cmd", metavar="command")

    p = sub.add_parser("init", help="create a project folder with project.yaml and slots.yaml")
    p.add_argument("project")
    p.add_argument("--species", required=True, help="species name, e.g. mouse")
    p.add_argument("--describe", default="", help='short description, e.g. "laboratory, home cage and open field"')
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("studio", help="Animal World Studio: the whole pipeline in one window (tkinter)")
    p.add_argument("project", nargs="?")
    sub.add_parser("zeroshot", help="the engine of Animal World Run without a window: VIDEOS... --out DIR")
    p = sub.add_parser("weights", help="where the pretrained model files are looked for, and which are present")
    p.add_argument("--verify", action="store_true", help="also check the SHA-256 of every file found (slow)")
    sub.add_parser("run", help="Animal World Run: pick mouse videos, press Start (tkinter). Same as `mouseworld`")
    sub.add_parser("demo", help="fetch the small demo clips and open Animal World Run with them (--headless --out DIR: no window)")
    p = sub.add_parser("pipeline", help="run all stages of a project in order (was `animalworld run PROJECT` before 0.2.0)")
    p.add_argument("project")
    p.add_argument("--dry-run", action="store_true", help="plan every stage, run nothing")
    p.add_argument("--auto-accept", action="store_true", help="accept every proposal that passes the gates (no human review)")
    p.add_argument("--train-dry-run", action="store_true", help="run everything but only print the training / export calls")
    p.add_argument("--force-train", action="store_true", help="train even below train.min_accepted_images")
    p.add_argument("--from", dest="start", choices=STAGES)
    p.add_argument("--until", choices=STAGES)

    p = sub.add_parser("status", help="counts per stage")
    p.add_argument("project")

    p = _stage_parser(sub, "literature", "search the literature for the species")
    p.add_argument("--backend", choices=registry.available("literature"))
    p.add_argument("--limit", type=int)
    p.add_argument("--query")
    p.add_argument("--force", action="store_true")
    _stage_parser(sub, "taxonomy", "keyword / taxonomy extraction from the abstracts")
    p = _stage_parser(sub, "prompts", "write diversified prompts")
    p.add_argument("--n", type=int)
    p.add_argument("--writer")
    p.add_argument("--force", action="store_true")
    p = _stage_parser(sub, "generate", "generate images / clips from the prompts")
    p.add_argument("--backend")
    p.add_argument("--limit", type=int)
    p = _stage_parser(sub, "propose", "proposal models label the images")
    p.add_argument("--backend")
    p.add_argument("--weights", help="weights for --backend model / pose: .pt, .onnx, or an RF-DETR .pth / .ckpt")
    p.add_argument("--limit", type=int)
    p.add_argument("--force", action="store_true", help="re-propose images that are not reviewed yet")
    p.add_argument("--include-reviewed", action="store_true")
    p = _stage_parser(sub, "review", "human-in-the-loop review (annotator window with --gui) or automatic acceptance")
    p.add_argument("--gui", action="store_true", help="open the annotator window (tkinter)")
    p.add_argument("--auto", action="store_true", default=None, help="accept every proposal that passes the gates")
    p = _stage_parser(sub, "dataset", "build the COCO + YOLO dataset from accepted labels")
    p.add_argument("--task", choices=("detect", "segment", "pose"))
    p = _stage_parser(sub, "train", "train a model (gated on the number of accepted images)")
    p.add_argument("--backend")
    p.add_argument("--task", choices=("detect", "segment", "pose"))
    p.add_argument("--force", action="store_true", help="train below the threshold (small model for the refine loop)")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a trainer option, e.g. --set epochs=1")
    p = _stage_parser(sub, "export", "export trained weights (ONNX)")
    p.add_argument("--weights")
    p.add_argument("--format", dest="fmt")
    p = _stage_parser(sub, "predict", "run detection / pose on a recording")
    p.add_argument("--source", help="video file or folder of images")
    p.add_argument("--weights")
    p.add_argument("--backend")
    p.add_argument("--name")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--conf", type=float)
    p = _stage_parser(sub, "track", "identity tracking on predictions")
    p.add_argument("--name")
    p.add_argument("--backend")
    p.add_argument("--n-animals", type=int)

    p = sub.add_parser("import-images", help="register real images for manual annotation")
    p.add_argument("project")
    p.add_argument("folder")
    p = sub.add_parser("import-labels", help="import labels from COCO / YOLO / DeepLabCut / SLEAP")
    p.add_argument("project")
    p.add_argument("--format", dest="fmt", required=True, choices=("coco", "yolo", "dlc", "sleap"))
    p.add_argument("path", help="COCO json | folder of YOLO txt | CollectedData_*.csv | .slp or COCO json")
    p.add_argument("--images", help="folder with the images (default: next to the labels)")
    p.add_argument("--status", choices=("accepted", "proposed"), default="accepted",
                   help="accepted (default): trusted as they are, origin 'import'; proposed: they wait for review")
    p = sub.add_parser("export-labels", help="export accepted labels to COCO / YOLO / DeepLabCut / SLEAP")
    p.add_argument("project")
    p.add_argument("--format", dest="fmt", required=True, choices=("coco", "yolo", "dlc", "sleap"))
    p.add_argument("--task", choices=("detect", "segment", "pose"), help="YOLO only (default: dataset.task)")
    p.add_argument("--out")
    p = sub.add_parser("demo-recording", help="write a small synthetic recording (frames) to try predict / track")
    p.add_argument("out")
    p.add_argument("--frames", type=int, default=40)
    p.add_argument("--n", type=int, default=2)
    p.add_argument("--seed", type=int, default=1)

    p = sub.add_parser("sheet", help="contact sheet of proposals / reviewed labels / predictions, to look at")
    p.add_argument("project")
    p.add_argument("--what", required=True, choices=("proposals", "reviewed", "predictions"))
    p.add_argument("--name", help="prediction run name (for --what predictions)")
    p.add_argument("--tracks", action="store_true", help="draw identities from `track`")
    p.add_argument("--out")
    p.add_argument("--limit", type=int, default=48)
    p.add_argument("--cols", type=int, default=6)
    p.add_argument("--every", type=int, default=1, help="use every N-th frame")

    p = sub.add_parser("backends", help="list the registered backends")
    p = sub.add_parser("credentials", help="store API tokens outside the project (names only are ever shown)")
    cs = p.add_subparsers(dest="ccmd", metavar="action")
    c = cs.add_parser("set", help="store a token (asked without echo, or read from stdin)")
    c.add_argument("name")
    c.add_argument("--stdin", action="store_true")
    cs.add_parser("list", help="list credential names (never values)")
    c = cs.add_parser("unset")
    c.add_argument("name")
    cs.add_parser("path", help="where the credentials file lives")
    return ap


def _coerce(v: str):
    try:
        return json.loads(v)
    except ValueError:
        return v


def _demo_recording(out: str, frames: int, n: int, seed: int) -> dict:
    from .backends.generators.mock import render, simulate
    o = Path(out)
    o.mkdir(parents=True, exist_ok=True)
    W, H = 320, 240
    apps = (["black", "white", "brown", "grey"] * 3)[:n]
    truth = []
    for f, animals in enumerate(simulate(n, frames, W, H, seed)):
        img, inst = render(W, H, animals, apps, [], seed, f)
        img.save(o / f"frame_{f:05d}.jpg", quality=92)
        truth.append({"frame": f, "instances": [{"box": i["box"], "track_id": k} for k, i in enumerate(inst)]})
    (o / "truth.json").write_text(json.dumps(truth), encoding="utf-8")
    return {"frames": frames, "animals": n, "folder": str(o), "note": "synthetic shapes with known identities (truth.json)"}


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # The windows and the zero-shot engine have their own argument parsers.
    if argv[:1] == ["zeroshot"]:
        from .zeroshot.cli import main as zmain
        return zmain(argv[1:])
    if argv[:1] == ["demo"]:
        from .zeroshot.demo import main as dmain
        return dmain(argv[1:])
    if argv[:1] == ["studio"]:
        from .studio.app import main as smain
        return smain(argv[1:])
    if argv[:1] == ["run"]:
        pos = [x for x in argv[1:] if not x.startswith("-")]
        if pos and (Path(pos[0]) / "project.yaml").exists():         # the pre-0.2.0 meaning: `animalworld run PROJECT`
            print("animalworld: `animalworld run PROJECT` is now `animalworld pipeline PROJECT`; running that.", file=sys.stderr)
            argv = ["pipeline"] + argv[1:]
        else:
            from .zeroshot.app import main as amain
            return amain(argv[1:])
    if argv[:1] == ["review"] and "--gui" in argv:
        from .annotator.tkapp import main as gmain
        return gmain([x for x in argv[1:] if x != "--gui"])
    ap = build_parser()
    a = ap.parse_args(argv)
    if not a.cmd:
        ap.print_help()
        return 0
    try:
        if a.cmd == "init":
            pr = Project.init(a.project, a.species, a.describe, force=a.force)
            _print({"created": str(pr.root), "species": pr.species, "config": str(pr.root / "project.yaml"),
                    "slots": str(pr.root / "slots.yaml"), "next": f"animalworld pipeline {pr.root} --dry-run"})
            return 0
        if a.cmd == "weights":
            from . import weights as W
            print("looked for in: " + " , ".join(str(d) for d in W.search_dirs()))
            print("download address of the MouseWorld weights: " + (W.base_url({"own": True}) or "(not set: BASE_URL is still the placeholder)"))
            for r in W.status(verify=a.verify):
                ok = "found  " if r["found"] else "MISSING"
                chk = "" if "sha256_ok" not in r else ("  sha256 ok" if r["sha256_ok"] else "  SHA256 MISMATCH")
                print(f"  {ok} {r['title']:<42} {r['path'] or r['file']}{chk}")
            return 0
        if a.cmd == "backends":
            _print({k: registry.available(k) for k in registry.KINDS})
            return 0
        if a.cmd == "credentials":
            if a.ccmd == "set":
                if a.stdin:
                    value = sys.stdin.readline().rstrip("\n")
                else:
                    import getpass
                    value = getpass.getpass(f"value for {a.name} (input hidden): ")
                path = credentials.set_value(a.name, value)
                print(f"stored {a.name} in {path} (mode 600)")
            elif a.ccmd == "list":
                rows = credentials.names()
                print("\n".join(f"{r['name']}\t({r['source']})" for r in rows) or "no credentials stored")
            elif a.ccmd == "unset":
                print("removed" if credentials.unset(a.name) else "not found")
            elif a.ccmd == "path":
                print(credentials.credentials_path())
            else:
                ap.parse_args(["credentials", "--help"])
            return 0
        if a.cmd == "demo-recording":
            _print(_demo_recording(a.out, a.frames, a.n, a.seed))
            return 0
        pr = Project.load(a.project)
        from . import pipeline
        if a.cmd == "status":
            _print(pipeline.status(pr))
        elif a.cmd == "pipeline":
            def echo(name, res):
                print(f"=== {name}: {res.get('mode')}", flush=True)
                _print({k: v for k, v in res.items() if k not in ("stage", "mode")})
            pipeline.run_all(pr, dry_run=a.dry_run, auto_accept=True if a.auto_accept else None,
                             train_dry_run=a.train_dry_run, force_train=a.force_train, until=a.until, start=a.start, echo=echo)
        elif a.cmd == "sheet":
            from . import sheets
            _print(sheets.make(pr, a.what, a.name, Path(a.out) if a.out else None, a.limit, a.cols, a.tracks, a.every))
        elif a.cmd == "import-images":
            from .stages import review
            _print(review.add_images(pr, Path(a.folder)))
        elif a.cmd == "import-labels":
            from .stages import review
            _print(review.import_labels(pr, a.fmt, Path(a.path), Path(a.images) if a.images else None, status=a.status))
        elif a.cmd == "export-labels":
            from .stages import review
            _print(review.export_labels(pr, a.fmt, Path(a.out) if a.out else None, task=a.task))
        elif a.cmd in STAGES:
            kw = {k: v for k, v in vars(a).items() if k not in ("cmd", "project", "set") and v is not None}
            if a.cmd == "train" and a.set:
                kw["overrides"] = {s.split("=", 1)[0]: _coerce(s.split("=", 1)[1]) for s in a.set}
            _print(pipeline.run_stage(pr, a.cmd, **kw))
        else:
            ap.error(f"unknown command {a.cmd}")
    except (ConfigError, registry.BackendError, credentials.CredentialError, FileNotFoundError, RuntimeError, ValueError) as e:
        print(credentials.redact(f"animalworld: error: {e}"), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
