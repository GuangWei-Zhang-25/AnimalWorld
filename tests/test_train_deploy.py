import json
import math
import random
import itertools

import numpy as np
import pytest
import yaml
from PIL import Image

from animalworld import pipeline, registry
from animalworld.backends.deploy.base import nms
from animalworld.backends.deploy.onnx import decode_yolo, letterbox
from animalworld.backends.generators import looks
from animalworld.backends.generators.mock import render, simulate
from animalworld.backends.trackers.iou import IoUTracker, estimate_n, hungarian
from animalworld.config import Project
from animalworld.formats import coco as C
from animalworld.formats import labels as L
from animalworld.utils import read_json, read_jsonl


def test_dataset_layout_coco_and_yolo(reviewed_project):
    pr = reviewed_project
    D = pr.path("dataset")
    y = yaml.safe_load((D / "data.yaml").read_text())
    assert y["names"] == {0: "mouse"} and y["train"] == "train/images"
    imgs = {s: sorted(p.stem for p in (D / s / "images").glob("*.jpg")) for s in ("train", "val")}
    lbls = {s: sorted(p.stem for p in (D / s / "labels").glob("*.txt")) for s in ("train", "val")}
    assert imgs == lbls and len(imgs["train"]) + len(imgs["val"]) == 24 and imgs["val"]
    assert not set(imgs["train"]) & set(imgs["val"])
    for s in ("train", "val"):
        c = json.loads((D / "coco" / f"instances_{s}.json").read_text())
        assert len(c["images"]) == len(imgs[s]) and all((D / i["file_name"]).exists() for i in c["images"])
        assert all(a["segmentation"] and len(a["keypoints"]) == 21 for a in c["annotations"])
        yolo_n = sum(len((D / s / "labels" / f"{i}.txt").read_text().splitlines()) for i in imgs[s])
        assert yolo_n == len(c["annotations"])
    line = next(l for i in imgs["train"] for l in (D / "train" / "labels" / f"{i}.txt").read_text().splitlines())
    v = [float(x) for x in line.split()[1:]]
    assert len(v) >= 6 and all(0 <= x <= 1 for x in v)
    assert len((D / "manifest.csv").read_text().splitlines()) == 25


def test_dataset_follows_review_changes_and_pose_crops(reviewed_project):
    pr = reviewed_project
    from animalworld.stages import review as R
    rows = {r["id"]: r for r in read_jsonl(pr.path("generated", "index.jsonl"))}
    victim = next(d["id"] for d in R.accepted_docs(pr) if d["instances"])
    R.save_review(pr, rows[victim], "rejected", [], "human")
    cfg = yaml.safe_load((pr.root / "project.yaml").read_text())
    cfg["dataset"] = {"task": "pose", "pose_crops": True, "apply_looks": True, "formats": ["coco", "yolo"]}
    (pr.root / "project.yaml").write_text(yaml.safe_dump(cfg))
    pr = Project.load(pr.root)
    r = pipeline.run_stage(pr, "dataset")
    assert r["train"] + r["val"] == 23 and r["stale_files_removed"] == 2
    assert not list(pr.path("dataset").rglob(f"{victim}.*"))
    assert r["pose_crops"]["train"] > 0
    PC = pr.path("dataset", "pose_crops")
    y = yaml.safe_load((PC / "data.yaml").read_text())
    assert y["kpt_shape"] == [7, 3] and y["flip_idx"] == [0, 2, 1, 3, 5, 4, 6]
    crop = next((PC / "train" / "images").glob("*.jpg"))
    assert Image.open(crop).size == (256, 256)
    assert len((PC / "train" / "labels" / (crop.stem + ".txt")).read_text().split()) == 5 + 21


def test_looks_keep_geometry():
    im = Image.fromarray((np.random.default_rng(0).random((60, 80, 3)) * 255).astype(np.uint8))
    for look in looks.LOOKS:
        out = looks.apply(im, looks.params(look, "unit1"), 3)
        assert out.size == im.size
    assert looks.params("mono", "u") == looks.params("mono", "u") and looks.params("mono", "u") != looks.params("mono", "v")
    inv = np.asarray(looks.apply(im, looks.params("invert", "u")))
    assert (inv.astype(int) + np.asarray(im).astype(int) == 255).all()
    with pytest.raises(ValueError):
        looks.params("sepia", "u")


def test_training_gate_and_dry_run(reviewed_project):
    pr = reviewed_project
    r = pipeline.run_stage(pr, "train", dry_run=True)
    assert r["mode"] == "dry-run" and r["gate_passed"] and "YOLO('yolo11n-seg.pt').train(" in r["python"]
    assert r["kwargs"]["data"].endswith("dataset/data.yaml") and not pr.path("models").exists()
    cfg = yaml.safe_load((pr.root / "project.yaml").read_text())
    cfg["train"]["min_accepted_images"] = 500
    (pr.root / "project.yaml").write_text(yaml.safe_dump(cfg))
    pr = Project.load(pr.root)
    r = pipeline.run_stage(pr, "train")
    assert r["mode"] == "waiting" and "476 more" in r["next"] and not pr.path("models").exists()
    r = pipeline.run_stage(pr, "train", dry_run=True, backend="rfdetr", task="segment")
    assert "RFDETRSegLarge(resolution=768).train(" in r["python"] and r["kwargs"]["dataset_file"] == "yolo"
    assert r["kwargs"]["early_stopping_patience"] == 3 and r["kwargs"]["lr"] == 1e-4 and r["kwargs"]["resume"] is None
    r = pipeline.run_stage(pr, "train", dry_run=True, task="pose", overrides={"pose_crops": True})
    assert r["kwargs"]["imgsz"] == 256 and r["kwargs"]["degrees"] == 180 and r["kwargs"]["data"].endswith("pose_crops/data.yaml")
    r = pipeline.run_stage(pr, "export", dry_run=True)
    assert r["mode"] == "dry-run" and "format='onnx'" in r["python"]
    assert pipeline.run_stage(pr, "export")["mode"] == "waiting"


def test_sleap_dlc_export_trainer_runs(reviewed_project):
    pr = reviewed_project
    r = pipeline.run_stage(pr, "train", backend="sleap_dlc", task="pose")
    assert r["mode"] == "executed" and r["dlc"]["images"] == 24 and r["sleap"]["images"] == 24
    assert pr.path("models", "sleap_dlc_pose", "dlc", "config_snippet.yaml").exists()


def test_hungarian_matches_brute_force():
    rng = random.Random(0)
    for n, m in [(1, 1), (2, 3), (3, 3), (4, 6), (5, 5)]:
        for _ in range(20):
            c = [[rng.uniform(0, 10) for _ in range(m)] for _ in range(n)]
            best = min(sum(c[i][p[i]] for i in range(n)) for p in itertools.permutations(range(m), n))
            a = hungarian(c)
            assert len(set(a)) == n and abs(sum(c[i][a[i]] for i in range(n)) - best) < 1e-9


def test_nms_and_yolo_decode():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=float)
    assert nms(boxes, np.array([0.9, 0.8, 0.7]), 0.5) == [0, 2]
    out = np.zeros((4 + 1 + 6, 100), dtype=np.float32)                 # pose head: 1 class, 2 keypoints
    out[:, 5] = [100, 80, 40, 20, 0.9, 90, 75, 0.8, 110, 85, 0.2]
    out[:, 6] = [101, 81, 40, 20, 0.6, 0, 0, 0, 0, 0, 0]                # duplicate, removed by NMS
    b, s, c, extra = decode_yolo(out, 1, "pose", 0.25)
    assert len(b) == 1 and list(b[0]) == [80, 70, 120, 90] and abs(s[0] - 0.9) < 1e-6 and extra.shape == (1, 6)
    e2e = np.array([[10, 10, 20, 20, 0.8, 0], [0, 0, 5, 5, 0.1, 0]] + [[0] * 6] * 298, dtype=np.float32)
    b, s, c, _ = decode_yolo(e2e, 1, "detect", 0.25)
    assert len(b) == 1 and list(b[0]) == [10, 10, 20, 20]
    arr, r, px, py = letterbox(Image.new("RGB", (200, 100), (255, 0, 0)), (64, 64))
    assert arr.shape == (3, 64, 64) and abs(r - 0.32) < 1e-9 and (px, py) == (0, 16)
    assert abs(arr[0, 32, 32] - 1.0) < 1e-6 and abs(arr[0, 2, 2] - 114 / 255) < 1e-6


def test_blob_predictor_finds_mock_animals():
    pred = registry.create("predictors", "blob", {"category": "mouse"})
    found = total = 0
    for seed in range(8):
        animals = simulate(3, 1, 320, 240, seed)[0]
        img, inst = render(320, 240, animals, ["black", "white", "black"], [], seed)
        det = pred.predict(img)
        total += 3
        found += sum(1 for t in inst if any(L.iou(t["box"], d["box"]) > 0.4 for d in det))
        assert len(det) <= 4
    assert found >= total - 2, (found, total)


def test_tracker_keeps_identities_on_simulated_recording():
    seq = simulate(3, 80, 320, 240, 5)
    frames, truth = [], []
    rng = random.Random(1)
    for f, animals in enumerate(seq):
        _, inst = render(64, 48, [], [], [], 0)
        _, inst = render(320, 240, animals, ["black", "white", "brown"], [], 5, f)
        order = list(range(3)); rng.shuffle(order)                         # detections arrive in random order
        if f in (20, 21, 50):
            order = order[:2]                                              # missed detections
        frames.append([L.instance("mouse", inst[k]["box"], score=0.9, appearance=[0.1, 0.9, 0.4][k]) for k in order])
        truth.append(order)
    assert estimate_n(frames) == 3
    tr = IoUTracker({})
    ids = tr.track(frames)
    assert tr.n == 3
    maps = [dict() for _ in range(3)]
    for row, t in zip(ids, truth):
        assert len(set(row)) == len(row) and all(i >= 0 for i in row)
        for i, k in zip(row, t):
            maps[k][i] = maps[k].get(i, 0) + 1
    assert all(len(m) == 1 for m in maps), maps                             # every animal keeps exactly one identity
    assert len({next(iter(m)) for m in maps}) == 3


def test_predict_track_stages_and_rules_adapter(reviewed_project, tmp_path, monkeypatch):
    from animalworld.cli import main
    pr = reviewed_project
    assert main(["demo-recording", str(tmp_path / "rec"), "--frames", "25", "--n", "2"]) == 0
    assert pipeline.run_stage(pr, "predict")["mode"] == "waiting"
    r = pipeline.run_stage(pr, "predict", source=str(tmp_path / "rec"), backend="blob")
    assert r["frames"] == 25 and r["instances"] >= 45
    assert main(["predict", str(pr.root), "--source", str(tmp_path / "rec"), "--backend", "blob", "--name", "again", "--limit", "5"]) == 0
    assert len(read_jsonl(pr.path("predictions", "again", "frames.jsonl"))) == 5
    assert main(["track", str(pr.root), "--name", "rec", "--n-animals", "2"]) == 0
    t = pipeline.run_stage(pr, "track", name="rec")
    assert t["identities"] == 2 and t["frames_with_all_ids"] >= 23
    rows = pr.path("tracks", "rec", "tracks.csv").read_text().splitlines()
    assert rows[0].startswith("frame,id,x0,y0,x1,y1,score,nose_x") and len(rows) > 45
    truth = json.loads((tmp_path / "rec" / "truth.json").read_text())
    tracks = read_jsonl(pr.path("tracks", "rec", "tracks.jsonl"))
    seen = {0: set(), 1: set()}
    for tf, pf in zip(truth, tracks):
        for ti in tf["instances"]:
            for pi in pf["instances"]:
                if L.iou(ti["box"], pi["box"]) > 0.4:
                    seen[ti["track_id"]].add(pi["track_id"])
    assert all(len(s) == 1 for s in seen.values()) and seen[0] != seen[1]
    # rules adapter: exports detections, explains itself without an entry point, calls a stand-in callable
    with pytest.raises(registry.BackendError, match="det_export.npz"):
        pipeline.run_stage(pr, "track", backend="rules", name="rec")
    d = np.load(pr.path("tracks", "rec", "det_export.npz"))
    assert int(d["T"]) == 25 and d["box"].shape[1] == 4 and len(d["t"]) == r["instances"]
    mod = tmp_path / "my_rules.py"
    mod.write_text("import numpy as np\n"
                   "def run(work_dir, n, options):\n"
                   "    d = np.load(work_dir + '/det_export.npz'); T = int(d['T'])\n"
                   "    return [[int(k) for k in range(int((d['t'] == t).sum()))] for t in range(T)]\n")
    cfg = yaml.safe_load((pr.root / "project.yaml").read_text())
    cfg["track"] = {"backend": "rules", "options": {"entry": "my_rules:run", "pythonpath": str(tmp_path)}}
    (pr.root / "project.yaml").write_text(yaml.safe_dump(cfg))
    t = pipeline.run_stage(Project.load(pr.root), "track", name="rec")
    assert t["mode"] == "executed" and t["identities"] >= 2


def test_cli_end_to_end_mouse_and_status(tmp_path, capsys):
    from animalworld.cli import main
    p = str(tmp_path / "mouse_demo")
    assert main(["init", p, "--species", "mouse", "--describe", "laboratory, home cage and open field"]) == 0
    assert main(["pipeline", p, "--dry-run"]) == 0
    assert not (tmp_path / "mouse_demo" / "literature" / "records.jsonl").exists()      # a dry run writes no stage output
    assert main(["pipeline", p, "--auto-accept", "--train-dry-run"]) == 0
    out = capsys.readouterr().out
    for s in ("literature", "taxonomy", "prompts", "generate", "propose", "review", "dataset"):
        assert f"=== {s}: executed" in out
    assert "=== train: dry-run" in out and "=== export: dry-run" in out and "=== predict: waiting" in out
    assert main(["status", p]) == 0
    st = json.loads(capsys.readouterr().out)
    assert st["literature"]["records"] == 30 and st["prompts"]["items"] == 40 and st["prompts"]["passed"]
    assert st["generate"]["images"] == 40 and st["review"]["accepted"] == 40 and st["dataset"]["train"] + st["dataset"]["val"] == 40
    assert st["train"]["gate_passed"] is False and st["last_modes"]["train"] == "dry-run"
    assert main(["train", p]) == 0                                               # gate: 40 < 200 -> waiting, nothing trained
    assert json.loads(capsys.readouterr().out)["mode"] == "waiting"
    assert main(["generate", p, "--backend", "nope"]) == 2


def test_contact_sheets(reviewed_project, tmp_path):
    from animalworld.cli import main
    pr = reviewed_project
    assert main(["sheet", str(pr.root), "--what", "proposals", "--limit", "12", "--cols", "4"]) == 0
    with Image.open(pr.path("evidence", "proposals_sheet.jpg")) as im:
        assert im.width == 4 * 360 and im.height > 300
    assert main(["demo-recording", str(tmp_path / "rec"), "--frames", "6", "--n", "2"]) == 0
    assert main(["predict", str(pr.root), "--source", str(tmp_path / "rec"), "--backend", "blob", "--name", "r"]) == 0
    assert main(["track", str(pr.root), "--name", "r"]) == 0
    assert main(["sheet", str(pr.root), "--what", "predictions", "--name", "r", "--tracks", "--cols", "3"]) == 0
    assert pr.path("evidence", "tracks_r_sheet.jpg").exists()
    assert main(["sheet", str(pr.root), "--what", "reviewed"]) == 0


def test_options_belong_to_the_configured_backend(reviewed_project):
    pr = reviewed_project
    cfg = yaml.safe_load((pr.root / "project.yaml").read_text())
    cfg["train"] = {"backend": "yolo", "task": "segment", "min_accepted_images": 10,
                    "options": {"imgsz": 640, "device": 0, "plots": True}, "options_for": {"rfdetr": {"epochs": 3}}}
    cfg["generate"] = {"backend": "qwen_image", "options": {"model_path": "/some/qwen"}}
    (pr.root / "project.yaml").write_text(yaml.safe_dump(cfg))
    pr = Project.load(pr.root)
    y = pipeline.run_stage(pr, "train", dry_run=True)
    assert y["kwargs"]["imgsz"] == 640 and y["kwargs"]["plots"] is True
    r = pipeline.run_stage(pr, "train", dry_run=True, backend="rfdetr")
    assert "imgsz" not in r["kwargs"] and "plots" not in r["kwargs"] and r["kwargs"]["epochs"] == 3
    w = pipeline.run_stage(pr, "generate", dry_run=True, backend="wan_video")
    assert w["model_path"] is None                                  # the Qwen path does not leak into the Wan backend


def test_sheet_of_predictions_without_a_name_uses_the_newest_run(project, tmp_path):
    """Regression (20 Nov 2026): `animalworld sheet --what predictions` without --name crashed with a TypeError."""
    from animalworld.cli import main, _demo_recording
    _demo_recording(str(tmp_path / "rec"), 12, 2, 1)
    p = str(project.root)
    assert main(["sheet", p, "--what", "predictions"]) == 2                      # nothing predicted yet: a message, exit 2
    assert main(["predict", p, "--source", str(tmp_path / "rec"), "--backend", "blob"]) == 0 and main(["track", p]) == 0
    assert main(["sheet", p, "--what", "predictions", "--tracks", "--out", str(tmp_path / "s.jpg")]) == 0 and (tmp_path / "s.jpg").stat().st_size > 0
