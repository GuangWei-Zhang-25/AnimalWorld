import base64
import io
import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
import yaml
from PIL import Image

from animalworld import credentials, pipeline, registry
from animalworld.config import Project
from animalworld.formats import labels as L
from animalworld.stages import review as R
from animalworld.utils import read_json, read_jsonl


def _upto(project, stage):
    return pipeline.run_all(project, auto_accept=False, until=stage)


def test_mock_generation_is_resumable(project):
    _upto(project, "prompts")
    r = pipeline.run_stage(project, "generate", limit=5)
    assert r["units_generated"] == 5
    r = pipeline.run_stage(project, "generate")
    assert r["already_done"] == 5 and r["units_generated"] == 19 and r["images_total"] == 24
    r = pipeline.run_stage(project, "generate")
    assert r["todo"] == 0 and r["units_generated"] == 0
    rows = read_jsonl(project.path("generated", "index.jsonl"))
    assert len({x["id"] for x in rows}) == 24
    with Image.open(project.path(rows[0]["file"])) as im:
        assert im.size == (rows[0]["W"], rows[0]["H"]) and max(im.size) == 320
    assert not list(project.path("generated", ".claims").glob("*.lock"))


def test_claimed_items_are_skipped_by_other_workers(project):
    from animalworld.utils import Claims
    _upto(project, "prompts")
    items = read_jsonl(project.path("prompts", "manifest.jsonl"))
    Claims(project.path("generated", ".claims")).claim(items[0]["id"])       # another worker holds item 0
    r = pipeline.run_stage(project, "generate")
    assert r["units_generated"] == 23


def test_mock_clip_frames_share_one_unit(project):
    cfg = yaml.safe_load((project.root / "project.yaml").read_text())
    cfg["generate"] = {"backend": "mock", "options": {"frames": 3}}
    cfg["prompts"]["n"] = 6
    (project.root / "project.yaml").write_text(yaml.safe_dump(cfg))
    pr = Project.load(project.root)
    pipeline.run_all(pr, auto_accept=True, until="dataset")
    rows = read_jsonl(pr.path("generated", "index.jsonl"))
    assert len(rows) == 18 and len({r["unit"] for r in rows}) == 6
    split = {}
    for line in open(pr.path("dataset", "manifest.csv")).read().splitlines()[1:]:
        f = line.split(",")
        split.setdefault(f[2], set()).add(f[3])
    assert all(len(s) == 1 for s in split.values())            # no clip is split across train and val


def test_real_generators_plan_without_loading_anything(project, tmp_path):
    import sys
    items = [{"id": "a", "W": 1280, "H": 960, "seed": 1, "prompt": "p", "negative_prompt": "n", "n": 1}]
    q = registry.create("generators", "qwen_image", {"model_path": str(tmp_path / "missing")}, project=project)
    p = q.plan(items)
    assert p["needs_gpu"] and p["steps"] == 30 and p["true_cfg_scale"] == 4.0 and p["model_path_exists"] is False
    with pytest.raises(registry.BackendError, match="model_path"):
        list(q.generate(items, tmp_path))
    w = registry.create("generators", "wan_video", {}, project=project)
    p = w.plan(items)
    assert (p["num_frames"], p["steps"], p["fps"], p["size"]) == (121, 40, 16, "832x480") and w.kind == "video"
    s = registry.create("proposers", "sam", {}, project=project).plan([])
    assert s["text_prompt"] == "mouse" and s["needs_gpu"]
    assert "torch" not in sys.modules and "diffusers" not in sys.modules


def test_http_api_generator_against_local_stub(project):
    token = "tok-HTTP-SECRET-777"
    credentials.set_value("stub_api", token)
    seen = []
    buf = io.BytesIO(); Image.new("RGB", (64, 48), (120, 130, 140)).save(buf, "PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((self.headers.get("Authorization"), body))
            if self.headers.get("Authorization") != "Bearer " + token:
                self.send_response(401); self.end_headers(); self.wfile.write(b"bad token " + (self.headers.get("Authorization") or "").encode()); return
            out = json.dumps({"data": [{"b64_json": b64}]}).encode()
            self.send_response(200); self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out)
    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        cfg = yaml.safe_load((project.root / "project.yaml").read_text())
        cfg["prompts"]["n"] = 3
        cfg["generate"] = {"backend": "http_api", "options": {
            "endpoint": f"http://127.0.0.1:{srv.server_address[1]}/v1/images", "token_credential": "stub_api", "rate_limit_s": 0,
            "body": {"prompt": "{prompt}", "seed": "{seed}", "size": "{width}x{height}"}, "response": {"b64_path": "data.0.b64_json"}}}
        (project.root / "project.yaml").write_text(yaml.safe_dump(cfg))
        pr = Project.load(project.root)
        _upto(pr, "prompts")
        plan = pipeline.run_stage(pr, "generate", dry_run=True)
        assert plan["token_available"] is True and token not in json.dumps(plan)
        r = pipeline.run_stage(pr, "generate")
        assert r["units_generated"] == 3 and len(seen) == 3
        assert isinstance(seen[0][1]["seed"], int) and "x" in seen[0][1]["size"] and seen[0][1]["prompt"].startswith("A realistic")
        leaked = [p for p in pr.root.rglob("*") if p.is_file() and p.suffix in (".json", ".jsonl", ".yaml", ".csv", ".md") and token in p.read_text(errors="ignore")]
        assert not leaked                                           # the token never reaches the project folder
        # a failing endpoint must not echo the token either
        credentials.set_value("stub_api", "tok-WRONG-888888")
        (pr.path("generated", "index.jsonl")).unlink()
        with pytest.raises(registry.BackendError) as e:
            pipeline.run_stage(pr, "generate")
        assert "tok-WRONG-888888" not in str(e.value) and "401" in str(e.value)
    finally:
        srv.shutdown()


def test_http_api_presets_are_declared_untested():
    from animalworld.backends.generators import http_api
    assert "UNTESTED" in http_api.__doc__ and set(http_api.PRESETS) == {"openai_images", "generic_url"}
    g = registry.create("generators", "http_api", {"preset": "openai_images"})
    assert g.plan([])["token_credential"] == "openai_api_key" and g.plan([])["token_available"] is False


def test_propose_gates_and_count_gate(project):
    cfg = yaml.safe_load((project.root / "project.yaml").read_text())
    cfg["propose"] = {"backend": "mock", "options": {"miss_rate": 0.3}}
    (project.root / "project.yaml").write_text(yaml.safe_dump(cfg))
    pr = Project.load(project.root)
    _upto(pr, "generate")
    r = pipeline.run_stage(pr, "propose")
    assert r["proposed"] == 24 and r["failed_gates"] > 0 and "count" in r["fail_reasons"]
    assert r["passed_gates"] + r["failed_gates"] == 24
    assert pipeline.run_stage(pr, "propose")["todo"] == 0
    res = pipeline.run_stage(pr, "review", auto=True)
    assert res["auto_accepted"] == r["passed_gates"] and res["auto_rejected"] == r["failed_gates"] and res["pending"] == 0


def test_gate_rules_unit():
    inst = lambda b, s=0.9: L.instance("mouse", b, [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]], None, s)
    g = {"min_score": 0.5, "count_gate": True, "min_area_frac": 2e-4, "max_area_frac": 0.2, "max_border_frac": 0.4}
    ok = L.apply_gates(L.label("a", "a.jpg", 100, 100, [inst([10, 10, 30, 30])], "t"), 1, g)
    assert ok["kept"]
    low = L.apply_gates(L.label("a", "a.jpg", 100, 100, [inst([10, 10, 30, 30], 0.3)], "t"), 1, g)
    assert not low["kept"] and low["reason"] == "count 0 != 1"
    big = L.apply_gates(L.label("a", "a.jpg", 100, 100, [inst([0, 0, 90, 90])], "t"), 1, g)
    assert not big["kept"]
    empty = L.apply_gates(L.label("a", "a.jpg", 100, 100, [inst([10, 10, 30, 30])], "t"), 0, g)
    assert not empty["kept"] and "empty scene" in empty["reason"]
    border = L.apply_gates(L.label("a", "a.jpg", 100, 100, [inst([0, 0, 20, 20])], "t"), 1, g)
    assert border["reason"] == "border"
    nogate = L.apply_gates(L.label("a", "a.jpg", 100, 100, [inst([10, 10, 30, 30])] * 2, "t"), None, g)
    assert nogate["kept"]                                         # real images: no prompted count, no count gate


def _get(port, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}") as r:
        return r.status, r.read(), r.headers.get("Content-Type")


def _post(port, path, obj):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(obj).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def test_purely_manual_annotation_of_real_images(tmp_path):
    pr = Project.init(tmp_path / "manual", "zebrafish")
    src = tmp_path / "frames"; src.mkdir()
    for k in range(3):
        Image.new("RGB", (80, 60), (k * 40, 90, 90)).save(src / f"img{k}.png")
    assert R.add_images(pr, src)["images_added"] == 3 and R.add_images(pr, src)["images_added"] == 0
    rows = {r["id"]: r for r in read_jsonl(pr.path("generated", "index.jsonl"))}
    R.save_review(pr, rows["img0"], "accepted", [L.instance("zebrafish", [10, 10, 40, 30])], "human")
    r = pipeline.run_stage(pr, "dataset", task="detect")
    assert r["mode"] == "executed" and r["train"] + r["val"] == 1
    assert pipeline.run_stage(pr, "review")["pending"] == 2
