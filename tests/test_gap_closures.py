"""Regression tests for the 0.2.1 changes that close the gaps between the manuscript's Methods and the code."""
import base64
import json
import sys
import threading
import types
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
import yaml

from animalworld import credentials, pipeline, registry
from animalworld.cli import main as cli
from animalworld.config import Project
from animalworld.formats.labels import apply_gates, instance, label
from animalworld.stages import review as R
from animalworld.stages.generate import index_rows
from animalworld.utils import read_json, read_jsonl


def _upto(pr, stage):
    return pipeline.run_all(pr, auto_accept=False, until=stage)


def _set(pr, **sections):
    cfg = yaml.safe_load((pr.root / "project.yaml").read_text())
    for k, v in sections.items():
        cfg[k] = {**(cfg.get(k) or {}), **v}
    (pr.root / "project.yaml").write_text(yaml.safe_dump(cfg))
    return Project.load(pr.root)


# ---------------------------------------------------------------- literature
PUBMED_XML = """<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>7</PMID><Article>
<Journal><JournalIssue><Volume>21</Volume><Issue>7</Issue><PubDate><Year>2024</Year></PubDate></JournalIssue><Title>J Test</Title></Journal>
<ArticleTitle>Own paper</ArticleTitle><Pagination><MedlinePgn>1329-1339</MedlinePgn></Pagination>
<Abstract><AbstractText>Text.</AbstractText></Abstract>
<AuthorList><Author><LastName>Weinreb</LastName><ForeName>Caleb</ForeName><Initials>C</Initials></Author>
<Author><LastName>Datta</LastName><Initials>SR</Initials></Author><Author><CollectiveName>The Consortium</CollectiveName></Author></AuthorList>
</Article></MedlineCitation><PubmedData>
<ArticleIdList><ArticleId IdType="pubmed">7</ArticleId><ArticleId IdType="doi">10.1000/own</ArticleId></ArticleIdList>
<ReferenceList><Reference><Citation>A cited paper.</Citation><ArticleIdList><ArticleId IdType="doi">10.9999/cited-first</ArticleId></ArticleIdList></Reference>
<Reference><Citation>Another.</Citation><ArticleIdList><ArticleId IdType="doi">10.9999/cited-last</ArticleId></ArticleIdList></Reference></ReferenceList>
</PubmedData></PubmedArticle>
<PubmedArticle><MedlineCitation><PMID>8</PMID><Article><Journal><Title>J</Title></Journal><ArticleTitle>No own DOI</ArticleTitle>
</Article></MedlineCitation><PubmedData><ArticleIdList><ArticleId IdType="pubmed">8</ArticleId></ArticleIdList>
<ReferenceList><Reference><ArticleIdList><ArticleId IdType="doi">10.9999/only-cited</ArticleId></ArticleIdList></Reference></ReferenceList>
</PubmedData></PubmedArticle></PubmedArticleSet>"""


def test_pubmed_keeps_the_articles_own_doi_not_a_cited_one():
    from animalworld.backends.literature.pubmed import parse_efetch
    a, b = parse_efetch(PUBMED_XML)
    assert a["doi"] == "10.1000/own"                       # was 10.9999/cited-last before 0.2.1
    assert b["doi"] == ""                                  # a record without its own DOI gets none, not a reference's
    assert a["authors"] == ["Weinreb C", "Datta SR", "The Consortium"]
    assert (a["volume"], a["issue"], a["pages"], a["year"], a["journal"]) == ("21", "7", "1329-1339", "2024", "J Test")
    assert b["authors"] == [] and b["volume"] == "" and b["pages"] == ""


def test_europepmc_bibliographic_fields_and_old_cache_pages(tmp_path, monkeypatch):
    page = {"resultList": {"result": [
        {"id": "1", "source": "MED", "title": "T", "abstractText": "A", "pubYear": "2020", "doi": "10.1/a",
         "authorString": "Smith J, Doe A.", "pageInfo": "11-19",
         "journalInfo": {"volume": "5", "issue": "2", "journal": {"title": "J"}}},
        {"id": "2", "source": "PPR", "title": "old page without the new keys"}]}}

    class Resp:
        status_code = 200
        def json(self): return page
        def raise_for_status(self): pass
    import requests
    monkeypatch.setattr(requests, "request", lambda *a, **k: Resp())
    be = registry.create("literature", "europepmc", {"rate_limit_s": 0}, cache_dir=tmp_path / "c")
    a, b = be.search("q", 2)
    assert a["authors"] == ["Smith J", "Doe A"] and (a["volume"], a["issue"], a["pages"]) == ("5", "2", "11-19")
    assert b["authors"] == [] and b["volume"] == b["issue"] == b["pages"] == "" and b["doi"] == ""


def test_offline_records_carry_the_same_keys(tmp_path):
    f = tmp_path / "a.csv"
    f.write_text("title,abstract,authors,volume\nT,A,\"Smith J; Doe A\",3\n")
    r = registry.create("literature", "offline", {"source": str(f)}).search("", 0)[0]
    assert r["authors"] == ["Smith J", "Doe A"] and r["volume"] == "3" and r["pages"] == ""


# ---------------------------------------------------------------- taxonomy candidates
def test_candidates_ignore_case_check_all_slots_and_drop_non_names(tmp_path):
    pr = Project.init(tmp_path / "p", "mouse")
    recs = [
        {"id": "1", "title": "the zebra maze", "abstract": "Mice were placed on the test arena. It was to test them."},
        {"id": "2", "title": "The Zebra Maze", "abstract": "A marble burying test. One chamber was used while test ran."},
        {"id": "3", "title": "A ZEBRA MAZE", "abstract": "The marble burying test again; two per chamber; EPM task."},
        {"id": "4", "title": "x", "abstract": "The EPM task, and the Wobble Assay."},
        {"id": "5", "title": "x", "abstract": "the wobble assay"},
    ]
    (pr.root / "literature").mkdir(exist_ok=True)
    (pr.root / "literature" / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    pipeline.run_stage(pr, "taxonomy")
    c = {x["phrase"]: x["records"] for x in read_json(pr.path("taxonomy", "taxonomy.json"))["candidates"]["arena"]}
    assert c["zebra maze"] == 3 and c["wobble assay"] == 2          # one candidate whatever the capitals
    assert "marble burying test" not in c                          # covered by a BEHAVIOUR rule ("marble"), not an arena rule
    assert not any(p.split()[0] in ("was", "on", "one", "while", "per", "two", "it", "placed") for p in c)
    assert "EPM task" not in c                                     # covered by the plus-maze rule (\bEPM\b)
    pr = _set(pr, taxonomy={"candidate_stop_words": ["wobble"]})   # the stop list is configurable
    pipeline.run_stage(pr, "taxonomy")
    assert "wobble assay" not in {x["phrase"] for x in read_json(pr.path("taxonomy", "taxonomy.json"))["candidates"]["arena"]}


def test_strain_patterns_stay_case_sensitive(tmp_path):
    pr = Project.init(tmp_path / "p", "mouse")
    recs = [{"id": str(k), "title": "t", "abstract": "FVB/NJ mice and/or controls, open/closed arms."} for k in range(3)]
    (pr.root / "literature").mkdir(exist_ok=True)
    (pr.root / "literature" / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    pipeline.run_stage(pr, "taxonomy")
    c = {x["phrase"] for x in read_json(pr.path("taxonomy", "taxonomy.json"))["candidates"]["appearance"]}
    assert "FVB/NJ" in c and "and/or" not in c and "open/closed" not in c


# ---------------------------------------------------------------- gates
def test_border_gate_uses_the_box_when_there_is_no_outline():
    G = {"min_score": 0.5, "count_gate": True, "min_area_frac": 2e-4, "max_area_frac": 0.5, "max_border_frac": 0.4}
    def kept(box):
        return apply_gates(label("i", "f", 100, 100, [instance("m", box, None, score=0.9)], "t"), 1, G)
    assert kept([40, 40, 60, 60])["kept"]
    assert kept([0, 40, 20, 60])["kept"]                    # one side on the border: 0.25 of the perimeter
    d = kept([0, 0, 20, 20])                                # a corner: 0.5
    assert not d["kept"] and d["reason"] == "border"
    assert kept([0, 0, 100, 30])["reason"] == "border"      # a strip along the top edge


# ---------------------------------------------------------------- generation log, frames
def test_generation_log_has_duration_and_sizes_for_every_backend(project):
    _upto(project, "generate")
    meta = read_jsonl(project.path("generated", "meta.jsonl"))
    man = {m["id"]: m for m in read_jsonl(project.path("prompts", "manifest.jsonl"))}
    idx = {r["id"]: r for r in index_rows(project)}
    assert len(meta) == len(man)
    for m in meta:
        assert m["generator"] == "mock" and isinstance(m["sec"], float) and m["sec"] >= 0 and m["utc"]
        assert (m["W"], m["H"]) == (idx[m["id"]]["W"], idx[m["id"]]["H"])                  # what was written
        assert (m["requested_W"], m["requested_H"]) == (man[m["id"]]["W"], man[m["id"]]["H"])   # what was asked for
        assert m["W"] <= 320 < m["requested_W"] or m["H"] <= 320 < m["requested_H"]


def test_frames_of_a_video_keep_unit_and_frame_index(tmp_path):
    cv2 = pytest.importorskip("cv2")
    import numpy as np
    from animalworld.stages.generate import _extract_frames
    v = tmp_path / "unitA.mp4"
    w = cv2.VideoWriter(str(v), cv2.VideoWriter_fourcc(*"mp4v"), 16, (160, 120))
    if not w.isOpened():
        pytest.skip("this OpenCV build cannot write mp4")
    for i in range(40):
        w.write(np.full((120, 160, 3), i * 5, np.uint8))
    w.release()
    out = tmp_path / "frames"
    out.mkdir()
    fs = _extract_frames(v, out, "unitA", 8)
    assert [(f["id"], f["frame"]) for f in fs] == [(f"unitA_f{k:04d}", k) for k in (0, 8, 16, 24, 32)]
    assert all((out / f"{f['id']}.jpg").exists() for f in fs)
    means = [float(cv2.imread(f["file"]).mean()) for f in fs]
    assert means == sorted(means) and means[-1] - means[0] > 100       # the frames really are 0, 8, 16, ... of the clip


# ---------------------------------------------------------------- trained checkpoints as proposers
def test_model_proposer_finds_rfdetr_checkpoints_and_picks_the_predictor(project, tmp_path):
    import os
    from animalworld.backends.proposers.yolo import newest_weights, predictor_for
    assert [predictor_for(x) for x in ("a/best.pt", "m.onnx", "checkpoint_best_total.pth", "mouseworld-det.ckpt")] == \
        ["ultralytics", "onnx", "rfdetr", "rfdetr"]
    d = project.path("models", "rfdetr_segment")
    d.mkdir(parents=True)
    (d / "checkpoint_best_total.pth").write_bytes(b"x")
    assert newest_weights(project).endswith("checkpoint_best_total.pth")
    y = project.path("models", "yolo_segment", "train", "weights")
    y.mkdir(parents=True)
    (y / "best.pt").write_bytes(b"x")
    os.utime(y / "best.pt", (2e9, 2e9))
    assert newest_weights(project).endswith("best.pt")                  # the newest of the two
    ck = tmp_path / "mouseworld-det.ckpt"
    ck.write_bytes(b"x")
    assert {"model", "yolo", "vlm"} <= set(registry.available("proposers"))
    plan = pipeline.run_stage(project, "propose", dry_run=True, backend="model", weights=str(ck))
    assert plan["predictor"] == "rfdetr" and plan["weights"] == str(ck) and plan["weights_exist"]
    assert "rfdetr" not in sys.modules and "torch" not in sys.modules     # planning loads nothing


# ---------------------------------------------------------------- label exchange, review records
def test_export_labels_yolo_and_import_as_proposed(reviewed_project, tmp_path, capsys):
    pr = reviewed_project
    assert cli(["export-labels", str(pr.root), "--format", "yolo", "--task", "detect"]) == 0
    out = json.loads(capsys.readouterr().out)
    n = len(R.accepted_docs(pr))
    files = sorted((pr.root / "exchange" / "yolo" / "labels").glob("*.txt"))
    assert out["images"] == n == len(files) and (pr.root / "exchange" / "yolo" / "classes.txt").read_text() == "mouse\n"
    full = [f for f in files if f.read_text().strip()]
    assert len(full[0].read_text().splitlines()[0].split()) == 5        # class cx cy w h
    # the same label files come back into a fresh project as PROPOSALS that still wait for review ...
    new = Project.init(tmp_path / "new", "mouse")
    assert cli(["import-labels", str(new.root), "--format", "yolo", str(files[0].parent), "--images",
                str(pr.root / "generated" / "images"), "--status", "proposed"]) == 0
    res = json.loads(capsys.readouterr().out)
    c = R.counts(new)
    assert res["status"] == "proposed" and res["imported"] == n and c["proposed"] == n and c["accepted"] == 0 and c["pending"] == n
    assert R.proposed(new, files[0].stem)["source"] == "import:yolo"
    # ... and by default as accepted labels marked origin=import
    new2 = Project.init(tmp_path / "new2", "mouse")
    assert cli(["import-labels", str(new2.root), "--format", "yolo", str(files[0].parent), "--images",
                str(pr.root / "generated" / "images")]) == 0
    capsys.readouterr()
    d = R.reviewed(new2, files[0].stem)
    assert d["status"] == "accepted" and d["origin"] == "import" and d["proposal_source"] is None and d["edited"] is None


def test_review_record_keeps_proposal_source_and_edited_flag(project):
    from animalworld.annotator.model import Doc, clean_instances
    _upto(project, "propose")
    rows = [r for r in index_rows(project) if R.proposed(project, r["id"])["instances"]]
    cats, kp = project.categories(), project.keypoints()
    def human(row, edit):
        d = Doc(R.proposed(project, row["id"])["instances"], row["W"], row["H"], cats[0], kp)
        if edit:
            d.checkpoint(); d.move_instance(0, 3, 2)
        return R.save_review(project, row, "accepted", clean_instances(d.instances, cats, kp), "human", "annotator")
    a, b = human(rows[0], False), human(rows[1], True)
    assert (a["origin"], a["source"], a["proposal_source"], a["edited"]) == ("human", "annotator", "mock", False)
    assert (b["proposal_source"], b["edited"]) == ("mock", True) and b["reviewed_utc"]
    R.auto_accept(project)
    c = R.counts(project)
    auto = R.reviewed(project, rows[2]["id"])
    assert (auto["origin"], auto["proposal_source"], auto["edited"]) == ("auto", "mock", False)
    assert c["accepted_by_human"] == 2 and c["accepted_after_edit"] == 1


def test_dlc_export_reports_why_no_h5_was_written(reviewed_project, tmp_path, monkeypatch):
    from animalworld.formats import dlc
    fake = types.ModuleType("pandas")
    def boom(*a, **k):
        raise ValueError("numpy.dtype size changed, may indicate binary incompatibility")
    fake.read_csv = boom
    monkeypatch.setitem(sys.modules, "pandas", fake)
    r = dlc.export_dlc(R.accepted_docs(reviewed_project), reviewed_project.keypoints(), tmp_path / "d", reviewed_project.root)
    assert r["h5"] is None and "binary incompatibility" in r["note"] and "ValueError" in r["note"] and "not installed" not in r["note"]


def test_export_after_sleap_dlc_says_what_to_do(reviewed_project, tmp_path):
    w = tmp_path / "w.pt"
    w.write_bytes(b"x")
    pr = _set(reviewed_project, train={"backend": "sleap_dlc"})
    plan = pipeline.run_stage(pr, "export", dry_run=True)
    assert "nothing to export" in plan["note"]
    with pytest.raises(registry.BackendError, match="trains no model here"):
        pipeline.run_stage(pr, "export", weights=str(w))
    assert cli(["export", str(pr.root), "--weights", str(w)]) == 2          # a message, not a traceback


# ---------------------------------------------------------------- VLM proposer (stub server only)
def test_vlm_proposer_against_local_stub(project):
    token = "tok-VLM-SECRET-4242"
    credentials.set_value("stub_vlm", token)
    seen = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((self.path, self.headers.get("Authorization"), body))
            text = body["messages"][0]["content"][1]["text"]
            k = len(seen)
            if k == 1:      # a good answer wrapped in prose, with one malformed instance and an unknown keypoint
                ans = ('Here you go: {"instances": [{"box": [10, 12, 60, 70], "score": 0.9, "keypoints": {"nose": [20, 20], '
                       '"wing": [1, 1]}}, {"box": [1, 2, 3]}, {"box": [70, 20, 95, 60]}]} hope it helps')
            elif k == 2:
                ans = '{"instances": []}'
            else:
                ans = "I cannot see the image."
            out = json.dumps({"choices": [{"message": {"role": "assistant", "content": ans}}], "echo": text[:10]}).encode()
            self.send_response(200); self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out)
    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        pr = _set(project, prompts={"n": 3}, generate={"options": {"max_side": 200}},
                  propose={"backend": "vlm", "options": {"base_url": f"http://127.0.0.1:{srv.server_address[1]}/v1",
                                                        "model": "stub-vlm", "token_credential": "stub_vlm", "max_side": 100}})
        _upto(pr, "generate")
        plan = pipeline.run_stage(pr, "propose", dry_run=True)
        assert plan["token_available"] is True and token not in json.dumps(plan) and "not yet run" in plan["note"]
        rows = index_rows(pr)
        res = pipeline.run_stage(pr, "propose", limit=2)
        assert res["proposed"] == 2 and len(seen) == 2
        path, auth, body = seen[0]
        assert path == "/v1/chat/completions" and auth == "Bearer " + token and body["model"] == "stub-vlm"
        img, txt = body["messages"][0]["content"]
        assert img["image_url"]["url"].startswith("data:image/jpeg;base64,") and base64.b64decode(img["image_url"]["url"].split(",", 1)[1])[:2] == b"\xff\xd8"
        assert "Find every mouse" in txt["text"] and "nose" in txt["text"] and "The scene was set up with" in txt["text"]
        d = R.proposed(pr, rows[0]["id"])
        s = max(rows[0]["W"], rows[0]["H"]) / 100.0                    # answers are scaled back from the 100 px image sent
        assert d["source"] == "vlm" and d["n_raw"] == 2 and "kept" in d and d["vlm"]["model"] == "stub-vlm"
        assert all(i["poly"] is None and i["category"] == "mouse" for i in d["instances"])
        from animalworld.backends.proposers.vlm import parse_answer
        inst = parse_answer('{"instances": [{"box": [10, 12, 60, 70], "score": 0.9, "keypoints": {"nose": [20, 20], "wing": [1, 1]}}]}',
                            rows[0]["W"], rows[0]["H"], "mouse", pr.keypoints(), 1 / s)
        assert inst[0]["box"] == [round(10 * s, 2), round(12 * s, 2), round(min(rows[0]["W"], 60 * s), 2), round(min(rows[0]["H"], 70 * s), 2)]
        assert inst[0]["keypoints"] == {"nose": [round(20 * s, 2), round(20 * s, 2), 2]} and inst[0]["poly"] is None
        assert R.proposed(pr, rows[1]["id"])["instances"] == []
        with pytest.raises(registry.BackendError, match="no JSON object"):          # strict: prose is refused
            pipeline.run_stage(pr, "propose")
        leaked = [p for p in pr.root.rglob("*") if p.is_file() and p.suffix in (".json", ".jsonl", ".yaml") and token in p.read_text(errors="ignore")]
        assert not leaked
        from animalworld.backends.proposers import vlm
        assert "NOT YET RUN AGAINST A LIVE SERVICE" in vlm.__doc__ and vlm.VLMProposer.tested is False
    finally:
        srv.shutdown()


def test_vlm_answer_parsing_is_strict():
    from animalworld.backends.proposers.vlm import parse_answer
    for bad in ("", "no json", '{"boxes": []}', '{"instances": {"box": [1, 2, 3, 4]}}', "{not json}"):
        with pytest.raises(registry.BackendError):
            parse_answer(bad, 100, 100, "m", [])
    ok = parse_answer('{"instances": [{"box": [-5, 10, 300, 50], "score": 7}, {"box": [1, 1, "x", 5]}, {"box": [5, 5, 5, 9]},'
                      ' {"box": [1, 1, 9, 9], "score": true}, {"box": [1, 1, 9, 9, 3]}]}', 100, 100, "m", [])
    assert len(ok) == 1 and ok[0]["box"] == [0.0, 10.0, 100.0, 50.0] and ok[0]["score"] == 1.0


# ---------------------------------------------------------------- identities across the frames of a unit
def test_propose_links_identities_over_the_frames_of_a_unit(project):
    pr = _set(project, prompts={"n": 12}, generate={"options": {"frames": 5}})
    _upto(pr, "generate")
    res = pipeline.run_stage(pr, "propose")
    assert res["units_linked"] >= 8 and res["frames_linked"] == res["units_linked"] * 5
    by_unit = {}
    for r in index_rows(pr):
        by_unit.setdefault(r["unit"], []).append(r)
    checked = 0
    for unit, rs in by_unit.items():
        maps = []
        for r in sorted(rs, key=lambda r: r["frame"]):
            d = R.proposed(pr, r["id"])
            truth = read_json(pr.path("generated", "truth", r["id"] + ".json"))["instances"]
            if not d["instances"]:
                continue
            assert d["identity_source"] == "iou_link"
            ids = [i["track_id"] for i in d["instances"]]
            assert len(set(ids)) == len(ids) and None not in ids                 # one id per animal on a frame
            # the mock proposer keeps the truth order, so instance k is truth animal k
            maps.append({t["track_id"]: i["track_id"] for t, i in zip(truth, d["instances"])})
        if maps:
            assert all(m == maps[0] for m in maps), unit                         # the same animal keeps its id on all frames
            checked += 1
    assert checked >= 8
    pr = _set(pr, propose={"link_identities": False})
    assert "units_linked" not in pipeline.run_stage(pr, "propose", force=True)


def test_annotator_identity_edit_swaps_and_undoes():
    from animalworld.annotator.model import Doc, clean_instances
    mk = lambda b, t: {"category": "m", "box": b, "poly": None, "keypoints": None, "score": 1.0, "track_id": t}   # noqa: E731
    d = Doc([mk([0, 0, 10, 10], 0), mk([20, 20, 40, 40], 1), mk([50, 50, 70, 70], None)], 100, 100, "m", [])
    d.set_track_id(0, 1)                                    # an identity swap is repaired with one key
    assert [i["track_id"] for i in d.instances] == [1, 0, None]
    d.set_track_id(2, 5)
    assert [i["track_id"] for i in d.instances] == [1, 0, 5] and d.dirty
    assert [i["track_id"] for i in clean_instances(d.instances, ["m"], [])] == [1, 0, 5]     # and it is saved
    assert d.undo() and d.undo() and [i["track_id"] for i in d.instances] == [0, 1, None]
    d.set_track_id(0, 0)
    assert not d.undo()                                     # setting the id it already has records nothing


def test_annotator_window_shows_and_edits_identities(project):
    """Needs a display (skipped without one). A clip of 3 frames: the window draws the proposed ids, a digit key
    changes the identity of the selected animal, and the accepted record keeps it and is marked as edited."""
    from tkutil import make_root, pump
    root = make_root()
    try:
        root.geometry("200x100+0+0")
        from animalworld.annotator.tkapp import AnnotatorWindow
        pr = _set(project, prompts={"n": 6}, generate={"options": {"frames": 3}})
        _upto(pr, "propose")
        w = AnnotatorWindow(root, pr, geometry="1180x760+0+0")
        pump(root, settle=0.4)
        k = next(k for k, r in enumerate(w.shown) if len(R.proposed(pr, r["id"])["instances"]) >= 2)
        w._select_index(k, load=True)
        pump(root, settle=0.2)
        iid = w.cur["id"]
        ids = [i["track_id"] for i in w.doc.instances]
        assert None not in ids and len(set(ids)) == len(ids)
        texts = [w.canvas.itemcget(t, "text") for t in w.canvas.find_all() if w.canvas.type(t) == "text"]
        assert all(any(f"id {t}" in x for x in texts) for t in ids)             # the ids are drawn at the boxes
        w.doc.selected = 0
        w.canvas.focus_force()
        w.canvas.event_generate(f"<KeyPress-{ids[1]}>")                         # give animal 1 the id of animal 2: a swap
        pump(root, settle=0.1)
        assert [i["track_id"] for i in w.doc.instances][:2] == [ids[1], ids[0]]
        w.canvas.event_generate("<KeyPress-a>")
        pump(root, settle=0.2)
        d = R.reviewed(pr, iid)
        assert d["status"] == "accepted" and d["origin"] == "human" and d["edited"] is True and d["proposal_source"] == "mock"
        assert [i["track_id"] for i in d["instances"]][:2] == [ids[1], ids[0]]
    finally:
        root.destroy()
