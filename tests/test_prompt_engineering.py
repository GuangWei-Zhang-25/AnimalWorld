import json

import pytest
import yaml

from animalworld import pipeline, registry
from animalworld.backends.literature.pubmed import parse_efetch
from animalworld.backends.writers.llm import LLMWriter, parse_answer
from animalworld.backends.writers.template import TemplateWriter, distinct_report
from animalworld.stages.prompts import DistinctError
from animalworld.utils import read_json, read_jsonl


def test_offline_literature_and_resume(project):
    r = pipeline.run_stage(project, "literature")
    assert r["mode"] == "executed" and r["records"] == 30 and r["requests_made"] == 0
    assert pipeline.run_stage(project, "literature")["mode"] == "skipped"
    recs = read_jsonl(project.path("literature", "records.jsonl"))
    assert all(x["source"] == "synthetic-sample" for x in recs)


def test_offline_csv(tmp_path):
    f = tmp_path / "a.csv"
    f.write_text("pmid,title,abstract,publication_year\n1,T one,An abstract about pigs,2020\n2,T two,Another,2021\n")
    be = registry.create("literature", "offline", {"source": str(f)})
    recs = be.search("", 10)
    assert [r["id"] for r in recs] == ["1", "2"] and recs[0]["year"] == "2020"


def test_europepmc_paging_cache_and_no_personal_data(tmp_path, monkeypatch):
    calls = []

    class Resp:
        status_code = 200
        def __init__(self, d): self._d = d
        def json(self): return self._d
        def raise_for_status(self): pass

    def fake(method, url, headers=None, timeout=None, **kw):
        calls.append((url, headers, kw))
        cur = kw["params"]["cursorMark"]
        page = {"*": ("c2", ["1", "2"]), "c2": ("c3", ["3"]), "c3": ("c3", [])}[cur]
        return Resp({"nextCursorMark": page[0], "resultList": {"result": [
            {"id": i, "source": "MED", "title": f"t{i}", "abstractText": "mouse open field " + i, "pubYear": "2020",
             "journalInfo": {"journal": {"title": "J"}}} for i in page[1]]}})
    import requests
    monkeypatch.setattr(requests, "request", fake)
    opts = {"rate_limit_s": 0, "page_size": 2}
    be = registry.create("literature", "europepmc", opts, cache_dir=tmp_path / "cache")
    recs = be.search("mouse", 3)
    assert [r["id"] for r in recs] == ["MED:1", "MED:2", "MED:3"] and be.requests_made == 2
    ua = calls[0][1]["User-Agent"]
    assert ua.startswith("animalworld/") and "@" not in ua
    assert "email" not in calls[0][2]["params"]                 # contact e-mail is opt-in and empty by default
    be2 = registry.create("literature", "europepmc", opts, cache_dir=tmp_path / "cache")
    assert len(be2.search("mouse", 3)) == 3 and be2.requests_made == 0      # served from the disk cache
    be3 = registry.create("literature", "europepmc", dict(opts, contact_email="me@example.org"), cache_dir=tmp_path / "c3")
    be3.search("mouse", 1)
    assert calls[-1][2]["params"]["email"] == "me@example.org"


def test_pubmed_xml_parse():
    xml = """<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>42</PMID><Article><Journal><JournalIssue><PubDate>
    <Year>2019</Year></PubDate></JournalIssue><Title>J Test</Title></Journal><ArticleTitle>Pigs <i>in</i> pens</ArticleTitle>
    <Abstract><AbstractText Label="METHODS">We filmed.</AbstractText><AbstractText>Then tracked.</AbstractText></Abstract>
    </Article></MedlineCitation><PubmedData><ArticleIdList><ArticleId IdType="doi">10.1/x</ArticleId></ArticleIdList>
    </PubmedData></PubmedArticle></PubmedArticleSet>"""
    r = parse_efetch(xml)[0]
    assert r["id"] == "PMID:42" and r["abstract"] == "METHODS: We filmed. Then tracked." and r["doi"] == "10.1/x"
    assert r["title"] == "Pigs in pens" and r["year"] == "2019"


def test_default_queries(project):
    q = registry.create("literature", "europepmc", {}).default_query("mouse", ["mice"], "")
    assert '"mouse" OR "mice"' in q and "LANG:eng" in q
    assert "mouse[Title/Abstract]" in registry.create("literature", "pubmed", {}).default_query("mouse", [], "")


def test_taxonomy_counts_and_weights(project):
    pipeline.run_stage(project, "literature")
    r = pipeline.run_stage(project, "taxonomy")
    assert r["records"] == 30 and r["rules"] > 20
    tax = read_json(project.path("taxonomy", "taxonomy.json"))
    arena = {x["id"]: x for x in tax["slots"]["arena"]}
    assert arena["open_field"]["hits"] == 3 and arena["open_field"]["weight"] > arena["open_field"]["base_weight"]
    assert project.path("taxonomy", "report.md").exists()
    # since 0.2.1 a phrase covered by the rule of ANY slot is not a candidate: "marble burying test" is matched by the
    # behaviour rule "marble" and "resident-intruder assay" by a group-size rule (tests/test_gap_closures.py has the rest)
    assert not any("marble burying test" in c["phrase"] for c in tax["candidates"]["arena"])


def test_prompts_distinct_deterministic_and_append(project):
    pipeline.run_stage(project, "literature"); pipeline.run_stage(project, "taxonomy")
    r = pipeline.run_stage(project, "prompts")
    assert r["passed"] and r["distinct_fraction"] >= 0.95 and r["items"] == 24
    a = read_jsonl(project.path("prompts", "manifest.jsonl"))
    assert pipeline.run_stage(project, "prompts")["mode"] == "skipped"
    r2 = pipeline.run_stage(project, "prompts", n=60)
    b = read_jsonl(project.path("prompts", "manifest.jsonl"))
    assert r2["new"] == 36 and b[:24] == a and len({m["id"] for m in b}) == 60
    pipeline.run_stage(project, "prompts", n=60, force=True)
    assert read_jsonl(project.path("prompts", "manifest.jsonl")) == b            # same seed -> same prompts
    for m in b:
        assert "{" not in m["prompt"] and m["W"] % 16 == 0 and m["split"] in ("train", "val")
        if m["n"] == 0:
            assert "empty" in m["prompt"]
        else:
            assert f"exactly {['', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight'][m['n']]} " in m["prompt"]
            assert len(m["appearance"]) == m["n"]
    assert any(m["n"] == 0 for m in b) and any(m["n"] > 1 for m in b)
    rep = read_json(project.path("prompts", "report.json"))
    assert rep["literature_weights_used"] is True


def test_prompts_large_set_is_distinct(project):
    w = TemplateWriter()
    items = [w.item(i, project.slots(), "mouse") for i in range(3000)]
    rep = distinct_report(items)
    assert rep["distinct_fraction_non_empty"] >= 0.95, rep


def test_distinct_gate_fails_on_tiny_vocabulary(project):
    s = project.slots()
    for k in list(s["slots"]):
        s["slots"][k] = s["slots"][k][:1]
    (project.root / "slots.yaml").write_text(yaml.safe_dump(s))
    with pytest.raises(DistinctError):
        pipeline.run_stage(project, "prompts")
    assert read_json(project.path("prompts", "report.json"))["passed"] is False
    assert not project.path("prompts", "manifest.jsonl").exists()


def test_new_species_needs_only_yaml(tmp_path):
    from animalworld.config import Project
    pr = Project.init(tmp_path / "pig", "pig", overrides={"literature": {"backend": "offline", "source": None}})
    items = [TemplateWriter().item(i, pr.slots(), "pig") for i in range(50)]
    assert all("pig" in m["prompt"] and "mouse" not in m["prompt"] for m in items)
    assert any("two pink pigs" in m["prompt"] or "pigs" in m["prompt"] for m in items)


def test_llm_writer_expands_slots_with_stub(project, monkeypatch):
    seen = {}

    def fake_ask(self, prompt):
        seen["prompt"] = prompt
        return 'Sure. {"arena": ["a round glass cylinder on a {floor} table", "bad {unknown} value", "a square open-field arena with a {floor} floor and {wall} walls"], "behaviour": ["climbing the cage lid"]}'
    monkeypatch.setattr(LLMWriter, "ask", fake_ask)
    w = LLMWriter({"per_slot": 5}, candidates={"arena": [{"phrase": "rotarod test", "records": 3}]})
    out = w.expand_slots(project.slots(), "mouse", "home cage")
    texts = [e["text"] if isinstance(e, dict) else e for e in out["slots"]["arena"]]
    assert "a round glass cylinder on a {floor} table" in texts and not any("unknown" in t for t in texts)
    assert len(texts) == len(project.slots()["slots"]["arena"]) + 1          # duplicate and bad placeholder dropped
    assert "rotarod test" in seen["prompt"] and "mouse" in seen["prompt"]
    assert parse_answer('{"arena": ["ok value"], "x": 1}', ["arena", "camera"]) == {"arena": ["ok value"], "camera": []}
    # the full stage with the llm writer writes the expanded vocabulary for the user to read
    cfg = yaml.safe_load((project.root / "project.yaml").read_text()); cfg["prompts"]["writer"] = "llm"
    (project.root / "project.yaml").write_text(yaml.safe_dump(cfg))
    from animalworld.config import Project
    r = pipeline.run_stage(Project.load(project.root), "prompts")
    assert r["passed"] and project.path("prompts", "slots_expanded.yaml").exists()


def test_llm_openai_compatible_sends_token_only_in_header(monkeypatch):
    from animalworld import credentials
    credentials.set_value("llm_tok", "tok-LLM-9999")
    got = {}

    class Resp:
        status_code = 200
        def json(self): return {"choices": [{"message": {"content": '{"arena": ["x y z"]}'}}]}

    def post(url, headers=None, json=None, timeout=None):
        got.update(url=url, headers=headers, body=json)
        return Resp()
    import requests
    monkeypatch.setattr(requests, "post", post)
    w = LLMWriter({"provider": "openai_compatible", "base_url": "http://localhost:9/v1", "model": "m", "token_credential": "llm_tok"})
    assert "x y z" in w.ask("hello")
    assert got["url"].endswith("/chat/completions") and got["headers"]["Authorization"] == "Bearer tok-LLM-9999"
    assert "tok-LLM-9999" not in json.dumps(got["body"])
