"""Animal World Studio without a window: the core the tkinter GUI is a thin layer over. Secrets, the argument lists it
builds, the job runner, and a whole mock pipeline driven through `Studio.run` (the same subprocess commands the
window's Run buttons start). No display needed."""
import json
import stat
import sys
import time
from pathlib import Path

import pytest

from animalworld import credentials
from animalworld.annotator.model import clean_instances
from animalworld.config import Project
from animalworld.stages import review as R
from animalworld.stages.generate import index_rows
from animalworld.studio import core as S

SECRET = "sk-test-0123456789-very-secret-value"
HOSTILE = ['"; rm -rf ~ #', "$(id)", "`id`", "a; touch PWNED", "x && echo hi", "--force"]


def wait(st, res, timeout=180):
    job = st.jobs[res["job"]]
    t0 = time.time()
    while job.status == "running":
        assert time.time() - t0 < timeout, "job did not finish"
        time.sleep(0.1)
    return dict(job.info(0), status=job.status)


def run(st, action, values=None, params=None):
    return wait(st, st.run(action, values, params or {}))


@pytest.fixture
def studio(tmp_path, monkeypatch):
    monkeypatch.setenv("ANIMALWORLD_CONFIG_DIR", str(tmp_path / "cfg"))     # the job subprocesses see the same isolated folder
    return S.Studio(None), tmp_path


def test_no_server_code_left():
    src = Path(S.__file__).read_text()
    assert "http.server" not in src and "shell=True" not in src and "os.system" not in src and "token_urlsafe" not in src
    app = Path(S.__file__).with_name("app.py").read_text()
    assert "http" not in app.replace("https://api.example.com", "").replace("http_api", "") and "shell=True" not in app
    assert "ours" not in app.lower().split()


def test_secrets_never_reach_project_or_logs(studio):
    st, tmp = studio
    proj = tmp / "p"
    with pytest.raises(S.FormError):
        st.run("literature", None, {})                                      # no project yet
    assert st.open_project({"path": str(proj), "species": "mouse", "description": "lab"})["created"] is True
    r = st.set_credential({"name": "my_image_api", "value": SECRET})
    assert r["stored"] == "my_image_api" and SECRET not in json.dumps(r)
    f = credentials.credentials_path()
    assert f.exists() and stat.S_IMODE(f.stat().st_mode) == 0o600 and SECRET in f.read_text()
    assert proj.resolve() not in f.resolve().parents
    st.save("generate", {"backend": "http_api", "endpoint": "http://127.0.0.1:9/v1/images", "model": "some-model",
                         "style": "openai_images", "credential": "my_image_api"})
    cfg = Project.load(proj).cfg["generate"]
    assert cfg["backend"] == "http_api" and cfg["options"]["token_credential"] == "my_image_api"
    assert cfg["options"]["body"]["model"] == "some-model" and cfg["options"]["response"] == {"b64_path": "data.0.b64_json"}
    state = st.state()
    assert "my_image_api" in state["credentials"] and state["values"]["generate"]["credential"] == "my_image_api"
    assert SECRET not in json.dumps(state)
    j = run(st, "generate", state["values"]["generate"], {"limit": 1})       # nothing listens on that port; the key is never printed
    assert j["status"] in ("failed", "done") and SECRET not in json.dumps(j)
    log = Path(j["log"])
    with open(log, "a") as fh:
        fh.write(f"leak {SECRET}\n")                                         # worst case: a tool echoes the key
    shown = st.jobs[j["id"]].info(0)["text"]
    assert SECRET not in shown and "leak ***" in shown
    for p in proj.rglob("*"):
        if p.is_file() and p != log:
            assert SECRET not in p.read_text(errors="replace"), p
    text = (proj / "project.yaml").read_text()
    assert text.startswith("# animalworld project.") and "Never put API tokens here" in text.splitlines()[1]
    for bad in ({"name": "x y", "value": SECRET}, {"name": "ok", "value": ""}, {"name": "ok", "value": "a\nb"}):
        with pytest.raises(S.FormError):
            st.set_credential(bad)
    for bad in ({"backend": "http_api", "endpoint": "ftp://x", "credential": "a"},
                {"backend": "http_api", "endpoint": "https://x", "credential": "bad name"}):
        with pytest.raises(S.FormError):
            st.save("generate", bad)
    with pytest.raises(S.FormError):
        S.save_config(Project.load(proj), [("generate.options", {"api_key": SECRET}, True)])
    assert SECRET not in (proj / "project.yaml").read_text()


def test_argv_is_a_list_and_hostile_input_is_inert(tmp_path):
    pr = Project.init(tmp_path / "proj", "mouse", "lab")
    vid = tmp_path / "v.mp4"
    vid.write_bytes(b"0")
    P = str(pr.root)
    exe = [sys.executable, "-m", "animalworld"]
    assert S.build_commands("prompts", {"force": True}, pr) == [exe + ["taxonomy", P], exe + ["prompts", P, "--force"]]
    assert S.build_commands("generate", {"limit": "5"}, pr) == [exe + ["generate", P, "--limit", "5"]]
    assert S.build_commands("train", {"force": True, "epochs": "3", "imgsz": 320}, pr) == [
        exe + ["train", P, "--force", "--set", "epochs=3", "--set", "imgsz=320"]]
    assert S.build_commands("deploy_pretrained", {"videos": str(vid), "n_animals": "2", "engine": "mock", "out": str(tmp_path / "o")}, pr) == [
        exe + ["zeroshot", str(vid), "--out", str(tmp_path / "o"), "--yes", "--n-animals", "2", "--engine", "mock"]]
    assert S.build_commands("deploy_pretrained", {"videos": [str(vid)]}, pr)[0][-6:] == [str(vid), "--out", str(pr.path("zeroshot")), "--yes", "--engine", "auto"]
    assert S.build_commands("deploy_project", {"videos": str(vid), "n_animals": 3}, pr) == [
        exe + ["predict", P, "--source", str(vid)], exe + ["track", P, "--n-animals", "3"],
        exe + ["sheet", P, "--what", "predictions", "--tracks", "--every", "30", "--limit", "24"]]
    for h in HOSTILE:
        for action, params in (("generate", {"limit": h}), ("train", {"epochs": h}), ("train", {"imgsz": h}),
                               ("deploy_pretrained", {"videos": str(vid), "n_animals": h}),
                               ("deploy_pretrained", {"videos": str(vid), "engine": h}),
                               ("import_labels", {"format": h, "path": str(vid)}), ("export_labels", {"format": h})):
            with pytest.raises(S.FormError):
                S.build_commands(action, params, pr)
        for action, params in (("deploy_pretrained", {"videos": h}), ("deploy_project", {"videos": h}),
                               ("import_labels", {"format": "coco", "path": h}), ("propose_keypoints", {"pose_weights": h})):
            with pytest.raises(S.FormError):
                S.build_commands(action, params, pr)
        if h.startswith("-"):
            with pytest.raises(S.FormError):
                S.build_commands("export_labels", {"format": "coco", "out": h}, pr)
        else:
            (cmd,) = S.build_commands("export_labels", {"format": "coco", "out": h}, pr)
            assert isinstance(cmd, list) and cmd[-2:] == ["--out", h] and all(isinstance(a, str) for a in cmd)
        with pytest.raises(S.FormError):
            S.stage_values("literature", {"backend": h, "limit": 5}, pr)
        with pytest.raises(S.FormError):
            S.stage_values("generate", {"backend": "qwen_image", "model_id": h}, pr)
        with pytest.raises(S.FormError):
            S.stage_values("generate", {"backend": "qwen_image", "model_id": "Qwen/Qwen-Image", "pipeline_class": h}, pr)
        for k in ("partition", "gres", "mem", "time", "cpus"):
            with pytest.raises(S.FormError):
                S.slurm_options({k: h})
    with pytest.raises(S.FormError):
        S.build_commands("shell", {}, pr)
    with pytest.raises(S.FormError):
        S.build_commands("deploy_pretrained", {"videos": "a\nb"}, pr)


def test_hostile_path_runs_nothing(studio):
    st, tmp = studio
    st.open_project({"path": str(tmp / "p"), "species": "mouse", "description": ""})
    canary = tmp / "PWNED"
    out = f"{tmp}/out; touch {canary} #"
    j = run(st, "export_labels", None, {"format": "coco", "out": out})
    assert j["status"] == "done", j["text"]
    assert not canary.exists() and (Path(out) / "annotations.json").exists()


def test_qwen_form_defaults_and_model_id(tmp_path):
    pr = Project.init(tmp_path / "proj", "mouse", "lab")
    v = S.form_values(pr)["generate"]
    assert (v["model_id"], v["pipeline_class"]) == ("Qwen/Qwen-Image-2.1", "QwenImage21Pipeline")   # the setup of the real run
    assert v["model_path"].endswith("models/Qwen-Image-2.1")                                        # default: the cache location
    assert (v["num_inference_steps"], v["true_cfg_scale"], v["dtype"], v["split"]) == (30, 4.0, "bfloat16", True)
    assert S.form_values(pr)["prompts"]["megapixels"] == 1.25 and S.form_values(pr)["prompts"]["size_multiple"] == 32
    pr = S.save_config(pr, S.stage_values("generate", dict(v, backend="qwen_image"), pr))
    o = pr.cfg["generate"]["options"]
    assert o["model_path"].endswith("Qwen-Image-2.1") and o["model_id"] == "Qwen/Qwen-Image-2.1" and o["num_inference_steps"] == 30
    from animalworld import registry
    plan = registry.create("generators", "qwen_image", pr.options("generate", "qwen_image"), project=pr).plan([])
    assert plan["model_id"] == "Qwen/Qwen-Image-2.1" and plan["pipeline_class"] == "QwenImage21Pipeline"
    # a local folder keeps whatever class the form says (the lab's run: a Qwen-Image-2.1 folder + QwenImage21Pipeline)
    pr = S.save_config(pr, S.stage_values("generate", dict(v, backend="qwen_image", model_path=str(tmp_path), model_id="",
                                                           pipeline_class=S.QWEN_LOCAL_CLASS), pr))
    assert pr.cfg["generate"]["options"]["pipeline_class"] == "QwenImage21Pipeline"
    assert S.form_values(pr)["generate"]["model_path"] == str(tmp_path)
    with pytest.raises(S.FormError):
        S.stage_values("generate", {"backend": "qwen_image"}, pr)            # neither folder nor id


def test_mock_pipeline_through_studio_jobs(studio):
    st, tmp = studio
    proj = tmp / "proj"
    st.open_project({"path": str(proj), "species": "mouse", "description": "home cage"})
    with pytest.raises(S.FormError):
        st.open_project({"path": str(tmp / "nospecies"), "species": "", "description": ""})
    st.open_project({"path": str(proj)})                                    # opening an existing project needs no species
    s = st.state()
    assert s["project"] == str(proj.resolve()) and s["status"]["prompts"]["items"] == 0
    assert run(st, "literature", {"backend": "offline", "limit": 50})["status"] == "done"
    j = run(st, "prompts", {"n": 24, "seed": 3, "megapixels": 1.25, "size_multiple": 32})
    assert j["status"] == "done" and "$ animalworld taxonomy" in j["text"] and "$ animalworld prompts" in j["text"]
    assert run(st, "generate", {"backend": "mock"})["status"] == "done"
    assert run(st, "propose", {"backend": "mock"})["status"] == "done"
    s = st.state()["status"]
    assert s["prompts"]["items"] == 24 and s["generate"]["images"] == 24 and s["propose"]["proposed"] == 24
    pr = Project.load(proj)
    assert pr.cfg["prompts"]["seed"] == 3
    # human review with the functions the annotator window calls
    rows = index_rows(pr)
    for r in rows[:3]:
        inst = R.proposed(pr, r["id"])["instances"] + [{"category": "mouse", "box": [5, 5, 60, 40], "poly": None,
                                                         "keypoints": {"nose": [10, 10, 2], "bogus": [1, 1, 2]}, "score": 1}]
        R.save_review(pr, r, "accepted", clean_instances(inst, pr.categories(), pr.keypoints()), "human", "annotator")
    R.save_review(pr, rows[3], "rejected", [], "human", "annotator")
    assert run(st, "review_auto")["status"] == "done"
    rv = st.state()["status"]["review"]
    assert rv["accepted_by_human"] == 3 and rv["rejected"] >= 1 and rv["pending"] == 0 and rv["accepted"] + rv["rejected"] == 24
    acc = rv["accepted"]
    assert run(st, "dataset", {"task": "segment"})["status"] == "done"
    j = run(st, "train", {"backend": "yolo", "task": "segment", "min_accepted_images": 200})
    assert j["status"] == "done" and f"{acc} accepted images < train.min_accepted_images = 200" in j["text"]
    assert st.state()["status"]["train"]["gate_passed"] is False
    j = run(st, "train", {"backend": "yolo", "task": "segment", "min_accepted_images": 10}, {"dry_run": True, "epochs": 1})
    assert j["status"] == "done" and '"mode": "dry-run"' in j["text"] and '"gate_passed": true' in j["text"]
    assert run(st, "export_labels", None, {"format": "dlc"})["status"] == "done" and (proj / "exchange" / "dlc").is_dir()
    j = run(st, "export_labels", None, {"format": "coco"})
    coco = proj / "exchange" / "coco" / "annotations.json"
    assert j["status"] == "done" and coco.exists()
    j = run(st, "import_labels", None, {"format": "coco", "path": str(coco), "images": str(proj / "generated" / "images")})
    assert j["status"] == "done" and f'"imported": {acc}' in j["text"]
    logs = sorted((proj / "logs").glob("studio_*.log"))
    assert len(logs) >= 10 and all(p.stat().st_size > 0 for p in logs)


def test_stop_and_single_job(studio, monkeypatch):
    st, tmp = studio
    st.open_project({"path": str(tmp / "p"), "species": "mouse", "description": ""})
    real = S.build_commands
    monkeypatch.setattr(S, "build_commands", lambda action, params, project, python=None:
                        [[sys.executable, "-c", "import time; print('started', flush=True); time.sleep(120)"]] if action == "export" else real(action, params, project, python))
    job = st.jobs[st.run("export", None, {})["job"]]
    for _ in range(100):
        if "started" in job.info(0)["text"]:
            break
        time.sleep(0.1)
    with pytest.raises(S.FormError, match="still running"):
        st.run("literature", None, {})
    job.stop()
    for _ in range(100):
        if job.status != "running":
            break
        time.sleep(0.1)
    assert job.status == "stopped"


def test_slurm_is_an_argument_list_and_the_wrapper_runs_locally(tmp_path, monkeypatch, capfd):
    """Nothing is submitted here: only the sbatch argument list, the job file and the wrapper's own run are checked."""
    monkeypatch.setenv("ANIMALWORLD_CONFIG_DIR", str(tmp_path / "cfg"))
    pr = Project.init(tmp_path / "proj", "mouse", "lab")
    cmds = S.build_commands("literature", {}, pr)
    log = tmp_path / "x.log"
    job = S.SlurmJob("literature_1", "literature", cmds, log, pr.root, {"partition": "gpu-a100,gpu-h100", "gres": "shard:3", "cpus": "2", "mem": "40G", "time": "00:30:00"})
    argv = job.submit_argv()
    assert argv[:2] == ["sbatch", "--parsable"] and "--partition=gpu-a100,gpu-h100" in argv and "--gres=shard:3" in argv
    assert "--cpus-per-task=2" in argv and "--mem=40G" in argv and "--time=00:30:00" in argv and f"--output={log}" in argv
    assert argv[-3:] == [str(S.SLURM_WRAPPER), sys.executable, str(job.job_json)] and all(isinstance(a, str) for a in argv)
    text = S.SLURM_WRAPPER.read_text()
    assert 'exec "$1" -m animalworld.studio.runjob "$2"' in text and len(text.splitlines()) == 3
    assert "--gres" not in " ".join(S.slurm_command({"gres": ""}, log, job.job_json, "a"))
    job.job_json.write_text(json.dumps({"cmds": cmds, "cwd": str(pr.root)}))
    from animalworld.studio import runjob
    assert runjob.main([str(job.job_json)]) == 0
    out = capfd.readouterr().out
    assert "$ animalworld literature" in out and "[exit 0]" in out
    assert Project.load(pr.root).state()["literature"]["mode"] == "executed"
    job.job_json.write_text(json.dumps({"cmds": "rm -rf /", "cwd": "."}))
    assert runjob.main([str(job.job_json)]) == 2
