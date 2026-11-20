"""Animal World Studio: the real tkinter window, driven through its real widgets (Run buttons invoked, form variables
set, mouse and key events on the annotator canvas), with the mock backends. Skipped when no display can be opened.
Set STUDIO_SHOTS=<folder> to save screenshots."""
import stat
import sys

import pytest

from animalworld import credentials
from animalworld.config import Project
from animalworld.stages import review as R

from tkutil import box_grab_point, make_root, pump, shot

SECRET = "sk-test-0123456789-very-secret-value"


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("ANIMALWORLD_CONFIG_DIR", str(tmp_path / "cfg"))     # the job subprocesses see the same isolated folder
    root = make_root()
    from animalworld.studio.app import StudioApp
    a = StudioApp(root, geometry="1200x800+0+0")
    a.confirm = lambda title, text: True                                    # no modal dialogs in a test
    pump(root, settle=0.4)
    yield a, root, tmp_path
    if a.busy:
        a.job.stop()
    root.destroy()


def run(a, root, stage, button="run", timeout=180):
    row = a.rows[stage]
    b = row.run_button if button == "run" else row.buttons[button]
    b.invoke()
    assert a.job is not None, f"{stage}: job did not start: {row.msg_var.get()}"
    assert str(a.stop_button.cget("state")) == "normal" and str(a.rows["dataset"].run_button.cget("state")) == "disabled"
    pump(root, lambda: a.job is None, timeout)
    assert str(a.stop_button.cget("state")) == "disabled" and str(a.rows["dataset"].run_button.cget("state")) == "normal"
    return row.log_content()


def scroll_to(a, root, stage):
    a.show({"dataset": "train", "export": "train", "deploy": "inference"}.get(stage, stage))     # the stage's page (sidebar)
    root.update()
    a.canvas.yview_moveto(max(0.0, (a.rows[stage].frame.winfo_y() - 8) / max(1, a.inner.winfo_height())))
    pump(root)


def test_studio_window_end_to_end_with_mock_backends(app, monkeypatch):
    a, root, tmp = app
    R_ = a.rows
    assert list(R_) == ["project", "literature", "prompts", "generate", "propose", "review", "dataset", "train", "export", "deploy"]
    assert R_["project"].status_var.get() == "no project yet"
    # the Qwen-Image worked example is pre-filled, with a matching id / class pair
    g = R_["generate"].vars
    assert (g["model_id"].get(), g["pipeline_class"].get(), g["num_inference_steps"].get(), g["true_cfg_scale"].get(),
            g["dtype"].get(), g["split"].get()) == ("Qwen/Qwen-Image-2.1", "QwenImage21Pipeline", "30", "4.0", "bfloat16", True)
    assert (R_["prompts"].vars["megapixels"].get(), R_["prompts"].vars["size_multiple"].get()) == ("1.25", "32")
    # a stage cannot run before a project exists; the message is shown, nothing starts
    R_["literature"].run_button.invoke()
    pump(root)
    assert a.job is None and "create or open a project first" in R_["literature"].msg_var.get()
    # 1. project
    proj = tmp / "mouse_proj"
    R_["project"].vars["path"].set(str(proj))
    R_["project"].vars["species"].set("mouse")
    R_["project"].vars["description"].set("home cage and open field")
    R_["project"].run_button.invoke()
    pump(root)
    assert (proj / "project.yaml").exists() and "mouse" in R_["project"].status_var.get() and a.status_var.get().startswith("Created")
    # 2. literature, 3. prompts (target number of images = 24)
    log = run(a, root, "literature")
    assert "$ animalworld literature" in log and "[exit 0]" in log and R_["literature"].log_shown
    assert "records (offline)" in R_["literature"].status_var.get() and not R_["literature"].status_var.get().startswith("0 ")
    R_["prompts"].vars["n"].set("24")
    R_["prompts"].vars["seed"].set("3")
    log = run(a, root, "prompts")
    assert "$ animalworld taxonomy" in log and "$ animalworld prompts" in log
    assert R_["prompts"].status_var.get().startswith("24 prompts") and Project.load(proj).cfg["prompts"]["seed"] == 3
    g["backend"].set("qwen_image")
    a._gen_source()
    scroll_to(a, root, "generate")
    shot(root, "studio_generate")
    # 4. generate: an API key typed into the form ends up ONLY in the credentials file
    g["backend"].set("http_api")
    a._gen_source()
    g["endpoint"].set("http://127.0.0.1:9/v1/images")
    g["model"].set("some-model")
    g["credential"].set("my_image_api")
    g["api_key"].set(SECRET)
    R_["generate"].buttons["save"].invoke()
    pump(root)
    assert g["api_key"].get() == "" and "Saved generate" in a.status_var.get()
    f = credentials.credentials_path()
    assert SECRET in f.read_text() and stat.S_IMODE(f.stat().st_mode) == 0o600 and proj.resolve() not in f.resolve().parents
    cfg = Project.load(proj).cfg["generate"]
    assert cfg["backend"] == "http_api" and cfg["options"]["token_credential"] == "my_image_api" and cfg["options"]["body"]["model"] == "some-model"
    for p in proj.rglob("*"):
        if p.is_file():
            assert SECRET not in p.read_text(errors="replace"), p
    g["credential"].set("bad name")                                          # a refused form: message, nothing saved, nothing started
    R_["generate"].run_button.invoke()
    pump(root)
    assert a.job is None and "credential name" in R_["generate"].msg_var.get()
    g["credential"].set("my_image_api")
    g["backend"].set("mock")
    a._gen_source()
    log = run(a, root, "generate")
    assert "[exit 0]" in log and R_["generate"].status_var.get().startswith("24 images; 24 of 24 prompts done (mock)")
    # 5. propose
    log = run(a, root, "propose")
    assert "[exit 0]" in log and R_["propose"].status_var.get().startswith("24 proposed")
    assert R_["review"].status_var.get().startswith("24 pending, 0 accepted")
    a.show("propose")
    pump(root)
    shot(root, "studio")
    # 6. review: the annotator window, real mouse and key events on its canvas
    w = a.open_annotator()
    w.geometry("1180x760+10+20")
    pump(root, settle=0.5)
    assert a.open_annotator() is w                                           # a second click reuses the window
    w.canvas.focus_force()
    pump(root)
    pr = Project.load(proj)
    first = w.cur["id"]
    n0 = len(R.proposed(pr, first)["instances"])
    c = w.canvas
    c.event_generate("<KeyPress-b>")
    p0, p1 = w.view.to_canvas(8, 8), w.view.to_canvas(72, 52)
    c.event_generate("<ButtonPress-1>", x=int(p0[0]), y=int(p0[1]))
    c.event_generate("<B1-Motion>", x=int((p0[0] + p1[0]) / 2), y=int((p0[1] + p1[1]) / 2))
    c.event_generate("<B1-Motion>", x=int(p1[0]), y=int(p1[1]))
    c.event_generate("<ButtonRelease-1>", x=int(p1[0]), y=int(p1[1]))
    c.event_generate("<KeyPress-k>")
    for pt in ((20, 20), (32, 16)):
        q = w.view.to_canvas(*pt)
        c.event_generate("<ButtonPress-1>", x=int(q[0]), y=int(q[1]))
        c.event_generate("<ButtonRelease-1>", x=int(q[0]), y=int(q[1]))
    pump(root, settle=0.3)
    assert len(w.doc.instances) == n0 + 1
    shot(w, "studio_review")
    c.event_generate("<KeyPress-Escape>")
    w.buttons["accept"].invoke()
    pump(root)
    d = R.reviewed(pr, first)
    assert d["status"] == "accepted" and d["origin"] == "human" and len(d["instances"]) == n0 + 1
    assert set(d["instances"][-1]["keypoints"]) == {"nose", "left_ear"}
    second = w.cur["id"]
    w.buttons["reject"].invoke()
    pump(root)
    assert R.reviewed(pr, second)["status"] == "rejected"
    third = w.cur["id"]                                                      # drag an existing box, then accept with the key
    i = 0
    w.doc.selected = i
    before = list(w.doc.instances[i]["box"])
    cx, cy = w.view.to_canvas(*box_grab_point(w.doc, i, 8 / w.view.scale))
    dx = -24 if before[0] > w.cur["W"] / 2 else 24
    scale = w.view.scale                                                     # the next image may be shown at another scale
    c.event_generate("<ButtonPress-1>", x=int(cx), y=int(cy))
    c.event_generate("<B1-Motion>", x=int(cx + dx / 2), y=int(cy + 5))
    c.event_generate("<B1-Motion>", x=int(cx + dx), y=int(cy + 10))
    c.event_generate("<ButtonRelease-1>", x=int(cx + dx), y=int(cy + 10))
    c.event_generate("<KeyPress-a>")
    pump(root)
    saved = R.reviewed(pr, third)["instances"][i]["box"]
    assert saved[0] - before[0] == pytest.approx(dx / scale, abs=1.0) and saved != pytest.approx(before, abs=0.5)
    assert R_["review"].status_var.get().startswith("21 pending, 2 accepted (2 by a person), 1 rejected")   # Studio saw it at once
    w.destroy()
    pump(root)
    # accept the rest without a human look (explicit, confirmed), then export labels for DeepLabCut
    asked = []
    a.confirm = lambda title, text: asked.append(text) or False
    R_["review"].buttons["auto"].invoke()
    pump(root)
    assert a.job is None and asked and "NO person" in asked[0]               # declined: nothing runs
    a.confirm = lambda title, text: True
    log = run(a, root, "review", "auto")
    assert "$ animalworld review" in log and "--auto" in log
    c_ = R.counts(pr)
    assert c_["pending"] == 0 and c_["accepted_by_human"] == 2 and c_["accepted"] + c_["rejected"] == 24
    R_["review"].vars["export_format"].set("dlc")
    log = run(a, root, "review", "export")
    assert "[exit 0]" in log and (proj / "exchange" / "dlc").is_dir()
    # 7. dataset, 8. train (the image-count gate), 9. export is not run here
    log = run(a, root, "dataset")
    assert "[exit 0]" in log and "train /" in R_["dataset"].status_var.get() and not R_["dataset"].status_var.get().startswith("0 train")
    acc = c_["accepted"]
    assert R_["train"].status_var.get().startswith(f"{acc} accepted of 200 needed (gate closed)")
    log = run(a, root, "train")
    assert f"{acc} accepted images < train.min_accepted_images = 200" in log
    R_["train"].vars["min_accepted_images"].set("10")
    R_["train"].vars["dry_run"].set(True)
    R_["train"].vars["epochs"].set("1")
    log = run(a, root, "train")
    assert '"mode": "dry-run"' in log and '"gate_passed": true' in log
    assert R_["train"].status_var.get().startswith(f"{acc} accepted of 10 needed (gate passed)")
    R_["deploy"].vars["videos"].set(["/data/lab_videos/cage1_day3.mp4", "/data/lab_videos/open_field"])
    a.show("inference")
    pump(root)
    shot(root, "studio_end")
    R_["train"].vars["epochs"].set("many")
    R_["train"].run_button.invoke()
    pump(root)
    assert a.job is None and "epochs" in R_["train"].msg_var.get()
    # 10. run on your videos: the argument list, and a missing video is refused with a message
    d_ = R_["deploy"]
    d_.vars["videos"].set(str(tmp / "nothing.mp4"))
    d_.run_button.invoke()
    pump(root)
    assert a.job is None and "does not exist" in d_.msg_var.get()
    vid = tmp / "clip.mp4"
    vid.write_bytes(b"0")
    d_.vars["videos"].set([str(vid)])
    d_.vars["n_animals"].set("2")
    d_.vars["engine"].set("mock")
    from animalworld.studio import core
    assert core.build_commands("deploy_pretrained", a.params("deploy_pretrained"), pr) == [
        [sys.executable, "-m", "animalworld", "zeroshot", str(vid), "--out", str(pr.path("zeroshot")), "--yes", "--n-animals", "2", "--engine", "mock"]]
    d_.vars["model"].set("project")
    assert core.build_commands("deploy_project", a.params("deploy_project"), pr)[0][3:] == ["predict", str(pr.root), "--source", str(vid)]
    # SLURM option: only the validation is exercised (nothing is ever submitted from a test)
    assert a.slurm() is None
    a.slurm_vars["on"].set(True)
    a.slurm_vars["partition"].set("gpu; rm -rf ~")
    R_["dataset"].run_button.invoke()
    pump(root)
    assert a.job is None and "SLURM partition" in R_["dataset"].msg_var.get()
    a.slurm_vars["on"].set(False)
    # "Open Animal World Run" starts the other app as its own process (argument list only checked here)
    started = []
    from animalworld.studio import app as app_mod
    monkeypatch.setattr(app_mod.subprocess, "Popen", lambda argv, **kw: started.append((argv, kw)))
    a.run_app_button.invoke()
    assert started[0][0] == [sys.executable, "-m", "animalworld", "run"] and started[0][1]["shell"] is False
    logs = sorted((proj / "logs").glob("studio_*.log"))
    assert len(logs) >= 9
    for p in proj.rglob("*"):
        if p.is_file():
            assert SECRET not in p.read_text(errors="replace"), p


def test_stop_button_and_window_stays_responsive(app, monkeypatch):
    a, root, tmp = app
    from animalworld.studio import core
    a.rows["project"].vars["path"].set(str(tmp / "p"))
    a.rows["project"].run_button.invoke()
    pump(root)
    real = core.build_commands
    monkeypatch.setattr(core, "build_commands", lambda action, params, project, python=None:
                        [[sys.executable, "-c", "import time\nfor i in range(600):\n    print('tick', i, flush=True); time.sleep(0.2)"]]
                        if action == "export" else real(action, params, project, python))
    a.rows["export"].run_button.invoke()
    assert a.busy
    pump(root, lambda: "tick 3" in a.rows["export"].log_content(), 30)       # the log is live while the job runs
    a.rows["literature"].run_button.invoke()                                 # buttons are disabled while a job runs
    pump(root)
    assert a.job.action == "export"
    n = [0]
    root.after(10, lambda: n.__setitem__(0, 1))                              # the event loop is free during the job
    pump(root, lambda: n[0] == 1, 5)
    a.stop_button.invoke()
    pump(root, lambda: a.job is None, 20)
    assert a.status_var.get().startswith("Stopped: export") and str(a.stop_button.cget("state")) == "disabled"


def test_open_existing_project_and_main_refuses_bad_folder(tmp_path, monkeypatch, capsys):
    from animalworld.studio import app as app_mod
    assert app_mod.main([str(tmp_path / "missing")]) == 2
    assert "not an animalworld project" in capsys.readouterr().err
    monkeypatch.setenv("ANIMALWORLD_CONFIG_DIR", str(tmp_path / "cfg"))
    pr = Project.init(tmp_path / "proj", "pig", "farm pen", overrides={"prompts": {"n": 7}, "train": {"min_accepted_images": 55}})
    root = make_root()
    try:
        a = app_mod.StudioApp(root, str(pr.root))
        pump(root)
        assert a.rows["project"].vars["species"].get() == "pig" and a.rows["prompts"].vars["n"].get() == "7"
        assert a.rows["train"].vars["min_accepted_images"].get() == "55" and "of 55 needed" in a.rows["train"].status_var.get()
    finally:
        root.destroy()


def test_no_display_message(monkeypatch, capsys):
    import tkinter as tk
    from animalworld.studio import app as app_mod

    def boom(*a, **k):
        raise tk.TclError("no display name and no $DISPLAY environment variable")
    monkeypatch.setattr(app_mod.tk, "Tk", boom)
    assert app_mod.main([]) == 2
    err = capsys.readouterr().err
    assert "needs a display" in err and "ssh -X" in err
