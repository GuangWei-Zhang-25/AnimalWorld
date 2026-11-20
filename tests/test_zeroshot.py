"""Zero-shot engine (mock engine, CPU) and the Animal World Run window. No model weights are used here."""
import json
import os
import stat

import numpy as np
import pytest

from tkutil import make_root, pump, shot


def _video(path, n_animals=3, frames=90):
    cv2 = pytest.importorskip("cv2")
    from animalworld.backends.generators.mock import render, simulate
    W, H = 320, 240
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
    apps = ["black", "white", "brown", "grey"][:n_animals]
    for f, animals in enumerate(simulate(n_animals, frames, W, H, 1)):
        img, _ = render(W, H, animals, apps, [], 1, f)
        vw.write(np.asarray(img)[:, :, ::-1].copy())
    vw.release()
    return path


def test_windows_and_seam_stitch():
    from animalworld.zeroshot import pipeline as P
    assert P.windows(200, 10, 60, 2) == [(0, 200)]
    w = P.windows(2000, 10, 60, 2)
    assert w[0] == (0, 600) and w[1][0] == 580 and w[-1][0] + w[-1][1] == 2000
    assert all(b[0] < a[0] + a[1] for a, b in zip(w, w[1:]))            # consecutive windows overlap
    prev = np.array([[[10, 10], [200, 50], [90, 300]]] * 5, float)
    new = prev[:, [2, 0, 1]] + 1.0                                      # same animals, other order
    perm, worst, margin = P.stitch(prev, new, bl=50.0)
    assert perm == [2, 0, 1] and worst < 0.1 and margin > 1


def test_count_rule_matches_agent_rule():
    from animalworld.zeroshot import pipeline as P
    t, c = [], []
    for f in range(40):                                                 # 3 animals, far apart, one missed on a third of frames
        for k in range(3 if f % 3 else 2):
            t.append(f); c.append([100 + 300 * k, 100])
    D = dict(t=np.array(t), score=np.full(len(t), 0.9), area=np.full(len(t), 2500.0), c=np.array(c, float), T=40,
             box=np.zeros((len(t), 4)), W=1000, H=400, mbuf=np.zeros(0, np.uint8), moff=np.zeros(len(t) + 1, int))
    n, bl, per = P.count_rule(D)
    assert n == 3 and bl == pytest.approx(100.0) and per.max() == 3


def test_zeroshot_cli_mock_engine(tmp_path):
    from animalworld.cli import main
    from animalworld.zeroshot import pipeline as P
    v = _video(tmp_path / "three mice; $(id).mp4")                       # a hostile file name is only ever an argument
    out = tmp_path / "out"
    assert main(["zeroshot", str(v), "--out", str(out), "--yes", "--engine", "mock"]) == 0
    d = out / P.slug(v.stem)
    for f in ("tracks.csv", "masks.npz", "overlay.mp4", "contact_sheet.jpg", "count_sheet.jpg", "summary.html", "summary.json", "run.log"):
        assert (d / f).exists() and (d / f).stat().st_size > 0, f
    assert not (d / "_work").exists()                                    # temporary frames removed
    s = json.loads((d / "summary.json").read_text())
    assert s["animals"] == 3 and s["animals_auto_count"] == 3 and s["frames_analysed"] == 90 and s["output_every_nth_frame"] == 1   # every camera frame
    assert s["sam3"] is False                                              # the default needs nothing gated
    assert s["frames_with_all_animals_pct"] > 90 and "without a look" in s["count_confirmed_by"]
    rows = (d / "tracks.csv").read_text().splitlines()
    assert rows[0].startswith("video_frame,t_s,id,x,y,x0,y0,x1,y1,det_score,status,source,nose_x") and len(rows) > 240
    assert [r.split(",")[0] for r in rows[1:7:3]] == ["0", "1"] and {r.split(",")[11] for r in rows[1:]} == {"tracked"}
    assert main(["zeroshot", str(v), "--out", str(tmp_path / "out10"), "--yes", "--engine", "mock", "--analysis-frames-only"]) == 0
    assert json.loads((tmp_path / "out10" / P.slug(v.stem) / "summary.json").read_text())["frames_analysed"] == 30
    assert {r.split(",")[2] for r in rows[1:]} == {"1", "2", "3"}
    assert main(["zeroshot", str(tmp_path / "missing.mp4"), "--out", str(out), "--yes", "--engine", "mock"]) == 2


def test_run_window_mock_engine(tmp_path):
    """The real window, driven through its widgets: add, start, count shown, correct + confirm, results; then Stop."""
    root = make_root()
    from animalworld.zeroshot.app import RunApp
    v1, v2 = _video(tmp_path / "cage_a.mp4", 3), _video(tmp_path / "cage_b.mp4", 2)
    app = RunApp(root, engine="mock")
    try:
        pump(root, lambda: app.engine is not None, 30)
        assert "TEST engine" in app.device_var.get()
        app.start_button.invoke(); pump(root)
        assert not app.busy()                                            # nothing added: refused with a message
        assert app.add_paths([str(tmp_path)]) == 2 and app.add_paths([str(v1)]) == 0
        assert app.out_var.get().endswith("animalworld_results")
        app.out_var.set(str(tmp_path / "res"))
        shot(root, "run_start")
        app.start_button.invoke()
        pump(root, lambda: all(j.status == "confirm" for j in app.jobs.values()), 120)
        a, b = app.jobs["cage_a"], app.jobs["cage_b"]
        assert (a.n_auto, b.n_auto) == (3, 2) and a.count_sheet.exists()
        assert str(app.confirm_button["state"]) == "normal" and app._photo is not None
        shot(root, "run_confirm")
        app.tree.selection_set("cage_a"); pump(root)
        assert app.count_var.get() == "3"
        app.count_var.set("abc"); app.confirm_button.invoke(); pump(root)
        assert a.status == "confirm"                                     # not a number: refused
        app.count_var.set("3"); app.confirm_button.invoke()
        pump(root, lambda: a.status == "done", 120)
        assert (tmp_path / "res" / "cage_a" / "tracks.csv").exists() and b.status == "confirm"
        app.tree.selection_set("cage_b"); pump(root)
        app.count_var.set("2"); app.confirm_button.invoke()
        pump(root, lambda: not app.busy(), 120); pump(root, None, settle=0.5)
        assert b.status == "done" and str(app.start_button["state"]) == "normal"
        s = json.loads((tmp_path / "res" / "cage_b" / "summary.json").read_text())
        assert s["animals"] == 2 and s["count_confirmed_by"] == "user"
        app.tree.selection_set("cage_a"); pump(root)
        assert str(app.open_button["state"]) == "normal" and "tracks.csv" in app.confirm_var.get()
        shot(root, "run_done")
        # Stop while a video waits for its count: the worker ends and the window stays usable
        v3 = _video(tmp_path / "cage_c.mp4", 2)
        app.add_paths([str(v3)]); app.start_button.invoke()
        pump(root, lambda: app.jobs["cage_c"].status == "confirm", 120)
        app.stop_button.invoke()
        pump(root, lambda: not app.busy(), 30); pump(root, None, settle=0.5)
        assert app.jobs["cage_c"].status == "stopped" and str(app.start_button["state"]) == "normal"
        app.start_button.invoke()                                        # ... and pressing Start again resumes the stopped video
        pump(root, lambda: app.jobs["cage_c"].status == "confirm", 120)
        app.tree.selection_set("cage_c"); pump(root)
        app.count_var.set("2"); app.confirm_button.invoke()
        pump(root, lambda: not app.busy(), 120); pump(root, None, settle=0.5)
        assert app.jobs["cage_c"].status == "done" and (tmp_path / "res" / "cage_c" / "tracks.csv").exists()
        assert app.sam3_var.get() is False                               # the gated SAM 3.1 witness is off unless ticked
    finally:
        app.stop_all.set()
        root.destroy()


def test_studio_run_on_your_videos_starts_the_same_engine(tmp_path):
    """Studio's last row ("MouseWorld pretrained") runs `animalworld zeroshot` as a job: same engine as the Run window."""
    import time
    pytest.importorskip("cv2")
    from animalworld.studio import core as S
    v = _video(tmp_path / "pair.mp4", 2, 60)
    st = S.Studio()
    st.open_project({"path": str(tmp_path / "proj"), "species": "mouse", "description": "home cage"})
    r = st.run("deploy_pretrained", None, {"videos": [str(v)], "out": str(tmp_path / "zs"), "engine": "mock", "n_animals": "2"})
    job = st.jobs[r["job"]]
    assert job.cmds[0][1:4] == ["-m", "animalworld", "zeroshot"] and "--yes" in job.cmds[0]
    t0 = time.time()
    while job.status == "running" and time.time() - t0 < 180:
        time.sleep(0.2)
    assert job.status == "done", job.info()["text"][-800:]
    s = json.loads((tmp_path / "zs" / "pair" / "summary.json").read_text())
    assert s["animals"] == 2 and s["count_confirmed_by"] == "user" and (tmp_path / "zs" / "pair" / "overlay.mp4").exists()
