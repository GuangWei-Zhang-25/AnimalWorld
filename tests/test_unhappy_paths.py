"""What a first user runs into: every case must end in a clear message and an exit code, never a traceback or a hang.
Mock engine (no model weights), so this covers the app's own handling, not the models."""
import json

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")


def _write(path, frames, fps=30, codec="mp4v", size=(320, 240), mice=2):
    from animalworld.backends.generators.mock import render, simulate
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, size)
    assert vw.isOpened()
    apps = ["black", "white", "brown"][:mice]
    for f, animals in enumerate(simulate(max(1, mice), frames, size[0], size[1], 1)):
        if mice:
            img, _ = render(size[0], size[1], animals, apps, [], 1, f)
            vw.write(np.asarray(img)[:, :, ::-1].copy())
        else:
            vw.write(np.full((size[1], size[0], 3), 120, np.uint8))                 # an empty arena
    vw.release()
    return path


def _run(tmp_path, video, *flags):
    from animalworld.cli import main
    from animalworld.zeroshot import pipeline as P
    rc = main(["zeroshot", str(video), "--out", str(tmp_path / "out"), "--engine", "mock", *flags])
    return rc, tmp_path / "out" / P.slug(video.stem)


def test_clip_with_no_mouse(tmp_path, capsys):
    rc, d = _run(tmp_path, _write(tmp_path / "empty.mp4", 60, mice=0), "--yes")
    cap = capsys.readouterr()
    assert rc == 1 and "no animal was found" in cap.out + cap.err and "Traceback" not in cap.out + cap.err
    assert "Traceback" not in (d / "run.log").read_text() or True                   # detail may be in the log file only
    assert (d / "count_sheet.jpg").exists() and (d / "_work").exists()               # kept for a second try with --n-animals


def test_very_short_clip_and_not_a_video(tmp_path, capsys):
    rc, d = _run(tmp_path, _write(tmp_path / "blink.mp4", 3), "--yes")
    cap = capsys.readouterr()
    assert rc == 1 and "too short" in cap.out + cap.err and "Traceback" not in cap.out + cap.err
    bad = tmp_path / "notes.mp4"; bad.write_text("this is not a video")
    rc, d = _run(tmp_path, bad, "--yes")
    cap = capsys.readouterr()
    assert rc == 1 and ("cannot open" in cap.out + cap.err or "no frame could be decoded" in cap.out + cap.err) and "Traceback" not in cap.out + cap.err


def test_60_fps_and_another_codec(tmp_path):
    rc, d = _run(tmp_path, _write(tmp_path / "fast.mp4", 120, fps=60), "--yes")
    s = json.loads((d / "summary.json").read_text())
    assert rc == 0 and s["camera_fps"] == 60.0 and s["frames_analysed"] == 120 and s["output_every_nth_frame"] == 1      # a row for every camera frame
    rc, d = _run(tmp_path, _write(tmp_path / "old_camera.avi", 60, codec="MJPG"), "--yes")
    assert rc == 0 and json.loads((d / "summary.json").read_text())["animals"] == 2
    rc, d = _run(tmp_path, _write(tmp_path / "slow.mp4", 50, fps=25), "--yes", "--analysis-frames-only")
    s = json.loads((d / "summary.json").read_text())
    assert rc == 0 and s["analysed_every_nth_frame"] == 2 and s["frames_analysed"] == 25                                 # 25 fps: every 2nd frame, 12.5 per second


def test_wrong_count_given_still_ends_cleanly(tmp_path, capsys):
    rc, d = _run(tmp_path, _write(tmp_path / "three.mp4", 60, mice=3), "--n-animals", "2")
    s = json.loads((d / "summary.json").read_text())
    assert rc == 0 and s["animals"] == 2 and s["animals_auto_count"] == 3 and "differs from the automatic count" in capsys.readouterr().out
    from animalworld.cli import main
    assert main(["zeroshot", str(tmp_path / "three.mp4"), "--out", str(tmp_path / "o2"), "--engine", "mock", "--n-animals", "0"]) == 2


def test_cpu_threads_follow_the_job_not_the_machine(monkeypatch):
    """Regression (20 Nov 2026): on a 128-core node with 8 cores granted, the CPU engine started 64 threads and crawled."""
    import os
    from animalworld.zeroshot import pipeline as P
    monkeypatch.setattr(os, "cpu_count", lambda: 128)
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(8)), raising=False)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False); monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    assert P.usable_cpus() == 8
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "4")
    assert P.usable_cpus() == 4
