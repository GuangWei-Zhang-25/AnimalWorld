"""Weights registry: local folders first, placeholder address, verified / resumed / retried downloads, detector switch."""
import hashlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


def test_registry_local_first_and_placeholder_address(tmp_path, monkeypatch):
    from animalworld import weights as W
    assert W.BASE_URL == "https://github.com/GuangWei-Zhang-25/AnimalWorld/releases/download/weights-v1/"
    assert W.base_url({"own": True}) == ""                                  # the test address is the placeholder: nothing is downloaded
    monkeypatch.delenv("ANIMALWORLD_WEIGHTS_URL"); assert W.base_url({"own": True}) == W.BASE_URL; monkeypatch.setenv("ANIMALWORLD_WEIGHTS_URL", "https://github.com/OWNER/REPO/x/")
    monkeypatch.setenv("ANIMALWORLD_WEIGHTS", str(tmp_path / "w"))
    monkeypatch.setenv("ANIMALWORLD_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    with pytest.raises(W.WeightsError) as e:
        W.resolve("mouseworld-det")
    assert "placeholder" in str(e.value) and "ANIMALWORLD_WEIGHTS_URL" in str(e.value)
    with pytest.raises(W.WeightsError) as e:
        W.resolve("sam3.1-multiplex")
    assert "huggingface.co/facebook/sam3.1" in str(e.value)
    (tmp_path / "w").mkdir()
    (tmp_path / "w" / "mouseworld-det.ckpt").write_bytes(b"x")
    assert W.resolve("mouseworld-det") == tmp_path / "w" / "mouseworld-det.ckpt"
    assert [r for r in W.status(["mouseworld-det"], verify=True)][0]["sha256_ok"] is False
    for e_ in W.REGISTRY.values():
        assert len(e_["sha256"]) == 64 and "/" not in e_["file"]
        assert not e_["own"] or e_["size"] < 2 * 1000 ** 3                 # release assets must stay under 2 GB
    env = W.step_env()
    assert env["YOLO_CONFIG_DIR"].startswith(str(tmp_path / "cache")) and env["YOLO_OFFLINE"] == "true"     # never the user's Ultralytics settings


class _Site:
    """A stand-in release site on 127.0.0.1 with HTTP Range support and switchable faults."""

    def __init__(self, files):
        site = self
        self.files, self.requests, self.cut_after, self.corrupt_first = files, [], None, False

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                name = self.path.lstrip("/")
                site.requests.append((name, self.headers.get("Range"), self.headers.get("User-Agent")))
                if name not in site.files:
                    self.send_error(404)
                    return
                data = site.files[name]
                if site.corrupt_first:
                    site.corrupt_first = False
                    data = b"X" + data[1:]
                start = 0
                rng = self.headers.get("Range")
                if rng:
                    start = int(rng.split("=")[1].split("-")[0])
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
                else:
                    self.send_response(200)
                body = data[start:]
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if site.cut_after is not None:
                    n, site.cut_after = site.cut_after, None
                    self.wfile.write(body[:n])
                    self.wfile.flush()
                    self.connection.close()
                    return
                self.wfile.write(body)
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}/"


def test_download_sha256_resume_and_retry(tmp_path, monkeypatch):
    """The download path against a local stand-in site: verified install, an interrupted download (continued with a Range
    request), a corrupted file (set aside, fetched again once), a wrong file (refused), a missing file (clear error)."""
    from animalworld import weights as W
    data = bytes(range(256)) * 8000                                      # 2 MB
    monkeypatch.setitem(W.REGISTRY["mouseworld-pose"], "sha256", hashlib.sha256(data).hexdigest())
    monkeypatch.setitem(W.REGISTRY["mouseworld-pose"], "size", len(data))
    site = _Site({"mouseworld-pose.pt": data})
    monkeypatch.setenv("ANIMALWORLD_WEIGHTS_URL", site.url)
    monkeypatch.setenv("ANIMALWORLD_WEIGHTS", str(tmp_path / "none"))
    cache = tmp_path / "cache"
    monkeypatch.setenv("ANIMALWORLD_CACHE", str(cache))
    try:
        site.cut_after = 500000                                          # 1. interrupted after 0.5 MB
        p = W.resolve("mouseworld-pose")
        assert p == cache / "weights" / "mouseworld-pose.pt" and p.read_bytes() == data and not p.with_name(p.name + ".part").exists()
        assert [r[1] for r in site.requests] == [None, "bytes=500000-"] and all(r[2].startswith("animalworld/") for r in site.requests)
        assert W.resolve("mouseworld-pose") == p and len(site.requests) == 2          # second call: cache hit, no request
        p.unlink(); site.requests.clear()                                # 2. a partial file left by an earlier run is continued
        p.with_name(p.name + ".part").write_bytes(data[:700000])
        assert W.resolve("mouseworld-pose").read_bytes() == data and site.requests[0][1] == "bytes=700000-"
        p.unlink(); site.requests.clear(); site.corrupt_first = True     # 3. corrupted once: set aside, one fresh download
        assert W.resolve("mouseworld-pose").read_bytes() == data and len(site.requests) == 2
        assert p.with_name(p.name + ".badsum").exists()
        p.unlink(); site.files["mouseworld-pose.pt"] = b"not the model" * 1000          # 4. the site's file is wrong: refused
        monkeypatch.setitem(W.REGISTRY["mouseworld-pose"], "size", len(site.files["mouseworld-pose.pt"]))
        with pytest.raises(W.WeightsError) as e:
            W.resolve("mouseworld-pose")
        assert "SHA-256" in str(e.value) and not p.exists()
        site.requests.clear()                                            # 5. a file the site does not have
        with pytest.raises(W.WeightsError):
            W.download("mouseworld-det")
        assert len(site.requests) == 1
    finally:
        site.srv.shutdown()


def test_tracker_detector_switch(tmp_path, monkeypatch):
    from animalworld import weights as W
    from animalworld.zeroshot import pipeline as P
    monkeypatch.setenv("ANIMALWORLD_WEIGHTS", str(tmp_path))
    for e in W.REGISTRY.values():
        (tmp_path / e["file"]).write_bytes(b"x")
    assert P.Options().tracker_detector == "scored_pair" == P.DEFAULT_TRACKER_DETECTOR      # the configuration that scores best
    assert P.Options().sam3 is False and P.Options().all_frames is True and "sam3.1-multiplex" not in W.FULL       # nothing gated by default
    assert "sam31_witness.py" not in [s[2] for s in P.tracker_steps(tmp_path / "r", P.Options())[3]]
    assert "sam31_witness.py" in [s[2] for s in P.tracker_steps(tmp_path / "r", P.Options(sam3=True))[3]]
    assert [s[4]["MWT_DET_CKPT"].rsplit("/", 1)[1] for s in P.tracker_steps(tmp_path / "r", P.Options())[3] if "MWT_DET_CKPT" in s[4]][1] == "mouseworld-track-det-a.ckpt"
    monkeypatch.setenv("ANIMALWORLD_TRACKER_DETECTOR", "strongest")
    assert P.Options().tracker_detector == "strongest"
    monkeypatch.delenv("ANIMALWORLD_TRACKER_DETECTOR")
    A, B, Dm, steps = P.tracker_steps(tmp_path / "r", P.Options(tracker_detector="strongest"))
    assert B is None and Dm == A
    assert [s[2] for s in steps] == ["detect_dense.py", "track_fb.py", "track_local.py", "detect_local.py", "run_supervisor.py"]
    assert all(s[4]["MWT_DET_CKPT"].endswith("mouseworld-det.ckpt") for s in steps if "MWT_DET_CKPT" in s[4]) and "MWT_SEED2" not in steps[-1][4]
    assert abs(sum(s[5] for s in steps) - 1) < 1e-9
    A, B, Dm, steps = P.tracker_steps(tmp_path / "r", P.Options(tracker_detector="scored_pair"))
    ck = [s[4]["MWT_DET_CKPT"].rsplit("/", 1)[1] for s in steps if "MWT_DET_CKPT" in s[4]]
    assert ck == ["mouseworld-det.ckpt", "mouseworld-track-det-a.ckpt", "mouseworld-track-det-a.ckpt", "mouseworld-track-det-b.ckpt"]
    assert steps[-1][4]["MWT_SEED2"].endswith("agent_b/work") and Dm != A
    A, B, Dm, steps = P.tracker_steps(tmp_path / "r", P.Options(tracker_detector="strongest_two_seeds"))
    ck = [s[4]["MWT_DET_CKPT"].rsplit("/", 1)[1] for s in steps if "MWT_DET_CKPT" in s[4]]
    assert ck == ["mouseworld-det.ckpt", "mouseworld-det.ckpt", "mouseworld-track-det-b.ckpt"] and Dm == A
    assert steps[-1][4]["MWT_SEED2"].endswith("agent_b/work") and len(steps) == 7
    A, B, Dm, steps = P.tracker_steps(tmp_path / "r", P.Options(tracker_detector="strongest_recut"))        # experimental, never the default
    assert [s[2] for s in steps].count("rescale_scores.py") == 2 and steps[1][4]["MWT_SEED_CUT"] == "0.35" and B is None
    assert abs(sum(s[5] for s in steps) - 1) < 1e-9
    with pytest.raises(ValueError):
        P.Options(tracker_detector="other")


def test_fill_rows_between_tracked_frames():
    """Rows for every camera frame: tracked on analysed frames, filled from the nearest detection in between, interpolated
    (and marked so) when no detection can be attached; two animals side by side keep their own detections."""
    import numpy as np
    from animalworld.zeroshot import pipeline as P
    X = np.array([[[100.0, 100.0], [160.0, 100.0]], [[130.0, 100.0], [190.0, 100.0]]])      # two animals 60 px apart, both moving +30 px
    prior, src = P.prior_positions(X, 0, 3)
    assert src == "tracked" and prior[0, 0] == 100
    prior, src = P.prior_positions(X, 1, 3)
    assert src == "between" and prior[0, 0] == pytest.approx(110) and prior[1, 0] == pytest.approx(170)
    dets = [(np.array([172.0, 101.0]), 7), (np.array([111.0, 99.0]), 3)]
    rows = P.fill_rows([0, 1], prior, src, dets, 100.0, False)
    assert [(r["id"], r["source"], r["det"]) for r in rows] == [(1, "filled", 3), (2, "filled", 7)] and rows[0]["x"] == 111.0
    rows = P.fill_rows([0, 1], prior, src, dets[:1], 100.0, True)                           # one animal has no detection
    assert [(r["source"], r["det"], r["status"]) for r in rows] == [("interpolated", None, "flagged"), ("filled", 7, "flagged")]
    assert rows[0]["x"] == pytest.approx(110)
    assert np.isnan(P.prior_positions(X, 4, 3)[0]).all()                                    # after the last analysed frame: nothing
    rows = P.fill_rows([0, 1], X[0], "tracked", dets, 100.0, False)
    assert [r["source"] for r in rows] == ["tracked", "tracked"] and rows[0]["x"] == 100.0  # tracked rows keep the tracker's position
