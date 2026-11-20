"""`animalworld demo`: clips are found locally, copied from a folder or downloaded, always SHA-256 checked; headless run."""
import hashlib
import json

import pytest


def _clip(tmp_path, name="demo_a.mp4", data=b"0123456789" * 1000):
    d = tmp_path / "site"
    d.mkdir(exist_ok=True)
    (d / name).write_bytes(data)
    return dict(file=name, sha256=hashlib.sha256(data).hexdigest(), size=len(data), animals=2, about="test clip")


def test_demo_registry_and_placeholder(tmp_path, monkeypatch, capsys):
    from animalworld.zeroshot import demo as D
    assert D.DEMO_URL == "https://github.com/GuangWei-Zhang-25/AnimalWorld/releases/download/demo-v1/" and D.demo_url() == ""   # the test address is the placeholder
    for e in D.CLIPS:
        assert len(e["sha256"]) == 64 and e["file"].endswith(".mp4") and e["size"] < 15 * 1000 ** 2 and 1 <= e["animals"] <= 10
    monkeypatch.setenv("ANIMALWORLD_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(D, "CLIPS", [_clip(tmp_path)])
    assert D.main([]) == 2 and "placeholder" in capsys.readouterr().err          # nothing local, no address: a clear message
    assert D.main(["--headless"]) == 2
    assert D.main(["--list"]) == 0 and "demo_a.mp4" in capsys.readouterr().out


def test_demo_fetch_from_folder_file_url_and_http(tmp_path, monkeypatch):
    from animalworld import weights as W
    from animalworld.zeroshot import demo as D
    e = _clip(tmp_path)
    monkeypatch.setattr(D, "CLIPS", [e])
    site = tmp_path / "site"
    for k, url in enumerate((str(site), "file://" + str(site))):                    # a local folder, as a path and as file://
        monkeypatch.setenv("ANIMALWORLD_CACHE", str(tmp_path / f"cache{k}"))
        monkeypatch.setenv("ANIMALWORLD_DEMO_URL", url)
        p = D.fetch()[0]
        assert p == tmp_path / f"cache{k}" / "demo" / e["file"] and W.sha256(p) == e["sha256"]
        assert D.fetch()[0] == p                                                     # second call: cache hit
    import threading
    from functools import partial
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
    srv = ThreadingHTTPServer(("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=str(site)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        monkeypatch.setenv("ANIMALWORLD_CACHE", str(tmp_path / "cache_http"))
        monkeypatch.setenv("ANIMALWORLD_DEMO_URL", f"http://127.0.0.1:{srv.server_address[1]}")
        assert W.sha256(D.fetch()[0]) == e["sha256"]
        (site / e["file"]).write_bytes(b"x" * e["size"])                             # the site now serves a wrong file
        monkeypatch.setenv("ANIMALWORLD_CACHE", str(tmp_path / "cache_bad"))
        with pytest.raises(W.WeightsError) as ex:
            D.fetch()
        assert "SHA-256" in str(ex.value) and not (tmp_path / "cache_bad" / "demo" / e["file"]).exists()
    finally:
        srv.shutdown()
    monkeypatch.setenv("ANIMALWORLD_DEMO_URL", str(site))                            # a wrong file in a local folder is refused too
    monkeypatch.setenv("ANIMALWORLD_CACHE", str(tmp_path / "cache_bad2"))
    with pytest.raises(W.WeightsError):
        D.fetch()
    own = tmp_path / "own"; own.mkdir(); (own / e["file"]).write_bytes(b"0123456789" * 1000)
    monkeypatch.setenv("ANIMALWORLD_DEMO", str(own))                                 # a folder the user already has wins
    assert D.fetch()[0] == own / e["file"]


def test_demo_headless_runs_the_engine(tmp_path, monkeypatch):
    pytest.importorskip("cv2")
    from test_zeroshot import _video
    from animalworld.zeroshot import demo as D
    site = tmp_path / "site"; site.mkdir()
    v = _video(site / "pair.mp4", 2, 60)
    e = dict(file="pair.mp4", sha256=hashlib.sha256(v.read_bytes()).hexdigest(), size=v.stat().st_size, animals=2, about="synthetic")
    monkeypatch.setattr(D, "CLIPS", [e])
    monkeypatch.setenv("ANIMALWORLD_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("ANIMALWORLD_DEMO_URL", str(site))
    assert D.main(["--headless", "--out", str(tmp_path / "out"), "--engine", "mock"]) == 0
    s = json.loads((tmp_path / "out" / "pair" / "summary.json").read_text())
    assert s["animals"] == 2 and s["count_confirmed_by"] == "user" and (tmp_path / "cache" / "demo" / "pair.mp4").exists()
    from animalworld.cli import main
    assert main(["demo", "--list"]) == 0
