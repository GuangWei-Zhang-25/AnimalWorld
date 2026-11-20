import json
import os
import stat
import subprocess
import sys

import pytest
import yaml

from animalworld import credentials, registry
from animalworld.config import ConfigError, Project


def test_import_is_light():
    code = ("import sys, animalworld, animalworld.cli, animalworld.pipeline, animalworld.registry;"
            "bad=[m for m in ('torch','diffusers','ultralytics','onnxruntime','cv2','rfdetr','sam3','anthropic') if m in sys.modules];"
            "print(bad); sys.exit(1 if bad else 0)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path)))
    assert r.returncode == 0, r.stdout + r.stderr


def test_all_builtin_backends_resolve_without_heavy_imports():
    code = ("import sys\nfrom animalworld import registry\n"
            "for k in registry.KINDS:\n    for n in registry.available(k):\n        registry.get(k, n)\n"
            "bad=[m for m in ('torch','diffusers','ultralytics','onnxruntime','rfdetr','sam3') if m in sys.modules]\n"
            "print(bad); sys.exit(1 if bad else 0)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path)))
    assert r.returncode == 0, r.stdout + r.stderr


def test_registry_custom_backend_and_unknown():
    class Dummy:
        def __init__(self, options, **kw):
            self.opt = options
    registry.register("generators", "dummy", Dummy)
    assert "dummy" in registry.available("generators")
    assert registry.create("generators", "dummy", {"a": 1}).opt == {"a": 1}
    with pytest.raises(registry.BackendError):
        registry.get("generators", "nope")
    with pytest.raises(registry.BackendError):
        registry.register("nonsense", "x", Dummy)


def test_credentials_roundtrip_mode_and_names_only(capsys):
    secret = "sk-test-SECRET-123456"
    path = credentials.set_value("my_api", secret)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert credentials.get("my_api") == secret
    assert [r["name"] for r in credentials.names()] == ["my_api"]      # the package's own settings (download addresses ...) are not credentials
    assert os.environ.get("ANIMALWORLD_WEIGHTS_URL") and credentials._is_setting("PY_DET") and not credentials._is_setting("MY_API")
    from animalworld.cli import main
    assert main(["credentials", "list"]) == 0
    out = capsys.readouterr().out
    assert "my_api" in out and secret not in out
    assert secret not in credentials.redact(f"error talking to api with {secret}")
    assert credentials.unset("my_api") and credentials.get("my_api", required=False) is None


def test_credentials_refuse_world_readable(tmp_path):
    path = credentials.set_value("a", "value-1234")
    os.chmod(path, 0o644)
    with pytest.raises(credentials.CredentialError, match="chmod 600"):
        credentials.get("a")


def test_credentials_env_wins_and_missing(monkeypatch):
    credentials.set_value("tok", "from-file-0000")
    monkeypatch.setenv("ANIMALWORLD_TOK", "from-env-1111")
    assert credentials.get("tok") == "from-env-1111"
    with pytest.raises(credentials.CredentialError):
        credentials.get("absent")
    with pytest.raises(credentials.CredentialError):
        credentials.get("bad name!")


def test_credentials_never_inside_a_project(project, monkeypatch):
    monkeypatch.setenv("ANIMALWORLD_CONFIG_DIR", str(project.root / "secrets"))
    with pytest.raises(credentials.CredentialError, match="project"):
        credentials.set_value("x", "value-1234")


def test_credentials_stdin_cli(monkeypatch, capsys):
    import io
    from animalworld.cli import main
    monkeypatch.setattr("sys.stdin", io.StringIO("tok-ABCDEFGH\n"))
    assert main(["credentials", "set", "svc", "--stdin"]) == 0
    assert "tok-ABCDEFGH" not in capsys.readouterr().out
    assert credentials.get("svc") == "tok-ABCDEFGH"


def test_project_yaml_with_literal_secret_is_rejected(project):
    f = project.root / "project.yaml"
    cfg = yaml.safe_load(f.read_text())
    cfg["generate"] = {"backend": "http_api", "options": {"api_key": "sk-live-abc"}}
    f.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ConfigError, match="secrets"):
        Project.load(project.root)
    cfg["generate"]["options"] = {"token_credential": "my_api"}     # a NAME is fine
    f.write_text(yaml.safe_dump(cfg))
    Project.load(project.root)


def test_init_mouse_and_new_species(tmp_path):
    m = Project.init(tmp_path / "m", "mouse", "x")
    assert m.keypoints()[0] == "nose" and m.categories() == ["mouse"]
    assert (m.root / "literature" / "abstracts_sample.jsonl").exists()
    z = Project.init(tmp_path / "z", "zebrafish", "larvae in a dish")
    assert z.categories() == ["zebrafish"] and z.keypoints() == []
    with pytest.raises(ConfigError):
        Project.init(tmp_path / "z", "zebrafish")
    p = Project.init(tmp_path / "p", "pig")
    assert p.slots()["animal"]["plural"] == "pigs" and len(p.keypoints()) == 6


def test_cli_help_and_backends(capsys):
    from animalworld.cli import main
    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0
    assert "literature" in capsys.readouterr().out
    assert main(["backends"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert {"mock", "qwen_image", "wan_video", "http_api"} <= set(d["generators"])
    assert main(["status", "/nonexistent/project"]) == 2
