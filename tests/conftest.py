import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
SRC = Path(__file__).resolve().parents[1] / "src"
if os.environ.get("ANIMALWORLD_TEST_SOURCE", "1") == "1" and str(SRC) not in sys.path:
    try:
        import animalworld  # noqa: F401  (installed copy wins when present)
    except ImportError:
        sys.path.insert(0, str(SRC))


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    """Never touch the real ~/.config/animalworld; drop any real ANIMALWORLD_* variables."""
    monkeypatch.setenv("ANIMALWORLD_CONFIG_DIR", str(tmp_path / "cfg"))
    for k in list(os.environ):
        if k.startswith("ANIMALWORLD_") and k != "ANIMALWORLD_CONFIG_DIR":
            monkeypatch.delenv(k, raising=False)
    # tests never contact the release addresses: both downloads point at the "no address" placeholder unless a test sets its own
    monkeypatch.setenv("ANIMALWORLD_WEIGHTS_URL", "https://github.com/OWNER/REPO/releases/download/weights-v1/")
    monkeypatch.setenv("ANIMALWORLD_DEMO_URL", "https://github.com/OWNER/REPO/releases/download/demo-v1/")
    yield


@pytest.fixture
def project(tmp_path):
    from animalworld.config import Project
    return Project.init(tmp_path / "proj", "mouse", "laboratory, home cage and open field",
                        overrides={"prompts": {"n": 24}, "train": {"min_accepted_images": 10}})


@pytest.fixture
def reviewed_project(project):
    from animalworld import pipeline
    pipeline.run_all(project, auto_accept=True, until="dataset")
    return project
