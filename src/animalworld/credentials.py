"""API tokens for optional commercial backends.

Lookup order for a credential NAME:
  1. environment variable ANIMALWORLD_<NAME> (upper-case), then the bare <NAME>;
  2. the file ~/.config/animalworld/credentials.toml (or $ANIMALWORLD_CONFIG_DIR / $XDG_CONFIG_HOME).

Rules enforced here:
  * the file is created with mode 0600 and is refused if group/other can read it;
  * the file may never live inside a project directory;
  * values are never printed, logged or written into project files; `names()` returns names only;
  * `redact()` removes any known value from a string before it is shown.
"""
from __future__ import annotations

import os
import re
import stat
from pathlib import Path
from typing import Dict, List, Optional

_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
_LOADED_VALUES: set = set()


class CredentialError(RuntimeError):
    pass


def config_dir() -> Path:
    if os.environ.get("ANIMALWORLD_CONFIG_DIR"):
        return Path(os.environ["ANIMALWORLD_CONFIG_DIR"])
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "animalworld"


def credentials_path() -> Path:
    return config_dir() / "credentials.toml"


def _check_name(name: str) -> str:
    if not _NAME.match(name or ""):
        raise CredentialError("credential names are letters, digits and underscores, starting with a letter")
    return name


def _check_location(path: Path) -> None:
    p = path.resolve()
    for parent in [p.parent, *p.parent.parents]:
        if (parent / "project.yaml").exists():
            raise CredentialError(
                f"refusing to keep credentials inside an animalworld project directory ({parent}); "
                "use ~/.config/animalworld/credentials.toml or environment variables")


def _check_mode(path: Path) -> None:
    if os.name == "nt":
        return
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise CredentialError(
            f"{path} is readable by other users (mode {mode:o}); run: chmod 600 {path}")


def _parse(text: str) -> Dict[str, str]:
    """Minimal TOML subset: flat `name = "value"` lines, comments, optional [section] headers (ignored)."""
    out: Dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("["):
            continue
        m = re.match(r'^([A-Za-z][A-Za-z0-9_]*)\s*=\s*"((?:[^"\\]|\\.)*)"\s*(#.*)?$', line)
        if not m:
            m2 = re.match(r"^([A-Za-z][A-Za-z0-9_]*)\s*=\s*'([^']*)'\s*(#.*)?$", line)
            if not m2:
                raise CredentialError("credentials.toml: each line must be  name = \"value\"")
            out[m2.group(1)] = m2.group(2)
            continue
        out[m.group(1)] = m.group(2).replace('\\"', '"').replace("\\\\", "\\")
    return out


def _dump(d: Dict[str, str]) -> str:
    head = "# animalworld credentials. Keep this file private (mode 600). Never commit it.\n"
    return head + "".join(
        '%s = "%s"\n' % (k, v.replace("\\", "\\\\").replace('"', '\\"')) for k, v in sorted(d.items()))


def _load_file() -> Dict[str, str]:
    path = credentials_path()
    if not path.exists():
        return {}
    _check_location(path)
    _check_mode(path)
    d = _parse(path.read_text(encoding="utf-8"))
    _LOADED_VALUES.update(v for v in d.values() if v)
    return d


def get(name: str, required: bool = True) -> Optional[str]:
    """Return the secret for `name`, or None (or raise if `required`). Never log the result."""
    _check_name(name)
    for env in (f"ANIMALWORLD_{name.upper()}", name.upper(), name):
        v = os.environ.get(env)
        if v:
            _LOADED_VALUES.add(v)
            return v
    v = _load_file().get(name) or _load_file().get(name.lower())
    if v:
        return v
    if required:
        raise CredentialError(
            f"credential {name!r} not found: set the environment variable ANIMALWORLD_{name.upper()} "
            f"or run `animalworld credentials set {name}`")
    return None


def set_value(name: str, value: str) -> Path:
    _check_name(name)
    if not value:
        raise CredentialError("empty value")
    path = credentials_path()
    _check_location(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path.parent, 0o700)
    except OSError:
        pass
    d = {}
    if path.exists():
        os.chmod(path, 0o600)
        d = _parse(path.read_text(encoding="utf-8"))
    d[name] = value
    fd = os.open(str(path) + ".tmp", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(_dump(d))
    os.chmod(str(path) + ".tmp", 0o600)
    os.replace(str(path) + ".tmp", path)
    return path


def unset(name: str) -> bool:
    path = credentials_path()
    if not path.exists():
        return False
    d = _parse(path.read_text(encoding="utf-8"))
    if name not in d:
        return False
    del d[name]
    fd = os.open(str(path) + ".tmp", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(_dump(d))
    os.replace(str(path) + ".tmp", path)
    return True


# ANIMALWORLD_* variables that are settings of the package, not credentials: never listed as credential names.
SETTINGS = frozenset(("CONFIG_DIR", "CACHE", "WEIGHTS", "WEIGHTS_URL", "DEMO", "DEMO_URL", "THEME", "UI_SCALE", "SAM3", "TRACKER_DETECTOR",
                      "ENV", "PACKAGE", "SAM3_SOURCE", "TORCH_INDEX", "LD_PRELOAD", "TEST_SOURCE", "FFMPEG", "SEED_CUT"))


def _is_setting(name: str) -> bool:
    return name in SETTINGS or name.startswith("PY_")


def names() -> List[dict]:
    """Names only (and where each one comes from). Values are never returned here."""
    out = [{"name": k, "source": "file"} for k in sorted(_load_file())]
    for k in sorted(os.environ):
        if k.startswith("ANIMALWORLD_") and os.environ[k] and not _is_setting(k[len("ANIMALWORLD_"):]):
            out.append({"name": k[len("ANIMALWORLD_"):].lower(), "source": "environment"})
    return out


def redact(text: str) -> str:
    """Remove every secret value seen in this process from a string."""
    s = str(text)
    for v in sorted(_LOADED_VALUES, key=len, reverse=True):
        if v and len(v) >= 4:
            s = s.replace(v, "***")
    return s


SECRET_KEY = re.compile(r"(api[_-]?key|token|secret|password|passwd|authorization)$", re.I)


def find_inline_secrets(obj, path: str = "") -> List[str]:
    """Paths in a config mapping whose key looks like a secret and whose value is a non-empty literal.

    Config files hold the NAME of a credential (e.g. ``token_credential: my_api``), never its value.
    Keys ending in ``_credential``/``_env`` are therefore allowed.
    """
    bad: List[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else str(k)
            if isinstance(v, (dict, list)):
                bad += find_inline_secrets(v, p)
            elif isinstance(v, str) and v and SECRET_KEY.search(str(k)):
                bad.append(p)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            bad += find_inline_secrets(v, f"{path}[{i}]")
    return bad
