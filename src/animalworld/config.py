"""Project directory model: one folder per species project, described by project.yaml.

Every stage reads its inputs from, and writes its outputs to, this folder, so a run can be
stopped and resumed at any stage and inspected with ordinary tools.
"""
from __future__ import annotations

import copy
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from . import credentials
from .utils import now_utc, read_json, slug, write_json

CONFIG_NAME = "project.yaml"
SLOTS_NAME = "slots.yaml"

DEFAULTS: Dict[str, Any] = {
    "animalworld": 1,
    "species": "animal",
    "description": "",
    "literature": {
        "backend": "offline",          # offline | europepmc | pubmed
        "query": None,                 # None -> built from species + description
        "limit": 200,
        "source": None,                # offline: CSV/JSONL of abstracts; None -> bundled sample for the species
        "contact_email": "",           # optional, sent to the literature API only if you fill it in
        "rate_limit_s": 0.5,
    },
    "taxonomy": {"min_hits": 1, "max_candidates": 30, "literature_weight": 1.0,
                 "candidate_stop_words": []},          # extra words that disqualify a candidate phrase
    "prompts": {
        "n": 40, "seed": 0, "writer": "template", "writer_options": {},
        "min_distinct": 0.95, "empty_fraction": 0.05, "val_fraction": 0.05,
        "megapixels": 1.25, "size_multiple": 32,
    },
    "generate": {"backend": "mock", "options": {}, "frame_stride": 8},
    "propose": {
        "backend": "mock", "options": {},
        "link_identities": True,       # multi-frame units: proposed identity (track_id) per instance, by box linking
        "gates": {"min_score": 0.5, "count_gate": True, "min_area_frac": 2e-4, "max_area_frac": 0.2,
                  "max_border_frac": 0.4},
    },
    "review": {"auto_accept": False, "host": "127.0.0.1", "port": 8765},
    "dataset": {"formats": ["coco", "yolo"], "task": "segment", "apply_looks": False, "pose_crops": False},
    "train": {"backend": "yolo", "task": "detect", "min_accepted_images": 200, "options": {}},
    "export": {"format": "onnx", "options": {}},
    "predict": {"backend": "onnx", "weights": None, "source": None, "conf": 0.25, "options": {}},
    "track": {"backend": "iou", "n_animals": None, "options": {}},
}

GENERIC_SLOTS: Dict[str, Any] = {
    "animal": {"singular": "animal", "plural": "animals", "definition": "", "negatives": []},
    "keypoints": [], "flip_idx": [], "skeleton": [],
    "slots": {
        "appearance": [{"id": "typical", "text": "{singular} of typical appearance"}],
        "group_size": [{"id": 1, "weight": 50}, {"id": 2, "weight": 30}, {"id": 3, "weight": 20}],
        "behaviour": ["walking", "standing still", "resting", "feeding", "exploring"],
        "lighting": ["even overhead light", "light from one side with soft shadows", "dim light", "bright daylight"],
        "camera": [{"id": "top", "text": "Seen straight down from above; no camera in the picture."},
                   {"id": "oblique", "text": "Seen from above at an oblique angle; no camera in the picture."},
                   {"id": "side", "text": "Seen from the side at the animal's height; no camera in the picture."}],
        "arena": ["a plain enclosure with a {floor} floor", "an outdoor pen", "an indoor housing pen with bedding"],
        "scale": [{"id": "small", "text": "each {singular} small, about one tenth of the image width"},
                  {"id": "medium", "text": "each {singular} about one fifth of the image width"},
                  {"id": "large", "text": "each {singular} about one third of the image width"}],
        "occlusion": [{"id": "none", "text": "whole bodies visible", "weight": 3},
                      {"id": "behind", "text": "one {singular} partly hidden behind an object"},
                      {"id": "border", "text": "one {singular} partly cut off by the edge of the picture"}],
        "quality": ["ordinary research recording", "surveillance-style monitoring frame", "machine-vision camera frame"],
        "floor": ["white", "grey", "black", "beige"], "wall": ["white", "grey", "clear acrylic"],
        "bedding": ["wood shavings", "straw", "paper bedding"],
        "aspect": [1.0, 1.3333, 1.7778], "look": ["none"],
    },
    "template": ("A realistic photograph, {quality}: {arena}. {camera} {lighting}. There {is_are} exactly "
                 "{count_word} {animal} and no other animals: {who}, {definition}, {occlusion}, {scale}, {behaviour}."),
    "empty_template": ("A realistic photograph, {quality}: {arena}. {camera} {lighting}. The scene is completely "
                       "empty: no {singular}, no animal of any kind."),
    "negative_prompt": "cartoon, illustration, painting, 3d render, text, watermark, people, camera, tripod, "
                       "extra animals, deformed animals, merged animals",
}


class ConfigError(RuntimeError):
    pass


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def examples_dir() -> Optional[Path]:
    """Bundled example species (installed wheel: animalworld/examples; source tree: <repo>/examples)."""
    here = Path(__file__).resolve().parent
    for cand in (here / "examples", here.parents[1] / "examples"):
        if cand.is_dir():
            return cand
    return None


def species_preset(species: str) -> Optional[Path]:
    d = examples_dir()
    if d and (d / slug(species)).is_dir():
        return d / slug(species)
    return None


class Project:
    """A project directory plus its merged configuration."""

    def __init__(self, root: Path, cfg: Dict[str, Any]):
        self.root = Path(root).resolve()
        self.cfg = cfg

    # ---- loading / creating -------------------------------------------------------------
    @classmethod
    def load(cls, root) -> "Project":
        root = Path(root)
        f = root / CONFIG_NAME
        if not f.exists():
            raise ConfigError(f"{root} is not an animalworld project (no {CONFIG_NAME}); run `animalworld init {root}`")
        raw = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
        bad = credentials.find_inline_secrets(raw)
        if bad:
            raise ConfigError(
                "project.yaml must not contain secrets (found a literal value at: " + ", ".join(bad) + "). "
                "Store the token with `animalworld credentials set NAME` and reference it by name "
                "(e.g. token_credential: NAME).")
        return cls(root, deep_merge(DEFAULTS, raw))

    @classmethod
    def init(cls, root, species: str, description: str = "", overrides: dict = None, force: bool = False) -> "Project":
        root = Path(root)
        if (root / CONFIG_NAME).exists() and not force:
            raise ConfigError(f"{root / CONFIG_NAME} already exists (use --force to overwrite the config)")
        root.mkdir(parents=True, exist_ok=True)
        user: Dict[str, Any] = {"animalworld": 1, "species": species, "description": description}
        preset = species_preset(species)
        slots = copy.deepcopy(GENERIC_SLOTS)
        slots["animal"]["singular"] = species
        slots["animal"]["plural"] = species + "s"
        if preset:
            if (preset / CONFIG_NAME).exists():
                p = yaml.safe_load((preset / CONFIG_NAME).read_text(encoding="utf-8")) or {}
                p.pop("species", None)
                if description:
                    p.pop("description", None)
                user = deep_merge(user, p)
            if (preset / SLOTS_NAME).exists():
                slots = yaml.safe_load((preset / SLOTS_NAME).read_text(encoding="utf-8"))
            for extra in preset.glob("abstracts_sample.*"):
                (root / "literature").mkdir(exist_ok=True)
                shutil.copy(extra, root / "literature" / extra.name)
                user = deep_merge(user, {"literature": {"source": f"literature/{extra.name}"}})
        user = deep_merge(user, overrides or {})
        text = ("# animalworld project. Edit freely; every stage re-reads this file.\n"
                "# Never put API tokens here: use `animalworld credentials set NAME`.\n"
                + yaml.safe_dump(user, sort_keys=False, allow_unicode=True))
        (root / CONFIG_NAME).write_text(text, encoding="utf-8")
        if not (root / SLOTS_NAME).exists() or force:
            (root / SLOTS_NAME).write_text(
                "# Slot vocabulary for the prompt writer. One prompt = one independent draw from every slot.\n"
                "# Add values, weights and literature keywords (regex) here; `animalworld taxonomy` reports\n"
                "# which values the literature supports and proposes new candidates.\n"
                + yaml.safe_dump(slots, sort_keys=False, allow_unicode=True, width=120), encoding="utf-8")
        (root / ".gitignore").write_text("generated/\nmodels/\n*.lock\ncredentials.toml\n.env\n", encoding="utf-8")
        return cls.load(root)

    # ---- paths --------------------------------------------------------------------------
    def path(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    def resolve(self, p) -> Path:
        """A path from the config: absolute, or relative to the project folder."""
        p = Path(str(p)).expanduser()
        return p if p.is_absolute() else self.root / p

    @property
    def species(self) -> str:
        return str(self.cfg["species"])

    def slots(self) -> Dict[str, Any]:
        f = self.root / SLOTS_NAME
        if not f.exists():
            raise ConfigError(f"missing {f}")
        return yaml.safe_load(f.read_text(encoding="utf-8"))

    def categories(self) -> List[str]:
        return [self.slots().get("animal", {}).get("singular") or self.species]

    def keypoints(self) -> List[str]:
        return list(self.slots().get("keypoints") or [])

    def options(self, section: str, backend: str) -> Dict[str, Any]:
        """Options for `backend` in a config section. The section's `options` block belongs to the configured
        backend only; a different backend chosen on the command line gets `options_for.<backend>` (or nothing)."""
        cfg = self.cfg[section]
        if backend == cfg.get("backend"):
            return dict(cfg.get("options") or {})
        return dict((cfg.get("options_for") or {}).get(backend) or {})

    # ---- stage state --------------------------------------------------------------------
    def state(self) -> Dict[str, Any]:
        return read_json(self.root / "state.json", {}) or {}

    def record(self, stage: str, mode: str, summary: dict) -> None:
        """mode: executed | dry-run | skipped | waiting."""
        st = self.state()
        st[stage] = {"mode": mode, "utc": now_utc(), "summary": summary}
        write_json(self.root / "state.json", st)
