"""Backend registry.

Every pluggable part of the pipeline (literature source, prompt writer, generator, proposal model,
trainer, predictor, tracker) is looked up by (kind, name). Built-in backends are listed below as
lazy "module:attribute" strings so nothing heavy is imported until a backend is used. Third-party
packages add backends through entry points in the group ``animalworld.<kind>`` or by calling
``register(kind, name, obj)``. See docs/BACKENDS.md.
"""
from __future__ import annotations

import importlib
from typing import Any, Dict, List

KINDS = ("literature", "writers", "generators", "proposers", "trainers", "predictors", "trackers")

_B = "animalworld.backends."
_BUILTIN: Dict[str, Dict[str, str]] = {
    "literature": {
        "offline": _B + "literature.offline:OfflineLiterature",
        "europepmc": _B + "literature.europepmc:EuropePMC",
        "pubmed": _B + "literature.pubmed:PubMed",
    },
    "writers": {
        "template": _B + "writers.template:TemplateWriter",
        "llm": _B + "writers.llm:LLMWriter",
    },
    "generators": {
        "mock": _B + "generators.mock:MockGenerator",
        "qwen_image": _B + "generators.qwen_image:QwenImageGenerator",
        "wan_video": _B + "generators.wan_video:WanVideoGenerator",
        "http_api": _B + "generators.http_api:HTTPAPIGenerator",
    },
    "proposers": {
        "mock": _B + "proposers.mock:MockProposer",
        "sam": _B + "proposers.sam:SAMProposer",
        "yolo": _B + "proposers.yolo:YOLOProposer",          # a trained model (Ultralytics, ONNX or RF-DETR weights)
        "model": _B + "proposers.yolo:YOLOProposer",         # the same class under the name that says what it is
        "pose": _B + "proposers.pose:TopDownPoseProposer",
        "vlm": _B + "proposers.vlm:VLMProposer",
    },
    "trainers": {
        "yolo": _B + "trainers.yolo:YOLOTrainer",
        "rfdetr": _B + "trainers.rfdetr:RFDETRTrainer",
        "sleap_dlc": _B + "trainers.sleap_dlc:SleapDlcExport",
    },
    "predictors": {
        "blob": _B + "deploy.blob:BlobPredictor",
        "onnx": _B + "deploy.onnx:ONNXPredictor",
        "ultralytics": _B + "deploy.native:UltralyticsPredictor",
        "rfdetr": _B + "deploy.native:RFDETRPredictor",
    },
    "trackers": {
        "iou": _B + "trackers.iou:IoUTracker",
        "rules": _B + "trackers.rules:RuleTrackerAdapter",
    },
}
_EXTRA: Dict[str, Dict[str, Any]] = {k: {} for k in KINDS}


class BackendError(RuntimeError):
    pass


def register(kind: str, name: str, obj: Any) -> None:
    """Register a backend class (or a 'module:attr' string) at run time."""
    if kind not in KINDS:
        raise BackendError(f"unknown backend kind {kind!r}; kinds: {', '.join(KINDS)}")
    _EXTRA[kind][name] = obj


def _entry_points(kind: str) -> Dict[str, Any]:
    try:
        from importlib import metadata
        eps = metadata.entry_points()
        group = f"animalworld.{kind}"
        sel = eps.select(group=group) if hasattr(eps, "select") else eps.get(group, [])
        return {ep.name: ep for ep in sel}
    except Exception:
        return {}


def available(kind: str) -> List[str]:
    names = set(_BUILTIN.get(kind, {})) | set(_EXTRA.get(kind, {})) | set(_entry_points(kind))
    return sorted(names)


def get(kind: str, name: str) -> Any:
    """Return the backend class for (kind, name). Imports only that backend's module."""
    if kind not in KINDS:
        raise BackendError(f"unknown backend kind {kind!r}")
    obj = _EXTRA[kind].get(name)
    if obj is None:
        obj = _BUILTIN[kind].get(name)
    if obj is None:
        ep = _entry_points(kind).get(name)
        if ep is not None:
            obj = ep.load()
    if obj is None:
        raise BackendError(f"no {kind} backend named {name!r}; available: {', '.join(available(kind))}")
    if isinstance(obj, str):
        mod, attr = obj.split(":")
        obj = getattr(importlib.import_module(mod), attr)
    return obj


def create(kind: str, name: str, options: dict = None, **context: Any) -> Any:
    """Instantiate a backend: ``Backend(options, **context)``."""
    return get(kind, name)(dict(options or {}), **context)


def need(module: str, extra: str, purpose: str = "") -> Any:
    """Import an optional dependency or fail with the pip command that provides it."""
    try:
        return importlib.import_module(module)
    except ImportError as e:
        raise BackendError(
            f"{purpose or 'this backend'} needs the optional package '{module}' "
            f"(pip install 'animalworld[{extra}]'): {e}") from e
