"""animalworld: species name in, deployable detection / pose / tracking models out.

Importing this package is cheap: it never imports torch, diffusers, ultralytics or
onnxruntime. Heavy libraries are imported inside the backend that needs them.
"""
__version__ = "0.2.0"

STAGES = ("literature", "taxonomy", "prompts", "generate", "propose", "review",
          "dataset", "train", "export", "predict", "track")

__all__ = ["__version__", "STAGES"]
