# Changelog

## 0.2.0 (unreleased; prepared 20 November 2026)

First version meant for other people. Nothing is published yet.

- **Animal World Studio** (`animalworld studio`): the whole pipeline in one desktop window: literature, prompts,
  generation (Qwen-Image 2.1 as the worked example, or an image API), label proposals, human review in a canvas
  annotator, dataset, training, export, inference on your videos. Every stage is also a command.
- **Animal World Run** (`animalworld run`, `mouseworld`): zero-shot analysis of mouse videos with the pretrained
  MouseWorld models: animal count to confirm, identity tracking, keypoints; `tracks.csv` with a row for every camera
  frame, `masks.npz`, `overlay.mp4`, a summary page. `animalworld zeroshot` is the same engine without a window.
- **`animalworld demo`**: six short example clips and their expected results.
- Both windows and the annotator share one look (`animalworld/ui.py`): tkinter / ttk only, light and dark, a scale
  for dense displays; no browser, no server, no extra GUI package.
- The default tracker needs nothing gated: MouseWorld-DET, MouseWorld-Track (scored detector pair, SAM 2.1
  witnesses) and MouseWorld-Pose. SAM 3.1 is an option (`--sam3`).
- Without a usable GPU: MouseWorld-DET-Lite with a simple tracker (lower tracking quality, and the app says so).
- Model files and demo clips are separate downloads with SHA-256 checks (`animalworld weights`).
- `animalworld pipeline PROJECT` runs the stages of a project in order (formerly `animalworld run PROJECT`).

Known limits: the pretrained models are for top-view laboratory mice; a model trained in Studio is tracked with a
simple tracker, not MouseWorld-Track; tested on Linux only (Windows and macOS untested); long videos are stitched across
60 s windows by a step that has not been scored.

## 0.1.0

Internal: the command-line pipeline with mock backends.
