# Architecture

## Figure 1a, box by box

```
 species + description
        |
        v
 [1] PROMPT ENGINEERING      literature  ->  taxonomy  ->  prompts
        |                    records.jsonl   taxonomy.json  manifest.jsonl (+ distinct-prompt report)
        v
 [2] GENERATION              generate    image / edit models, video models, optional commercial APIs
        |                    generated/images, generated/videos, index.jsonl, meta.jsonl
        v
 [3] ANNOTATION / CURATION   propose  ->  review  ->  dataset
        |        ^           labels/proposed   labels/reviewed   dataset/ (COCO + YOLO)
        |        |  refine: a trained model becomes the proposer for the next round
        v        |
 [4] MODEL TRAINING          train (gated on accepted images)  ->  export
        |                    models/<trainer>_<task>/, latest.json
        v
 [5] DEPLOYMENT              predict  ->  track
                             predictions/<recording>/frames.jsonl, tracks/<recording>/tracks.csv
```

| Figure 1a element | module | project folder |
|---|---|---|
| literature search | `stages/literature.py`, `backends/literature/*` | `literature/records.jsonl`, `literature/cache/` |
| keywords, taxonomy | `stages/taxonomy.py` | `taxonomy/taxonomy.json`, `taxonomy/report.md` |
| prompts | `stages/prompts.py`, `backends/writers/*` | `prompts/manifest.jsonl`, `prompts/report.json` |
| image, edit, video models; APIs | `stages/generate.py`, `backends/generators/*` | `generated/` |
| proposal models (SAM, pose) | `stages/propose.py`, `backends/proposers/*` | `labels/proposed/` |
| human-in-the-loop, refine | `stages/review.py`, `annotator/` | `labels/reviewed/`, `exchange/` |
| species datasets | `stages/dataset.py`, `formats/*` | `dataset/` |
| model training | `stages/train.py`, `backends/trainers/*` | `models/` |
| deployment (boxes, keypoints, IDs) | `stages/export.py`, `stages/predict.py`, `stages/track.py`, `backends/deploy/*`, `backends/trackers/*` | `predictions/`, `tracks/` |

## Principles

* **Project directory.** `project.yaml` + `slots.yaml` describe a project; every stage is a function
  `run(project, dry_run=False, ...) -> summary` that reads and writes files under the project folder.
  `state.json` records the last mode of each stage (`executed`, `dry-run`, `skipped`, `waiting`).
* **Resumable.** Stages skip finished work: literature pages are cached, the prompt manifest only
  grows (earlier rows and their images stay valid), generation is a claimed work queue (several
  workers may share it; stale claims are reaped), proposals are written per image, a human review
  decision is never overwritten by a new proposal.
* **Backends are plugins.** `registry.py` maps (kind, name) to a class. Built-ins are lazy strings;
  third-party packages use entry points `animalworld.<kind>`. A backend's `plan()` must not load a
  model, so `--dry-run` works on any machine.
* **Light core.** `import animalworld` never imports torch, diffusers, ultralytics, onnxruntime or
  OpenCV (tested). Heavy imports live inside the backend method that needs them.
* **Species agnostic.** All wording, keypoints and literature keywords come from `slots.yaml`.
* **Labels are reviewed, not assumed.** Proposals carry gate results (score, area band, count equal
  to the prompted count, border). Only `labels/reviewed/*.json` with status `accepted` reaches the
  dataset. The train / val split is by generation unit (prompt or clip), never by frame.
* **Secrets stay out.** Tokens live in the user's config folder or environment; project files hold
  only credential names; `project.yaml` with a literal secret is refused.

## Data formats

* `prompts/manifest.jsonl`: `id, index, cell, n, appearance[], slots{}, look, W, H, seed, prompt,
  negative_prompt, u, split`.
* `generated/index.jsonl`: `id, unit, file, frame, W, H` (one row per image; `unit` = manifest id).
* label document (`labels/*/ID.json`): see `formats/labels.py`.
* `predictions/NAME/frames.jsonl`, `tracks/NAME/tracks.jsonl`: `frame, name, instances[]`;
  `tracks.csv`: `frame, id, x0, y0, x1, y1, score, <keypoint>_x, <keypoint>_y ...`.
