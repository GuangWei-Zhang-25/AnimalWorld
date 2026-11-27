# Writing a backend

A backend is a class with a tiny interface. Register it with an entry point in your own package:

```toml
[project.entry-points."animalworld.generators"]
my_generator = "my_package.gen:MyGenerator"
```

or at run time: `animalworld.registry.register("generators", "my_generator", MyGenerator)`.
Kinds: `literature`, `writers`, `generators`, `proposers`, `trainers`, `predictors`, `trackers`.
`animalworld backends` lists what is registered.

Rules for every backend:

1. Import heavy libraries inside methods (use `registry.need("torch", "generate", "my backend")` for
   a clear error message), never at module top level.
2. `plan()` describes what would run and must not load a model or touch the network.
3. Write files atomically (temporary name, then rename).
4. Take secrets from `animalworld.credentials.get(name)`; never accept them as plain options.

## Generator

```python
from animalworld.backends.generators.base import Generator

class MyGenerator(Generator):
    name, kind, ext, needs_gpu = "my_generator", "image", ".jpg", True   # kind: image | video | frames
    def plan(self, items):
        return dict(super().plan(items), model=self.opt.get("model_path"))
    def generate(self, items, out_dir):
        for m in items:        # m: id, prompt, negative_prompt, seed, W, H, n, look, ...
            path = out_dir / f"{m['id']}.jpg"
            ...                # write the image
            yield {"id": m["id"], "files": [{"id": m["id"], "file": str(path), "frame": -1}]}
            # video backends yield {"id": ..., "video": str(path)}; the stage extracts frames
```

The stage handles the queue, claims, resumption, `index.jsonl` and `meta.jsonl`.

## Proposal model

```python
from animalworld.backends.proposers.base import Proposer
from animalworld.formats import labels as L

class MyProposer(Proposer):
    name = "my_proposer"
    def propose(self, rows):   # rows: id, path, file, W, H, unit, frame
        for r in rows:
            inst = [L.instance(self.category, [x0, y0, x1, y1], poly, keypoints, score)]
            yield L.label(r["id"], r["file"], r["W"], r["H"], inst, source=self.name)
```

The stage applies the gates and writes `labels/proposed/<id>.json`. The gates: instances below
`min_score` or outside the area band are dropped; the image fails when the count differs from the prompted
count, or when an instance lies on the image border (share of the outline's points on the border, or, for an
instance without an outline, share of the box perimeter; `max_border_frac`).

Built-in proposers: `mock` (tests), `sam` (SAM 3.1 text prompt, or SAM 2.1 box to mask), `model` (a trained
checkpoint; `yolo` is the same class), `pose` (top-down crop pose model on existing boxes) and `vlm`.

### Vision-language model (`vlm`)

```yaml
propose:
  backend: vlm
  options:
    provider: openai_compatible          # any server with POST {base_url}/chat/completions and image input,
    base_url: http://localhost:8000/v1   # e.g. a local open-weight VLM behind vLLM; or provider: anthropic
    model: my-vlm
    token_credential: my_vlm_token       # a NAME (animalworld credentials set my_vlm_token); omit for a local server
    max_side: 1024                       # longer side of the image that is sent
    tell_count: true                     # the prompt states the animal count the image was generated for
```

The prompt is built from `slots.yaml` (species, definition, keypoint names) and the image size; the answer must
be one JSON object `{"instances": [{"box": [x0, y0, x1, y1], "score": s, "keypoints": {name: [x, y]}}]}` and is
parsed strictly. The result is a label document with boxes (no outlines) and optional keypoints, `source: vlm`,
and it passes the same gates as any other proposal. With `tell_count: true` the count gate checks the model
against a number it was told; set it to `false` if you want the count gate to be an independent check.

**Not yet run against a live service.** Request building, parsing and the mapping into label documents are
tested against a local stub server. No real vision-language model has been called from this package, and the
quality of boxes or keypoints from such models on generated animal images has not been measured.

## Literature source

`search(query, limit) -> [ {id, source, title, abstract, year, journal, doi, authors, volume, issue, pages} ]`
(`authors`: list of "Surname Initials"; the last four keys exist since 0.2.1, read older records with `.get`), plus
`default_query(species, synonyms, description)`. Use `self._http()` (rate limit, retries, generic
User-Agent) and `self._cached()/_store()` (disk cache).

## Prompt writer

Subclass `TemplateWriter` and override `expand_slots(slots, species, description)` to add slot
values; the deterministic slot filling and the distinct-prompt check stay the same.

## Trainer

`plan(dataset_dir, out_dir)`, `train(dataset_dir, out_dir) -> {"weights": path}`,
`export_plan(weights, fmt)`, `export(weights, fmt) -> {"file": path}`. `dataset_dir` holds
`data.yaml`, `train/{images,labels}`, `val/{images,labels}` and `coco/instances_{train,val}.json`.

## Predictor and tracker

`Predictor.predict(PIL image) -> [instance, ...]`;
`Tracker.track(frames) -> ids` (same nesting as `frames`, `-1` = no identity).

## Commercial generator APIs (`http_api`)

Describe the request in YAML; see the module docstring of
`animalworld/backends/generators/http_api.py`. Presets `openai_images` and `generic_url` are
**untested examples that require your own account**; the adapter itself is tested against a local
stub server. Check the provider's terms for training on generated images.

## Rule-based tracker adapter (`rules`)

The lab's MouseWorld-Track rule set is not bundled (it reads SAM 2 witness files this package does
not produce). The adapter exports this project's detections to `tracks/<name>/det_export.npz`
(`t, box, score, c, area, T, W, H`) and calls your function:

```yaml
track:
  backend: rules
  options: {entry: "my_module:run", pythonpath: /path/to/folder}
```
```python
def run(work_dir, n_animals, options):      # -> one list of identities per frame
    ...
```

## What has been run, and what has not (state of 0.2.1, 26 November 2026)

"Run" means executed with the real model or service, not a dry run and not a stub. Runs on 26 November used the
edited source tree on the lab cluster; their projects and logs were kept outside the repository.

| part | state | what the evidence is |
|---|---|---|
| `europepmc`, `pubmed` literature | RUN | live queries (200 and 5 records); second run answered from the cache with 0 requests. PubMed DOI and author / volume / issue / pages parsing: fixture test |
| `template` prompt writer, taxonomy | RUN | every pipeline run; tests |
| `llm` prompt writer | **NOT RUN against a live service** | stubbed model answer and stubbed HTTP call only |
| `qwen_image` generator | RUN on 18 November (30 images, split path, 30 steps) | not re-run for 0.2.1; `compile: true`, `split: false`, `model_id` download: not run |
| `wan_video` generator | RUN once on 26 November | ONE clip, Wan 2.2 T2V-A14B at the adapter defaults (832 x 480, 121 frames, 40 steps, guidance 4.0, `offload: model`), 878 s on an H100; the generate stage then extracted 16 frames (every 8th) with unit id and frame index. The clip showed the two prompted mice but almost no movement and an oblique rather than top view: one clip says nothing about how often Wan follows a prompt |
| `http_api` generator | adapter RUN against a local stub server only | the `openai_images` and `generic_url` presets were never sent to a real service |
| `sam` proposer | SAM 3.1 text prompt RUN on 18 November (24 images) | SAM 2.1 box-to-mask variant and the annotator's "Mask from box" button: not run with a model |
| `model` proposer (trained checkpoint) | RUN on 26 November with an RF-DETR checkpoint and with a YOLO pose checkpoint | released `mouseworld-det.ckpt` through `--backend model --weights` on 30 Qwen-Image pictures: 30 proposals with outlines, 25 passed the gates, 5 failed the count gate. Looked at as a contact sheet, not scored against ground truth. ONNX weights as proposer: not run |
| `pose` proposer | RUN on 26 November on MOCK renders | keypoints from the pose model trained below on 40 new mock images (91 animals): median error 0.04 of the longer box side against the known mock geometry. Not run on generated or real animal images |
| `vlm` proposer | **NOT RUN against a live service** | local stub server only; no real vision-language model was called |
| identity linking in `propose` | RUN on mock clips (test) and on the 16 frames of the one Wan clip | box linking only. On the Wan clip the two animals kept their ids on the frames where both were found, but the animals hardly moved, the detector missed the second animal on 4 of 16 frames and reported the tether cable as an extra animal on 8, so this is not a test of identities through crossings. No SAM 2.1 video propagation in the annotation workflow |
| annotator window | RUN under a virtual display (Xvfb) with real mouse and key events (tests) | accept, reject, draw box, keypoints, move / resize, identity keys. Not used by a person on a full dataset for this release |
| `yolo` trainer | detect and segment RUN on 18 November (21 images); **pose with top-down crops RUN on 26 November on MOCK renders** | 320 accepted mock images (gate of 200 passed without `--force`), 733 / 25 crops, `yolo11n-pose.pt`, 12 epochs, 148 s on an A100; the checkpoint was then the proposer of a second round (row above). Shapes on a plain background: this shows the path works, not what accuracy to expect on animals. `resume`: not run |
| `rfdetr` trainer | ONE epoch RUN on 18 November (21 images) | a full training, resume and ONNX export: not run. The released MouseWorld-DET was trained with the lab's own chunked script, not with this trainer |
| `sleap_dlc` | export RUN (tests; `.slp` read back with sleap-io) | the files were not loaded into a SLEAP GUI or a DeepLabCut installation |
| `onnx`, `ultralytics`, `blob` predictors, `iou` tracker | RUN (18 November and tests) | `rfdetr` predictor: RUN on 26 November as the proposer above; RF-DETR ONNX layout: not run |
| `rules` tracker adapter | stand-in function only | the lab's rule set is not bundled |
