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

The stage applies the gates and writes `labels/proposed/<id>.json`.

## Literature source

`search(query, limit) -> [ {id, source, title, abstract, year, journal, doi} ]`, plus
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
