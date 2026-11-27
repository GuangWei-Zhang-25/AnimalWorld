# Mouse: the worked example

Files here: `project.yaml` (configuration), `slots.yaml` (vocabulary and literature keywords),
`abstracts_sample.jsonl` (30 SYNTHETIC abstracts for offline tests; not real publications).
`animalworld init <dir> --species mouse` copies all three into a new project.

## 1. Mock run (CPU, minutes; this is what the test-suite executes)

```bash
animalworld init mouse_demo --species mouse --describe "laboratory, home cage and open field"
animalworld pipeline mouse_demo --auto-accept --train-dry-run
animalworld status mouse_demo
```

This run makes 40 mock images, proposes and accepts their labels automatically and builds the dataset. **It does
not train a model**: `--train-dry-run` only prints the training and export calls, and without it `train` would stop
anyway, because 40 accepted images are below `train.min_accepted_images` (200). Training is planned, not executed,
until you have that many accepted images (or pass `--force` for a small model). `predict` and `track` report
`waiting` for the same reason.

## 2. Real literature

```yaml
# project.yaml
literature:
  backend: europepmc        # or pubmed
  limit: 2000
  contact_email: ""         # optional; sent to the API only if you fill it in
```
```bash
animalworld literature mouse_demo --force && animalworld taxonomy mouse_demo
less mouse_demo/taxonomy/report.md      # which slot values the literature supports; candidate terms to add
```

## 3. Real generation (needs a GPU; NOT executed in the package tests)

```yaml
generate:
  backend: qwen_image
  options:
    model_path: /path/to/Qwen-Image-2.1   # local diffusers folder; nothing is downloaded
    split: true                            # text encoder and generator on the GPU one after the other
    compile: false
prompts:
  n: 2000
```
```bash
animalworld generate mouse_demo --dry-run   # checks the configuration, loads nothing
animalworld generate mouse_demo             # resumable; several jobs may share the queue
```
Video clips: `backend: wan_video` with `model_path: /path/to/Wan2.2-T2V-A14B` (832x480, 121 frames,
40 steps; frames are extracted every `generate.frame_stride` frames).

## 4. Proposals and review

```yaml
propose:
  backend: sam              # SAM 3.1 text prompt "mouse"; needs the sam3 package and a GPU
```
```bash
animalworld propose mouse_demo
animalworld review mouse_demo --gui         # accept / correct / reject in the annotator window
animalworld export-labels mouse_demo --format sleap    # or dlc, coco, yolo; correct there, then
animalworld import-labels mouse_demo --format dlc path/to/CollectedData_x.csv
# imported labels are stored as accepted with origin "import" (not reviewed here); add --status proposed to review them
animalworld propose mouse_demo --backend model --weights path/to/mouseworld-det.ckpt   # a trained or released checkpoint
```

## 5. Dataset, training, deployment

```bash
animalworld dataset mouse_demo
animalworld train mouse_demo --dry-run      # prints the exact Ultralytics / RF-DETR call
animalworld train mouse_demo                # runs once >= train.min_accepted_images are accepted
animalworld export mouse_demo               # ONNX
animalworld predict mouse_demo --source my_recording.mp4
animalworld track mouse_demo --n-animals 4
```
