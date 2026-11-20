# Animal World Studio

`animalworld studio [PROJECT]` opens one desktop window (tkinter / ttk; standard library + Pillow, no browser, no
server, no extra GUI package). A sidebar lists the stages (Project, Literature, Prompts, Generate, Propose labels,
Review, Train, Inference), each with a status dot: grey = nothing yet, green = has results, orange = running, red =
the last run failed. A stage is one page: a one-line status, a form of rows (label left, control right), one accented
button, quiet secondary buttons and a log you can open. Dataset, Train and Export share the Train page; "Where the
stages run" (this computer or SLURM) is on the Project page. Light and dark: the link at the bottom of the sidebar,
or `ANIMALWORLD_THEME`; `ANIMALWORLD_UI_SCALE=2` for a dense display. Screenshots: `docs/img/studio.png`,
`studio_generate.png`, `studio_review.png`, `studio_end.png`, `studio_dark.png` (real captures of the running window
on a headless X server; shown paths are under a neutral home folder).

**One code path.** A Run button saves the row's form into `project.yaml` and starts exactly the command the command
line runs, `python -m animalworld <stage> <project> ...`, as a child process of the window (an argument list, never a
shell string). Output goes to `<project>/logs/studio_<stage>_<utc>.log` and is shown live; the window stays usable
and **Stop** ends the job. One job runs at a time. Everything can be done without the window from the CLI.

| row | fields | what Run starts |
|---|---|---|
| 1 Project | folder, species, description | creates the project (`project.yaml`, `slots.yaml`) or opens an existing one |
| 2 Literature | source (`offline`, `europepmc`, `pubmed`), records | `literature` |
| 3 Prompts | **target number of images** (= `prompts.n`), seed, megapixels, size multiple | `taxonomy`, then `prompts` |
| 4 Generate | source: open weights (Qwen-Image), API, mock, Wan video | `generate` |
| 5 Propose labels | proposal model (`sam` = SAM 3.1 text prompt, `yolo`, `mock`); optional keypoint model | `propose` (and `propose --backend pose`) |
| 6 Review | open the annotator; accept-all (explicit, confirmed); import / export COCO, YOLO, DeepLabCut, SLEAP | `review --auto`, `import-labels`, `export-labels` |
| 7 Dataset | task | `dataset` |
| 8 Train | trainer (`yolo`, `rfdetr`, `sleap_dlc`), task, image-count gate, "train anyway", epochs, image size | `train` |
| 9 Export | | `export` (ONNX) |
| 10 Run on your videos | MouseWorld pretrained, or this project's model; videos, output folder, animals | `zeroshot ...`, or `predict` + `track` |

**Generate, the Qwen-Image example.** The form is pre-filled with the setup of the lab's real run: Qwen-Image 2.1 from
a local diffusers folder, class `QwenImage21Pipeline`, 30 steps, true CFG 4.0, bfloat16, split path, 1.25 MP, sizes in
multiples of 32. The weights folder is a field; its default is the cache location
`~/.cache/animalworld/models/Qwen-Image-2.1`. Official source: https://huggingface.co/Qwen/Qwen-Image-2.1 (about 31 GB;
Qwen Research License: non-commercial, and models trained on its images must say "Built with Qwen" or "Improved using
Qwen"). If the folder does not exist, the backend loads the Hugging Face id in the next field, which downloads the
weights at first use (that route is untested here). The window route with the local folder was run for real on a GPU
(STATUS.md).

**API keys.** With source "API" you give the endpoint, the model name, a credential *name* and the key. The key is
stored with `animalworld credentials` in a file outside the project (mode 600) and the field is cleared;
`project.yaml` only holds `token_credential: NAME`. The key is never written to a log; log text shown in the window
is passed through the same redaction as the CLI.

**Review (human in the loop).** "Open annotator" opens the desktop annotator (`animalworld review PROJECT --gui`
opens the same window on its own; the earlier browser annotator was removed from the package): image list with status and filter, zoom (wheel, + / -) and pan (middle-drag or
space-drag), accept **A**, reject **R**, draw a box **B**, an outline **P** (Enter closes), keypoints **K** (in the
species' order), click to select, drag a box to move it, a corner to resize it, a keypoint or outline point to move
it, **V** visible / hidden, **Del**, **Ctrl+Z**. Every decision is written at once to `labels/reviewed/<id>.json`
with origin `human`. "Mask from box (SAM 2.1)" replaces the selected animal's outline using SAM 2.1's box prompt; it
needs `propose.options_for.sam.config` and `.checkpoint` in `project.yaml` plus torch + sam2, says so when they are
missing, and has not been run with a real SAM model here. There is no free-hand mask brush.

**The training gate.** Row 8 shows "N accepted of M needed". Below the threshold `train` only reports how many more
accepted images are needed; "train anyway" passes `--force` (a small model for the refine loop).

**Run on your videos.** "MouseWorld pretrained" runs the zero-shot engine (MouseWorld-DET + MouseWorld-Track +
MouseWorld-Pose) through `animalworld zeroshot`. The full MouseWorld-Track strategy currently runs only with the
MouseWorld pretrained detectors; a model trained in this project is run with `predict` + `track` (IoU tracker), one
video at a time. "Open Animal World Run" starts the minimal zero-shot app in its own window.

**Run on SLURM (optional).** Tick the box in the bottom bar and the same commands are submitted with `sbatch`
(partition, gres, CPUs, memory, time; each value must match a strict pattern). The batch script is a fixed
three-line wrapper; the commands travel as argument lists in a JSON file, so no form value is ever placed in shell
text. The window tails the same log file and Stop calls `scancel`. **Not tested on a real queue**: only the argument
list, the job file and the wrapper's local run are covered by tests.

**No display?** `animalworld studio` prints one sentence and exits with code 2. Use a desktop session or `ssh -X`.
