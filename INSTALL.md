# Installing Animal World

Repository: https://github.com/GuangWei-Zhang-25/AnimalWorld (public). Releases: `weights-v2` (the six MouseWorld model
files) and `demo-v1` (the six demo clips and `expected.zip`); the package downloads from both by itself. Not on PyPI yet.

**Python:** the CPU engine works on Python 3.9 or newer; the full engine (`--gpu`) needs **Python 3.10 or newer**
(`rfdetr>=1.12`, behind MouseWorld-DET, is published for 3.10+ only). Where `python3` is 3.9 (e.g. Rocky / RHEL 9) use
`PYTHON=python3.11 bash install.sh --gpu` (on a cluster typically after `module load python/3.11`).

## One line

What "works today?" means: checked on 23 Nov 2026 as a new user would, on Linux (Rocky 9, Athena cluster), each time
from a fresh `git clone` of the public repository into a new folder, with a new HOME, an empty pip cache and empty
Animal World weight / demo caches, and NO `ANIMALWORLD_WEIGHTS_URL` / `ANIMALWORLD_DEMO_URL`, so everything came from
the public GitHub releases (and SAM 2.1 from Meta's address).

| | command | works today? |
|---|---|---|
| CPU engine, from a clone | `bash install.sh` | **yes** (system Python 3.9.21; 135 s). `animalworld demo --headless --engine lite` then ran all six clips on 8 CPU cores in 973 s, peak 1.1 GB RAM; scored with `score_demo.py`: five clips 100 % identity with 0 fixes, `mouse_x2_cage` 295 / 300 with 2 fixes (the simple tracker during the contact), 0 missed and 0 false on all six: the same numbers as the CPU row of `demo_release/benchmark/benchmark.csv` |
| full engine for an NVIDIA GPU, from a clone | `bash install.sh --gpu` | **yes with Python 3.10+** (Python 3.11.6; 356 s; torch 2.14.1+cu126, rfdetr 1.12.1, sam2 1.1.0, ultralytics 8.4.216; nothing gated installed). With Python 3.9 it FAILED in step 3 (`No matching distribution found for rfdetr>=1.12`); the script now stops at the start with the reason instead |
| the same + the optional SAM 3.1 code | `bash install.sh --gpu --sam3` | not run as a script (the SAM 3 code was installed by hand from a copy of Meta's GitHub repository) |
| demo | `~/animalworld-env/bin/animalworld demo --headless --out results` | **yes** (SLURM, one H100 shard, 4 CPU cores): fetched the six clips from `demo-v1` and the five weight files it needs from `weights-v2` + SAM 2.1 from Meta, full engine, all six clips 100 % identity, 0 fixes, 0 missed, 0 false, and every `tracks.csv` byte-identical to `demo_release/expected/`; 2,877 s for the six clips (the 20 Nov run with more cores: 1,370 s), peak 14.5 GB RAM, peak 2.5 GB GPU memory in its own processes. With 2 cores a first run was slower and hit its 50-min job limit during the last clip; the five finished clips were identical too. The `expected.zip` uploaded to `demo-v1` (built 20 Nov) is OLDER than `demo_release/expected/`: same `accuracy.json` and scorer, but its `tracks.csv` / `summary.json` lack the zoom / geometry columns of 0.2.0 and will not match a new run byte for byte; a rebuilt `demo_release/expected.zip` (sha256 `34244b4b...`) is ready to replace it |
| your video | `~/animalworld-env/bin/animalworld zeroshot VIDEO.mp4 --out results --n-animals 2` | **yes**, exactly as written (a demo clip saved as `VIDEO.mp4`): 404 s; `results/video/` holds `tracks.csv` (identical to the demo run of that clip), `masks.npz`, `overlay.mp4`, `overlay_masks.mp4`, `summary.json`, `summary.html`. Note the folder name is the lower-case file stem |
| the windows | `animalworld run`, `animalworld demo` (no `--headless`) | the commands exist (`animalworld run --help` works); the windows themselves were not opened in this check (no display) |
| from the repository, no clone | `pip install "animalworld[run] @ git+https://github.com/GuangWei-Zhang-25/AnimalWorld.git"` | **yes** (Python 3.9.21): import, `animalworld --help`, all six MouseWorld files downloaded from `weights-v2` (8 s) and `animalworld weights --verify` = sha256 ok for all six |
| one line | `curl -fsSL https://raw.githubusercontent.com/GuangWei-Zhang-25/AnimalWorld/main/install.sh \| bash` | **no** with the `install.sh` of commit f3a10b7 (the first public one): it fell back to `animalworld` from PyPI, which does not exist (`No matching distribution`), and setting `ANIMALWORLD_PACKAGE=animalworld@git+...` built an invalid requirement. Fixed in `install.sh` (falls back to the GitHub repository; `name@URL` handled); the fixed script, piped into `bash` from an empty folder, installed the CPU engine from GitHub. Works for everyone once the fix is pushed |
| public, after a PyPI upload | `pip install "animalworld[run]"` | **no**: nothing is on PyPI (checked: pypi.org/pypi/animalworld returns 404) |
| from a local build | `pip install "dist/animalworld-0.2.0-py3-none-any.whl[run]"` | **yes** (19-20 Nov, Python 3.9 and 3.12); `dist/` is not in the repository |
| Windows | `install.bat` | **untested** (written on Linux; CPU engine only) |

After installing:

```bash
~/animalworld-env/bin/animalworld studio    # Animal World Studio: the whole pipeline in one window
~/animalworld-env/bin/animalworld run       # Animal World Run: pick mouse videos, press Start   (same as: mouseworld)
```

Both are plain desktop windows (Python's built-in tkinter; no browser, no server). They need a display: on a remote
machine use `ssh -X` or a remote desktop. Without a display everything is available from the command line
(`animalworld zeroshot VIDEOS... --out FOLDER`, `animalworld pipeline PROJECT`, the stage commands).

## What gets installed

| install | adds | enough for |
|---|---|---|
| `animalworld` (core) | numpy, pillow, pyyaml, requests | Studio with the mock / offline backends, the annotator, label import and export |
| `[run]` | opencv, onnxruntime, imageio-ffmpeg | Animal World Run on any computer: the CPU engine (MouseWorld-DET-Lite + simple tracker) |
| `[run-gpu]` | torch, rfdetr, ultralytics, sam2, scipy | the full engine: MouseWorld-DET + MouseWorld-Track + MouseWorld-Pose |
| + SAM 3 code from GitHub (`install.sh --gpu --sam3`) | sam3 | only for the optional `--sam3` witness |
| `[generate]`, `[train]`, `[annotate]`, `[llm]` | diffusers, ultralytics, rfdetr, sleap-io, ... | the real generation / training backends of Studio |

tkinter comes with Python on Windows and macOS (python.org installer). On Linux it may be a separate system package
(`python3-tk` / `python3-tkinter`).

**SAM 3.1 is optional.** Its weights are gated by Meta's licence (accept it at https://huggingface.co/facebook/sam3.1,
then `huggingface-cli download facebook/sam3.1 sam3.1_multiplex.pt`) and 3.5 GB, and the `sam3` package on PyPI (0.1.4
checked on 19 Nov 2026) has no SAM 3.1 video predictor, so the code comes from Meta's repository
(`bash install.sh --gpu --sam3`). The default install and the default run need none of this. With `--sam3` (or
`ANIMALWORLD_SAM3=1`, or the checkbox in the Run window) the tracker gets one more witness; on the 34 public benchmark
clips that changed 20 identity fixes into 21, i.e. nothing measurable. If `--sam3` is asked for and the code or the
weights are missing, the command stops with the reason; the window continues without it and says so.

## Model files

Model files are never inside the wheel. They are a separate download: the folder `weights_release/` of this project is
the upload-ready bundle (six MouseWorld files, 539 MB in total, each far below GitHub's 2 GB per-file limit, with
`manifest.json`, `SHA256SUMS`, `README.md` and licence files), uploaded as the GitHub release `weights-v2`.

`animalworld weights` lists what is looked for and what is present (`--verify` also checks the checksums).

Search order: `$ANIMALWORLD_WEIGHTS/<file>`, then `~/.cache/animalworld/weights/<file>` (or `$ANIMALWORLD_CACHE`), then
a download into the cache with a SHA-256 check. An interrupted download is continued; a file with the wrong checksum
is set aside as `<file>.badsum` and never used.

| file | model | where it comes from |
|---|---|---|
| `mouseworld-det.ckpt` (146 MB) | MouseWorld-DET | the weights release |
| `mouseworld-pose.pt` (20 MB) | MouseWorld-Pose | the weights release |
| `mouseworld-det-lite.onnx` (41 MB), `mouseworld-pose-lite.onnx` (40 MB) | CPU engine | the weights release |
| `mouseworld-track-det-a.ckpt`, `-b.ckpt` (146 MB each) | the two detectors behind MouseWorld-Track in the default configuration (`scored_pair`) | the weights release |
| `sam2.1_hiera_large.pt` (898 MB) | SAM 2.1, Apache-2.0 | Meta's own address, fetched by the app at first use (checked 23 Nov into an empty cache: downloaded and used) |
| `sam3.1_multiplex.pt` (3.5 GB) | OPTIONAL (`--sam3`): SAM 3.1, SAM License, gated | accept the licence at huggingface.co/facebook/sam3.1 and fetch it yourself; the Hugging Face cache is searched too |

**The address is one constant:** `BASE_URL` in `src/animalworld/weights.py`
(`https://github.com/GuangWei-Zhang-25/AnimalWorld/releases/download/weights-v2/`); the demo clips use `DEMO_URL` in
`src/animalworld/zeroshot/demo.py` (`.../releases/download/demo-v1/`). To use a mirror, a lab server or a local folder set
`ANIMALWORLD_WEIGHTS_URL` / `ANIMALWORLD_DEMO_URL`. If a download fails the app names the file and the folders it looked in.

Upload (done for `weights-v2` and `demo-v1`): `tools/release_upload.py` creates a release by tag through the GitHub REST API and
uploads the files. It reads the token from the environment variable `GITHUB_TOKEN` only and never prints it.

```bash
python tools/release_upload.py --dry-run weights     # lists the files, sizes and SHA-256; contacts nothing
python tools/release_upload.py --dry-run demo
GITHUB_TOKEN=... python tools/release_upload.py weights    # tag weights-v2
GITHUB_TOKEN=... python tools/release_upload.py demo       # tag demo-v1
```

The same with the `gh` command line tool, where it is installed:

```bash
cd weights_release
gh release create weights-v2 --repo GuangWei-Zhang-25/AnimalWorld --title "MouseWorld weights v2" --notes-file RELEASE_NOTES.md \
  mouseworld-det.ckpt mouseworld-pose.pt mouseworld-det-lite.onnx mouseworld-pose-lite.onnx \
  mouseworld-track-det-a.ckpt mouseworld-track-det-b.ckpt \
  manifest.json SHA256SUMS README.md NOTICE.md LICENSE-AGPL-3.0.txt LICENSE-Apache-2.0.txt
cd ../demo_release
gh release create demo-v1 --repo GuangWei-Zhang-25/AnimalWorld --title "Animal World demo clips v1" --notes-file RELEASE_NOTES.md \
  mouse_x1_cage.mp4 mouse_x2_cage.mp4 mouse_x2_arena_a.mp4 mouse_x2_arena_b.mp4 mouse_x4_crossing.mp4 mouse_x5_huddle.mp4 \
  manifest.json SHA256SUMS README.md NOTICE.md expected.zip
```

## Demo clips

`animalworld demo` looks for the six clips in `$ANIMALWORLD_DEMO`, then in `~/.cache/animalworld/demo`, then fetches
them from `$ANIMALWORLD_DEMO_URL` or `DEMO_URL` (an https address, a `file://` address or a plain local folder) and
checks each against the SHA-256 recorded in the package. `demo_release/README.md` lists the clips, their source and
licence, and the expected results.

In the developer's own account on the Athena cluster the files are found without any download: `~/.cache/animalworld/weights`
is a symbolic link to `weights_local/`, whose MouseWorld entries point at `weights_release/`. The 23 Nov check above
used a new HOME and empty caches so that none of this was used.

## Which engine runs

`animalworld run` checks the computer when it starts and says what it found.

* **Full engine**: needs an NVIDIA GPU of the Ampere generation or newer, the libraries of `[run-gpu]`, and five
  weight files, none of them gated (MouseWorld-DET, MouseWorld-Pose, the two tracker detectors, SAM 2.1). Tested on an H100 (80 GB). Without SAM 3.1 (the default) the demo peaked at
  2.5 GB of GPU memory in its own processes (sampled every 2 s, 23 Nov), so smaller cards should do, but none was tried.
  With `--sam3`, SAM 3.1 alone holds about 30 GB in our earlier runs.
* **CPU engine**: everything else. It says plainly that tracking quality is lower. It uses MouseWorld-DET-Lite and a
  simple position tracker, not MouseWorld-Track.

* **Which detector feeds the tracker** (`--tracker-detector`, or `ANIMALWORLD_TRACKER_DETECTOR`). Measured on the 34
  public benchmark clips (16.5 min, animal count given), while the SAM 3.1 witness was still on by default (20 Nov):

  | option | what | identity fixes | per hour |
  |---|---|---|---|
  | `scored_pair` (default) | the two checkpoints the published tracking numbers were measured with; MouseWorld-DET still gives counts, boxes and outlines | 21 (20 without SAM 3.1) | 76 (73 without SAM 3.1) |
  | `strongest` | MouseWorld-DET alone: one detector, about 25 % less computing, two weight files fewer | 42 | 153 |
  | `strongest_two_seeds` | MouseWorld-DET plus the older `mid_1408` checkpoint as second seed | 39 | 142 |
  | `strongest_recut` (experimental) | MouseWorld-DET alone with its confidence cut moved from 0.40 to 0.35; the value was chosen on these same clips | 25 | 91 |

A step can run in another Python environment (for example SAM 3.1 in its own): set `ANIMALWORLD_PY_SAM3`
(also `_DET`, `_SAM2`, `_POSE`, `_CPU`) or write `~/.config/animalworld/runtimes.toml` with lines like
`sam3 = "/path/to/python"`.

## Appearance of the windows

Nothing to install: the look is drawn by `animalworld/ui.py` with tkinter only. Light or dark follows the system on
macOS, Windows and GNOME, can be switched in the window, or forced with `ANIMALWORLD_THEME=light` / `dark`. The scale
follows the display's pixels per inch, or `ANIMALWORLD_UI_SCALE` (for example `2`). The font is the system's UI font
(on Linux the first of Inter, Nimbus Sans, DejaVu Sans that is installed). A Python whose Tk was built without Xft
(some standalone Python builds on Linux) draws bitmap fonts: everything works, but the text is not smoothed and does
not scale; use the distribution's Python (`python3-tkinter`) for the windows. Dropping files onto the Run window works
only if the optional package `tkinterdnd2` is installed (untested); the buttons always work.

## Small animals, wide frames, stable outlines

* `--zoom auto` (default). The app first looks at about 40 frames across the clip. If the animals are at least about 64 px
  long at the detector's 504 px input and the frame is one camera, nothing else happens. Otherwise it also looks at the
  frame in overlapping pieces and at what moves, finds the regions where animals are active, and analyses each region as
  its own video, cropped and enlarged up to 4 times; results come back in full-frame pixels with a `region` and a `zoom`
  column, and `overlay_regions.mp4` shows the regions side by side. Two cameras recorded in one frame are never mixed.
  `--zoom off` keeps the whole frame; `--zoom FILE` takes a JSON list `[{"box": [x0, y0, x1, y1], "animals": n}, ...]`.
* `--fill sam2` (default). After the tracker has decided who is who (about 10 times per second), one more SAM 2.1 pass at
  the camera rate gives the box, outline and position of every row, so consecutive frames come from one source. Rows the
  propagation could not cover keep the detector's box (`geometry` = `detector`), a detector box that would move an animal
  impossibly far is rejected (`geometry` = `rejected`), and `detector_agrees` says whether the detector saw the same thing.
  The tracker's own position on analysed frames is kept in `track_x`, `track_y`. `--fill detector` is the earlier behaviour.
* Count the animals that move freely. An animal under a cup or behind a grid is not counted. When the regions are
  zoomed, the count you confirm is split over the regions (it always adds up to your number; each region keeps at least
  one animal while your number allows). A region in which no animal is detected is dropped and the run says so.
* Every run also writes `overlay_masks.mp4`: the segmentation alone.

## Temporary files

Each video gets `<results>/<video>/_work/` next to its results (never `/tmp`): every camera frame as a JPEG, written
by decoding the file once from start to end (no seeking, so no dropped or repeated frames), plus the tracker's
intermediates. It is removed when the video finishes successfully, kept when it fails (for debugging), and kept
always with `--keep-work`. Frames are files, not memory or a pipe, because each model step is a separate process and
SAM 2.1 / SAM 3.1 read a folder of JPEGs; rough size: 0.25 x width x height bytes per camera frame.

## Privacy

Nothing is sent anywhere by the windows or the zero-shot engine. Telemetry of ONNX Runtime is switched off before it
is imported; Ultralytics runs with automatic installs off, in offline mode, and with its own settings folder inside the Animal World cache
(analytics "sync" off), so your personal Ultralytics settings are neither used nor changed. The only network use is what you ask
for: a live literature search, a generation API you configure, or a weight download from the addresses above.
