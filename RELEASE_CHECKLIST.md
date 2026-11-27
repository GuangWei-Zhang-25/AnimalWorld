# Release checklist: animalworld 0.2.0

Status of every item that stands between this folder and a public release, as of 20 Nov 2026.
**done** = finished and checked here; **needs the user** = a decision or an action only the lab head can take;
**needs work** = not finished, with what is missing. Nothing has been published, uploaded or pushed.

## 1. Where things are published

| item | status | detail |
|---|---|---|
| Repository | needs the user (push) | https://github.com/GuangWei-Zhang-25/AnimalWorld exists, public and empty; `git init` is done locally, nothing is committed or pushed. `project.urls` point at it |
| Weights address | needs the user (upload) | `BASE_URL` is set to the release `weights-v2` of the repository; the release does not exist yet. Upload `weights_release/` with `tools/release_upload.py weights` (prepared, never run against GitHub) |
| Demo-clip address | needs the user (upload) | `DEMO_URL` is set to the release `demo-v1`; upload `demo_release/` with `tools/release_upload.py demo` |
| PyPI name `animalworld` | needs the user (upload) | reported free by the coordinator on 20 Nov (I did not query PyPI); not registered; `twine check dist/*` passes; `pip install "animalworld[run]"` works only after an upload |
| Public `curl ... \| bash` one-liner | needs the user | needs a hosted `install.sh`; the script itself works from a local copy |
| Download from GitHub itself | needs work | the downloader was tested against local servers and folders only (resume, wrong checksum, missing file); a real GitHub release was never contacted |

## 2. Licences

| item | status | detail |
|---|---|---|
| Code: AGPL-3.0-or-later | needs the user (confirm) | chosen because the trainer and predictors call Ultralytics; a permissive core + AGPL plugin split is possible but not done |
| Weights trained with Ultralytics (pose, DET-Lite, pose-lite) | done | must be AGPL-3.0; stated in `weights_release/README.md` and `manifest.json` |
| RF-DETR-derived weights (MouseWorld-DET, tracker detectors A and B) | needs the user | marked AGPL-3.0 to keep one licence for the bundle (the lab's 13 Nov decision); RF-DETR and its Seg weights are Apache-2.0, so a permissive licence is possible. Decide, then edit manifest / README |
| Real training data behind the weights | needs the user | the detectors were trained on MouseWorld frames "together with labelled real mouse datasets"; whether each of those datasets allows releasing a model trained on it (e.g. non-commercial terms) was not audited |
| Qwen: "Improved using Qwen" | done / needs the user | the notice is in `weights_release/NOTICE.md`. Open question for the lab: Qwen-Image 2.1 is under a non-commercial research licence; whether weights trained on its images may be released under AGPL (which permits commercial use) is a legal question I cannot settle |
| SAM 2.1 | done | Apache-2.0 (code and checkpoints); not redistributed, fetched from Meta's address at first use. That fetch was exercised only if the GPU node could reach the address (see STATUS) |
| SAM 3.1 | done | gated, SAM License; NOT required by default. `--sam3` / checkbox turns the extra witness on; a user without access loses nothing measurable (20 vs 21 fixes on the benchmark). If asked for and missing: the command stops with the reason, the window continues without it and says so |
| Demo clips | done | six clips, all excerpts of the publishing laboratory's own recordings; licence CC BY 4.0, attribution "Guangwei Zhang laboratory" (the user's approval of 20 Nov). No clip from another laboratory or from a public dataset is included. File names and texts carry no recording or animal identifier; the internal sources are listed only in a file that is not committed |
| Public-dataset clips | done (not included) | the user's decision of 20 Nov: none is distributed |
| Clips from other laboratories | done (not included) | the user decided on 20 Nov: only the publishing laboratory's own recordings; no clip of another laboratory and no public-dataset clip is included |

## 3. Installation on other machines

| item | status | detail |
|---|---|---|
| Linux, clean venv, CPU (`bash install.sh`) | done | run here |
| Linux, clean venv, GPU (`bash install.sh --gpu`) | done | run here end to end with an empty pip cache; then used for the demo on an H100 |
| `install.sh --gpu --sam3` | needs work | not run as a script |
| A Linux machine that is not this cluster | needs work | never tried. The wheel contains no `/lustre` path and no user name (checked by scanning the built wheel), but every run so far was on this cluster's Python 3.12 / 3.9 |
| Windows (`install.bat`) | needs work | written, never run. The full engine uses symbolic links (falls back to copying) and was never tried on Windows |
| macOS | needs work | never run; the CPU engine should work, the full engine needs CUDA |
| tkinter missing on the target Python | done (message) | `install.sh` warns and names the package to install |
| GPUs other than H100 (A100, 24 GB, 16 GB cards) | needs work | not tried; without SAM 3.1 the peak GPU memory is far lower than before (see STATUS), which makes smaller cards plausible but untested |
| Xvfb with `/tmp/xkb` | done (test-only) | the private headless X server under `_work/xvfb/` exists only to test the windows on this cluster; it is not part of the package and no user needs it |
| Capability check on a slow disk | done | the import check could time out and push a GPU node to the CPU engine; limit raised, message names the cause |

## 4. Package hygiene

| item | status | detail |
|---|---|---|
| Version | needs the user | `0.2.0`, alpha classifier |
| CHANGELOG | done | `CHANGELOG.md` |
| Citation (`CITATION.cff`) | done / needs the user | `CITATION.cff` with the software entry and the author name only; replace by the manuscript reference when it exists |
| Author / contact metadata | done | `authors = [{ name = "Guangwei Zhang" }]`; no e-mail address anywhere in the package or docs |
| Tests | done | both Python suites pass on the built wheel (counts in STATUS.md); window tests need a display |
| CI | needs work | no workflow file; tests need a display for the window tests (Xvfb on a runner) |
| Telemetry | done | ONNX Runtime telemetry off before import; Ultralytics offline with its own settings folder; no request carries personal data |
| Screenshots in `docs/img/` | done | all re-shot on 20 Nov from the restyled windows under a neutral home folder (paths shown as `~/...`); the old ones are in `_moved_aside/` (not committed) |
| Demo clip `mouse_x5_huddle` | done, with a named defect | replaced on 20 Nov by another 20 s window of the same recording: 100 % against its reference; boxes that also cover another animal's centre on 8 of 200 reference frames (the earlier window: 16). Still not fully clean by eye (one frame with a box over a touching neighbour, a few tall boxes); the README names it |
| Demo bar "easy clips" | needs the user | two arena clips with a contact each were added, but the comparison with other methods (SAM 2.1 alone matches the tracker) was measured on the four cage clips of the first bundle only and was not repeated on the new clips |
| `_work/`, `_moved_aside/`, `runs/`, `weights_local/` | needs the user | not part of the wheel or sdist; decide what to keep before pushing the folder (they contain cluster paths) |

## 5. What the README may claim (manuscript gaps, STATUS section D)

| claim | status | detail |
|---|---|---|
| "Detection, identity tracking and keypoints for mouse videos, zero-shot" | done | with the stated limits: top-view mice, a frame with all animals apart is needed to start, no pups, one arena |
| "The tracker of the paper" | done, with a caveat | same frozen rules and the scored detector pair; the default now runs WITHOUT the SAM 3.1 witness (benchmark: 20 vs 21 fixes); the paper should say that the scored configuration used SAM 2.1 and SAM 3.1 |
| "Re-identification", "trajectory stitching" (Fig. 1e) | needs work | no appearance Re-ID model exists; stitching of long videos across 60 s windows is new and unscored. The README must not claim either as validated |
| "Rows for every frame" | done, with a caveat | tracked at about 10 per second, filled in between from per-frame detections; public references exist only at the tracked rate, so the filled rows were scored indirectly (STATUS) |
| "Works for any species" (Studio) | needs work | the pipeline runs with mock and a small real mouse run; only Qwen-Image was run as a real generator; a new species without a detector needs SAM 3.1 (gated) or hand-drawn first boxes; a Studio-trained model is tracked with the simple tracker, not MouseWorld-Track; no pig weights are shipped |
| Literature map (47,358 papers, UMAP) | needs work | the package counts keywords; no embedding map |
| Training options HRNet / DeepLabCut / SLEAP | needs work | label export only; YOLO and RF-DETR are the trainers |
| Accuracy numbers | needs the user | the README quotes the 34-clip benchmark measured here; the paper's 46 fixes per hour was not re-measured with the package |
