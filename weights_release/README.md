# MouseWorld weights, version 1

Model files for Animal World Run (`animalworld run`, `mouseworld`, `animalworld zeroshot`). They are a separate download
from the `animalworld` Python package. All files are inference-only (no optimizer or training state) and contain no
local paths.

| file | model | size | needed for | what it is |
|---|---|---|---|---|
| `mouseworld-det.ckpt` | MouseWorld-DET | 146.0 MB | required: full engine | RF-DETR-Seg-L (rfdetr 1.12.0, class RFDETRSegLarge) fine-tune b_full_lr5e5_cos_c04 (selection F1 0.924). Inference-only copy of the training checkpoint. |
| `mouseworld-pose.pt` | MouseWorld-Pose | 19.8 MB | required: full engine | YOLO11s-pose on 256 px crops, 7 keypoints (nose, ears, neck, hips, tail base); run crop_pose_r3bottom. Ultralytics checkpoint. |
| `mouseworld-det-lite.onnx` | MouseWorld-DET-Lite | 40.8 MB | required: CPU engine | YOLO11s-seg exported to ONNX (768 px). |
| `mouseworld-pose-lite.onnx` | MouseWorld-Pose (CPU, earlier checkpoint) | 40.1 MB | required: CPU engine | YOLO11s-pose crop model exported to ONNX (256 px). An EARLIER checkpoint than mouseworld-pose.pt. |
| `mouseworld-track-det-a.ckpt` | MouseWorld-Track detector A ('epoch5') | 146.0 MB | required: full engine (default tracker configuration scored_pair) | Earlier RF-DETR-Seg-L checkpoint; first detector of the default tracker configuration (with detector B: the pair the published tracking numbers were measured with). |
| `mouseworld-track-det-b.ckpt` | MouseWorld-Track detector B ('mid_1408') | 146.0 MB | required: full engine (default tracker configuration scored_pair) | Earlier RF-DETR-Seg-L checkpoint; second, independent seed of the SAM 2.1 witnesses in the default tracker configuration. |

Total: 539 MB. The full (GPU) engine needs four of the files (458 MB), the CPU engine the two ONNX files. Every file is far below
GitHub's 2 GB limit per release asset. Checksums: `SHA256SUMS` (check with `sha256sum -c SHA256SUMS`); details: `manifest.json`.

The tracking rule set of MouseWorld-Track (v7j) is program code inside the `animalworld` package, not a file here.
By default the tracker's witnesses are fed by detectors A and B, the pair that scores best (21 identity fixes on the
34 public benchmark clips; MouseWorld-DET alone: 42); MouseWorld-DET gives the counts, boxes and outlines.

## How the app gets them

Automatically: the app downloads a missing file at first use from the release address, checks its SHA-256 and keeps
it in `~/.cache/animalworld/weights/`. An interrupted download is continued the next time.

By hand (offline computer, shared lab folder): download the files and put them, with these exact names, in either

* `~/.cache/animalworld/weights/` (Windows: `%USERPROFILE%\.cache\animalworld\weights\`), or
* any folder, and set the environment variable `ANIMALWORLD_WEIGHTS` to it.

`animalworld weights` lists what is found; `animalworld weights --verify` also checks the checksums.

## Not in this bundle (third-party; fetched from their own sources)

| file | size | licence | where it comes from |
|---|---|---|---|
| `sam2.1_hiera_large.pt` (SAM 2.1, Meta) | 898 MB | Apache-2.0 | downloaded by the app at first use from https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt |
| `sam3.1_multiplex.pt` (SAM 3.1, Meta) | 3.5 GB | SAM License, gated | accept the licence at https://huggingface.co/facebook/sam3.1 , then `huggingface-cli download facebook/sam3.1 sam3.1_multiplex.pt`; the app finds it in the Hugging Face cache or in the weights folder |

Both are needed only for the full (GPU) engine. The CPU engine needs neither.

## Licences

* `mouseworld-pose.pt`, `mouseworld-det-lite.onnx`, `mouseworld-pose-lite.onnx` were trained with Ultralytics YOLO11,
  which is AGPL-3.0. These files are therefore **AGPL-3.0** (`LICENSE-AGPL-3.0.txt`): if you distribute them, or a
  program or network service built on them, you must offer the corresponding source under the same licence. A
  different licence for commercial use is Ultralytics' to grant, not ours.
* `mouseworld-det.ckpt` and the two tracker detector files are fine-tunes of RF-DETR-Seg-L. RF-DETR's code and its
  released Seg weights are **Apache-2.0** (`LICENSE-Apache-2.0.txt`; Copyright Roboflow). These three files are
  released here under AGPL-3.0 as well, so that the whole bundle has one licence.
* Improved using Qwen: part of the training images were generated with Qwen-Image.
* See `NOTICE.md` for the upstream notices.
