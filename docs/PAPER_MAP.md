# Names in the paper and files in this release

This page says which released file is behind each model name in the manuscript, and where the packaged program
(Animal World Run) differs from the settings used for the figures. Checksums are in `weights_release/SHA256SUMS`.

## Models

| Name in the paper | Released file | Notes |
|---|---|---|
| MouseWorld-DET (RF-DETR-Seg-L, epoch-5 checkpoint, score threshold 0.20) | `mouseworld-track-det-a.ckpt` | The checkpoint behind the detection figures and the external-laboratory evaluations. In the package it is "MouseWorld-Track detector A". |
| MouseWorld-DET as used by Animal World Run for counts, boxes and outlines | `mouseworld-det.ckpt` | A later fine-tune started from the epoch-5 checkpoint (`b_full_lr5e5_cos_c04`, selection F1 0.924 against 0.911). It is the program's default detector; it is not the checkpoint scored in the detection figures. |
| MouseWorld-Track (rule set v7j) | program code: `src/animalworld/zeroshot/mwtrack/sup_v7j.py` | Not a weight file. Source hashes: `src/animalworld/zeroshot/mwtrack/PROVENANCE.md`. |
| Detectors behind MouseWorld-Track in the tracking benchmark | `mouseworld-track-det-a.ckpt` (epoch 5) and `mouseworld-track-det-b.ckpt` (`mid_1408`, second seed) | The default tracker configuration `scored_pair` uses exactly this pair. |
| MouseWorld-Pose (YOLO11s-pose, configuration `r3_b_g050`) | `mouseworld-pose.pt` | `r3_b_g050` = training run `crop_pose_r3bottom` with gamma 0.5 applied to dark crops at inference; the package applies the same rule (`mwtrack/pose_frames.py`). |
| MouseWorld-DET-Lite (YOLO11s-seg) | `mouseworld-det-lite.onnx` | ONNX export for computers without a GPU. |
| SAM 2.1 (Hiera-L) | downloaded from Meta at first use | Not redistributed here. |

`mouseworld-pose-lite.onnx` (CPU engine) is an earlier pose checkpoint and is not a model evaluated in the paper.

## To run the checkpoint scored in the paper as MouseWorld-DET

```python
from PIL import Image
from rfdetr.detr import RFDETR                  # rfdetr 1.12
from animalworld.weights import resolve         # downloads the file at first use and checks its SHA-256
net = RFDETR.from_checkpoint(str(resolve("mouseworld-track-det-a")), trust_checkpoint=True)
detections = net.predict(Image.open("frame.jpg").convert("RGB"), threshold=0.20)   # .xyxy, .confidence, .mask
```

## Settings: figures and packaged program

| Setting | Figures (Methods) | Animal World Run (this package) |
|---|---|---|
| Detection score threshold | 0.20 (MouseWorld-DET benchmark), 0.10 (MouseWorld-DET-Lite stored benchmark predictions) | 0.25 for every detector pass of the full engine and for the CPU engine; 0.40 counts as a confident detection for safe frames |
| Analysis rate of the tracker | about 10 frames per second | `round(camera rate / 10)` frame step: 10 per second at 30 or 60 frames per second, 12.5 at 25 |
| Window | one clip | 60 s windows with 2 s overlap, linked by position at the seams (the linking of long recordings has not been scored) |
| Position in the output | the supervisor's position | centre of a final SAM 2.1 outline at the output frame rate (`--fill sam2`, default); the supervisor's position is kept in `track_x`, `track_y` |
| Box used for the pose crop | MouseWorld-DET box | box of the final SAM 2.1 outline (`--fill sam2`) or the detector box (`--fill detector`) |
| Zoom | none in the benchmark | `--zoom auto` (default): acts only when animals are small in the frame or two cameras are side by side; `--zoom off` disables it |
| SAM 3.1 witness | not used in the default route | off unless `--sam3` |

## Example recordings

Each of the six demo clips comes with a reference (`reference.csv`: centre and box per animal, 10 per second) and the
expected output of the default route. The reference of `mouse_x2_cage` was made by the pipeline and has not been
confirmed by a person; see `demo_release/NOTICE.md`.

## Licences

Source code: AGPL-3.0-or-later. MouseWorld model files: AGPL-3.0. Demo clips: CC BY 4.0 (Guangwei Zhang laboratory).
