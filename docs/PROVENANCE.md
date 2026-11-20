# Provenance

What was adapted from the lab's existing (read-only) code, and what was written fresh. Paths are
relative to the lab's project folder. "Adapted" means re-implemented in this package after
reading the source; no file was copied verbatim except the licence text.

| package module | source studied | what was taken |
|---|---|---|
| `backends/literature/europepmc.py` | `SoraProject/literature search/mine_pubmed.py` | REST endpoint, `cursorMark` paging, `resultType=core`, query shape |
| `backends/literature/pubmed.py` | `SoraProject/literature search/mine_mouse_behavior_pubmed.py`, `Abstract_UMAP_analysis/scripts/02_fetch_abstracts.py` | esearch + efetch, batches of 200, retry back-off, structured abstracts joined as "Label: text" |
| `stages/taxonomy.py` | `SoraProject/literature search/behavior_terms.csv` (55 rules: name, category, regex), manuscript Methods "Literature map and behaviour taxonomy" | curated case-insensitive keyword rules counted per record |
| `backends/writers/template.py`, `examples/mouse/slots.yaml` | `mw_det_syn/gap_targeted/scripts/prompts_div.py`, `make_div_round.py` | independent draw per slot, matched empty scenes, neighbouring counts in the negative prompt, distinct-prompt report; mouse vocabulary (generic wording) |
| `utils.Claims`, `stages/generate.py` | `mw_det_syn/gap_targeted/scripts/gen_qwen.py` | O_EXCL claim files, stale-claim reaping, atomic writes, per-job metadata log |
| `backends/generators/qwen_image.py` | `mw_det_syn/gap_targeted/scripts/gen_qwen_split.py`, `qwen/GEN_SETTINGS.json` | split encode -> generate path, 30 steps, true CFG 4.0, bfloat16, torch.compile with items ordered by size |
| `backends/generators/wan_video.py` | `mw_det_syn/sched/wan_worker.py`, `gap_targeted/scripts/wan.sbatch` | AutoencoderKLWan + WanPipeline call, 832x480 / 121 frames / 16 fps / 40 steps / guidance 4.0 |
| `backends/generators/looks.py` | `mw_det_syn/gap_targeted/scripts/look.py` | per-unit camera parameters, mono / low-light / action-camera looks (numpy + PIL port; no timestamp, no glare) |
| `formats/labels.apply_gates`, `backends/proposers/sam.py`, `proposers/base.mask_to_polygon` | `mw_det_syn/gap_targeted/scripts/label_img.py` | SAM 3.1 image calls (PIL input), score / area / count / border gates, duplicate merge, mask -> polygon |
| `stages/dataset.py` | `mw_det_syn/gap_targeted/scripts/build.py` | YOLO-seg label lines, split by unit, idempotent build |
| `annotator/server.py`, `annotator/index.html` | `sam2-Tracking/src/sam2_tracking/server.py`, `web/index.html` (V14), an internal annotation script | architecture only (stdlib server, JSON API, single page, no embedded data, id checks); the page is new and much smaller | **Removed from the package on 19 Nov 2026** (moved to `_moved_aside/2026-11-19_web_annotator/`); the annotator is now `annotator/tkapp.py`. |
| `backends/trainers/rfdetr.py` | `mw_det_syn/finetune_v2/train_e.py`, `chunk_arm.sbatch` | RF-DETR class and `.train()` arguments, YOLO-layout dataset, resume from the newest checkpoint |
| `backends/trainers/yolo.py` | `TrackPose_demonstration/long_runs/synth_setup/yolo_arm.py`, `pose_topdown/train.py` | Ultralytics calls and the crop-pose recipe |
| `backends/proposers/pose.py`, `formats/yolo.pose_crop` | `TrackPose_demonstration/long_runs/pose_topdown/build_crops.py`, `pose_topdown.py` | 256-px square crops, 1.3 x box, grey padding, central-instance rule, decode to frame |
| `backends/deploy/onnx.py` | `mouseworld_app/src/mouseworld/ops.py` (AGPL-3.0) | tensor layouts and thresholds for YOLO / RF-DETR ONNX outputs |
| `backends/trackers/iou.py` | `mouseworld_app/src/mouseworld/track.py` (AGPL-3.0) | ideas: N from >= 25 % of frames, start at an all-apart frame, forward + backward, coat-brightness cue. Much simpler than the original. |
| `backends/trackers/rules.py` | `TrackPose_demonstration/long_runs/agent_v7/supervise7_frozen_v7b_20261112.py` | calling convention and det.npz field names only; the rule set is NOT bundled |
| `LICENSE` | `mouseworld_app/LICENSE` | GNU AGPL v3 text |

Written fresh: `config.py`, `registry.py`, `credentials.py`, `pipeline.py`, `cli.py`,
`backends/generators/mock.py`, `backends/generators/http_api.py`, `backends/writers/llm.py`,
`backends/deploy/blob.py`, `formats/coco.py`, `formats/dlc.py`, `formats/sleap.py`, all tests.

`examples/mouse/abstracts_sample.jsonl` is SYNTHETIC text written for tests (each record says so).
It is not taken from, and does not describe, any publication.

Not carried over (cluster or study specific): SLURM job scripts and self-resubmission, walltime
budgets, per-stratum quotas and pilots, the CLIP rat screen, DINOv2 coverage reports, the
literature UMAP / HDBSCAN map, polarity-inverted dataset copies (available as the `invert` look).

## Added in 0.2.0 (19 Nov 2026)

| package file | comes from | what changed |
|---|---|---|
| `zeroshot/mwtrack/*.py` (core_base, core, detect_dense, detect_local, track_fb, track_local, ext_tracks, sup_v7j, sup_v7b) | `TrackPose_demonstration/long_runs/PAPER_FIGS/fig8/track_wu/agent_ext7/` (the reference-free copy of the MouseWorld-Track agent; `sup_v7j` = frozen `supervise5_frozen_v7j_20261112.py`) | paths only: work folder, clip list and checkpoints come from `MWT_*` environment variables; the lab's internal window lists and cluster paths were removed from `core_base.py`. Source checksums: `zeroshot/mwtrack/PROVENANCE.md` |
| `zeroshot/mwtrack/sam31_witness.py` | the `sam31` branch of `track_wu/run_ext.py` | stand-alone script, checkpoint path from the environment |
| `zeroshot/mwtrack/run_supervisor.py` | `track_wu/run_supervisors.py` | one rule version per call |
| `zeroshot/mwtrack/pose_frames.py` | `pose_topdown/improve/r3_infer.py` with the selected headline options (mult 1.3, min 16, dark-crop gamma 0.5, no TTA) | request / result files instead of benchmark files |
| `zeroshot/pipeline.py` | new | frame decoding, count sheet, windows and seam linking, box / outline attachment, output files |
| `weights.py` | idea and layout from `mouseworld_app/src/mouseworld/weights.py` | new registry (eight files), empty release address, runtime roles |
| `zeroshot/app.py`, `studio/app.py`, `annotator/tkapp.py`, `annotator/model.py` | new (tkinter) | |
| `install.sh`, `install.bat` | layout from `mouseworld_app/install.sh` | no model download step |

Not reused from `mouseworld_app`: its light tracker (it failed on real lab videos and is not shipped), its web GUI.
