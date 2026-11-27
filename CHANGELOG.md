# Changelog

## 0.2.1 (27 November 2026)

The zero-shot engine (Animal World Run, MouseWorld-Track rule set v7j) is unchanged from 0.2.0; the demo's expected
results were recorded with 0.2.0 and still apply.

Added
- `docs/PAPER_MAP.md`: which released file is behind each model name in the manuscript (the checkpoint scored as
  MouseWorld-DET in the detection figures is `mouseworld-track-det-a.ckpt`), and where the packaged program's settings
  differ from those of the figures.
- `zeroshot/mwtrack/PROVENANCE.md` now gives the hash of the frozen v7j rule file and of every vendored tracker file as
  shipped; `tests/test_provenance.py` checks them.

Changes that make the package do what the manuscript's Methods ("Software architecture and workflow") says.
What was executed for real and what was not is listed in `docs/BACKENDS.md` ("What has been run, and what has not").

Fixed
- **PubMed records carried the DOI of a cited paper.** The parser walked every `ArticleId`, including the reference
  list, and kept the last one (3 of 5 records in a live check). It now reads the article's own
  `PubmedData/ArticleIdList` (and `ELocationID`) only.
- The border gate skipped instances without an outline. It now falls back to the share of the box perimeter on the
  image border, so box-only proposals (a detector, the VLM proposer) are filtered too.
- DeepLabCut export reported "pandas + tables not installed" for any failure; it now reports the real error.
- `animalworld export` after the `sleap_dlc` backend gives an explicit message instead of `NotImplementedError`.
- RF-DETR predictor: works with rfdetr versions that export `RFDETR` from `rfdetr.detr` only.

Added
- Literature records keep `authors`, `volume`, `issue`, `pages` (PubMed, Europe PMC, offline files). Cached pages are
  raw answers, so `literature --force` re-parses them with the new fields; older `records.jsonl` files stay readable.
- Taxonomy candidates: matched without regard to case (wrap a pattern in `(?-i:...)` to keep it case-sensitive, as the
  strain-name patterns of the mouse example now do); a phrase covered by the rule of ANY slot is no longer a candidate;
  phrases containing verbs, prepositions, numerals or pronouns ("was to test", "on the pen", "one pen") are dropped;
  `taxonomy.candidate_stop_words` extends that list. Behaviour change: the bundled mouse sample now yields no
  candidates, because "marble burying test" and "resident-intruder assay" are covered by behaviour / group-size rules.
- Generation log (`generated/meta.jsonl`): `sec`, `W`, `H` (files written), `requested_W`, `requested_H` (manifest) for
  every backend, the mock generator included; the Wan adapter records `video_W`, `video_H`, `steps`.
- `propose --backend model` (alias of `yolo`): the newest weights under `models/` now include RF-DETR checkpoints;
  `.pth` / `.ckpt` files go to the `rfdetr` predictor, so `--weights mouseworld-det.ckpt` works.
- `vlm` proposer: an image and a prompt built from the project (species, animal count, keypoint names) go to an
  OpenAI-compatible vision chat endpoint or to the Anthropic API; strict JSON parsing into boxes and optional
  keypoints; same gates as other proposals. **Not yet run against a live service** (local stub server only).
- Proposed identities: for units with several frames, `propose` links the boxes over the frames (the `iou` tracker) and
  writes a `track_id` per instance (`propose.link_identities`). The annotator draws the id and the keys 0-9 set it
  (a swap when the id is taken). Box linking only; no SAM 2.1 video propagation, no clip timeline.
- Review records keep `proposal_source` and `edited`; `status` reports `accepted_after_edit`.
- `export-labels --format yolo` (label files; `dataset` still writes the full YOLO dataset) and
  `import-labels --status proposed|accepted` (default `accepted`, stored with `origin: import`).
- `examples/pig/README.md`; the mouse README says that the mock run plans training but does not execute it.
- Tests: `tests/test_gap_closures.py` (PubMed fixture with reference-list DOIs, video frame extraction from a
  synthetic mp4, VLM stub server, identity linking, annotator identity keys, and the items above).

Real runs for this version (26 November 2026): one Wan 2.2 clip through `generate` with frame extraction; the released
RF-DETR checkpoint as proposal model; YOLO pose training with top-down crops on mock renders and that checkpoint as
the proposer of a second round; the annotator window tests under a virtual display.

## 0.2.0 (unreleased; prepared 20 November 2026)

First version meant for other people. Nothing is published yet.

- **Animal World Studio** (`animalworld studio`): the whole pipeline in one desktop window: literature, prompts,
  generation (Qwen-Image 2.1 as the worked example, or an image API), label proposals, human review in a canvas
  annotator, dataset, training, export, inference on your videos. Every stage is also a command.
- **Animal World Run** (`animalworld run`, `mouseworld`): zero-shot analysis of mouse videos with the pretrained
  MouseWorld models: animal count to confirm, identity tracking, keypoints; `tracks.csv` with a row for every camera
  frame, `masks.npz`, `overlay.mp4`, a summary page. `animalworld zeroshot` is the same engine without a window.
- **`animalworld demo`**: six short example clips and their expected results.
- Both windows and the annotator share one look (`animalworld/ui.py`): tkinter / ttk only, light and dark, a scale
  for dense displays; no browser, no server, no extra GUI package.
- The default tracker needs nothing gated: MouseWorld-DET, MouseWorld-Track (scored detector pair, SAM 2.1
  witnesses) and MouseWorld-Pose. SAM 3.1 is an option (`--sam3`).
- Without a usable GPU: MouseWorld-DET-Lite with a simple tracker (lower tracking quality, and the app says so).
- Model files and demo clips are separate downloads with SHA-256 checks (`animalworld weights`).
- `animalworld pipeline PROJECT` runs the stages of a project in order (formerly `animalworld run PROJECT`).

- **Automatic zoom** (`--zoom auto`, default): the whole clip is looked at first; when the animals are small in the frame
  (below about 64 px at the detector's 504 px input) or the frame is two cameras side by side, each region where animals are
  active is analysed zoomed in and mapped back to full-frame pixels (`region` and `zoom` columns, `overlay_regions.mp4`).
  `--zoom off`, `--zoom REGIONFILE`. Otherwise nothing changes.
- **One geometry source for every output frame** (`--fill sam2`, default): a final SAM 2.1 propagation at the output frame
  rate, anchored on the resolved identities, gives box, outline and position of every row; the detector only confirms
  (`detector_agrees`). `--fill detector` is the earlier, faster behaviour. `summary.json` reports a frame-to-frame jitter
  metric. `overlay_masks.mp4` shows the segmentation alone.
- When MouseWorld-Track cannot start (no frame with all animals apart) the run no longer ends with empty tracks: a simple
  tracker fills in, every such row is flagged and the status says so. A video with no detection ends as "no animal found".
- `frames_with_all_animals_pct` now counts frames in which every animal has a box.
- Zoom, regions and counts: the confirmed total is split over the regions so that it always sums to the total; an extra
  animal of the proposal is taken, one at a time, from the region whose current count is seen least often on the sample
  frames. A region in which no animal is detected on any frame is dropped with a warning instead of stopping the tracker
  (an empty detection file now has the shapes the tracker reads).
- Camera seams: a step at the middle column / row counts as two cameras only if each half is a landscape camera image
  (width / height 1 to 2) and no animal-sized detection lies across it on 10 % or more of the sample frames. Found on our own
  demo clip `mouse_x2_cage` (a cage edge on the middle row) and on one public benchmark clip (800 x 600 operant chamber, a
  straight edge on the middle column: 5 identity fixes instead of 0), where one arena had been cut in two.
- Stable geometry: the SAM 2.1 mask is used on an analysed frame only if it contains the tracker's own position of that
  identity, and on the frames in between only if both neighbouring analysed frames passed (two touching mice at the start
  of `mouse_x2_cage`: a mask on the wrong mouse had replaced the tracked position).
- MouseWorld-Pose crops: an enlarged crop (small animal) is now bicubic, a shrunk crop stays bilinear. The earlier
  `INTER_AREA` flag was bilinear in OpenCV's warp. On the six demo clips shrunk to 20-60 px animals, keypoints within 0.1
  body length of the full-size answer: 76-95 % bilinear, 78-98 % bicubic.
- Not changed, measured: tiling a large region to enlarge an animal that is still 45-55 px at the detector input after the
  zoom. On our demo clips shrunk to that size MouseWorld-DET finds 98.8-99.3 % of the animals, at twice that size 100 %.

Known limits: the pretrained models are for top-view laboratory mice; a model trained in Studio is tracked with a
simple tracker, not MouseWorld-Track; tested on Linux only (Windows and macOS untested); long videos are stitched across
60 s windows by a step that has not been scored.

## 0.1.0

Internal: the command-line pipeline with mock backends.
