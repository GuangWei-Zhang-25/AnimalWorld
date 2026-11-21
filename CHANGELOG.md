# Changelog

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
