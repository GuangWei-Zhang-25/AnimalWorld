# Animal World demo clips, version 1

6 short top-view clips of laboratory mice for `animalworld demo` (95 s of video, 59 MB). They are a separate
download from the Python package, like the weights.

| file | mice | length | size | what it shows | setting | reference used for the accuracy numbers |
|---|---|---|---|---|---|---|
| `mouse_x1_cage.mp4` | 1 | 15 s | 9.6 MB | one mouse walking around the cage | home cage, top view | reviewed reference (person-corrected) |
| `mouse_x2_cage.mp4` | 2 | 15 s | 11.2 MB | two mice, contact at the start, then apart | home cage, top view | pipeline-made reference, NOT confirmed by a person |
| `mouse_x2_arena_a.mp4` | 2 | 15 s | 5.9 MB | two black mice in a grey arena, one short contact | open arena with reflecting walls, top view | reviewed reference (person-corrected) |
| `mouse_x2_arena_b.mp4` | 2 | 15 s | 8.3 MB | two black mice in a bright arena, one short contact | open arena with reflecting walls, top view | reviewed reference (person-corrected) |
| `mouse_x4_crossing.mp4` | 4 | 15 s | 12.1 MB | four mice with crossings | home cage, top view | reviewed reference (person-corrected) |
| `mouse_x5_huddle.mp4` | 5 | 20 s | 11.6 MB | five mice with close contacts | home cage, top view | reviewed reference (person-corrected) |

All clips are excerpts of the publishing laboratory's own recordings: 1920x1080, 30 fps, H.264, no audio, metadata
removed. Licence: Creative Commons Attribution 4.0 International (CC BY 4.0); attribution: Guangwei Zhang laboratory.
Checksums: `SHA256SUMS`; details: `manifest.json`. No clip of another laboratory and no public-dataset clip is included.

**These are familiar clips for this software.** The cages and arenas are of the kind the detectors were trained on, and
none of the recordings is a held-out test recording. They show that the installation works and what the output looks
like; they do not show how the tracker copes with a new laboratory. The windows were chosen from the reference tracks
before any run of the package (rule in `manifest.json`). The two arena clips each contain one contact between the two
animals, with reflections of the animals on the walls.

## Run them

```bash
animalworld demo                        # window: press Start, confirm the counts
animalworld demo --headless --out DIR   # no window
```

`animalworld demo` looks in `$ANIMALWORLD_DEMO`, then `~/.cache/animalworld/demo`, then fetches the clips from the
demo address and checks each SHA-256. By hand: put the files in `~/.cache/animalworld/demo/`.

## Expected results

`expected/<clip>/` holds what the default engine wrote for each clip: `summary.json`, `tracks.csv`, a frame of the
overlay (`overlay_still.jpg`), a small copy of the overlay video, the contact sheet, the reference positions
(`reference.csv`, `reference.json`), and `accuracy.json` written by `expected/score_demo.py` (numpy only):

```bash
python expected/score_demo.py DIR/mouse_x4_crossing expected/mouse_x4_crossing
```

`benchmark/` holds the speed, memory and accuracy comparison that was measured on the four cage clips of the first
version of this bundle (the two arena clips and any replaced clip are not part of it). `expected.zip` is `expected/` and
`benchmark/` in one file for the release upload.

## Measured on these clips (20 Nov 2026, default settings, one H100)

| clip | reference mouse-frames | identity correct | identity fixes | missed / false | filled rows correct | seen by eye |
|---|---|---|---|---|---|---|
| `mouse_x1_cage` | 150 | 150 | 0 | 0 / 0 | 298 / 298 | clean |
| `mouse_x2_cage` | 300 | 300 | 0 | 0 / 0 | 596 / 596 | identities kept; the two boxes overlap while the animals touch (0-1 s) |
| `mouse_x2_arena_a` | 300 | 300 | 0 | 0 / 0 | 596 / 596 | identities kept through the contact at 7-8 s; nothing wrong seen |
| `mouse_x2_arena_b` | 300 | 300 | 0 | 0 / 0 | 596 / 596 | identities kept through the contact at 7-8 s; nothing wrong seen |
| `mouse_x4_crossing` | 600 | 600 | 0 | 0 / 0 | 1,192 / 1,192 | identities kept; a tall box at 1.0-1.5 s and 5.5 s |
| `mouse_x5_huddle` | 1,000 | 1,000 | 0 | 0 / 0 | 1,990 / 1,990 | identities kept through several contacts; NOT fully clean: at 9.5 s the box of one animal also covers the animal it touches, and tall boxes at 10.0, 14.0 and 16.0 s |

The references are 10 per second; the rows in between ("filled") were compared with the straight line between
neighbouring reference positions, which is an approximation. Keypoints have no reference and were not scored. The "seen
by eye" column is the developer's own look at one overlay frame every 0.5 s, not a review by a second person. Boxes
include the tail, as the detector draws them. On the four cage clips of the first version, SAM 2.1 alone, started from
the reference boxes on the first frame, was as accurate as the full tracker (`benchmark/benchmark.csv`): short clips
like these do not show how the tracker compares with other methods on hard video.
