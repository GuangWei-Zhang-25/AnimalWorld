# Animal World · MouseWorld

Detect, segment, track and pose laboratory mice in your own videos, with models trained on
MouseWorld (1.32 million generated mouse frames).

## 1. Install

```bash
git clone https://github.com/GuangWei-Zhang-25/AnimalWorld.git
cd AnimalWorld
bash install.sh --gpu      # NVIDIA GPU (full engine). No GPU: bash install.sh
```

The full engine needs Python 3.10 or newer (the CPU engine: 3.9). If `python3` is older:
`PYTHON=python3.11 bash install.sh --gpu`.

## 2. Run the demo

```bash
~/animalworld-env/bin/animalworld demo                          # window: press Start
~/animalworld-env/bin/animalworld demo --headless --out results # no window
```

Model weights and six demo clips download automatically on first use and are checked by SHA-256.

## 3. Your videos

```bash
~/animalworld-env/bin/animalworld run                           # window: pick videos, press Start
~/animalworld-env/bin/animalworld zeroshot VIDEO.mp4 --out results --n-animals 2   # or --yes to accept the automatic count
```

Each video gets `tracks.csv` (identity, box and keypoints for every frame), `masks.npz`, `overlay.mp4` and a summary.

## Models

| Model | What it does |
|---|---|
| MouseWorld-DET | detection and segmentation (RF-DETR-Seg-L) |
| MouseWorld-DET-Lite | fast detection and segmentation, runs on CPU (YOLO11s-seg) |
| MouseWorld-Track | identities across frames, through contacts |
| MouseWorld-Pose | seven keypoints per mouse |

SAM 3.1 is optional (`--sam3`; gated licence from Meta) and off by default.

## More

Names in the paper and released files: [docs/PAPER_MAP.md](docs/PAPER_MAP.md) · full guide: [docs/README_full.md](docs/README_full.md) · install details: [INSTALL.md](INSTALL.md) · Studio (build models for
another species): [docs/STUDIO.md](docs/STUDIO.md).

Licence: code AGPL-3.0-or-later, model files AGPL-3.0. Improved using Qwen. Demo clips: CC BY 4.0, Guangwei Zhang laboratory.
