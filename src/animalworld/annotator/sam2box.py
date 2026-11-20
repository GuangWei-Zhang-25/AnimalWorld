"""Box -> mask with SAM 2.1's image predictor, for the desktop annotator's "Mask from box" button.

    python -m animalworld.annotator.sam2box REQUEST.json RESULT.json

REQUEST: {"image": path, "boxes": [[x0, y0, x1, y1], ...], "config": sam2 config name, "checkpoint": path, "device": "cuda"}
RESULT:  {"polys": [[[x, y], ...] | null, ...], "scores": [...]}

Same calls as the `sam` proposer's variant `sam2` (backends/proposers/sam.py). Runs in its own process so the window
stays responsive and torch is never imported into the GUI. NOT EXECUTED in this package's tests (needs sam2, its
weights and normally a GPU).
"""
from __future__ import annotations

import json
import sys


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    with open(argv[0], encoding="utf-8") as f:
        req = json.load(f)
    try:
        import numpy as np
        import torch
        from PIL import Image
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError as e:
        print(f"sam2box: SAM 2 is not installed in this Python ({e}). Install torch and sam2 (pip install sam2).", file=sys.stderr)
        return 3
    from ..backends.proposers.base import mask_to_polygon
    device = req.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    pred = SAM2ImagePredictor(build_sam2(req["config"], str(req["checkpoint"]), device=device))
    polys, scores = [], []
    with torch.inference_mode():
        pred.set_image(np.asarray(Image.open(req["image"]).convert("RGB")))
        for b in req["boxes"]:
            masks, sc, _ = pred.predict(box=np.array(b, dtype=np.float32), multimask_output=False)
            poly, _, _ = mask_to_polygon(masks[0] > 0)
            polys.append(poly)
            scores.append(float(sc[0]))
    with open(argv[1], "w", encoding="utf-8") as f:
        json.dump({"polys": polys, "scores": scores}, f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
