"""DeepLabCut label exchange: labeled-data/<video>/CollectedData_<scorer>.csv (+ .h5 when pandas/tables exist).

Single-animal files have three header rows (scorer, bodyparts, coords); multi-animal files have four
(scorer, individuals, bodyparts, coords). The row index is written in the current three-column form
(labeled-data, <video>, <image>); the older single-column form "labeled-data/<video>/<image>" is
read as well.
"""
from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from . import labels as L


def export_dlc(docs: List[dict], kp_names: List[str], out_dir: Path, image_root: Path, scorer: str = "animalworld",
               video: str = "animalworld", multi: Optional[bool] = None, copy_images: bool = True) -> dict:
    out_dir = Path(out_dir)
    folder = out_dir / "labeled-data" / video
    folder.mkdir(parents=True, exist_ok=True)
    nmax = max((len(d["instances"]) for d in docs), default=1)
    multi = (nmax > 1) if multi is None else multi
    inds = [f"animal{k + 1}" for k in range(max(1, nmax))] if multi else [None]
    ncol = len(inds) * len(kp_names) * 2
    rows = [["scorer", "", ""] + [scorer] * ncol]
    if multi:
        rows.append(["individuals", "", ""] + [i for i in inds for _ in range(2 * len(kp_names))])
    rows.append(["bodyparts", "", ""] + [k for _ in inds for k in kp_names for _ in range(2)])
    rows.append(["coords", "", ""] + ["x", "y"] * (len(inds) * len(kp_names)))
    n = 0
    for d in docs:
        name = Path(d["file"]).name
        if copy_images:
            src = Path(image_root) / d["file"]
            if src.exists() and not (folder / name).exists():
                shutil.copy(src, folder / name)
        vals: List[str] = []
        for k in range(len(inds)):
            inst = d["instances"][k] if k < len(d["instances"]) else None
            for x, y, v in (L.kp_array(inst, kp_names) if inst else [(0, 0, 0)] * len(kp_names)):
                vals += [f"{x:.2f}", f"{y:.2f}"] if v > 0 else ["", ""]
        rows.append(["labeled-data", video, name] + vals)
        n += 1
    csv_path = folder / f"CollectedData_{scorer}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    h5 = None
    try:                                       # optional: DLC reads the .h5; `deeplabcut.convertcsv2h5` also makes it
        import pandas as pd
        nh = 4 if multi else 3
        df = pd.read_csv(csv_path, header=list(range(nh)), index_col=[0, 1, 2])
        h5 = folder / f"CollectedData_{scorer}.h5"
        df.to_hdf(h5, key="df_with_missing", mode="w")
    except Exception:
        h5 = None
    (out_dir / "config_snippet.yaml").write_text(
        "# Paste into your DeepLabCut config.yaml (or create the project with these body parts).\n"
        f"scorer: {scorer}\n" + ("multianimalproject: true\nindividuals:\n" + "".join(f"- {i}\n" for i in inds) +
                                  "multianimalbodyparts:\n" if multi else "bodyparts:\n") +
        "".join(f"- {k}\n" for k in kp_names) + f"video_sets:\n  {video}: {{crop: '0, 0, 0, 0'}}\n", encoding="utf-8")
    return {"csv": str(csv_path), "h5": str(h5) if h5 else None, "images": n, "multi_animal": bool(multi),
            "note": None if h5 else "no .h5 written (pandas + tables not installed): run deeplabcut.convertcsv2h5(config)"}


def import_dlc(csv_path: Path, sizes: Dict[str, tuple] = None, category: str = "animal", margin: float = 0.25) -> dict:
    """-> {"keypoints": [names], "docs": [label documents]}. Image sizes are read from the files next to the CSV."""
    csv_path = Path(csv_path)
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    head = {}
    k = 0
    while k < len(rows) and rows[k] and rows[k][0] in ("scorer", "individuals", "bodyparts", "coords"):
        head[rows[k][0]] = rows[k]
        k += 1
    body = rows[k:]
    first = next((j for j, v in enumerate(head["coords"]) if v == "x"), 1)
    parts = head["bodyparts"][first:]
    inds = head.get("individuals", [""] * len(head["bodyparts"]))[first:]
    names: List[str] = []
    for p in parts:
        if p not in names:
            names.append(p)
    docs = []
    for r in body:
        if not r or not any(r):
            continue
        fname = r[first - 1] if first >= 3 else r[0].replace("\\", "/").rsplit("/", 1)[-1]
        vals = r[first:]
        per: Dict[str, Dict[str, list]] = {}
        for j in range(0, len(vals) - 1, 2):
            ind = inds[j] or "animal1"
            x, y = vals[j], vals[j + 1]
            per.setdefault(ind, {})[parts[j]] = [float(x), float(y), 2] if x not in ("", "nan", "NaN") and y not in ("", "nan", "NaN") else [0.0, 0.0, 0]
        W, H = (sizes or {}).get(fname, (0, 0))
        img = csv_path.parent / fname
        if not W and img.exists():
            from PIL import Image
            with Image.open(img) as im:
                W, H = im.size
        inst = []
        for t, (ind, kps) in enumerate(per.items()):
            pts = [(v[0], v[1]) for v in kps.values() if v[2] > 0]
            if not pts:
                continue
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            m = margin * max(max(xs) - min(xs), max(ys) - min(ys), 8.0)
            box = [max(0.0, min(xs) - m), max(0.0, min(ys) - m), max(xs) + m, max(ys) + m]
            if W:
                box[2], box[3] = min(float(W), box[2]), min(float(H), box[3])
            inst.append(L.instance(category, box, None, kps, track_id=t, box_from_keypoints=True))
        docs.append(L.label(fname.rsplit(".", 1)[0], fname, W, H, inst, source="dlc", status="accepted"))
    return {"keypoints": names, "docs": docs}
