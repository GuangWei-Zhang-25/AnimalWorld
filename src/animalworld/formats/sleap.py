"""SLEAP label exchange.

Route 1 (when `sleap-io` is installed): a native .slp file.
Route 2 (always available): a COCO keypoint JSON, which SLEAP imports (File > Import > COCO dataset;
`sleap_io.load_file(path, format="coco")` in recent sleap-io versions).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

from . import coco as C
from . import labels as L


def have_sleap_io() -> bool:
    try:
        import sleap_io  # noqa: F401
        return True
    except Exception:
        return False


def export_sleap(docs: List[dict], categories: List[str], kp_names: List[str], skeleton: List[List[int]],
                 out_dir: Path, image_root: Path, prefer_slp: bool = True) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    abs_docs = [dict(d, abs_file=str((Path(image_root) / d["file"]).resolve())) for d in docs]
    coco = C.to_coco(abs_docs, categories, kp_names, skeleton, file_key="abs_file")
    jpath = out_dir / "sleap_coco.json"
    jpath.write_text(json.dumps(coco), encoding="utf-8")
    res = {"coco_json": str(jpath), "slp": None, "route": "coco-json", "images": len(docs)}
    if prefer_slp and have_sleap_io():
        import numpy as np
        import sleap_io as sio
        skel = sio.Skeleton(nodes=list(kp_names), edges=[(kp_names[a], kp_names[b]) for a, b in (skeleton or [])])
        video = sio.Video.from_filename([d["abs_file"] for d in abs_docs])
        frames = []
        for k, d in enumerate(abs_docs):
            insts = []
            for i in d["instances"]:
                pts = np.array([[x, y] if v > 0 else [np.nan, np.nan] for x, y, v in L.kp_array(i, kp_names)], dtype=float)
                insts.append(sio.Instance.from_numpy(pts, skeleton=skel))
            frames.append(sio.LabeledFrame(video=video, frame_idx=k, instances=insts))
        slp = out_dir / "labels.slp"
        sio.Labels(labeled_frames=frames, videos=[video], skeletons=[skel]).save(str(slp))
        res.update(slp=str(slp), route="slp")
    return res


def import_sleap(path: Path, category: str = "animal", margin: float = 0.25) -> dict:
    """-> {"keypoints": [...], "docs": [...]} from a .slp (needs sleap-io) or a COCO keypoint JSON."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        coco = json.loads(path.read_text(encoding="utf-8"))
        docs = C.from_coco(coco, source="sleap-coco")
        for d in docs:
            d["status"] = "accepted"
        return {"keypoints": C.keypoint_names(coco), "docs": docs}
    if not have_sleap_io():
        raise RuntimeError("reading .slp files needs sleap-io (pip install 'animalworld[annotate]'); "
                           "or export COCO from SLEAP and import the .json")
    import numpy as np
    import sleap_io as sio
    labels = sio.load_slp(str(path))
    names = [n.name for n in labels.skeletons[0].nodes]
    docs = []
    for lf in labels.labeled_frames:
        fn = lf.video.filename
        fname = fn[lf.frame_idx] if isinstance(fn, (list, tuple)) else f"{Path(str(fn)).stem}_f{lf.frame_idx:06d}.jpg"
        shape = lf.video.shape
        H, W = (int(shape[1]), int(shape[2])) if shape is not None else (0, 0)
        inst = []
        for t, i in enumerate(lf.instances):
            pts = np.asarray(i.numpy(), dtype=float)[:, :2]
            kps = {n: ([float(p[0]), float(p[1]), 2] if not np.isnan(p).any() else [0.0, 0.0, 0]) for n, p in zip(names, pts)}
            ok = pts[~np.isnan(pts).any(1)]
            if not len(ok):
                continue
            m = margin * max(float(np.ptp(ok[:, 0])), float(np.ptp(ok[:, 1])), 8.0)
            inst.append(L.instance(category, [max(0.0, ok[:, 0].min() - m), max(0.0, ok[:, 1].min() - m),
                                              ok[:, 0].max() + m, ok[:, 1].max() + m], None, kps, track_id=t,
                                   box_from_keypoints=True))
        docs.append(L.label(Path(str(fname)).stem, str(fname), W, H, inst, source="sleap", status="accepted"))
    return {"keypoints": names, "docs": docs}
