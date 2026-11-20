"""COCO detection / segmentation / keypoint JSON <-> internal label documents."""
from __future__ import annotations

from typing import Dict, List

from . import labels as L


def to_coco(docs: List[dict], categories: List[str], kp_names: List[str] = None, skeleton: List[List[int]] = None,
            file_key: str = "file") -> dict:
    kp_names = list(kp_names or [])
    cats = []
    for k, c in enumerate(categories, 1):
        cat = {"id": k, "name": c, "supercategory": "animal"}
        if kp_names:
            cat["keypoints"] = kp_names
            cat["skeleton"] = [[a + 1, b + 1] for a, b in (skeleton or [])]
        cats.append(cat)
    cid = {c: k for k, c in enumerate(categories, 1)}
    images, anns = [], []
    for n, d in enumerate(docs, 1):
        images.append({"id": n, "file_name": d[file_key], "width": d["W"], "height": d["H"]})
        for i in d["instances"]:
            x0, y0, x1, y1 = i["box"]
            a = {"id": len(anns) + 1, "image_id": n, "category_id": cid.get(i.get("category"), 1), "iscrowd": 0,
                 "bbox": [round(x0, 2), round(y0, 2), round(x1 - x0, 2), round(y1 - y0, 2)],
                 "area": round(L.poly_area(i["poly"]) if i.get("poly") else L.box_area(i["box"]), 2)}
            if i.get("poly"):
                a["segmentation"] = [[round(v, 2) for pt in i["poly"] for v in pt]]
            if kp_names:
                ks = L.kp_array(i, kp_names)
                a["keypoints"] = [round(float(v), 2) if j < 2 else int(v) for k in ks for j, v in enumerate(k)]
                a["num_keypoints"] = sum(1 for k in ks if k[2] > 0)
            if i.get("track_id") is not None:
                a["track_id"] = i["track_id"]
            anns.append(a)
    return {"info": {"description": "animalworld export", "version": "1.0"}, "licenses": [],
            "images": images, "annotations": anns, "categories": cats}


def from_coco(coco: dict, source: str = "coco") -> List[dict]:
    cats = {c["id"]: c for c in coco.get("categories", [])}
    by_img: Dict[int, list] = {}
    for a in coco.get("annotations", []):
        by_img.setdefault(a["image_id"], []).append(a)
    docs = []
    for im in coco.get("images", []):
        inst = []
        for a in by_img.get(im["id"], []):
            c = cats.get(a["category_id"], {})
            x, y, w, h = a["bbox"]
            seg = a.get("segmentation")
            poly = None
            if isinstance(seg, list) and seg and isinstance(seg[0], list) and len(seg[0]) >= 6:
                poly = [[seg[0][k], seg[0][k + 1]] for k in range(0, len(seg[0]) - 1, 2)]
            kps = None
            names = c.get("keypoints") or []
            if a.get("keypoints") and names:
                v = a["keypoints"]
                kps = {nm: [v[3 * k], v[3 * k + 1], int(v[3 * k + 2])] for k, nm in enumerate(names) if 3 * k + 2 < len(v)}
            inst.append(L.instance(c.get("name", "animal"), [x, y, x + w, y + h], poly, kps,
                                   score=a.get("score", 1.0), track_id=a.get("track_id")))
        stem = str(im["file_name"]).replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0]
        docs.append(L.label(stem, im["file_name"], im["width"], im["height"], inst, source=source))
    return docs


def keypoint_names(coco: dict) -> List[str]:
    for c in coco.get("categories", []):
        if c.get("keypoints"):
            return list(c["keypoints"])
    return []
