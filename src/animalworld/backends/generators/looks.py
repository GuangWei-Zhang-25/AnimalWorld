"""Exact per-pixel camera looks applied AFTER labelling (labels need no transform).

Reduced numpy/PIL port of mw_det_syn/gap_targeted/scripts/look.py (which uses OpenCV): no crop,
warp, resize or shift. Camera parameters are drawn once per source unit (seeded by its id) so all
frames of one clip share one camera; sensor noise is re-drawn per frame. The burnt-in timestamp
and glare blob of the original are not ported.
"""
from __future__ import annotations

import hashlib
import random

import numpy as np
from PIL import Image, ImageFilter

LOOKS = ("none", "mono", "lowlight", "actioncam", "invert")


def _seed(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)


def params(look: str, unit: str) -> dict:
    r = random.Random(_seed(unit + look))
    P = {"look": look, "q": 92}
    if look == "mono":            # near-infrared-like monochrome camera
        P.update(mix=(r.uniform(0.35, 0.6), r.uniform(0.25, 0.45)), lo=r.uniform(5, 30), hi=r.uniform(180, 245),
                 gamma=r.uniform(0.8, 1.2), noise=r.uniform(3, 8), vign=r.uniform(0.0, 0.5), blur=r.choice([0, 0.5, 0.8]),
                 q=r.randint(75, 95))
    elif look == "lowlight":      # colour kept: exposure x0.2-0.45 + gamma, sensor noise, JPEG
        P.update(gain=r.uniform(0.2, 0.45), gamma=r.uniform(1.1, 1.8), noise=r.uniform(3, 9), blur=r.choice([0, 0.5, 0.8]),
                 q=r.randint(40, 80))
    elif look == "actioncam":     # colour kept: veiling haze, slight blur, low resolution + compression
        P.update(haze=r.uniform(0.1, 0.35), hazev=r.uniform(170, 235), blur=r.choice([0.5, 0.8, 1.2]),
                 down=r.uniform(1.5, 3.0), noise=r.uniform(1, 5), q=r.randint(30, 70))
    elif look in ("none", "invert"):
        pass
    else:
        raise ValueError(f"unknown look {look!r}; looks: {', '.join(LOOKS)}")
    return P


def _blur(a: np.ndarray, sigma: float) -> np.ndarray:
    if not sigma:
        return a
    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    return np.asarray(im.filter(ImageFilter.GaussianBlur(sigma))).astype(np.float32)


def apply(img: Image.Image, P: dict, frame_seed: int = 0) -> Image.Image:
    look = P["look"]
    if look == "none":
        return img
    a = np.asarray(img.convert("RGB")).astype(np.float32)
    h, w = a.shape[:2]
    rng = np.random.default_rng(frame_seed)
    if look == "invert":
        return Image.fromarray((255 - a).astype(np.uint8))
    if look == "mono":
        rw, gw = P["mix"]
        g = a[..., 0] * rw + a[..., 1] * gw + a[..., 2] * (1 - rw - gw)
        lo, hi = np.percentile(g, 1), np.percentile(g, 99.5)
        g = np.clip((g - lo) / max(1.0, hi - lo), 0, 1) ** P["gamma"]
        g = P["lo"] + g * (P["hi"] - P["lo"])
        if P["vign"]:
            y, x = np.ogrid[:h, :w]
            g = g * (1 - P["vign"] * (((x - w / 2) / (w / 2)) ** 2 + ((y - h / 2) / (h / 2)) ** 2) / 2)
        g = _blur(np.repeat(g[..., None], 3, -1), P["blur"])[..., 0] + rng.normal(0, P["noise"], (h, w))
        out = np.repeat(g[..., None], 3, -1)
    elif look == "lowlight":
        out = 255 * (a / 255) ** P["gamma"] * P["gain"]
        out = _blur(out, P["blur"]) + rng.normal(0, P["noise"], (h, w))[..., None]
    else:                         # actioncam
        out = (1 - P["haze"]) * a + P["haze"] * P["hazev"]
        small = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).resize(
            (max(1, int(w / P["down"])), max(1, int(h / P["down"]))), Image.BOX)
        out = np.asarray(small.resize((w, h), Image.BILINEAR)).astype(np.float32)
        out = _blur(out, P["blur"]) + rng.normal(0, P["noise"], (h, w))[..., None]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
