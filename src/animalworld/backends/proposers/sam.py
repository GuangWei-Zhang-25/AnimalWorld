"""SAM adapter: SAM 3.1 text-prompted image labelling, or SAM 2.1 box-prompted mask refinement.

variant: sam3 (default)
    Adapted from mw_det_syn/gap_targeted/scripts/label_img.py:
        proc = Sam3Processor(build_sam3_image_model(), confidence_threshold=0.3)
        state = proc.set_image(PIL image)            # a PIL image, NOT a numpy array (an ndarray
        state = proc.set_text_prompt(state=state, prompt="mouse")   #  silently gives wrong masks)
        masks = state["masks"], scores = state["scores"], boxes = state["boxes"]
    Duplicates (intersection / smaller mask > 0.5) are merged keeping the higher score; a mask whose
    largest connected component is < 0.85 of the mask is marked fragmented.
    bfloat16 autocast is used on GPUs that support it (it fails on pre-Ampere cards: set autocast: false).

variant: sam2
    Turns existing BOXES (from a human or a detector) into masks with SAM 2.1's image predictor:
        predictor = SAM2ImagePredictor(build_sam2(config, checkpoint)); predictor.set_image(ndarray)
        masks, scores, _ = predictor.predict(box=[x0, y0, x1, y1], multimask_output=False)
    Needs options.config and options.checkpoint, and boxes in labels/reviewed or labels/proposed.

NOT EXECUTED in this package's tests (needs the sam3 / sam2 packages, their weights and a GPU).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List

from ...formats import labels as L
from ...registry import BackendError, need
from ...utils import read_json
from .base import Proposer, mask_to_polygon

DEFAULTS = {"variant": "sam3", "prompt": None, "confidence_threshold": 0.3, "autocast": True, "device": "cuda",
            "dedup_overlap": 0.5, "min_component_frac": 0.85, "config": None, "checkpoint": None}


class SAMProposer(Proposer):
    name = "sam"
    needs_gpu = True
    tested = False

    def __init__(self, options: Dict[str, Any] = None, project=None, **kw: Any):
        super().__init__({**DEFAULTS, **(options or {})}, project, **kw)

    def plan(self, rows: List[dict]) -> dict:
        o = self.opt
        return dict(super().plan(rows), variant=o["variant"], text_prompt=o["prompt"] or self.category,
                    confidence_threshold=o["confidence_threshold"], checkpoint=o.get("checkpoint"),
                    call=("Sam3Processor(build_sam3_image_model()).set_image(PIL) -> set_text_prompt(prompt)"
                          if o["variant"] == "sam3" else "SAM2ImagePredictor(build_sam2(config, checkpoint)).predict(box=...)"))

    def propose(self, rows: List[dict]) -> Iterator[dict]:
        if self.opt["variant"] == "sam3":
            yield from self._sam3(rows)
        elif self.opt["variant"] == "sam2":
            yield from self._sam2(rows)
        else:
            raise BackendError("sam: options.variant must be sam3 or sam2")

    def _sam3(self, rows):
        import numpy as np
        from PIL import Image
        torch = need("torch", "generate", "the SAM proposer")
        sam3 = need("sam3", "generate", "the SAM 3 proposer (install SAM 3 from its repository)")
        from sam3.model.sam3_image_processor import Sam3Processor
        o = self.opt
        if o["device"].startswith("cuda") and not torch.cuda.is_available():
            raise BackendError("sam: no CUDA device")
        proc = Sam3Processor(sam3.build_sam3_image_model(), confidence_threshold=float(o["confidence_threshold"]))
        prompt = o["prompt"] or self.category
        for r in rows:
            im = Image.open(r["path"]).convert("RGB")          # PIL, never ndarray
            W, H = im.size
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(o["autocast"])):
                st = proc.set_image(im)
                st = proc.set_text_prompt(state=st, prompt=prompt)
                ms = st["masks"].squeeze(1).cpu().numpy() > 0 if len(st["masks"]) else np.zeros((0, H, W), bool)
                sc = st["scores"].float().cpu().numpy()
            keep: List[int] = []
            for i in np.argsort(-sc):
                a = ms[i]
                if a.sum() == 0:
                    continue
                if any((a & ms[k]).sum() / max(1, min(a.sum(), ms[k].sum())) > float(o["dedup_overlap"]) for k in keep):
                    continue
                keep.append(int(i))
            inst = []
            for i in keep:
                poly, frac, edge = mask_to_polygon(ms[i])
                if poly is None:
                    continue
                ys, xs = np.where(ms[i])
                inst.append(L.instance(self.category, [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], poly, None,
                                       score=float(sc[i]), area=int(ms[i].sum()), edge=round(edge, 3),
                                       fragmented=bool(frac < float(o["min_component_frac"]))))
            yield L.label(r["id"], r["file"], W, H, inst, source="sam3")

    def _sam2(self, rows):
        import numpy as np
        from PIL import Image
        torch = need("torch", "generate", "the SAM proposer")
        need("sam2", "generate", "the SAM 2 proposer (install SAM 2 from its repository)")
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        o = self.opt
        if not o.get("config") or not o.get("checkpoint"):
            raise BackendError("sam (variant sam2): set options.config and options.checkpoint")
        pred = SAM2ImagePredictor(build_sam2(o["config"], str(o["checkpoint"]), device=o["device"]))
        root = self.project.root
        for r in rows:
            prev = read_json(root / "labels" / "reviewed" / f"{r['id']}.json") or read_json(root / "labels" / "proposed" / f"{r['id']}.json")
            if not prev:
                continue
            im = Image.open(r["path"]).convert("RGB")
            W, H = im.size
            inst = []
            with torch.inference_mode():
                pred.set_image(np.asarray(im))
                for i in prev["instances"]:
                    masks, scores, _ = pred.predict(box=np.array(i["box"], dtype=np.float32), multimask_output=False)
                    poly, frac, edge = mask_to_polygon(masks[0] > 0)
                    inst.append(dict(i, poly=poly, score=float(scores[0])) if poly else i)
            yield L.label(r["id"], r["file"], W, H, inst, source="sam2")
