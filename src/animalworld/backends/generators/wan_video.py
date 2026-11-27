"""Wan text-to-video through local diffusers weights (open weights; nothing is downloaded).

Adapted from mw_det_syn/sched/wan_worker.py: AutoencoderKLWan (float32) + WanPipeline, per-clip
seed on a CUDA generator, export_to_video. Defaults follow the recipe used for the MouseWorld
clips (Wan 2.2 T2V-A14B, 832x480, 121 frames, 16 fps, 40 steps, guidance 4.0). Do not lower frames
or steps to save time: clip quality drops.

NOT EXECUTED in this package's tests (needs a >= 40 GB GPU). `plan()` checks the configuration only.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List

from ...registry import BackendError, need
from .base import Generator

DEFAULTS = {"model_path": None, "width": 832, "height": 480, "num_frames": 121, "fps": 16, "num_inference_steps": 40,
            "guidance_scale": 4.0, "device": "cuda", "offload": "none"}   # offload: none | model | sequential


class WanVideoGenerator(Generator):
    name = "wan_video"
    kind = "video"
    ext = ".mp4"
    needs_gpu = True
    tested = False

    def __init__(self, options: Dict[str, Any] = None, project=None, **kw: Any):
        super().__init__({**DEFAULTS, **(options or {})}, project, **kw)

    def plan(self, items: List[dict]) -> dict:
        o = self.opt
        mp = o.get("model_path")
        return dict(super().plan(items), model_path=mp, model_path_exists=bool(mp and Path(mp).is_dir()),
                    size=f"{o['width']}x{o['height']}", num_frames=o["num_frames"], fps=o["fps"],
                    steps=o["num_inference_steps"], guidance_scale=o["guidance_scale"], offload=o["offload"],
                    call="WanPipeline(prompt, negative_prompt, height, width, num_frames, num_inference_steps, "
                         "guidance_scale, generator).frames[0] -> export_to_video(fps)")

    def generate(self, items: List[dict], out_dir: Path) -> Iterator[dict]:
        o = self.opt
        if not o.get("model_path") or not Path(o["model_path"]).is_dir():
            raise BackendError("wan_video: set generate.options.model_path to a local Wan diffusers folder")
        torch = need("torch", "generate", "the Wan generator")
        diffusers = need("diffusers", "generate", "the Wan generator")
        from diffusers.utils import export_to_video
        if not torch.cuda.is_available():
            raise BackendError("wan_video: no CUDA device (this generator does not run on CPU)")
        dt = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        vae = diffusers.AutoencoderKLWan.from_pretrained(o["model_path"], subfolder="vae", torch_dtype=torch.float32)
        pipe = diffusers.WanPipeline.from_pretrained(o["model_path"], vae=vae, torch_dtype=dt)
        if o["offload"] == "model":
            pipe.enable_model_cpu_offload()
        elif o["offload"] == "sequential":
            pipe.enable_sequential_cpu_offload()
        else:
            pipe.to(o["device"])
        gpu = torch.cuda.get_device_name(0)
        for m in items:
            t0 = time.time()
            gen = torch.Generator(device="cuda").manual_seed(int(m["seed"]))
            frames = pipe(prompt=m["prompt"], negative_prompt=m["negative_prompt"], height=int(o["height"]),
                          width=int(o["width"]), num_frames=int(o["num_frames"]),
                          num_inference_steps=int(o["num_inference_steps"]), guidance_scale=float(o["guidance_scale"]),
                          generator=gen).frames[0]
            p = Path(out_dir) / f"{m['id']}.mp4"
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = str(p)[:-4] + ".tmp.mp4"
            export_to_video(frames, tmp, fps=int(o["fps"]))
            os.replace(tmp, p)
            # The clip size comes from generate.options (width, height), not from the manifest's W / H, which are
            # image sizes; the size used is written to the generation log.
            yield {"id": m["id"], "video": str(p), "frames": len(frames), "fps": int(o["fps"]),
                   "video_W": int(o["width"]), "video_H": int(o["height"]), "steps": int(o["num_inference_steps"]),
                   "sec": round(time.time() - t0, 1), "gpu": gpu, "generator": "wan_video"}
