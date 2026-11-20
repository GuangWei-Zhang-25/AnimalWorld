"""Qwen-Image text-to-image through local diffusers weights (open weights; nothing is downloaded).

Adapted from mw_det_syn/gap_targeted/scripts/gen_qwen_split.py (split path: the text encoder and the
generator are on the GPU one after the other: encode all prompts -> free the encoder -> generate
from the embeddings; optional torch.compile of the transformer with items ordered by size) and
gen_qwen.py (single-pipeline path). Settings recorded in gap_targeted/qwen/GEN_SETTINGS.json:
30 steps, true_cfg_scale 4.0, torch.bfloat16, per-item seed on a CUDA generator.

NOT EXECUTED in this package's tests (needs a GPU and ~60 GB of weights). `plan()` and
`animalworld generate --dry-run` check the configuration without loading anything.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List

from ...registry import BackendError, need
from .base import Generator

DEFAULTS = {
    "model_path": None,                  # e.g. /path/to/Qwen-Image-2.1 (a local diffusers folder)
    "model_id": None,                    # OR a Hugging Face id, e.g. Qwen/Qwen-Image-2.1; used when model_path is empty or missing.
                                         # UNTESTED: the weights are then downloaded by diffusers from the official repo.
    "pipeline_class": "QwenImage21Pipeline",   # diffusers class name; "QwenImagePipeline" for Qwen-Image 1.x
    "num_inference_steps": 30,
    "true_cfg_scale": 4.0,
    "dtype": "bfloat16",
    "device": "cuda",
    "split": True,                       # encode prompts first, free the encoder, then generate
    "compile": False,                    # torch.compile(transformer); pays off for long runs
    "jpeg_quality": 95,
    "budget_s": 0,                       # stop starting new images when this many seconds have passed (0 = no limit)
}


class QwenImageGenerator(Generator):
    name = "qwen_image"
    kind = "image"
    needs_gpu = True
    tested = False

    def __init__(self, options: Dict[str, Any] = None, project=None, **kw: Any):
        super().__init__({**DEFAULTS, **(options or {})}, project, **kw)

    def plan(self, items: List[dict]) -> dict:
        mp = self.opt.get("model_path")
        return dict(super().plan(items), model_path=mp, model_path_exists=bool(mp and Path(mp).is_dir()),
                    model_id=self.opt.get("model_id"),
                    source=("local folder" if mp else ("Hugging Face download at first use (UNTESTED path)"
                                                       if self.opt.get("model_id") else "not set")),
                    model_index=bool(mp and (Path(mp) / "model_index.json").exists()),
                    pipeline_class=self.opt["pipeline_class"], steps=self.opt["num_inference_steps"],
                    true_cfg_scale=self.opt["true_cfg_scale"], dtype=self.opt["dtype"], split=self.opt["split"],
                    compile=self.opt["compile"], sizes=sorted({(m["W"], m["H"]) for m in items})[:12],
                    call="pipe(prompt_embeds=..., negative_prompt_embeds=..., true_cfg_scale, height, width, "
                         "num_inference_steps, generator=torch.Generator(device).manual_seed(seed))")

    def generate(self, items: List[dict], out_dir: Path) -> Iterator[dict]:
        o = self.opt
        src = o.get("model_path")
        if src and not Path(src).is_dir() and o.get("model_id"):
            print(f"qwen_image: the weights folder {src} does not exist; using the Hugging Face id instead", flush=True)
            src = None
        if src:
            if not Path(src).is_dir():
                raise BackendError(f"qwen_image: generate.options.model_path {src!r} is not a folder")
        elif o.get("model_id"):
            src = str(o["model_id"])
            print(f"qwen_image: no local model_path; loading '{src}' through diffusers. If it is not in your Hugging Face "
                  "cache yet, the weights are downloaded now from the official repository huggingface.co/" + src +
                  " (Qwen-Image 2.1 is about 31 GB: check disk space and the model licence; set HF_HOME to choose the "
                  "cache folder). This download path is UNTESTED in this package.", flush=True)
        else:
            raise BackendError("qwen_image: set generate.options.model_path (a local Qwen-Image diffusers folder) "
                               "or generate.options.model_id (a Hugging Face id such as Qwen/Qwen-Image)")
        torch = need("torch", "generate", "the Qwen-Image generator")
        diffusers = need("diffusers", "generate", "the Qwen-Image generator")
        if o["device"].startswith("cuda") and not torch.cuda.is_available():
            raise BackendError("qwen_image: no CUDA device (this generator does not run on CPU)")
        Pipe = getattr(diffusers, o["pipeline_class"], None)
        if Pipe is None:
            raise BackendError(f"qwen_image: diffusers has no class {o['pipeline_class']}; set generate.options.pipeline_class")
        dt = getattr(torch, o["dtype"])
        dev = o["device"]
        t_start = time.time()
        items = list(items)
        gpu = torch.cuda.get_device_name(0) if dev.startswith("cuda") else dev
        common = dict(true_cfg_scale=float(o["true_cfg_scale"]), num_inference_steps=int(o["num_inference_steps"]))

        if o["split"]:
            import gc
            enc = Pipe.from_pretrained(src, transformer=None, vae=None, torch_dtype=dt).to(dev)
            E, NEG = {}, {}
            with torch.no_grad():
                for m in items:
                    pe = enc.encode_prompt(prompt=m["prompt"], device=dev)
                    E[m["id"]] = (pe[0].cpu(), pe[1].cpu() if pe[1] is not None else None)
                    if m["negative_prompt"] not in NEG:
                        ne = enc.encode_prompt(prompt=m["negative_prompt"], device=dev)
                        NEG[m["negative_prompt"]] = (ne[0].cpu(), ne[1].cpu() if ne[1] is not None else None)
            del enc
            gc.collect()
            if dev.startswith("cuda"):
                torch.cuda.empty_cache()
            pipe = Pipe.from_pretrained(src, text_encoder=None, tokenizer=None, torch_dtype=dt).to(dev)
        else:
            pipe = Pipe.from_pretrained(src, torch_dtype=dt).to(dev)
        pipe.set_progress_bar_config(disable=True)
        if o["compile"]:
            import torch._dynamo as _dyn
            _dyn.config.recompile_limit = 32            # several image sizes x cond/uncond exceed the default of 8
            pipe.transformer = torch.compile(pipe.transformer)
            items.sort(key=lambda m: (m["W"], m["H"]))  # one compile per size
        mv = lambda t: None if t is None else t.to(dev)   # noqa: E731
        per: List[float] = []
        for m in items:
            if o["budget_s"] and per and time.time() - t_start + 3 * max(per[-10:]) + 60 > float(o["budget_s"]):
                break
            t0 = time.time()
            gen = torch.Generator(dev).manual_seed(int(m["seed"]))
            if o["split"]:
                (pe, pm), (ne, nm) = E[m["id"]], NEG[m["negative_prompt"]]
                img = pipe(prompt_embeds=mv(pe), prompt_embeds_mask=mv(pm), negative_prompt_embeds=mv(ne),
                           negative_prompt_embeds_mask=mv(nm), height=m["H"], width=m["W"], generator=gen, **common).images[0]
            else:
                img = pipe(prompt=m["prompt"], negative_prompt=m["negative_prompt"], height=m["H"], width=m["W"],
                           generator=gen, **common).images[0]
            p = Path(out_dir) / f"{m['id']}.jpg"
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp.jpg")
            img.convert("RGB").save(tmp, quality=int(o["jpeg_quality"]))
            os.replace(tmp, p)
            per.append(time.time() - t0)
            yield {"id": m["id"], "files": [{"id": m["id"], "file": str(p), "frame": -1}], "sec": round(per[-1], 1),
                   "gpu": gpu, "generator": "qwen_image",
                   "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2 ** 30, 2) if dev.startswith("cuda") else None, "path": ("split" if o["split"] else "single") + ("+compile" if o["compile"] else "")}
