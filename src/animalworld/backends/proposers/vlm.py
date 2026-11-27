"""Vision-language-model proposer: one image goes to a vision chat model, which answers with boxes (and,
optionally, keypoints) as JSON.

The request is built from the project: the species (slots.yaml `animal`), the keypoint names, the image size and,
unless options.tell_count is false, the animal count the image was generated for. The answer must be ONE JSON object

    {"instances": [{"box": [x0, y0, x1, y1], "score": 0.0-1.0, "keypoints": {"nose": [x, y], ...}}]}

in pixels of the image as it was sent. Anything else is refused: no JSON object, a missing "instances" list, a box
that is not four finite numbers with x1 > x0 and y1 > y0. Boxes are clipped to the image; keypoints with unknown
names or outside the image are dropped. The result is an ordinary label document (source "vlm", no outline), so
it passes the same gates as every other proposal (score, area, count, border on the box) and waits for review.

Providers (options.provider):
  openai_compatible  POST {base_url}/chat/completions with the image as a base64 data URL
                     (options.base_url, options.model, options.token_credential): any server that implements
                     the vision chat format, including a local open-weight VLM served by vLLM or similar.
  anthropic          the official `anthropic` SDK (pip install 'animalworld[llm]'); credentials: the SDK's own
                     resolution, or a credential named in options.token_credential.

Other options: max_side (default 1024: the longer image side that is sent; answers are scaled back),
tell_count (default true; note that the count gate then checks the model against a number it was told),
keypoints (default true when the project has keypoints), default_score (0.5, used when the model gives none),
timeout_s, max_tokens.

NOT YET RUN AGAINST A LIVE SERVICE. The request building, the strict parsing and the mapping into label documents
are tested against a local stub server only; nobody has measured how good the boxes or keypoints of any real
vision-language model are on generated animal images. Treat its output as unverified proposals.
"""
from __future__ import annotations

import base64
import io
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from ... import credentials
from ...formats import labels as L
from ...registry import BackendError, need
from ...utils import read_jsonl
from .base import Proposer

DEFAULTS = {"provider": "openai_compatible", "base_url": None, "model": None, "token_credential": None,
            "max_side": 1024, "tell_count": True, "keypoints": True, "default_score": 0.5, "timeout_s": 120,
            "max_tokens": 16000, "fallbacks": True}
DEFAULT_ANTHROPIC_MODEL = "claude-opus-5-5"


def build_prompt(singular: str, plural: str, definition: str, W: int, H: int, expected_n: Optional[int],
                 kp_names: List[str]) -> str:
    lines = [
        f"This is one frame of a research recording. Find every {singular} in it.",
        f"The image is {W} pixels wide and {H} pixels high; x runs from 0 at the left edge to {W}, y from 0 at the "
        f"top edge to {H}.",
    ]
    if definition:
        lines.append(f"What counts as a {singular}: {definition}.")
    if expected_n is not None:
        lines.append(f"The scene was set up with {expected_n} {singular if expected_n == 1 else plural}. Report what "
                     "you actually see: fewer if some are not visible, none if the scene is empty.")
    lines.append('Answer with one JSON object and nothing else: {"instances": [{"box": [x0, y0, x1, y1], "score": s'
                 + (', "keypoints": {...}' if kp_names else "") + "}]}.")
    lines.append(f"box: the tight box around the whole visible body of one {singular}, in pixels. "
                 "score: your confidence from 0 to 1 that this is one such animal.")
    if kp_names:
        lines.append("keypoints: an object that maps each of these body-part names to [x, y] in pixels: "
                     + ", ".join(kp_names) + ". Leave out a part you cannot locate; left and right are the animal's "
                     "own left and right.")
    lines.append('An empty scene is {"instances": []}.')
    return "\n".join(lines)


def _num(v: Any) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ValueError(f"not a finite number: {v!r}")
    return float(v)


def parse_answer(text: str, W: int, H: int, category: str, kp_names: List[str], scale: float = 1.0,
                 default_score: float = 0.5) -> List[dict]:
    """The model's text -> instances in full-image pixels (`scale` = sent size / original size). Strict: raises
    BackendError when the answer is not the requested JSON; single malformed instances are dropped."""
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        raise BackendError("vlm: the answer contained no JSON object")
    try:
        data = json.loads(m.group(0))
    except ValueError as e:
        raise BackendError(f"vlm: the answer is not valid JSON ({e})") from None
    if not isinstance(data, dict) or not isinstance(data.get("instances"), list):
        raise BackendError('vlm: the answer has no "instances" list')
    out = []
    for raw in data["instances"]:
        try:
            b = [_num(v) / scale for v in raw["box"]]
            if len(b) != 4:
                raise ValueError("box needs four numbers")
            x0, y0, x1, y1 = max(0.0, b[0]), max(0.0, b[1]), min(float(W), b[2]), min(float(H), b[3])
            if x1 - x0 < 1 or y1 - y0 < 1:
                raise ValueError("empty box")
            score = min(1.0, max(0.0, _num(raw["score"]))) if "score" in raw else float(default_score)
            kps = {}
            for k, v in (raw.get("keypoints") or {}).items() if isinstance(raw.get("keypoints"), dict) else []:
                if k in kp_names and isinstance(v, (list, tuple)) and len(v) >= 2:
                    try:
                        x, y = _num(v[0]) / scale, _num(v[1]) / scale
                    except ValueError:
                        continue
                    if 0 <= x <= W and 0 <= y <= H:
                        kps[k] = [x, y, 2]
        except (KeyError, TypeError, ValueError):
            continue
        out.append(L.instance(category, [x0, y0, x1, y1], None, kps or None, score=score))
    return out


class VLMProposer(Proposer):
    name = "vlm"
    needs_gpu = False
    tested = False       # stub server only; see the module docstring

    def __init__(self, options: Dict[str, Any] = None, project=None, **kw: Any):
        super().__init__({**DEFAULTS, **(options or {})}, project, **kw)
        an = (project.slots().get("animal") or {}) if project is not None else {}
        self.singular = an.get("singular") or self.category
        self.plural = an.get("plural") or self.singular + "s"
        self.definition = an.get("definition") or ""
        self.kp = self.kp_names if self.opt["keypoints"] else []

    def plan(self, rows: List[dict]) -> dict:
        o = self.opt
        cred = o.get("token_credential")
        return dict(super().plan(rows), provider=o["provider"], base_url=o.get("base_url"),
                    model=o.get("model") or (DEFAULT_ANTHROPIC_MODEL if o["provider"] == "anthropic" else None),
                    token_credential=cred, token_available=bool(cred and credentials.get(cred, required=False)),
                    max_side=o["max_side"], tell_count=bool(o["tell_count"]), keypoints=self.kp,
                    network=True, note="not yet run against a live service; proposals are unverified",
                    example_prompt=build_prompt(self.singular, self.plural, self.definition, 1024, 768,
                                                1 if o["tell_count"] else None, self.kp))

    # ---- providers -----------------------------------------------------------------------
    def _ask_openai_compatible(self, prompt: str, jpeg: bytes) -> str:
        import requests
        o = self.opt
        base = (o.get("base_url") or "").rstrip("/")
        if not base or not o.get("model"):
            raise BackendError("vlm (openai_compatible): set propose.options.base_url and propose.options.model")
        headers = {"Content-Type": "application/json"}
        if o.get("token_credential"):
            headers["Authorization"] = "Bearer " + credentials.get(o["token_credential"])
        url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
        body = {"model": o["model"], "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": prompt}]}]}
        try:
            r = requests.post(base + "/chat/completions", headers=headers, json=body, timeout=float(o["timeout_s"]))
        except requests.RequestException as e:
            raise BackendError(credentials.redact(f"vlm: request failed: {type(e).__name__}")) from None
        if r.status_code >= 400:
            raise BackendError(credentials.redact(f"vlm: HTTP {r.status_code} from the endpoint: {r.text[:300]}"))
        try:
            return r.json()["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError):
            raise BackendError("vlm: the endpoint did not answer in the chat-completions format") from None

    def _ask_anthropic(self, prompt: str, jpeg: bytes) -> str:
        anthropic = need("anthropic", "llm", "the Anthropic vision proposer")
        o = self.opt
        cred = o.get("token_credential")
        client = anthropic.Anthropic(api_key=credentials.get(cred)) if cred else anthropic.Anthropic()
        kw = dict(model=o.get("model") or DEFAULT_ANTHROPIC_MODEL, max_tokens=int(o["max_tokens"]),
                  messages=[{"role": "user", "content": [
                      {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                   "data": base64.standard_b64encode(jpeg).decode("utf-8")}},
                      {"type": "text", "text": prompt}]}])
        try:
            if o.get("fallbacks", True):
                # Server-side refusal fallback, as in the LLM prompt writer (propose.options.fallbacks: false turns it
                # off, e.g. on platforms that do not offer it).
                resp = client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kw)
            else:
                resp = client.messages.create(**kw)
        except anthropic.APIStatusError as e:
            raise BackendError(credentials.redact(f"vlm: Anthropic API error {e.status_code}")) from None
        except anthropic.APIConnectionError:
            raise BackendError("vlm: could not reach the Anthropic API") from None
        if resp.stop_reason == "refusal":
            raise BackendError("vlm: the model declined this request (stop_reason=refusal)")
        return "".join(b.text for b in resp.content if b.type == "text")

    def ask(self, prompt: str, jpeg: bytes) -> str:
        p = self.opt["provider"]
        if p == "openai_compatible":
            return self._ask_openai_compatible(prompt, jpeg)
        if p == "anthropic":
            return self._ask_anthropic(prompt, jpeg)
        raise BackendError(f"vlm: unknown provider {p!r} (openai_compatible | anthropic)")

    # ---- proposals ------------------------------------------------------------------------
    def propose(self, rows: List[dict]) -> Iterator[dict]:
        from PIL import Image
        n_by_unit = {}
        if self.project is not None and self.opt["tell_count"]:
            n_by_unit = {m["id"]: m["n"] for m in read_jsonl(Path(self.project.path("prompts", "manifest.jsonl")))}
        for r in rows:
            im = Image.open(r["path"]).convert("RGB")
            W, H = im.size
            scale = min(1.0, float(self.opt["max_side"]) / max(W, H))
            sent = im if scale == 1.0 else im.resize((max(1, round(W * scale)), max(1, round(H * scale))), Image.BILINEAR)
            buf = io.BytesIO()
            sent.save(buf, "JPEG", quality=90)
            prompt = build_prompt(self.singular, self.plural, self.definition, sent.size[0], sent.size[1],
                                  n_by_unit.get(r.get("unit")), self.kp)
            inst = parse_answer(self.ask(prompt, buf.getvalue()), W, H, self.category, self.kp, scale,
                                float(self.opt["default_score"]))
            yield L.label(r["id"], r["file"], W, H, inst, source="vlm",
                          vlm={"provider": self.opt["provider"], "model": self.opt.get("model") or
                               (DEFAULT_ANTHROPIC_MODEL if self.opt["provider"] == "anthropic" else None)})
