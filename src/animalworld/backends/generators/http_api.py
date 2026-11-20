"""Generic HTTP adapter for commercial image-generation APIs (optional; bring your own account).

You describe the request in project.yaml; the token is looked up BY NAME in the credential store
(environment variable or ~/.config/animalworld/credentials.toml) and is only ever placed in the
request header. It is never written to the project folder, the metadata log or an error message.

    generate:
      backend: http_api
      options:
        endpoint: https://api.example.com/v1/images
        token_credential: example_api          # a NAME, see `animalworld credentials set example_api`
        auth_header: Authorization             # default
        auth_scheme: "Bearer {token}"          # default
        body: {prompt: "{prompt}", negative_prompt: "{negative_prompt}", width: "{width}", height: "{height}", seed: "{seed}"}
        response: {b64_path: "data.0.b64_json"}   # or {url_path: "data.0.url"} or {raw: true}

The adapter itself is tested against a local stub server. The PRESETS below are UNTESTED examples
that require your own account; check the provider's current documentation and terms (including
whether generated images may be used to train models) before use.
"""
from __future__ import annotations

import base64
import copy
import io
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List

from ... import credentials
from ...registry import BackendError
from .base import Generator

PRESETS: Dict[str, Dict[str, Any]] = {
    # UNTESTED, requires your own account.
    "openai_images": {
        "endpoint": "https://api.openai.com/v1/images/generations",
        "token_credential": "openai_api_key",
        "body": {"model": "gpt-image-1", "prompt": "{prompt}", "size": "1024x1024", "n": 1},
        "response": {"b64_path": "data.0.b64_json"},
    },
    # UNTESTED, requires your own account. Generic shape used by several hosted diffusion services:
    # JSON in, a URL to the finished image out.
    "generic_url": {
        "endpoint": None,
        "token_credential": None,
        "body": {"prompt": "{prompt}", "negative_prompt": "{negative_prompt}", "width": "{width}", "height": "{height}", "seed": "{seed}"},
        "response": {"url_path": "images.0.url"},
    },
}
_INT_FIELDS = ("{width}", "{height}", "{seed}", "{n}")


def fill(template: Any, values: Dict[str, Any]) -> Any:
    if isinstance(template, dict):
        return {k: fill(v, values) for k, v in template.items()}
    if isinstance(template, list):
        return [fill(v, values) for v in template]
    if isinstance(template, str):
        if template in _INT_FIELDS:
            return values[template[1:-1]]
        out = template
        for k, v in values.items():
            out = out.replace("{" + k + "}", str(v))
        return out
    return template


def dig(obj: Any, path: str) -> Any:
    for part in path.split("."):
        obj = obj[int(part)] if isinstance(obj, list) else obj[part]
    return obj


class HTTPAPIGenerator(Generator):
    name = "http_api"
    kind = "image"
    needs_gpu = False
    tested = True      # the adapter; not the presets

    def __init__(self, options: Dict[str, Any] = None, project=None, **kw: Any):
        o = dict(options or {})
        if o.get("preset"):
            if o["preset"] not in PRESETS:
                raise BackendError(f"http_api: unknown preset {o['preset']!r}; presets: {', '.join(PRESETS)}")
            base = copy.deepcopy(PRESETS[o["preset"]])
            base.update({k: v for k, v in o.items() if v is not None})
            o = base
        o.setdefault("method", "POST")
        o.setdefault("auth_header", "Authorization")
        o.setdefault("auth_scheme", "Bearer {token}")
        o.setdefault("response", {"raw": True})
        super().__init__(o, project, **kw)

    def plan(self, items: List[dict]) -> dict:
        o = self.opt
        cred = o.get("token_credential")
        return dict(super().plan(items), endpoint=o.get("endpoint"), method=o["method"], preset=o.get("preset"),
                    token_credential=cred,
                    token_available=bool(cred and credentials.get(cred, required=False)),   # a yes/no, never the value
                    body_template=o.get("body"), response=o["response"],
                    note="commercial API: untested presets, requires your own account")

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json", "User-Agent": "animalworld-http-generator"}
        cred = self.opt.get("token_credential")
        if cred:
            h[self.opt["auth_header"]] = self.opt["auth_scheme"].replace("{token}", credentials.get(cred))
        return h

    def generate(self, items: List[dict], out_dir: Path) -> Iterator[dict]:
        import requests
        from PIL import Image
        o = self.opt
        if not o.get("endpoint"):
            raise BackendError("http_api: generate.options.endpoint is not set")
        headers = self._headers()
        pause = float(o.get("rate_limit_s", 1.0))
        for m in items:
            t0 = time.time()
            body = fill(o.get("body") or {"prompt": "{prompt}"},
                        {"prompt": m["prompt"], "negative_prompt": m["negative_prompt"], "width": m["W"],
                         "height": m["H"], "seed": m["seed"], "n": 1, "id": m["id"]})
            try:
                r = requests.request(o["method"], o["endpoint"], headers=headers, json=body,
                                     timeout=float(o.get("timeout_s", 300)))
            except requests.RequestException as e:
                raise BackendError(credentials.redact(f"http_api: request failed: {type(e).__name__}")) from None
            if r.status_code >= 400:
                raise BackendError(credentials.redact(f"http_api: HTTP {r.status_code} from the endpoint: {r.text[:300]}"))
            resp = o["response"]
            if resp.get("raw"):
                data = r.content
            elif resp.get("b64_path"):
                data = base64.b64decode(dig(r.json(), resp["b64_path"]))
            elif resp.get("url_path"):
                data = requests.get(dig(r.json(), resp["url_path"]), timeout=float(o.get("timeout_s", 300))).content
            else:
                raise BackendError("http_api: response needs one of raw / b64_path / url_path")
            img = Image.open(io.BytesIO(data)).convert("RGB")
            p = Path(out_dir) / f"{m['id']}.jpg"
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp.jpg")
            img.save(tmp, quality=95)
            os.replace(tmp, p)
            yield {"id": m["id"], "files": [{"id": m["id"], "file": str(p), "frame": -1}], "sec": round(time.time() - t0, 1),
                   "generator": "http_api", "endpoint_host": o["endpoint"].split("/")[2] if "//" in o["endpoint"] else ""}
            time.sleep(pause)
