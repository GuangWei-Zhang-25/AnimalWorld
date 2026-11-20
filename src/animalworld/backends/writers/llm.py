"""Optional LLM slot expansion.

The LLM never writes final prompts: it only proposes extra VALUES for the slots (new arenas,
appearances, behaviours ...) from the species name, the user's description and the taxonomy
candidates. The deterministic template writer then fills the slots, so the distinct-prompt check and
the seeds stay reproducible. The expanded vocabulary is written to prompts/slots_expanded.yaml for
the user to read and edit.

Providers (options.provider):
  anthropic          Anthropic API through the official `anthropic` SDK (pip install 'animalworld[llm]').
                     Credentials: the SDK's own resolution (ANTHROPIC_API_KEY, or an `ant auth login`
                     profile), or a credential named in options.token_credential.
  openai_compatible  Any endpoint that implements POST {base_url}/chat/completions
                     (options.base_url, options.model, options.token_credential).

UNTESTED against live services: only exercised with a stub client in the test-suite.
"""
from __future__ import annotations

import copy
import json
import re
from typing import Any, Dict, List

from ... import credentials
from ...registry import BackendError, need
from .template import TemplateWriter, norm_entries

DEFAULT_ANTHROPIC_MODEL = "claude-opus-5-5"
EXPANDABLE = ("appearance", "behaviour", "lighting", "camera", "arena", "occlusion", "quality")


def build_request(slots: Dict[str, Any], species: str, description: str, want: List[str], per_slot: int,
                  candidates: Dict[str, list]) -> str:
    current = {s: [e["text"] for e in norm_entries((slots.get("slots") or {}).get(s))] for s in want}
    return (
        "You are helping to write text-to-image prompts for synthetic training images of animals in research "
        "recordings.\n"
        f"Species: {species}\nUser description: {description or '(none)'}\n"
        f"Current slot values (JSON): {json.dumps(current, ensure_ascii=False)}\n"
        f"Candidate terms mined from the literature (may be noisy): {json.dumps(candidates, ensure_ascii=False)}\n\n"
        f"Propose up to {per_slot} NEW values for each slot, in the same style and grammar as the existing values "
        "(short noun phrases; arenas describe only the setting, never the animal; use the same {floor}, {wall}, "
        "{bedding}, {singular} placeholders where the existing values do). Do not repeat existing values. Do not "
        "describe any specific real laboratory or a named product.\n"
        "Answer with one JSON object only, mapping each slot name to a list of strings.")


def parse_answer(text: str, want: List[str]) -> Dict[str, List[str]]:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise BackendError("LLM answer contained no JSON object")
    data = json.loads(m.group(0))
    out: Dict[str, List[str]] = {}
    for s in want:
        vals = data.get(s) or []
        out[s] = [str(v).strip() for v in vals if isinstance(v, str) and 3 <= len(v.strip()) <= 300]
    return out


class LLMWriter(TemplateWriter):
    name = "llm"

    def __init__(self, options: Dict[str, Any] = None, candidates: Dict[str, list] = None, **kw: Any):
        super().__init__(options, **kw)
        self.candidates = candidates or {}

    # ---- providers -----------------------------------------------------------------------
    def _ask_anthropic(self, prompt: str) -> str:
        anthropic = need("anthropic", "llm", "the Anthropic LLM writer")
        cred = self.opt.get("token_credential")
        client = anthropic.Anthropic(api_key=credentials.get(cred)) if cred else anthropic.Anthropic()
        kw = dict(model=self.opt.get("model") or DEFAULT_ANTHROPIC_MODEL,
                  max_tokens=int(self.opt.get("max_tokens", 16000)),
                  messages=[{"role": "user", "content": prompt}])
        if self.opt.get("fallbacks", True):
            # Server-side refusal fallback (on by default; set writer_options.fallbacks: false to turn it off,
            # e.g. on platforms that do not offer it).
            resp = client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kw)
        else:
            resp = client.messages.create(**kw)
        if resp.stop_reason == "refusal":
            raise BackendError("the model declined this request (stop_reason=refusal)")
        return "".join(b.text for b in resp.content if b.type == "text")

    def _ask_openai_compatible(self, prompt: str) -> str:
        import requests
        base = (self.opt.get("base_url") or "").rstrip("/")
        if not base or not self.opt.get("model"):
            raise BackendError("openai_compatible needs writer_options.base_url and writer_options.model")
        headers = {"Content-Type": "application/json"}
        cred = self.opt.get("token_credential")
        if cred:
            headers["Authorization"] = "Bearer " + credentials.get(cred)
        r = requests.post(base + "/chat/completions", headers=headers, timeout=float(self.opt.get("timeout_s", 120)),
                          json={"model": self.opt["model"], "messages": [{"role": "user", "content": prompt}]})
        if r.status_code >= 400:
            raise BackendError(credentials.redact(f"LLM endpoint returned HTTP {r.status_code}"))
        return r.json()["choices"][0]["message"]["content"]

    def ask(self, prompt: str) -> str:
        provider = self.opt.get("provider", "anthropic")
        if provider == "anthropic":
            return self._ask_anthropic(prompt)
        if provider == "openai_compatible":
            return self._ask_openai_compatible(prompt)
        raise BackendError(f"unknown LLM provider {provider!r} (anthropic | openai_compatible)")

    # ---- slot expansion --------------------------------------------------------------------
    def expand_slots(self, slots: Dict[str, Any], species: str, description: str) -> Dict[str, Any]:
        want = [s for s in (self.opt.get("slots") or EXPANDABLE) if s in (slots.get("slots") or {})]
        per_slot = int(self.opt.get("per_slot", 8))
        answer = parse_answer(self.ask(build_request(slots, species, description, want, per_slot, self.candidates)), want)
        out = copy.deepcopy(slots)
        for s, vals in answer.items():
            have = {e["text"].lower() for e in norm_entries(out["slots"][s])}
            for v in vals[:per_slot]:
                if v.lower() not in have:
                    try:
                        v.format(singular="x", plural="x", floor="x", wall="x", bedding="x")
                    except (KeyError, IndexError, ValueError):
                        continue                                       # unknown placeholder: skip the value
                    out["slots"][s].append({"text": v, "weight": float(self.opt.get("new_weight", 1.0)), "origin": "llm"})
                    have.add(v.lower())
        return out
