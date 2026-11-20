"""Deterministic prompt writer: independent slot filling from a species vocabulary.

Method adapted from mw_det_syn/gap_targeted/scripts/prompts_div.py (`item()`): every item draws each
slot independently (arena, appearance per animal, group size, scale, occlusion, behaviour, light,
camera, image-quality word, aspect ratio, look); matched empty scenes; the negative prompt lists
the neighbouring animal counts. Nothing here is species specific: all wording comes from slots.yaml.
"""
from __future__ import annotations

import math
import random
import re
from collections import Counter
from typing import Any, Dict, List, Optional

from ...utils import md5_unit, seed_from, slug

NUM = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight",
       9: "nine", 10: "ten", 11: "eleven", 12: "twelve"}
SUB_SLOTS = ("floor", "wall", "bedding")


def norm_entries(values: List[Any]) -> List[dict]:
    """Slot values may be plain strings/numbers or dicts {id, text, weight, keywords}."""
    out = []
    for v in values or []:
        if isinstance(v, dict):
            e = dict(v)
            if "id" not in e:
                e["id"] = slug(e.get("text", "x"))[:24]
            e.setdefault("text", str(e["id"]))
        else:
            e = {"id": v if isinstance(v, (int, float)) else slug(v)[:24], "text": str(v)}
        e.setdefault("weight", 1.0)
        out.append(e)
    return out


def number_word(n: int) -> str:
    return NUM.get(n, str(n))


class TemplateWriter:
    name = "template"

    def __init__(self, options: Dict[str, Any] = None, **_: Any):
        self.opt = options or {}

    # hook for the LLM writer; the template writer changes nothing
    def expand_slots(self, slots: Dict[str, Any], species: str, description: str) -> Dict[str, Any]:
        return slots

    @staticmethod
    def _pick(rng: random.Random, entries: List[dict], weights: Optional[Dict[str, float]] = None) -> dict:
        w = [float((weights or {}).get(str(e["id"]), e.get("weight", 1.0))) for e in entries]
        if sum(w) <= 0:
            w = [1.0] * len(entries)
        return rng.choices(entries, w)[0]

    def item(self, index: int, slots: Dict[str, Any], species: str, seed: int = 0, empty_fraction: float = 0.05,
             val_fraction: float = 0.05, megapixels: float = 1.25, size_multiple: int = 16,
             weights: Optional[Dict[str, Dict[str, float]]] = None) -> dict:
        rng = random.Random(seed_from(seed, species, index))
        S = {k: norm_entries(v) for k, v in (slots.get("slots") or {}).items()}
        W = weights or {}
        an = slots.get("animal") or {}
        sing = an.get("singular") or species
        plur = an.get("plural") or sing + "s"
        fmt = {"singular": sing, "plural": plur}
        for sub in SUB_SLOTS:
            fmt[sub] = self._pick(rng, S[sub])["text"] if S.get(sub) else ""
        pick = lambda slot: self._pick(rng, S[slot], W.get(slot))        # noqa: E731
        text = lambda e: str(e["text"]).format(**fmt)                     # noqa: E731

        empty = rng.random() < empty_fraction
        arena = pick("arena"); quality = pick("quality"); camera = pick("camera"); light = pick("lighting")
        aspect = float(rng.choice([e["id"] for e in S.get("aspect") or [{"id": 1.0}]]))
        look = str(rng.choice([e["text"] for e in S.get("look") or [{"text": "none"}]]))
        n = 0 if empty else int(pick("group_size")["id"])
        apps = [pick("appearance") for _ in range(n)]
        if n > 3 and rng.random() < 0.5:
            apps = [apps[0]] * n                                          # a group of one type
        scale = pick("scale"); occ = pick("occlusion")
        behs = [pick("behaviour")] if n <= 3 else list({e["id"]: e for e in (pick("behaviour") for _ in range(3))}.values())

        chosen = {"arena": arena["id"], "quality": quality["id"], "camera": camera["id"], "lighting": light["id"],
                  "scale": scale["id"], "occlusion": occ["id"], "behaviour": [b["id"] for b in behs],
                  "appearance": [a["id"] for a in apps], "group_size": n,
                  **{sub: fmt[sub] for sub in SUB_SLOTS if fmt[sub]}}
        common = dict(quality=text(quality), arena=text(arena), camera=text(camera),
                      lighting=text(light)[:1].upper() + text(light)[1:], **fmt)
        if empty:
            prompt = str(slots["empty_template"]).format(**common)
            neg = slots.get("negative_prompt", "") + f", {sing}, {plur}"
        else:
            cnt = Counter(a["id"] for a in apps)
            by_id = {a["id"]: a for a in apps}
            who = " and ".join(
                f"{number_word(k)} " + (text(by_id[i]) if k == 1 else
                                        str(by_id[i].get("plural") or text(by_id[i]).replace(sing, plur, 1)))
                for i, k in cnt.items())
            occ_t = text(occ)
            if n == 1:
                occ_t = occ_t.replace(f"one {sing}", f"the {sing}").replace("whole bodies", "the whole body")
                scale_t = text(scale)
            else:
                scale_t = text(scale)
            beh_t = text(behs[0]) if len(behs) == 1 else ", ".join(text(b) for b in behs[:-1]) + " and " + text(behs[-1])
            prompt = str(slots["template"]).format(
                is_are="is" if n == 1 else "are", count_word=number_word(n), animal=sing if n == 1 else plur,
                who=who, definition=an.get("definition", ""), occlusion=occ_t, scale=scale_t, behaviour=beh_t, **common)
            others = [f"{number_word(k)} {sing if k == 1 else plur}" for k in range(1, 13) if k != n and abs(k - n) <= 3]
            neg = ", ".join([slots.get("negative_prompt", "")] + list(an.get("negatives") or []) + others)
        prompt = re.sub(r"\s+", " ", re.sub(r"(,\s*)+,", ",", prompt)).replace(": ,", ":").strip()
        neg = re.sub(r"\s+", " ", neg).strip(" ,")
        side = math.sqrt(megapixels * 1e6)
        m = max(1, int(size_multiple))
        Wd = max(m, int(round(side * math.sqrt(aspect) / m)) * m)
        Hd = max(m, int(round(side / math.sqrt(aspect) / m)) * m)
        iid = f"{slug(species)[:8]}_{index:06d}"
        u = md5_unit(iid)
        return {"id": iid, "index": index, "cell": str(arena["id"]), "n": n, "appearance": [a["id"] for a in apps],
                "slots": chosen, "look": look, "W": Wd, "H": Hd, "seed": rng.randrange(2 ** 31),
                "prompt": prompt, "negative_prompt": neg, "u": round(u, 6), "split": "val" if u < val_fraction else "train"}


def distinct_report(items: List[dict]) -> dict:
    """Share of distinct prompts (the >= 95 % check from make_div_round.py), overall and without empties."""
    non_empty = [m["prompt"] for m in items if m["n"] > 0]
    allp = [m["prompt"] for m in items]
    f = lambda xs: (len(set(xs)) / len(xs)) if xs else 1.0               # noqa: E731
    per = {}
    for m in items:
        per.setdefault(m["cell"], []).append(m["prompt"])
    return {"items": len(items), "non_empty": len(non_empty), "empties": len(items) - len(non_empty),
            "distinct": len(set(allp)), "distinct_fraction": round(f(allp), 4),
            "distinct_non_empty": len(set(non_empty)), "distinct_fraction_non_empty": round(f(non_empty), 4),
            "per_cell": {c: {"items": len(v), "distinct_fraction": round(f(v), 4)} for c, v in sorted(per.items())}}
