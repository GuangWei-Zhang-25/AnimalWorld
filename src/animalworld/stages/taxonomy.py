"""Stage 1b: keyword / taxonomy extraction from the abstracts.

For every slot value that carries `keywords` (regular expressions) the stage counts the records
whose title + abstract match, exactly like the 55 curated keyword rules used for the MouseWorld
literature map (behavior_terms.csv: canonical name, category, regex, case-insensitive). The counts
turn into sampling weights for the prompt writer, and frequent phrases that no rule covers are
listed as candidates for a human (or the optional LLM writer) to add to slots.yaml.
"""
from __future__ import annotations

import math
import re
from collections import Counter

from ..backends.writers.template import norm_entries
from ..config import Project
from ..utils import read_jsonl, write_json, atomic_write_text

DEFAULT_CANDIDATES = {
    "arena": [r"\b((?:[a-z]+[- ]){1,2}(?:test|maze|task|assay|arena|chamber|paradigm|pen|tank|enclosure))\b"],
}
# Leading words that are stripped from a candidate ("the open field test" -> "open field test").
STOP = {"the", "a", "an", "this", "that", "these", "those", "each", "every", "our", "their", "its", "his", "her", "and", "or"}
# A candidate that still contains one of these is not a name ("was to test", "on the pen", "pigs per pen", "one pen"):
# it is dropped. Extend the list with taxonomy.candidate_stop_words in project.yaml.
NOT_A_NAME = {
    "is", "are", "was", "were", "be", "been", "being", "has", "have", "had", "do", "does", "did", "can", "could", "may",
    "might", "will", "would", "should", "must", "used", "using", "use", "performed", "conducted", "tested", "placed",
    "in", "on", "at", "of", "for", "to", "from", "by", "with", "without", "per", "during", "while", "when", "where",
    "after", "before", "into", "onto", "under", "over", "between", "within", "across", "through", "than", "as", "but",
    "if", "then", "that", "which", "who", "whose", "not", "no", "also", "both", "all", "any", "some", "same", "other",
    "another", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "first", "second", "third",
    "we", "they", "it", "he", "she", "i", "you",
}


def _clean_candidate(raw: str, drop: set):
    """A matched phrase -> the candidate to count, or None. Words are lower-cased unless they carry a capital after
    their first letter (EPM, C57BL/6J, BALB/c), so "The Zebra Maze" and "the zebra maze" are one candidate."""
    words = raw.split()
    if len(words) > 1 and all(w.isupper() for w in words if any(ch.isalpha() for ch in w)):
        words = [w.lower() for w in words]                       # a phrase written in capitals (a heading)
    words = [w if any(ch.isupper() for ch in w[1:]) else w.lower() for w in words]
    multi = len(words) > 1
    while words and words[0].lower() in STOP:
        words = words[1:]
    if len(words) < (2 if multi else 1):
        return None
    if any(p.lower() in drop for w in words for p in w.split("-")):
        return None
    ph = " ".join(words)
    return ph if len(ph) >= 3 else None


def run(project: Project, dry_run: bool = False, **_) -> dict:
    cfg = project.cfg["taxonomy"]
    src = project.path("literature", "records.jsonl")
    out = project.path("taxonomy", "taxonomy.json")
    plan = {"stage": "taxonomy", "input": str(src), "output": str(out)}
    if dry_run:
        return dict(plan, mode="dry-run")
    recs = read_jsonl(src)
    texts = [(r.get("title", "") + ". " + r.get("abstract", "")) for r in recs]
    slots = project.slots()
    lw = float(cfg.get("literature_weight", 1.0))
    result, weights, n_rules = {}, {}, 0
    for slot, values in (slots.get("slots") or {}).items():
        entries = norm_entries(values)
        if not any(e.get("keywords") for e in entries):
            continue
        rows = []
        for e in entries:
            hits = 0
            if e.get("keywords"):
                n_rules += 1
                rx = re.compile("|".join(f"(?:{k})" for k in e["keywords"]), re.I)
                hits = sum(1 for t in texts if rx.search(t))
            rows.append({"id": e["id"], "text": e["text"], "base_weight": float(e["weight"]), "hits": hits,
                         "has_rule": bool(e.get("keywords"))})
        mx = max((r["hits"] for r in rows), default=0)
        for r in rows:
            boost = (math.log1p(r["hits"]) / math.log1p(mx)) if mx and r["hits"] >= int(cfg.get("min_hits", 1)) else 0.0
            r["weight"] = round(r["base_weight"] * (1.0 + lw * boost), 4)
        result[slot] = rows
        weights[slot] = {str(r["id"]): r["weight"] for r in rows}
    # Candidate phrases not covered by any rule. The patterns are configurable (slots.yaml: candidate_patterns,
    # one list of regular expressions per slot, the phrase in group 1); they are matched without regard to case
    # (wrap a pattern in (?-i:...) to keep it case-sensitive, as the strain-name patterns of the mouse example do).
    # A phrase is "covered" when the keyword rule of ANY slot matches it, not only a rule of its own slot.
    pats = dict(DEFAULT_CANDIDATES); pats.update(slots.get("candidate_patterns") or {})
    drop = NOT_A_NAME | {str(w).lower() for w in (cfg.get("candidate_stop_words") or [])}
    known = [re.compile("|".join(f"(?:{k})" for k in e["keywords"]), re.I)
             for values in (slots.get("slots") or {}).values() for e in norm_entries(values) if e.get("keywords")]
    cands = {}
    for slot, plist in pats.items():
        c = Counter()
        for t in texts:
            found = set()
            for p in plist:
                for m in re.finditer(p, t, re.I):
                    ph = _clean_candidate(m.group(1).strip(), drop)
                    if ph and not any(k.search(ph) for k in known):
                        found.add(ph)
            c.update(found)
        cands[slot] = [{"phrase": p, "records": n} for p, n in c.most_common(int(cfg.get("max_candidates", 30))) if n >= 2]
    tax = {"species": project.species, "records": len(recs), "rules": n_rules, "slots": result, "weights": weights,
           "candidates": cands}
    write_json(out, tax)
    lines = [f"# Taxonomy for {project.species}", "", f"{len(recs)} literature records, {n_rules} keyword rules.", ""]
    for slot, rows in result.items():
        lines += [f"## {slot}", "", "| value | records matching | base weight | weight used |", "|---|---|---|---|"]
        lines += [f"| {r['text'][:70]} | {r['hits'] if r['has_rule'] else 'no rule'} | {r['base_weight']:g} | {r['weight']:g} |" for r in rows]
        lines.append("")
    for slot, rows in cands.items():
        if rows:
            lines += [f"## Candidates not covered by a rule: {slot}", ""] + [f"- {r['phrase']} ({r['records']} records)" for r in rows] + [""]
    atomic_write_text(project.path("taxonomy", "report.md"), "\n".join(lines))
    return dict(plan, mode="executed", records=len(recs), rules=n_rules,
                slots_with_evidence=sorted(result), candidates={k: len(v) for k, v in cands.items()})
