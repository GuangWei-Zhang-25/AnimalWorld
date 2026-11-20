"""Offline literature backend: reads abstracts from a local CSV or JSONL file (no network)."""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import List

from .base import LiteratureBackend

_ALIASES = {
    "id": ("id", "pmid", "record_id", "record_index"),
    "title": ("title", "literature_name"),
    "abstract": ("abstract", "abstracttext", "description_raw_text", "text"),
    "year": ("year", "publication_year", "pubyear"),
    "journal": ("journal", "journaltitle"),
    "doi": ("doi",),
}


def normalise(row: dict, k: int, source: str) -> dict:
    low = {str(a).lower(): b for a, b in row.items()}
    out = {"source": low.get("source") or source}
    for key, names in _ALIASES.items():
        out[key] = next((low[n] for n in names if low.get(n) not in (None, "")), "")
    out["id"] = str(out["id"] or f"{source}:{k}")
    out["year"] = str(out["year"])[:4]
    return out


class OfflineLiterature(LiteratureBackend):
    name = "offline"

    def search(self, query: str, limit: int) -> List[dict]:
        src = self.opt.get("source")
        if not src:
            raise FileNotFoundError("literature.source is not set: give a CSV or JSONL file of abstracts "
                                    "(columns: title, abstract, optional id/year/journal/doi)")
        src = Path(src)
        rows = []
        if src.suffix.lower() in (".jsonl", ".ndjson"):
            with open(src, encoding="utf-8") as f:
                rows = [json.loads(l) for l in f if l.strip()]
        elif src.suffix.lower() == ".json":
            rows = json.loads(src.read_text(encoding="utf-8"))
        else:
            with open(src, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f, delimiter="\t" if src.suffix.lower() == ".tsv" else ","))
        recs = [normalise(r, k, "offline") for k, r in enumerate(rows)]
        if self.opt.get("filter_query") and query:
            words = [w for w in re.findall(r"[a-z]{3,}", query.lower())]
            recs = [r for r in recs if any(w in (r["title"] + " " + r["abstract"]).lower() for w in words)]
        return recs[:limit] if limit else recs
