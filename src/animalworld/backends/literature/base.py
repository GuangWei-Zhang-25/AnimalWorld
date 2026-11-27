"""Literature backends: ``search(query, limit) -> list of records``.

A record is a dict with keys: id, source, title, abstract, year, journal, doi, authors (list of
"Surname Initials"), volume, issue, pages. The last four were added in 0.2.1: records written by an
earlier version lack them, so read them with ``record.get(...)``.
Network backends cache every page under <project>/literature/cache so re-runs cost no requests; the
cache holds the raw answers, so `animalworld literature --force` re-parses old pages with the new fields.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ... import __version__

# Generic client name. No personal or lab details are ever sent. An e-mail address is sent only if
# the user sets literature.contact_email in project.yaml.
USER_AGENT = f"animalworld/{__version__} (python-requests; literature-search)"
TOOL_NAME = "animalworld"


class LiteratureBackend:
    name = "base"
    needs_network = False

    def __init__(self, options: Dict[str, Any], cache_dir: Optional[Path] = None, **_: Any):
        self.opt = options
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.requests_made = 0
        self._last = 0.0

    def default_query(self, species: str, synonyms: List[str], description: str) -> str:
        return species

    def search(self, query: str, limit: int) -> List[dict]:
        raise NotImplementedError

    # ---- helpers for network backends ------------------------------------------------------
    def _cache_file(self, key: str) -> Optional[Path]:
        if not self.cache_dir:
            return None
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir / f"{self.name}_{hashlib.sha1(key.encode()).hexdigest()[:16]}.json"

    def _cached(self, key: str):
        f = self._cache_file(key)
        if f and f.exists():
            return json.loads(f.read_text(encoding="utf-8"))
        return None

    def _store(self, key: str, value) -> None:
        f = self._cache_file(key)
        if f:
            f.write_text(json.dumps(value), encoding="utf-8")

    def _http(self, method: str, url: str, **kw):
        """Polite request: fixed pause between calls, retries with back-off on 429/5xx."""
        import requests
        pause = float(self.opt.get("rate_limit_s", 0.5))
        headers = {"User-Agent": USER_AGENT, "Accept": "application/json, text/xml"}
        for attempt in range(5):
            wait = pause - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            self.requests_made += 1
            try:
                r = requests.request(method, url, headers=headers, timeout=float(self.opt.get("timeout_s", 30)), **kw)
            except requests.RequestException:
                if attempt == 4:
                    raise
                time.sleep(min(30, 2 ** attempt + 0.5))
                continue
            if r.status_code in (429, 500, 502, 503, 504) and attempt < 4:
                time.sleep(min(30, 2 ** attempt + 0.5))
                continue
            r.raise_for_status()
            return r
        raise RuntimeError("unreachable")


def synonyms_clause(terms: List[str]) -> str:
    return " OR ".join('"%s"' % t for t in terms if t)
