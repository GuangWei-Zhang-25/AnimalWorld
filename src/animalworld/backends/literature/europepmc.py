"""Europe PMC REST search (public, no key). https://europepmc.org/RestfulWebService

Request shape adapted from SoraProject/literature search/mine_pubmed.py (cursorMark paging,
resultType=core so the abstract is returned). See docs/PROVENANCE.md.
"""
from __future__ import annotations

from typing import List

from .base import LiteratureBackend, synonyms_clause

URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


class EuropePMC(LiteratureBackend):
    name = "europepmc"
    needs_network = True

    def default_query(self, species, synonyms, description):
        who = synonyms_clause([species] + list(synonyms))
        return (f"({who}) AND (behavior OR behaviour) AND (video OR tracking OR \"pose estimation\" OR "
                f"\"home cage\" OR arena) AND (SRC:MED OR SRC:PMC OR SRC:PPR) AND LANG:eng")

    def search(self, query: str, limit: int) -> List[dict]:
        out: List[dict] = []
        cursor = "*"
        page = int(min(max(1, limit), int(self.opt.get("page_size", 100))))
        while len(out) < limit:
            params = {"query": query, "format": "json", "pageSize": page, "cursorMark": cursor, "resultType": "core"}
            if self.opt.get("contact_email"):
                params["email"] = self.opt["contact_email"]
            key = f"{query}|{page}|{cursor}"
            data = self._cached(key)
            if data is None:
                data = self._http("GET", URL, params=params).json()
                self._store(key, data)
            results = (data.get("resultList") or {}).get("result") or []
            for r in results:
                out.append({
                    "id": f"{r.get('source', 'EPMC')}:{r.get('id', '')}", "source": "europepmc",
                    "title": r.get("title") or "", "abstract": r.get("abstractText") or "",
                    "year": str(r.get("pubYear") or ""),
                    "journal": ((r.get("journalInfo") or {}).get("journal") or {}).get("title") or "",
                    "doi": r.get("doi") or "",
                })
            nxt = data.get("nextCursorMark")
            if not results or not nxt or nxt == cursor:
                break
            cursor = nxt
        return out[:limit]
