"""PubMed through NCBI E-utilities (public; no key needed at <= 3 requests per second).

esearch (ids) + efetch (XML with abstracts), as in SoraProject/literature search/
mine_mouse_behavior_pubmed.py and Abstract_UMAP_analysis/scripts/02_fetch_abstracts.py:
structured abstracts are joined as "Label: text". See docs/PROVENANCE.md.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import List

from .base import TOOL_NAME, LiteratureBackend

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"


def parse_efetch(xml_text: str) -> List[dict]:
    out = []
    root = ET.fromstring(xml_text)
    for art in root.iter("PubmedArticle"):
        pmid = (art.findtext(".//MedlineCitation/PMID") or "").strip()
        title = "".join(art.find(".//ArticleTitle").itertext()).strip() if art.find(".//ArticleTitle") is not None else ""
        parts = []
        for a in art.iter("AbstractText"):
            txt = "".join(a.itertext()).strip()
            if txt:
                parts.append(f"{a.get('Label')}: {txt}" if a.get("Label") else txt)
        year = art.findtext(".//JournalIssue/PubDate/Year") or (art.findtext(".//JournalIssue/PubDate/MedlineDate") or "")[:4]
        doi = ""
        for i in art.iter("ArticleId"):
            if i.get("IdType") == "doi":
                doi = (i.text or "").strip()
        out.append({"id": f"PMID:{pmid}", "source": "pubmed", "title": title, "abstract": " ".join(parts),
                    "year": year or "", "journal": art.findtext(".//Journal/Title") or "", "doi": doi})
    return out


class PubMed(LiteratureBackend):
    name = "pubmed"
    needs_network = True

    def default_query(self, species, synonyms, description):
        who = " OR ".join(f"{t}[Title/Abstract]" for t in [species] + list(synonyms))
        return (f"(({who}) AND (behavior*[Title/Abstract] OR behaviour*[Title/Abstract]) AND "
                f"(video[Title/Abstract] OR tracking[Title/Abstract] OR \"pose estimation\"[Title/Abstract]))")

    def _common(self) -> dict:
        p = {"tool": TOOL_NAME}
        if self.opt.get("contact_email"):
            p["email"] = self.opt["contact_email"]
        return p

    def search(self, query: str, limit: int) -> List[dict]:
        key = f"esearch|{query}|{limit}"
        ids = self._cached(key)
        if ids is None:
            params = dict(self._common(), db="pubmed", term=query, retmax=int(limit), retstart=0,
                          retmode="json", sort="relevance")
            ids = self._http("GET", BASE + "esearch.fcgi", params=params).json()["esearchresult"]["idlist"]
            self._store(key, ids)
        out: List[dict] = []
        batch = int(self.opt.get("batch_size", 200))
        for i in range(0, len(ids), batch):
            chunk = ids[i:i + batch]
            k2 = "efetch|" + ",".join(chunk)
            xml_text = self._cached(k2)
            if xml_text is None:
                data = dict(self._common(), db="pubmed", id=",".join(chunk), retmode="xml", rettype="abstract")
                xml_text = self._http("POST", BASE + "efetch.fcgi", data=data).text
                self._store(k2, xml_text)
            out += parse_efetch(xml_text)
        return out[:limit]
