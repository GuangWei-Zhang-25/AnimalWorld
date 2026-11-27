"""The vendored MouseWorld-Track files are the ones named in mwtrack/PROVENANCE.md ("Files as shipped")."""
import hashlib
import re
from pathlib import Path

M = Path(__file__).resolve().parents[1] / "src" / "animalworld" / "zeroshot" / "mwtrack"


def test_shipped_tracker_files_match_provenance():
    text = (M / "PROVENANCE.md").read_text().split("## Files as shipped")[1]
    rows = re.findall(r"^\| (\S+\.py) \| ([0-9a-f]{64}) \|$", text, flags=re.M)
    assert len(rows) == 9
    for name, want in rows:
        assert hashlib.sha256((M / name).read_bytes()).hexdigest() == want, name
