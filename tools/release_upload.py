#!/usr/bin/env python3
"""Create a GitHub release by tag and upload its files, through the GitHub REST API (standard library only).

    python tools/release_upload.py --dry-run weights      # list what would be uploaded: name, size, SHA-256. Contacts nothing.
    python tools/release_upload.py --dry-run demo
    GITHUB_TOKEN=... python tools/release_upload.py weights     # tag weights-v1, files of weights_release/
    GITHUB_TOKEN=... python tools/release_upload.py demo        # tag demo-v1,    files of demo_release/

The token is read from the environment variable GITHUB_TOKEN only (a fine-grained token with "Contents: read and write"
on the repository is enough). It is never printed, never written to a file and never put into a URL. Files already
present in the release are skipped, so an interrupted upload can be run again. This script is not part of the
installed package. It has only been run with --dry-run: the upload itself was never exercised against GitHub.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "GuangWei-Zhang-25/AnimalWorld"
API = "https://api.github.com"
BUNDLES = {
    "weights": dict(tag="weights-v1", title="MouseWorld weights v1", folder="weights_release", notes="RELEASE_NOTES.md",
                    files=["mouseworld-det.ckpt", "mouseworld-pose.pt", "mouseworld-det-lite.onnx", "mouseworld-pose-lite.onnx",
                           "mouseworld-track-det-a.ckpt", "mouseworld-track-det-b.ckpt", "manifest.json", "SHA256SUMS", "README.md",
                           "NOTICE.md", "LICENSE-AGPL-3.0.txt", "LICENSE-Apache-2.0.txt"]),
    "demo": dict(tag="demo-v1", title="Animal World demo clips v1", folder="demo_release", notes="RELEASE_NOTES.md",
                 files=["mouse_x1_cage.mp4", "mouse_x2_cage.mp4", "mouse_x2_arena_a.mp4", "mouse_x2_arena_b.mp4", "mouse_x4_crossing.mp4", "mouse_x5_huddle.mp4", "manifest.json",
                        "SHA256SUMS", "README.md", "NOTICE.md", "expected.zip"]),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_sums(folder: Path, files) -> list:
    """Problems found when the files are compared with the folder's SHA256SUMS (files not listed there are not compared)."""
    listed = {}
    sums = folder / "SHA256SUMS"
    if sums.is_file():
        for line in sums.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 2:
                listed[parts[1].lstrip("*")] = parts[0]
    return [f"{name}: SHA-256 differs from SHA256SUMS" for name, digest in files if name in listed and listed[name] != digest]


def api(method: str, url: str, token: str, data: bytes = None, content_type: str = "application/json"):
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", "Bearer " + token)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    req.add_header("User-Agent", "animalworld-release-upload")
    if data is not None:
        req.add_header("Content-Type", content_type)
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            body = r.read()
            return r.status, (json.loads(body) if body else {})
    except urllib.error.HTTPError as e:
        try:
            msg = json.loads(e.read()).get("message", "")
        except Exception:
            msg = ""
        return e.code, {"message": msg}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Create a GitHub release by tag and upload its files (token from GITHUB_TOKEN only).")
    ap.add_argument("bundle", choices=sorted(BUNDLES))
    ap.add_argument("--repo", default=REPO, help=f"owner/name (default: {REPO})")
    ap.add_argument("--dry-run", action="store_true", help="list the files with sizes and SHA-256; contact nothing")
    ap.add_argument("--draft", action="store_true", help="create the release as a draft")
    a = ap.parse_args(argv)
    b = BUNDLES[a.bundle]
    folder = ROOT / b["folder"]
    missing = [n for n in b["files"] + [b["notes"]] if not (folder / n).is_file()]
    if missing:
        print(f"missing in {folder}: {', '.join(missing)}", file=sys.stderr)
        return 2
    rows = [(n, (folder / n).stat().st_size, sha256(folder / n)) for n in b["files"]]
    print(f"release {b['tag']} of {a.repo}: \"{b['title']}\", {len(rows)} files, {sum(r[1] for r in rows) / 1e6:.1f} MB")
    for n, size, digest in rows:
        print(f"  {size:>12,d}  {digest}  {n}")
    bad = check_sums(folder, [(n, d) for n, _, d in rows])
    if bad:
        print("NOT uploading:\n  " + "\n  ".join(bad), file=sys.stderr)
        return 2
    if a.dry_run:
        print("dry run: nothing was contacted, nothing was uploaded")
        return 0
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        print("GITHUB_TOKEN is not set. Export a token with write access to the repository's contents and run again.", file=sys.stderr)
        return 2
    st, rel = api("GET", f"{API}/repos/{a.repo}/releases/tags/{urllib.parse.quote(b['tag'])}", token)
    if st == 404:
        body = json.dumps({"tag_name": b["tag"], "name": b["title"], "body": (folder / b["notes"]).read_text(encoding="utf-8"),
                           "draft": bool(a.draft), "prerelease": False}).encode()
        st, rel = api("POST", f"{API}/repos/{a.repo}/releases", token, body)
        if st != 201:
            print(f"could not create the release (HTTP {st}): {rel.get('message', '')}", file=sys.stderr)
            return 1
        print("created the release")
    elif st != 200:
        print(f"could not read the release (HTTP {st}): {rel.get('message', '')}", file=sys.stderr)
        return 1
    have = {x["name"]: x.get("size") for x in rel.get("assets", [])}
    upload = rel["upload_url"].split("{")[0]
    for n, size, _ in rows:
        if have.get(n) == size:
            print(f"  already there: {n}")
            continue
        if n in have:
            print(f"  {n} is in the release with another size; delete it there first", file=sys.stderr)
            return 1
        st, out = api("POST", upload + "?name=" + urllib.parse.quote(n), token, (folder / n).read_bytes(), "application/octet-stream")
        if st != 201:
            print(f"  upload of {n} failed (HTTP {st}): {out.get('message', '')}", file=sys.stderr)
            return 1
        print(f"  uploaded {n}")
    print(f"done: https://github.com/{a.repo}/releases/tag/{b['tag']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
