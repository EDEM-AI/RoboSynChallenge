#!/usr/bin/env python3
"""Download the extra object assets from the Hugging Face dataset and verify every sha256.

The meshes are not stored in git. `extras/assets_manifest.json` lists every file with its sha256, source and licence;
this script fetches them from the dataset `liyifreddy/robosyn-extra-assets` (public) and checks the hashes.
Standard library only.

    python3 extras/scripts/fetch_assets.py --list
    python3 extras/scripts/fetch_assets.py --out extras_assets                      # everything
    python3 extras/scripts/fetch_assets.py --out extras_assets --items rack_4x10_main_zc bell_02_77f2ea_84
    python3 extras/scripts/fetch_assets.py --out extras_assets --verify-only         # re-check files already on disk

A file whose hash does not match is deleted and reported; the exit code is 1 if anything failed.
--base-url overrides the download location (a mirror, or file:///path/to/local/copy of the dataset).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MANIFEST = os.path.join(os.path.dirname(HERE), "assets_manifest.json")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_url(man, rel, base_url=None, revision=None):
    rev = revision or man.get("revision", "main")
    base = base_url or f"https://huggingface.co/datasets/{man['repo_id']}/resolve/{rev}"
    return base.rstrip("/") + "/" + urllib.parse.quote(rel)


def download(url, dest, retries=3):
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    tmp = dest + ".part"
    last = None
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "robosyn-extras-fetch/1"})
            with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            os.replace(tmp, dest)
            return
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (k + 1))
    if os.path.exists(tmp):
        os.remove(tmp)
    raise RuntimeError(f"download failed: {url} ({type(last).__name__}: {last})")


def wanted_files(man, items=None, with_docs=True):
    names = items or sorted(man["items"])
    unknown = [n for n in names if n not in man["items"]]
    if unknown:
        raise SystemExit(f"unknown item(s): {', '.join(unknown)} (use --list)")
    out = []
    if with_docs:
        out += [(d["path"], d["sha256"]) for d in man["docs"]]
    for n in names:
        out += [(f["path"], f["sha256"]) for f in man["items"][n]["files"]]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--out", default="extras_assets", help="target directory (default ./extras_assets)")
    ap.add_argument("--items", nargs="*", help="item names (default: all)")
    ap.add_argument("--list", action="store_true", help="list items with licence and exit")
    ap.add_argument("--verify-only", action="store_true", help="do not download, only check files on disk")
    ap.add_argument("--revision", help="dataset revision (default from manifest: main)")
    ap.add_argument("--base-url", help="override download base URL (mirror or file:// path)")
    a = ap.parse_args(argv)
    man = json.load(open(a.manifest, encoding="utf-8"))
    if a.list:
        for n, it in sorted(man["items"].items()):
            print(f"{n:34s} {it['license']:9s} {it['origin']:12s} {it.get('author', '')}")
        return 0
    bad = 0
    todo = wanted_files(man, a.items)
    for i, (rel, want) in enumerate(todo, 1):
        dest = os.path.join(a.out, *rel.split("/"))
        if os.path.exists(dest) and sha256_file(dest) == want:
            print(f"[{i}/{len(todo)}] ok (cached)  {rel}")
            continue
        if a.verify_only:
            print(f"[{i}/{len(todo)}] MISSING/BAD  {rel}")
            bad += 1
            continue
        try:
            download(file_url(man, rel, a.base_url, a.revision), dest)
        except RuntimeError as e:
            print(f"[{i}/{len(todo)}] FAIL {rel}: {e}")
            bad += 1
            continue
        got = sha256_file(dest)
        if got != want:
            os.remove(dest)
            print(f"[{i}/{len(todo)}] SHA MISMATCH {rel}: expected {want[:16]}..., got {got[:16]}...; file removed")
            bad += 1
        else:
            print(f"[{i}/{len(todo)}] ok  {rel}")
    print(f"{len(todo) - bad}/{len(todo)} files verified" + (f", {bad} FAILED" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
