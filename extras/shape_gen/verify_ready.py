"""Verification helper for the ready-made marker set (for the code reviewer).

Does three things in one run:
  1. rebuilds every variant into a temporary directory and compares file sha256
     with ``out/marker_ready`` -> proves byte-identical output;
  2. prints the per-variant provenance table (base / licence / author / source
     sha256), so the licences can be checked item by item;
  3. cross-checks ``out/marker_ready/index.json`` against the files on disk.

    python3 verify_ready.py
"""

import hashlib
import json
import os
import shutil
import sys
import tempfile

import marker_ready as MR


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dir_shas(d):
    out = {}
    for root, _dirs, files in os.walk(d):
        for fn in files:
            p = os.path.join(root, fn)
            out[os.path.relpath(p, d)] = sha(p)
    return out


def main():
    print("== 1. determinism (rebuild into a temp dir, compare shas) ==")
    tmp = tempfile.mkdtemp(prefix="mr_verify_")
    try:
        for idx, (name, base, d, L) in enumerate(MR.VARIANTS, start=1):
            rec = MR.write_variant(idx, name, base, d, L, tmp)
            rec.pop("mesh", None)
        a = dir_shas(MR.OUT)
        b = dir_shas(tmp)
        only_a = sorted(k for k in set(a) - set(b) if os.path.basename(k) != "index.json")
        diff = sorted(k for k in set(a) & set(b) if a[k] != b[k])
        print(f"   files compared: {len(set(a) & set(b))}  identical: {len(set(a) & set(b)) - len(diff)}")
        print(f"   only in out/: {only_a if only_a else 'none'}")
        print(f"   differing  : {diff if diff else 'none'}")
        ok = not only_a and not diff
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n== 2. provenance / licence per variant ==")
    with open(os.path.join(MR.OUT, "index.json"), encoding="utf-8") as fh:
        idx = json.load(fh)
    for r in idx["rows"]:
        print(f"  {r['index']:02d} {r['variant']:26s} Ø{r['d']*1e3:5.1f} mm  "
              f"{r['license']:34s} {r['author']:16s} src {r['base_sha256'][:12]}")
    licences = sorted({r["license"] for r in idx["rows"]})
    print("   licences:", licences)
    bad = [l for l in licences if "NC" in l or "ND" in l]
    print("   non-free licences:", bad if bad else "none")

    print("\n== 3. index vs disk ==")
    bad = 0
    for r in idx["rows"]:
        for fn, want in r["files"].items():
            p = os.path.join(MR.OUT, r["variant"], fn)
            got = sha(p) if os.path.exists(p) else "MISSING"
            if got != want:
                bad += 1
                print(f"   MISMATCH {r['variant']}/{fn}")
    print(f"   checked {sum(len(r['files']) for r in idx['rows'])} files, mismatches: {bad}")

    print("\nRESULT:", "PASS" if (ok and bad == 0 and not [l for l in licences if "NC" in l]) else "FAIL")
    return 0 if ok and bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
