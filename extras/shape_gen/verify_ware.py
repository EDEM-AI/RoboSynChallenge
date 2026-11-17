"""One-shot verification for the library-import shapes (for the code reviewer).

  1. rebuild every variant into a temp dir and compare file sha256 -> byte-identical
  2. print the per-variant provenance / licence table (15-char sha prefixes)
  3. cross-check out/libimport/<cls>/index.json against the files on disk

    python3 verify_ware.py
"""

import hashlib
import json
import os
import shutil
import sys
import tempfile

import ware_ready as WR

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = WR.OUT


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
    with open(os.path.join(HERE, "selection.json"), encoding="utf-8") as fh:
        sel = json.load(fh)
    print("== 1. determinism (rebuild into a temp dir, compare shas) ==")
    tmp = tempfile.mkdtemp(prefix="wr_verify_")
    ok = True
    try:
        for row in sel:
            WR.write_variant(row["cls"], row["index"], row["name"], row["key"],
                             row["target_m"], os.path.join(tmp, row["cls"]))
        total = same = 0
        diff = []
        for cls in sorted({r["cls"] for r in sel}):
            a = dir_shas(os.path.join(OUT, cls))
            b = dir_shas(os.path.join(tmp, cls))
            keys = set(a) & set(b)
            total += len(keys)
            same += sum(1 for k in keys if a[k] == b[k])
            diff += [f"{cls}/{k}" for k in keys if a[k] != b[k]]
            only = sorted(k for k in set(a) - set(b) if os.path.basename(k) != "index.json")
            if only:
                print(f"   only on disk ({cls}): {only}")
                ok = False
        print(f"   files compared: {total}  identical: {same}")
        print(f"   differing  : {diff if diff else 'none'}")
        ok = ok and not diff
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n== 2. provenance / licence per variant ==")
    licences = set()
    rows = 0
    for cls in sorted({r["cls"] for r in sel}):
        with open(os.path.join(OUT, cls, "index.json"), encoding="utf-8") as fh:
            idx = json.load(fh)
        for r in idx["rows"]:
            rows += 1
            licences.add(r["license"])
            print(f"  {cls:6s} {r['variant']:24s} {r['target_m']*1e3:6.1f} mm "
                  f"{r['mass_kg']*1e3:7.1f} g  {r['license']:10s} {r['author'][:16]:17s} "
                  f"src {r['base_sha256'][:15]}")
    bad = [l for l in licences if ("NC" in l.upper() or "-SA" in l.upper() or "ND" in l.upper())]
    print("   licences:", sorted(licences))
    print("   non-free licences:", bad if bad else "none")

    print("\n== 3. index vs disk ==")
    mism = 0
    checked = 0
    for cls in sorted({r["cls"] for r in sel}):
        with open(os.path.join(OUT, cls, "index.json"), encoding="utf-8") as fh:
            idx = json.load(fh)
        for r in idx["rows"]:
            for fn, want in r["files"].items():
                p = os.path.join(OUT, cls, r["variant"], fn)
                checked += 1
                got = sha(p) if os.path.exists(p) else "MISSING"
                if got != want:
                    mism += 1
                    print(f"   MISMATCH {cls}/{r['variant']}/{fn}")
    print(f"   checked {checked} files, mismatches: {mism}")

    good = ok and mism == 0 and not bad
    print("\nRESULT:", "PASS" if good else "FAIL", f"({rows} variants)")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
