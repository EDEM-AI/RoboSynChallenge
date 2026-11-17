"""Import step for ready-made marker meshes (the ONLY stage that touches trimesh).

Normalises a base mesh into the frame the generator expects and writes it as
plain numpy arrays, together with the provenance needed by the registry:

    <key>.v.npy    float64 (N,3)   PCA-aligned, +x = long axis, tip towards +x
    <key>.f.npy    int64   (M,3)   triangle indices
    <key>.json     {"key","source_file","source_sha256","name","license","author",
                    "uid","url","faces"}

Everything downstream (marker_ready.py) reads these arrays only - numpy, no
trimesh, no simulator - so the generator side satisfies the "no sim import" and
"byte-identical output" rules.

    python3 marker_import.py            # build whatever is missing / stale
    python3 marker_import.py --force    # rebuild everything
"""

import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.expanduser("~/objaverse_marker")
GLB = os.path.join(CACHE, "glbs")
NORM = os.path.join(CACHE, "normalised")

# base meshes; licence/author verified 2026-11-07 through the Sketchfab API
BASES = {
    "poly_permanent": {"uid": "b671c8a542b245f3b0058f60457ee77f",
                       "name": "Permanent Marker from Poly by Google",
                       "license": "CC-BY-4.0", "author": "ie-niels"},
    "highlighter": {"uid": "600157f4ffe440f99eb56b08168e1a41", "name": "Highlighter",
                    "license": "CC-BY-4.0", "author": "cirby2180"},
    "stabilo_boss": {"uid": "25481a65cad54e2a956394ff2b2765cd", "name": "Stabilo Boss Highlighter",
                     "license": "CC-BY-4.0", "author": "lukematakitoga"},
    "marker_cap": {"uid": "163a461f22f2479fb88fe94b917c37e3", "name": "Marker (body + cap)",
                   "license": "CC-BY-4.0", "author": "sleepingburrito"},
    "sharpie_thin": {"uid": "7fc87671c0844fd9874658a9f767a74f", "name": "Sharpie Marker (thin)",
                     "license": "CC-BY-4.0", "author": "adim09"},
    "official_pen": {"file": os.path.join(os.environ.get("ROBOSYN_REPO", "RoboSynChallenge"), "assets", "Pen", "the_pen.obj"),
                     "name": "官方 assets/Pen/the_pen.obj",
                     "license": "Apache-2.0 (official RoboSynChallenge asset)",
                     "author": "RoboSynChallenge"},
}


def source_path(key):
    b = BASES[key]
    if "file" in b:
        return b["file"]
    return os.path.join(GLB, b["uid"] + ".glb")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pca_frame(v):
    c = v.mean(axis=0)
    _, _, vt = np.linalg.svd(v - c, full_matrices=False)
    axis = vt[0]
    e1 = np.cross(axis, [0.0, 0.0, 1.0])
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.cross(axis, [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    return c, np.stack([axis, e1, e2])


def normalise(key):
    import trimesh

    path = source_path(key)
    m = trimesh.load(path, force="mesh", process=False)
    if len(m.faces) == 0:
        raise ValueError(f"{key}: empty mesh")
    v = np.asarray(m.vertices, dtype=float)
    f = np.asarray(m.faces, dtype=np.int64)
    c, R = pca_frame(v)
    v = (v - c) @ R.T
    # tip towards +x: the end with the smaller median radius is the tip
    pos = v[:, 0] > 0
    r_pos = np.median(np.linalg.norm(v[pos][:, 1:], axis=1)) if pos.any() else 0.0
    r_neg = np.median(np.linalg.norm(v[~pos][:, 1:], axis=1)) if (~pos).any() else 0.0
    if r_pos < r_neg:
        v = v * np.array([-1.0, 1.0, 1.0])
        f = f[:, ::-1]
    lo, hi = v.min(0), v.max(0)
    v = v - np.array([(lo[0] + hi[0]) / 2.0, 0.0, 0.0])
    return v, f


def main(argv):
    force = "--force" in argv
    os.makedirs(NORM, exist_ok=True)
    made = 0
    for key, b in BASES.items():
        vp = os.path.join(NORM, key + ".v.npy")
        fp = os.path.join(NORM, key + ".f.npy")
        jp = os.path.join(NORM, key + ".json")
        if not force and all(os.path.exists(p) for p in (vp, fp, jp)):
            continue
        v, f = normalise(key)
        np.save(vp, v)
        np.save(fp, f)
        meta = {
            "key": key,
            "name": b["name"],
            "license": b["license"],
            "author": b["author"],
            "source_file": source_path(key),
            "source_sha256": sha256_file(source_path(key)),
            "faces": int(len(f)),
            "vertices": int(len(v)),
        }
        if "uid" in b:
            meta["uid"] = b["uid"]
            meta["url"] = f"https://sketchfab.com/3d-models/{b['uid']}"
        with open(jp, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=1, sort_keys=True)
        made += 1
        print(f"{key:16s} faces={len(f):6d} src_sha={meta['source_sha256'][:12]} -> {NORM}")
    if made == 0:
        print("nothing to do (use --force to rebuild)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
