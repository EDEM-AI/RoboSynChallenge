"""Import step for library shapes: bell / plate / fork / spoon.

Same split as marker_import.py -> marker_ready.py:

    ware_import.py   (this file, the ONLY place that touches trimesh)
        candidates from the Objaverse LVIS classes bell / plate / fork / spoon
        (+ YCB real scans for fork / spoon / plate)
        -> licence filter (CC-BY / CC0 only, provenance from the Objaverse
           metadata dump; Sketchfab's API is rate limited)
        -> download the mesh
        -> measure (PCA frame, extents, end radii, connected components)
        -> normalise into the class frame, write plain numpy:
             <key>.v.npy   float64 (N,3)
             <key>.f.npy   int64   (M,3)
             <key>.json    provenance + measurement
    ware_ready.py   (numpy only) -> variants, annotations, variant.yaml, sheets

Frames written here (ware_ready scales/centres on top of them):
    bell / plate : +z up, xy centred, z = 0 at the lowest point of the mesh
    fork / spoon : long axis +x with the head towards +x, face up +z, geometric
                   centre of the bbox at the origin
    (sign conventions: bell wide end down, plate cavity up, utensil wide end = head)

    python3 ware_import.py list                 # measure the pool, print it
    python3 ware_import.py fetch                # download the pool only
    python3 ware_import.py normalise            # build the .npy cache
"""

import gzip
import hashlib
import json
import os
import sys
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HOME = os.path.expanduser("~/objaverse_ware")
META = os.path.join(HOME, "meta_selected.json")
GLB = os.path.expanduser("~/objaverse_marker/glbs")
NORM = os.path.join(HOME, "normalised")
PATHS = os.path.expanduser("~/objaverse_marker/object-paths.json.gz")
YCB_DIR = os.path.join(HOME, "ycb")

# YCB real scans (CC BY 4.0): true scale, real metal / ceramic.
YCB = {
    "ycb_029_plate": ("029_plate", "google_16k"),
    "ycb_030_fork": ("030_fork", "google_16k"),
    "ycb_031_spoon": ("031_spoon", "google_16k"),
}
YCB_URL = "https://ycb-benchmarks.s3.amazonaws.com/data/%s/%s/textured.obj"
YCB_TEX = "https://ycb-benchmarks.s3.amazonaws.com/data/%s/%s/textured.mtl"

CLASSES = {
    "bell": {"lvis": ["bell"], "search": ["service bell", "call bell", "desk bell",
                                          "reception bell", "counter bell", "hotel bell"]},
    "plate": {"lvis": ["plate"], "search": ["dinner plate", "ceramic plate",
                                            "porcelain plate", "dinner dish"]},
    "fork": {"lvis": ["fork"], "search": ["dinner fork", "steel fork", "table fork",
                                          "metal fork"]},
    "spoon": {"lvis": ["spoon"], "search": ["table spoon", "metal spoon", "dinner spoon",
                                            "teaspoon", "soup spoon", "serving spoon",
                                            "dessert spoon", "wooden spoon"]},
}

# the LVIS classes are noisy (bell = church bell / lamp, spoon = bowling pin ...),
# so the name has to say what the object is.  Multilingual on purpose.
NAME_KW = {
    "bell": ["bell", "campana", "campan", "glocke", "cloche", "klocka", "铃", "鐘"],
    "plate": ["plate", "dish", "plato", "teller", "plat ", "assiette", "piatto", "тарелк",
              "piring", "皿", "borden"],
    "fork": ["fork", "tenedor", "fourchette", "gabel", "forchetta", "вилк", "叉"],
    "spoon": ["spoon", "cuchara", "löffel", "cuill", "cucchiaio", "ложк", "勺"],
}

# licence labels as they appear in the Objaverse dump
FREE = {"by", "cc0", "publicdomain", "pd"}
DROP = {"by-nc", "by-nc-sa", "by-nc-nd", "by-nd", "by-sa"}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
SEARCH_POOL = os.path.join(HOME, "search_pool.json")


def api_search(q, count=48):
    import urllib.parse
    url = ("https://api.sketchfab.com/v3/search?type=models&downloadable=true&count=%d&q=%s"
           % (count, urllib.parse.quote(q)))
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return json.load(r).get("results", [])
    except Exception as e:                                      # pragma: no cover
        print("search failed", q, str(e)[:40])
        return []


def search_pool(force=False):
    """Sketchfab keyword search, kept only where the uid is in Objaverse."""
    if os.path.exists(SEARCH_POOL) and not force:
        with open(SEARCH_POOL, encoding="utf-8") as fh:
            return json.load(fh)
    paths = json.load(gzip.open(PATHS))
    bad = ("helicopter", "supercobra", "huey", "ah-1", "agusta", "church", "cathedral",
           "tower", "stupa", "lamp", "curtain")
    out = {}
    for cls, spec in CLASSES.items():
        for q in spec["search"]:
            for m in api_search(q):
                uid = m.get("uid")
                name = str(m.get("name") or "")
                low = name.lower()
                if not uid or uid not in paths:
                    continue
                if any(b in low for b in bad):
                    continue
                if not any(k in low for k in NAME_KW[cls]):
                    continue
                lic = (m.get("license") or {}).get("label", "") if isinstance(
                    m.get("license"), dict) else str(m.get("license") or "")
                code = {"CC Attribution": "by", "CC0 Public Domain": "cc0",
                        "Public Domain": "publicdomain"}.get(lic, lic.lower())
                out[uid] = {"uid": uid, "classes": [cls], "name": name, "license": code,
                            "author": (m.get("user") or {}).get("username", ""),
                            "faces": m.get("faceCount"), "isDownloadable": True,
                            "tags": [t.get("name") for t in (m.get("tags") or [])][:8],
                            "url": f"https://sketchfab.com/3d-models/{uid}"}
            print(f"search {cls:6s} {q:20s} pool={len(out)}")
    with open(SEARCH_POOL, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    return out


def pool():
    """Candidate records for the four classes, licence already filtered."""
    with open(META, encoding="utf-8") as fh:
        metas = json.load(fh)
    SP = SEARCH_POOL
    if os.path.exists(SP):
        with open(SP, encoding="utf-8") as fh:
            have = {m["uid"] for m in metas}
            metas = metas + [m for m in json.load(fh).values() if m["uid"] not in have]
    out = []
    for m in metas:
        lic = str(m.get("license") or "").strip().lower()
        if lic not in FREE:
            continue
        if not m.get("isDownloadable"):
            continue
        f = m.get("faces") or 0
        if not (100 <= f <= 200000):
            continue
        cls = m["classes"][0]
        name = str(m.get("name") or "").lower()
        if not any(k in name for k in NAME_KW[cls]):
            continue
        out.append(m)
    return out


def ycb_records():
    cls_of = {"029_plate": "plate", "030_fork": "fork", "031_spoon": "spoon"}
    out = []
    for key, (obj, res) in YCB.items():
        out.append({"uid": key, "classes": [cls_of[obj]],
                    "name": f"YCB {obj} ({res}, real scan)",
                    "license": "CC-BY-4.0", "author": "YCB Benchmark",
                    "faces": None, "isDownloadable": True,
                    "url": "https://ycb-benchmarks.s3.amazonaws.com/index.html"})
    return out


def mesh_path(uid):
    if uid.startswith("ycb_"):
        return os.path.join(YCB_DIR, uid + ".obj")
    return os.path.join(GLB, uid + ".glb")


def fetch(recs, workers=6):
    paths = json.load(gzip.open(PATHS))
    os.makedirs(GLB, exist_ok=True)

    def one(m):
        uid = m["uid"]
        out = os.path.join(GLB, uid + ".glb")
        if os.path.exists(out) and os.path.getsize(out) > 1024:
            return uid, "cached"
        rel = paths.get(uid)
        if not rel:
            return uid, "no-path"
        try:
            with urllib.request.urlopen(
                    "https://huggingface.co/datasets/allenai/objaverse/resolve/main/" + rel,
                    timeout=120) as r, open(out, "wb") as fh:
                fh.write(r.read())
            return uid, "ok"
        except Exception as e:                                  # pragma: no cover
            return uid, "err " + str(e)[:50]

    with ThreadPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(one, recs))
    bad = [r for r in res if r[1].startswith("err")]
    print(f"fetch: {len(res) - len(bad)} ok/cached, {len(bad)} failed")
    return {r[0] for r in res if not r[1].startswith("err")}


def fetch_ycb(workers=3):
    import tarfile

    os.makedirs(YCB_DIR, exist_ok=True)

    def one(item):
        url, dst = item
        if os.path.exists(dst) and os.path.getsize(dst) > 512:
            return dst, "cached"
        try:
            with urllib.request.urlopen(url, timeout=300) as r, open(dst, "wb") as fh:
                fh.write(r.read())
            return dst, "ok"
        except Exception as e:                                  # pragma: no cover
            return dst, "err " + str(e)[:50]

    todo = []
    for key, (obj, res) in YCB.items():
        dst = os.path.join(YCB_DIR, obj + "_" + res + ".tgz")
        todo.append((f"https://ycb-benchmarks.s3.amazonaws.com/data/google/"
                     f"{obj}_{res}.tgz", dst))
    for dst, st in ThreadPoolExecutor(workers).map(one, todo):
        if st.startswith("err"):
            print("ycb", os.path.basename(dst), st)
            continue
        with tarfile.open(dst) as tf:
            tf.extractall(YCB_DIR)
    # the archives unpack to <obj>/<...>/textured.obj; copy to <key>.obj + .mtl + textures
    for key, (obj, res) in YCB.items():
        src = os.path.join(YCB_DIR, obj)
        found = None
        for root, _d, files in os.walk(src):
            if "textured.obj" in files:
                found = root
                break
        if not found:
            print("ycb", key, "no textured.obj")
            continue
        dst_obj = os.path.join(YCB_DIR, key + ".obj")
        # geometry only: drop the mtllib line so trimesh needs no PIL / texture
        with open(os.path.join(found, "textured.obj"), encoding="utf-8", errors="ignore") as src, \
                open(dst_obj, "w", encoding="utf-8") as dst:
            for line in src:
                if line.startswith("mtllib") or line.startswith("usemtl"):
                    continue
                dst.write(line)
    return os.path.join(YCB_DIR, "ycb_030_fork.obj")


# ---------------------------------------------------------------------------
def pca_frame(v):
    c = v.mean(axis=0)
    _, _, vt = np.linalg.svd(v - c, full_matrices=False)
    return c, vt


def area_normals_z(v, f):
    """Area-weighted sum of the outward face normals, z component."""
    a = v[f[:, 1]] - v[f[:, 0]]
    b = v[f[:, 2]] - v[f[:, 0]]
    n = np.cross(a, b)
    return float(n[:, 2].sum() * 0.5)


def radii_at(v, t, tol):
    sel = np.abs(v[:, 2] - t) < tol
    r = np.linalg.norm(v[sel][:, :2], axis=1)
    return (float(np.percentile(r, 90)) if len(r) >= 6 else None)


def oriented(v, f, cls):
    """PCA axes -> class frame; returns (points, faces, meta).

    PCA coordinates are already the projection onto the principal axes (sorted by
    variance); they are re-ordered by extent so that the longest axis is x and
    the shortest is z.  bell / plate then use z as up, utensils use x as the long
    axis and z as the face normal.
    """
    c, vt = pca_frame(v)
    u = (v - c) @ vt.T
    ext = u.max(0) - u.min(0)
    if cls in ("bell", "plate"):
        # symmetry axis: the one whose removal leaves the two most equal extents
        best = max(range(3), key=lambda j: min(ext[[k for k in range(3) if k != j]])
                   / max(ext[[k for k in range(3) if k != j]]))
        horiz = [k for k in range(3) if k != best]
        horiz.sort(key=lambda k: -ext[k])
        order = np.array([horiz[0], horiz[1], best])
    else:
        order = np.argsort(-ext)
    w = u[:, order]
    m = {}
    if cls in ("bell", "plate"):
        w = w - np.array([(w[:, 0].min() + w[:, 0].max()) / 2,
                          (w[:, 1].min() + w[:, 1].max()) / 2, 0.0])
        w[:, 2] -= w[:, 2].min()
        H = float(w[:, 2].max())
        if cls == "bell":
            # wide end down
            r_lo = radii_at(w, 0.02 * H, 0.02 * H)
            r_hi = radii_at(w, 0.98 * H, 0.02 * H)
            if r_lo is not None and r_hi is not None and r_hi > r_lo:
                w[:, 2] = H - w[:, 2]
                f = f[:, ::-1]
        else:
            # cavity up: more outward area faces up on the eating side
            if area_normals_z(w, f) < 0:
                w[:, 2] = H - w[:, 2]
                w[:, 2] -= w[:, 2].min()
                f = f[:, ::-1]
    else:
        # utensil: head (wider end) towards +x, face up +z
        L = float(w[:, 0].max() - w[:, 0].min())
        k = max(3, int(0.12 * len(w)))
        idx = np.argsort(w[:, 0])
        # the head is the wider end (fork tines / spoon bowl), measured in y
        w_neg = float(np.ptp(w[idx[:k], 1]))
        w_pos = float(np.ptp(w[idx[-k:], 1]))
        rad = np.linalg.norm(w[:, 1:], axis=1)
        if w_pos < w_neg:
            w[:, 0] = -w[:, 0]
            f = f[:, ::-1]
        if area_normals_z(w, f) < 0:
            w[:, 1] = -w[:, 1]
            w[:, 2] = -w[:, 2]
            f = f[:, ::-1]
        w = w - np.array([(w[:, 0].min() + w[:, 0].max()) / 2,
                          (w[:, 1].min() + w[:, 1].max()) / 2,
                          (w[:, 2].min() + w[:, 2].max()) / 2])
        m["L"], m["W"], m["T"] = L, float(np.ptp(w[:, 1])), float(np.ptp(w[:, 2]))
        # widths / thickness per slice along +x; the neck is where the width jumps
        xs = np.linspace(w[:, 0].min(), w[:, 0].max(), 25)
        prof = []
        for i in range(len(xs) - 1):
            sel = (w[:, 0] >= xs[i]) & (w[:, 0] < xs[i + 1])
            if not sel.any():
                continue
            prof.append((0.5 * (xs[i] + xs[i + 1]),
                         float(w[sel, 1].max() - w[sel, 1].min()),
                         float(w[sel, 2].max() - w[sel, 2].min())))
        m["profile_x_w"] = [(a, b, c) for a, b, c in prof]
        body = float(np.median([p[1] for p in prof[: max(2, len(prof) // 2)]]))
        # neck: scanning back from the head end, the first slice narrow enough to be
        # the handle.  Robust against a pommel / decorative bulge near the tail.
        neck = float(w[:, 0].max())
        for i in range(len(prof) - 1, -1, -1):
            if prof[i][1] <= 1.35 * body:
                neck = prof[i + 1][0] if i + 1 < len(prof) else prof[i][0]
                break
        m["neck_x"] = neck
        m["body_w"] = body
        m["head_w"] = float(np.percentile([p[1] for p in prof[-4:]], 85)) if len(prof) >= 4 else body
    return w, f, m


def w_axis(vt, i):
    return np.array(vt[i], dtype=float)


def load_obj_geometry(path):
    """Minimal OBJ reader (the import step may not rely on PIL for textures)."""
    vs, fs = [], []
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if line.startswith("v "):
                vs.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                tri = []
                for tok in line.split()[1:]:
                    i = int(tok.split("/")[0])
                    tri.append(i - 1 if i > 0 else len(vs) + i)
                for k in range(1, len(tri) - 1):
                    fs.append([tri[0], tri[k], tri[k + 1]])
    return np.asarray(vs, dtype=float), np.asarray(fs, dtype=np.int64)


def measure_shape(path):
    if path.endswith(".obj"):
        v, f = load_obj_geometry(path)
        if len(v) < 12 or len(f) < 12:
            raise ValueError("too few elements")
        _, f = weld(v, f)
        m = None
    else:
        import trimesh
        m = trimesh.load(path, force="mesh", process=False)
        if len(m.faces) == 0:
            raise ValueError("empty mesh")
        v = np.asarray(m.vertices, dtype=float)
        v = v[np.isfinite(v).all(axis=1)]
        f = np.asarray(m.faces, dtype=np.int64)
    if len(v) < 12 or len(f) < 12:
        raise ValueError("too few elements")
    ncomp, share, fc = component_stats(v, f)
    wt = bool(m.is_watertight) if m is not None else bool(geom_free_watertight(v, f))
    rec = {"faces": int(len(f)), "vertices": int(len(v)),
           "components": int(ncomp), "largest_component_share": round(float(share), 3),
           "watertight": wt}
    return v, f, rec, fc


def weld(v, f, eps=0.0):
    """Merge duplicate vertices (OBJ scans repeat them per face)."""
    if eps <= 0:
        key = np.round(v, 9)
    else:
        key = np.floor(v / eps)
    _, idx, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    return v[idx], inv[f]


def geom_free_watertight(v, f):
    e = np.sort(np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    return bool(np.all(counts == 2))


def component_stats(v, f):
    """Connected components without networkx (scipy only, import step only)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    e = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    g = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(v), len(v)))
    ncomp, lab = connected_components(g, directed=False)
    fc = lab[f[:, 0]]
    counts = np.bincount(fc)
    return ncomp, counts.max() / max(len(f), 1), fc


# ---------------------------------------------------------------------------
def build_one(rec, cls):
    uid = rec["uid"]
    path = mesh_path(uid)
    v, f, stats, fc = measure_shape(path)
    w, f2, m = oriented(v, f, cls)
    ext = w.max(0) - w.min(0)
    out = {
        "key": uid, "classes": rec.get("classes", [cls]),
        "name": rec.get("name", ""), "license": rec.get("license"),
        "author": rec.get("author", ""), "url": rec.get("url", ""),
        "source_file": path, "source_sha256": sha256_file(path),
        "source_kind": "ycb_real_scan" if uid.startswith("ycb_") else "objaverse_lvis",
        "extent": [round(float(x), 6) for x in ext],
        **stats, **{k: (round(v_, 6) if isinstance(v_, float) else v_)
                    for k, v_ in m.items() if k != "profile_x_w"},
    }
    out["profile_x_w"] = [[round(a, 6), round(b, 6), round(c, 6)]
                         for a, b, c in m.get("profile_x_w", [])]
    return w, f2, out, fc


def main(argv):
    mode = argv[1] if len(argv) > 1 else "list"
    os.makedirs(NORM, exist_ok=True)
    recs = pool()
    ycb = ycb_records()
    print(f"pool: {len(recs)} licence-clean Objaverse candidates "
          f"({', '.join(sorted({r['classes'][0] for r in recs}))}) + {len(ycb)} YCB")
    if mode == "fetch":
        fetch(recs)
        fetch_ycb()
        return 0
    if mode == "normalise":
        ok = fetch(recs)
        fetch_ycb()
        built = 0
        for r in [r for r in recs if r["uid"] in ok] + ycb:
            try:
                w, f, meta, fc = build_one(r, r["classes"][0])
            except Exception as e:
                print("skip", r["uid"][:12], str(e)[:60])
                continue
            np.save(os.path.join(NORM, r["uid"] + ".v.npy"), w)
            np.save(os.path.join(NORM, r["uid"] + ".f.npy"), f)
            np.save(os.path.join(NORM, r["uid"] + ".c.npy"), fc)
            with open(os.path.join(NORM, r["uid"] + ".json"), "w", encoding="utf-8") as fh:
                json.dump(meta, fh, indent=1, sort_keys=True, ensure_ascii=False)
            built += 1
        print("normalised", built)
        return 0
    if mode == "measure":
        ok = fetch(recs)
        fetch_ycb()
        rows = []
        for r in [r for r in recs if r["uid"] in ok] + ycb:
            try:
                w, f, meta, fc = build_one(r, r["classes"][0])
            except Exception as e:
                continue
            e = [float(x) for x in (w.max(0) - w.min(0))]
            rows.append({"uid": r["uid"], "cls": r["classes"][0], "name": r["name"],
                         "license": r["license"], "author": r["author"],
                         "faces": meta["faces"], "components": meta["components"],
                         "share": meta["largest_component_share"],
                         "watertight": meta["watertight"],
                         "extent": [round(x, 4) for x in e],
                         "neck_x": meta.get("neck_x"), "body_w": meta.get("body_w")})
        with open(os.path.join(HOME, "measured.json"), "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=1, ensure_ascii=False)
        for row in sorted(rows, key=lambda r: (r["cls"], r["extent"][0])):
            print(f"{row['cls']:6s} {row['uid'][:12]:13s} f={row['faces']:7d} c={row['components']:2d} "
                  f"sh={row['share']:.2f} ext={row['extent'][0]*1e3:6.1f}x"
                  f"{row['extent'][1]*1e3:6.1f}x{row['extent'][2]*1e3:6.1f}mm "
                  f"{row['license']:6s} {row['name'][:34]}")
        print("->", os.path.join(HOME, "measured.json"))
        return 0
    # list
    for r in sorted(recs, key=lambda r: (r["classes"][0], -(r.get("faces") or 0))):
        print(f"  {r['classes'][0]:6s} {r['uid'][:8]} f={r.get('faces'):7d} {r['license']:4s} "
              f"{r['author'][:14]:15s} {r['name'][:46]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
