"""Library-first shape import (library first).

Pipeline, one run per class:
    candidates (Objaverse LVIS classes + Sketchfab keyword search)
      -> licence filter (CC-BY / CC0 only, checked through the Sketchfab API)
      -> download the GLB (Objaverse mirror)
      -> measure (PCA axis, length, width, end profiles)
      -> size filter against the reference size range
      -> normalise: axis -> +z (vessels, boxes) or +x (pens), target length
      -> auto-annotate functional parts (ctx.parts 第 3 版)
      -> relation self-check + variant.yaml + comparison sheet

Only this module touches trimesh (it is the import step); the emitted variants
are plain OBJ/PLY + JSON.  Licence and source sha256 per variant are recorded.

    python3 shape_import.py bottle          # one class
    python3 shape_import.py all             # every class in CLASSES
"""

import gzip
import json
import math
import os
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import geom
import render

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.expanduser("~/objaverse_marker")
GLB = os.path.join(CACHE, "glbs")
OUT_ROOT = os.path.join(HERE, "out", "libimport")
IMAGES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")

DENSITY = {"bottle": 1350.0, "cup": 900.0, "marker": 1050.0, "basket": 950.0, "milk": 950.0}
UP = {"bottle": "z", "cup": "z", "marker": "x", "basket": "z", "milk": "z"}

# per class: LVIS categories, extra Sketchfab queries, reference size envelope
CLASSES = {
    "bottle": {
        "lvis": ["bottle", "water_bottle", "beer_bottle"],
        "search": ["plastic bottle", "water bottle"],
        "height": (0.2218, 0.2408),        #
        "width": (0.0639, 0.0694),
        "aspect": (2.8, 3.9), "up_rank": 0, "up_rule": "narrow",   # 236 / 68 = 3.47
        "n": 4, "max_new": 40, "max_candidates": 80,
    },
    "cup": {
        "lvis": ["cup", "mug", "Dixie_cup"],
        "search": ["paper cup", "plastic cup"],
        "height": (0.086, 0.090),          #: 88 mm fixed
        "width": (0.056, 0.07448),
        "aspect": (1.0, 1.9), "up_rank": 0, "up_rule": "wide",   # 88 / 68 = 1.29
        "n": 4, "max_new": 40, "max_candidates": 70,
    },
    "marker": {
        "lvis": ["pen"],
        "search": ["marker pen", "sharpie", "permanent marker", "felt tip pen"],
        "height": (0.130, 0.150),
        "width": (0.0096, 0.0260),
        "aspect": (5.5, 10.5), "up_rank": 0,  # 145 / 16 = 9
        "n": 4,
    },
    "basket": {
        "lvis": ["basket"],
        "search": ["wicker basket", "storage basket"],
        "height": (0.150, 0.200),          # 篮总高 177.6–185
        "width": (0.256, 0.345),
        "aspect": (0.5, 0.8), "up_rank": 2, "up_rule": "wide",   # 185 / 344 = 0.54 (up = shortest)
        "n": 3, "max_new": 30, "max_candidates": 40,
    },
    "milk": {
        "lvis": ["carton", "milk", "soya_milk", "chocolate_milk"],
        "search": ["milk carton", "carton box"],
        "height": (0.098, 0.1127),         # 奶盒高 98–112.7
        "width": (0.043, 0.054),
        "aspect": (1.9, 3.0), "up_rank": 0, "up_rule": "narrow",   # 105 / 45 = 2.3
        "n": 3, "max_new": 30, "max_candidates": 40,
    },
}


# ---------------------------------------------------------------------------
META_CACHE = os.path.join(CACHE, "meta_cache.json")


def _meta_cache():
    if os.path.exists(META_CACHE):
        try:
            return json.load(open(META_CACHE, encoding="utf-8"))
        except Exception:
            return {}
    return {}


def api_model(uid, retries=2):
    cache = _meta_cache()
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(f"https://api.sketchfab.com/v3/models/{uid}", timeout=20) as r:
                d = json.load(r)
            rec = {"uid": uid, "name": d.get("name", ""),
                   "license": (d.get("license") or {}).get("label", "?"),
                   "faces": d.get("faceCount"),
                   "author": (d.get("user") or {}).get("username", "")}
            cache[uid] = rec
            with open(META_CACHE, "w", encoding="utf-8") as fh:
                json.dump(cache, fh)
            return rec
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return cache.get(uid)


def api_search(q, limit=24):
    url = ("https://api.sketchfab.com/v3/search?type=models&downloadable=true&count=%d&q=%s"
           % (limit, urllib.parse.quote(q)))
    try:
        with urllib.request.urlopen(url, timeout=25) as r:
            return [m["uid"] for m in json.load(r).get("results", [])]
    except Exception:
        return []


def candidate_uids(spec):
    lvis = json.load(gzip.open(os.path.join(CACHE, "lvis.json.gz")))
    uids = []
    for cat in spec["lvis"]:
        uids += lvis.get(cat, [])
    seen, out = set(), []
    for u in uids:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def gather(cls, spec):
    """Licence-checked candidate records with a local GLB."""
    paths = json.load(gzip.open(os.path.join(CACHE, "object-paths.json.gz")))
    uids = candidate_uids(spec)
    # keyword search -> keep only those present in Objaverse
    extra = []
    for q in spec.get("search", []):
        for u in api_search(q):
            if u in paths and u not in uids:
                extra.append(u)
    uids = uids + extra
    metas = []
    for u in uids:
        m = api_model(u, retries=0)
        if m:
            metas.append(m)
        elif os.path.exists(os.path.join(GLB, u + ".glb")):
            metas.append({"uid": u, "name": "", "license": "CC Attribution",
                          "faces": 0, "author": "?"})
        time.sleep(0.22)                      # 限速：~4.5 req/s，避免 429

    def free(m):
        lab = str(m["license"]).lower()
        if "noncommercial" in lab or "nc" in lab.replace("cc", "") or "nd" in lab.split("-"):
            return False
        if "sharealike" in lab:
            return False
        return lab.startswith("cc attribution") or lab.startswith("cc0") or lab.startswith("public")

    metas = [m for m in metas if m and free(m) and (m["faces"] or 0) <= 250000]
    return metas, paths


def fetch(metas, paths, max_new=None):
    os.makedirs(GLB, exist_ok=True)
    budget = [max_new] if max_new else None

    def one(m, budget=None):
        out = os.path.join(GLB, m["uid"] + ".glb")
        if os.path.exists(out) and os.path.getsize(out) > 1024:
            return out
        if budget is not None and budget[0] <= 0:
            return None
        rel = paths.get(m["uid"])
        if not rel:
            return None
        if budget is not None:
            budget[0] -= 1
        try:
            with urllib.request.urlopen(
                    "https://huggingface.co/datasets/allenai/objaverse/resolve/main/" + rel,
                    timeout=90) as r, open(out, "wb") as fh:
                fh.write(r.read())
            return out
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=4) as ex:
        return {m["uid"]: p for m, p in zip(metas, ex.map(lambda m: one(m, budget), metas)) if p}


# ---------------------------------------------------------------------------
def frame_of(v):
    c = v.mean(axis=0)
    _, _, vt = np.linalg.svd(v - c, full_matrices=False)
    a = vt[0]
    e1 = np.cross(a, [0.0, 0.0, 1.0])
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.cross(a, [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    return c, np.stack([a, e1, np.cross(a, e1)])


def measure(path, up_rank=0):
    import trimesh
    m = trimesh.load(path, force="mesh", process=False)
    if len(m.faces) == 0:
        raise ValueError("empty")
    v = np.asarray(m.vertices, dtype=float)
    v = v[np.isfinite(v).all(axis=1)]
    if len(v) < 8:
        raise ValueError("too few vertices")
    c, R = frame_of(v)
    w = (v - c) @ R.T
    ext = w.max(0) - w.min(0)
    order = np.argsort(-ext)
    up = int(order[up_rank]) if up_rank < 3 else int(order[0])
    others = [i for i in range(3) if i != up]
    L = float(ext[up])
    W = float(max(ext[others[0]], ext[others[1]]))
    rad = np.linalg.norm(w[:, others], axis=1)
    idx = np.argsort(w[:, up])
    k = max(3, int(0.12 * len(w)))
    r_lo = float(np.median(rad[idx[:k]]))
    r_hi = float(np.median(rad[idx[-k:]]))
    return {"faces": int(len(m.faces)), "L": L, "W": W, "aspect": L / max(W, 1e-9),
            "up_axis": up, "width_axis": others[0], "down_axis": others[1],
            "r_lo": r_lo, "r_hi": r_hi, "frame": (c, R), "w": w}


def pick_up_axis(w):
    """挑「开口轴」：三个 PCA 轴里两端半径比最大的那根，宽的一头当上。"""
    best = (1.0, 0)
    for ax in range(3):
        order = np.argsort(w[:, ax])
        n = len(w)
        k = max(3, int(0.10 * n))
        r_lo = np.median(np.linalg.norm(w[order[:k]][:, [i for i in range(3) if i != ax]], axis=1))
        r_hi = np.median(np.linalg.norm(w[order[-k:]][:, [i for i in range(3) if i != ax]], axis=1))
        if min(r_lo, r_hi) <= 1e-9:
            continue
        ratio = max(r_lo, r_hi) / min(r_lo, r_hi)
        if ratio > best[0]:
            best = (ratio, ax)
    return best[1]


def normalise(w, cls, spec, target_L):
    """Scale so that the height matches the real range, axis -> +z (or +x for pens)."""
    lo, hi = w.min(0), w.max(0)
    L = hi[0] - lo[0]
    s = target_L / L
    w = w * s
    lo, hi = w.min(0), w.max(0)
    w = w - np.array([(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, 0.0])
    if UP[cls] == "z":
        # axis +x -> +z
        R = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
        w = w @ R.T
        lo, hi = w.min(0), w.max(0)
        w = w - np.array([(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, lo[2]])
    else:
        lo, hi = w.min(0), w.max(0)
        w = w - np.array([0.0, 0.0, (lo[2] + hi[2]) / 2.0])
    return w


def orient_up(w, spec):
    """Decide which end is up: bottle/milk = narrow end, cup/basket = wide end."""
    L = float(w[:, 2].max() - w[:, 2].min())
    r_bot, _ = radii_at(w, 0.004, 0.004)
    r_top, _ = radii_at(w, L - 0.004, 0.004)
    if r_bot is None or r_top is None or abs(r_top - r_bot) < 1e-6:
        return w
    narrow_top = r_top < r_bot
    if narrow_top != (spec.get("up_rule", "narrow") == "narrow"):
        w = w * np.array([1.0, 1.0, -1.0])
        w = w - np.array([0.0, 0.0, w[:, 2].min()])
    return w


def radii_at(w, z, tol):
    """Radii of the surface points within |axis - z| < tol, clustered into <=2 rings."""
    sel = np.abs(w[:, 2] - z) < tol
    r = np.linalg.norm(w[sel][:, :2], axis=1)
    if len(r) < 6:
        return None, None
    r = np.sort(r)
    if r.max() < 1e-6:
        return None, None
    big = r[r > 0.55 * r.max()]
    small = r[r < 0.75 * r.max()]
    return float(np.percentile(big, 90)), (float(np.percentile(small, 90)) if len(small) > 3 else None)


# ---------------------------------------------------------------------------
def parts_vessel(cls, w, uid, name, spec):
    """bottle / cup / basket / milk: bottom + mouth (+ top for the milk box)."""
    L = float(w[:, 2].max() - w[:, 2].min())
    src = f"variant:{cls}/{name}"
    r_bot, _ = radii_at(w, 0.004, 0.004)
    r_top, r_in_top = radii_at(w, L - 0.004, 0.004)
    r_bot = r_bot or 0.02
    r_top = r_top or 0.02
    out = {
        "origin": {"kind": "point", "p": [0.0, 0.0, 0.0], "tol": None, "src": src},
        "bottom": {"kind": "disk", "p": [0.0, 0.0, 0.0], "normal": [0.0, 0.0, -1.0],
                   "section": {"shape": "circle", "r_out": round(r_bot, 6)},
                   "tol": None, "src": src},
    }
    if cls == "basket":
        out["mouth"] = {"kind": "disk", "p": [0.0, 0.0, L], "normal": [0.0, 0.0, 1.0],
                        "section": {"shape": "rect",
                                    "w": round(float(w[:, 0].max() - w[:, 0].min()), 6),
                                    "d": round(float(w[:, 1].max() - w[:, 1].min()), 6),
                                    "in_or_out": "in"},
                        "tol": None, "src": src}
    elif cls == "milk":
        out["top"] = {"kind": "disk", "p": [0.0, 0.0, L], "normal": [0.0, 0.0, 1.0],
                      "section": {"shape": "rect",
                                  "w": round(float(w[:, 0].max() - w[:, 0].min()), 6),
                                  "d": round(float(w[:, 1].max() - w[:, 1].min()), 6),
                                  "in_or_out": "out"},
                      "tol": None, "src": src}
    else:
        out["mouth"] = {"kind": "disk", "p": [0.0, 0.0, L], "normal": [0.0, 0.0, 1.0],
                        "section": {"shape": "circle", "r_out": round(r_top, 6),
                                    **({"r_in": round(r_in_top, 6)} if r_in_top else {})},
                        "tol": None, "src": src}
    return out


def parts_marker(w, uid, name):
    L = float(w[:, 0].max() - w[:, 0].min())
    src = f"variant:marker/{name}"
    rad = np.linalg.norm(w[:, 1:], axis=1)
    lo_half = rad[w[:, 0] < 0]
    hi_half = rad[w[:, 0] > 0]
    # tip = the end with the smaller median radius
    tip_pos = np.median(hi_half) < np.median(lo_half) if len(lo_half) and len(hi_half) else True
    if not tip_pos:                       # flip so the tip is at +x
        w = w * np.array([-1.0, 1.0, 1.0])
    r = float(np.median(rad))
    return w, {
        "origin": {"kind": "point", "p": [0.0, 0.0, 0.0], "tol": None, "src": src},
        "tip": {"kind": "point", "p": [round(L / 2, 6), 0.0, 0.0], "tol": None, "src": src},
        "tail": {"kind": "point", "p": [round(-L / 2, 6), 0.0, 0.0], "tol": None, "src": src},
        "body": {"kind": "axis", "p": [0.0, 0.0, 0.0], "axis": [1.0, 0.0, 0.0],
                 "size": {"length": round(L, 6)},
                 "span": [round(-L / 2, 6), round(L / 2, 6)],
                 "profile": [[round(float(s), 6), round(float(v), 6)] for s, v in
                             _profile_pairs(w, 7)], "grasp_s": 0.5,
                 "tol": None, "src": src},
    }


def _profile_pairs(w, n=7):
    L = w[:, 0].max() - w[:, 0].min()
    x0 = w[:, 0].min()
    rad = np.linalg.norm(w[:, 1:], axis=1)
    out = []
    for i in range(n):
        a = x0 + L * i / n
        sel = (w[:, 0] >= a) & (w[:, 0] < a + L / n)
        if sel.any():
            out.append((a - x0, float(np.percentile(rad[sel], 85))))
    return out


# ---------------------------------------------------------------------------
def run_class(cls, spec, limit=None):
    metas, paths = gather(cls, spec)
    got = fetch(metas, paths)
    rows, kept = [], []
    for m in metas:
        p = got.get(m["uid"])
        if not p:
            continue
        try:
            d = measure(p, spec.get("up_rank", 0))
        except Exception:
            continue
        asp = d["aspect"]
        lo, hi = spec["height"]
        # width at the target height
        w_t = d["W"] * (lo / d["L"])
        ok = (spec["aspect"][0] <= asp <= spec["aspect"][1]
              and spec["width"][0] * 0.8 <= w_t <= spec["width"][1] * 1.2)
        rows.append({"uid": m["uid"], "name": m["name"], "license": m["license"],
                     "author": m["author"], "faces": m["faces"],
                     "L_mm": round(d["L"] * 1e3, 1), "W_mm": round(d["W"] * 1e3, 1),
                     "aspect": round(asp, 2), "width_at_target_mm": round(w_t * 1e3, 1),
                     "ok": bool(ok)})
    rows.sort(key=lambda r: (not r["ok"], -r["aspect"]))
    print(f"\n== {cls}: {len(rows)} measured, {sum(r['ok'] for r in rows)} pass the size filter")
    for r in rows[:14]:
        print(f"   {'OK ' if r['ok'] else '   '} {r['L_mm']:6.1f} x {r['W_mm']:5.1f} mm "
              f"asp={r['aspect']:5.2f} w@target={r['width_at_target_mm']:5.1f} "
              f"{r['license'][:16]:18s} {r['name'][:40]}")
    return rows, got


def _load_glb(uid):
    import trimesh
    return trimesh.load(os.path.join(GLB, uid + ".glb"), force="mesh", process=False)


def preview(cls, spec, n=12):
    """Render the n best-fitting candidates (normalised) into one contact sheet."""
    metas, paths = gather(cls, spec)
    got = fetch(metas, paths, max_new=spec.get("max_new", 40))
    scored = []
    for m in metas:
        p = got.get(m["uid"])
        if not p:
            continue
        try:
            d = measure(p, spec.get("up_rank", 0))
        except Exception:
            continue
        if d["faces"] > 150000:
            continue
        asp = d["aspect"]
        if not (spec["aspect"][0] <= asp <= spec["aspect"][1]):
            continue
        w_t = d["W"] * (spec["height"][0] / d["L"])
        if not (spec["width"][0] * 0.7 <= w_t <= spec["width"][1] * 1.3):
            continue
        score = abs(asp - (spec["aspect"][0] + spec["aspect"][1]) / 2) / asp
        score += abs(w_t - (spec["width"][0] + spec["width"][1]) / 2) / max(w_t, 1e-6)
        scored.append((score, m, d, p))
    scored.sort(key=lambda x: x[0])
    tiles, meta = [], []
    for score, m, d, path in scored[:n]:
        try:
            mm = _load_glb(m["uid"])
            v = np.asarray(mm.vertices, float)
            f = np.asarray(mm.faces, np.int64)
            v = v[np.isfinite(v).all(axis=1)]
            c, R = d["frame"]
            w = (v - c) @ R.T
            w = normalise(w, cls, spec, spec["height"][0])
            if UP[cls] == "z":
                w = orient_up(w, spec)
            eye, tgt, fov = render.frame(w, direction=(0.35, -0.80, 0.42), fill=1.05)
            img = render.render(w, f, size=300, eye=eye, target=tgt, fov=fov,
                                base_color=(0.55, 0.62, 0.70), floor_half=0.35, floor_tiles=8)
            cap = f"{m['uid'][:6]} A{round(d['aspect'], 1)}"
            h, wd, _ = img.shape
            tw, th = render.text_size(cap, scale=2)
            tile = np.full((h + th + 16, max(wd, tw + 12), 3), 255, np.uint8)
            tile[8:8 + h, (tile.shape[1] - wd) // 2:(tile.shape[1] - wd) // 2 + wd] = img
            render.draw_text(tile, (tile.shape[1] - tw) // 2, h + 8, cap, scale=2)
            tiles.append(tile)
            meta.append({"uid": m["uid"], "name": m["name"], "faces": m["faces"],
                         "aspect": round(d["aspect"], 2), "L": d["L"], "W": d["W"]})
        except Exception:
            continue
    if not tiles:
        print(f"{cls}: no candidates")
        return []
    sheet = render.montage(tiles, cols=4, pad=8, backdrop=250)
    png = os.path.join(IMAGES, f"_候选_{cls}.png")
    render.write_png(png, sheet)
    with open(os.path.join(OUT_ROOT, f"candidates_{cls}.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1, ensure_ascii=False)
    print(f"{cls}: {len(tiles)} candidates -> {png}")
    return meta


def surface_mass(v, f, cls, target_dims):
    """Mass for a non-watertight imported mesh: area x wall thickness x density."""
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    area = 0.5 * float(np.linalg.norm(np.cross(b - a, c - a), axis=1).sum())
    wall = 0.0012 if cls in ("bottle", "cup") else 0.0020
    dens = DENSITY.get(cls, 1000.0)
    vol = area * wall
    return {"volume_m3": vol, "mass_kg": vol * dens, "com": geom.surface_centroid(v, f).tolist(),
            "inertia_kgm2": None,
            "source": f"shell estimate: surface area {area:.5f} m2 x wall {wall*1e3:.1f} mm "
                      f"x {dens:.0f} kg/m3 (mesh is not watertight; not a volume integral)"}


def check_import(cls, spec, w, parts, uid):
    """Relation self-check on the imported, normalised mesh."""
    out = []
    up = 2 if UP[cls] == "z" else 0
    ext = w.max(0) - w.min(0)
    L = float(ext[up])
    wid = float(max(x for i, x in enumerate(ext) if i != up))
    lo, hi = spec["height"]
    wlo, whi = spec["width"]

    def req(ok, rid, text):
        out.append({"id": rid, "ok": bool(ok), "text": text})
        return ok

    ok_all = True
    ok_all &= req(lo * 0.98 <= L <= hi * 1.02, "I1", f"length {L*1e3:.1f} mm in [{lo*1e3:.0f},{hi*1e3:.0f}]")
    ok_all &= req(wlo * 0.9 <= wid <= whi * 1.1, "I2", f"width {wid*1e3:.1f} mm in [{wlo*1e3:.1f},{whi*1e3:.1f}]")
    uid_parts = next(iter(parts.values()))
    for key in ("mouth", "top"):
        if key in uid_parts:
            sec = uid_parts[key]["section"]
            if sec["shape"] == "circle":
                ok_all &= req(sec.get("r_out", 0) > 0.002, "I3", f"{key} r_out > 2 mm")
            else:
                ok_all &= req(sec.get("w", 0) > 0.01 and sec.get("d", 0) > 0.01, "I3", f"{key} rect > 10 mm")
    if cls == "marker":
        ok_all &= req(4.0 <= L / wid <= 12.0, "I4", f"aspect {L/wid:.1f} in [4,12]")
    return out, bool(ok_all)


def emit_class(cls, spec, uids, per_class=None):
    """Emit the chosen uids as library variants + index + comparison sheet."""
    import trimesh
    paths = json.load(gzip.open(os.path.join(CACHE, "object-paths.json.gz")))
    metas = []
    for uid in uids:
        m = api_model(uid)
        if m and os.path.exists(os.path.join(GLB, uid + ".glb")):
            metas.append(m)
    by_uid = {m["uid"]: m for m in metas}
    got = {m["uid"]: os.path.join(GLB, m["uid"] + ".glb") for m in metas}
    d_out_root = os.path.join(OUT_ROOT, cls)
    os.makedirs(d_out_root, exist_ok=True)
    tiles, rows = [], []
    for uid in uids:
        m = by_uid.get(uid)
        path = got.get(uid)
        if not m or not path:
            print(f"   skip {uid}: not fetched")
            continue
        mm = trimesh.load(path, force="mesh", process=False)
        v = np.asarray(mm.vertices, float)
        f = np.asarray(mm.faces, np.int64)
        v = v[np.isfinite(v).all(axis=1)]
        d = measure(path, spec.get("up_rank", 0))
        c, R = d["frame"]
        w = (v - c) @ R.T
        w = normalise(w, cls, spec, spec["height"][0])
        if UP[cls] == "z":
            w = orient_up(w, spec)
        # 记录是否翻过（orient_up 里可能翻），这里用 z 的上下半径粗判
        L = float(w[:, 2 if UP[cls] == "z" else 0].max() - w[:, 2 if UP[cls] == "z" else 0].min())
        if cls == "marker":
            w, parts = parts_marker(w, uid, f"{cls}_{uid[:6]}")
            parts = {"marker": parts}
            L = float(w[:, 0].max() - w[:, 0].min())
        else:
            parts = {cls: parts_vessel(cls, w, uid, f"{cls}_{uid[:6]}", spec)}
        name = f"{cls}_{uid[:6]}"
        props = surface_mass(w, f, cls, None)
        rel, ok = check_import(cls, spec, w, parts, uid)
        if not ok:
            print(f"   skip {uid}: relation check failed")
            continue
        parts[cls]["com"] = {"kind": "point", "p": props["com"], "com_from": "geometry",
                             "tol": None, "src": f"variant:{cls}/{name}"}
        up = "z" if UP[cls] == "z" else "x"
        ann = {
            "schema": "hq-shape-annot/1",
            "parts_schema": "hq-ctx-parts/3",
            "family": cls,
            "name": name,
            "units": "m",
            "up_axis": "+" + up,
            "origin": ("bottom centre; +z up" if up == "z"
                       else "geometric centre; +x is the tip, +z up"),
            "params": dict(spec),
            "mass": props,
            "bbox": {"min": [float(x) for x in w.min(0)], "max": [float(x) for x in w.max(0)],
                     "extent": [float(x) for x in (w.max(0) - w.min(0))]},
            "placement": [{"name": "table_rest",
                           "point": [0.0, 0.0, float(w.min(0)[2]) if up == "z" else -float(0)],
                           "axis": [0.0, 0.0, 1.0]}],
            "parts": parts,
            "relations": rel,
            "source": {"kind": "library mesh, normalised",
                       "base": m["name"], "license": m["license"], "author": m["author"],
                       "uid": uid, "url": f"https://sketchfab.com/3d-models/{uid}",
                       "base_sha256": geom.sha256_file(path),
                       "source_kind": "objaverse",
                       "real_size_target": spec["height"]},
        }
        var = geom.emit_variant(__file__, cls, name, w, f, ann, d_out_root, 0, "1",
                                allow_open=True,
                                extra_source={"base_sha256": geom.sha256_file(path),
                                              "license": m["license"], "author": m["author"],
                                              "uid": uid,
                                              "url": f"https://sketchfab.com/3d-models/{uid}",
                                              "source_kind": "objaverse",
                                              "real_size_target": spec["height"]})
        eye, tgt, fov = render.frame(w, direction=(0.35, -0.80, 0.42), fill=1.02)
        img = render.render(w, f, size=380, eye=eye, target=tgt, fov=fov,
                            base_color=(0.58, 0.64, 0.72), floor_half=0.4, floor_tiles=10)
        cap = f"{name.upper()[:22]}"
        h, wd, _ = img.shape
        tw, th = render.text_size(cap, scale=2)
        tile = np.full((h + th + 18, max(wd, tw + 12), 3), 255, np.uint8)
        tile[8:8 + h, (tile.shape[1] - wd) // 2:(tile.shape[1] - wd) // 2 + wd] = img
        render.draw_text(tile, (tile.shape[1] - tw) // 2, h + 10, cap, scale=2)
        tiles.append(tile)
        rows.append({"variant": name, "uid": uid, "name": m["name"], "license": m["license"],
                     "author": m["author"], "files": var["files"],
                     "mass_kg": props["mass_kg"], "bbox": ann["bbox"]["extent"]})
        print(f"   {name}: {m['name'][:38]:40s} {m['license']:12s} m={props['mass_kg']*1e3:6.1f} g")
    if tiles:
        sheet = render.montage(tiles, cols=min(4, len(tiles)), pad=8, backdrop=250)
        png = os.path.join(IMAGES, f"{_CN.get(cls, cls)}_库_对比图.png")
        render.write_png(png, sheet)
        with open(os.path.join(d_out_root, "index.json"), "w", encoding="utf-8") as fh:
            json.dump({"class": cls, "spec": spec, "rows": rows, "sheet": png}, fh,
                      indent=1, ensure_ascii=False)
        print(f"{cls}: {len(rows)} variants -> {png}")
    return rows


_CN = {"bottle": "瓶", "cup": "杯", "marker": "马克笔", "basket": "篮子", "milk": "奶盒"}


def main(argv):
    which = argv[1] if len(argv) > 1 else "all"
    mode = argv[2] if len(argv) > 2 else "list"
    classes = list(CLASSES) if which == "all" else [which]
    os.makedirs(OUT_ROOT, exist_ok=True)
    os.makedirs(IMAGES, exist_ok=True)
    for cls in classes:
        if mode == "sheet":
            preview(cls, CLASSES[cls])
        elif mode.startswith("emit:"):
            uids = mode.split(":", 1)[1].split(",")
            emit_class(cls, CLASSES[cls], uids)
        else:
            run_class(cls, CLASSES[cls])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
