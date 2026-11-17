"""Ready-made marker variants: real meshes, normalised, annotated.

Design: do not draw the marker by hand; take ready-made meshes and cover the
diameter range **from the thin simulation pen to a thick permanent marker** in
even bins with quotas (not random sampling).  Objaverse has no real units, so
every mesh is PCA-aligned by marker_import.py, its length is normalised and the
diameter is set by a radial scale - that is also how the thick end is reached.

This module is a *generator*: it reads plain numpy arrays (no trimesh, no
simulator) and is byte-identical for the same inputs.  The import step lives in
marker_import.py; provenance (including the source-mesh sha256) is copied into
every annotation and variant.yaml.

    python3 marker_import.py     # once, or when a base mesh changes
    python3 marker_ready.py      # build all variants + the contact sheet
"""

import json
import os
import sys

import numpy as np

import geom
import render

HERE = os.path.dirname(os.path.abspath(__file__))
NORM = os.path.expanduser("~/objaverse_marker/normalised")
OUT = os.path.join(HERE, "out", "marker_ready")
IMAGES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")

# (variant name, base key, target diameter m, target length m)
VARIANTS = [
    # thin bin 9-12 mm: the simulation pen is the reference for this end
    ("mk_thin_096_official", "official_pen", 0.0096, 0.1434),
    ("mk_thin_105_permanent", "poly_permanent", 0.0105, 0.1450),
    ("mk_thin_110_sharpie", "sharpie_thin", 0.0110, 0.1450),
    # mid bin 12-18 mm
    ("mk_mid_138_stabilo", "stabilo_boss", 0.0138, 0.1450),
    ("mk_mid_150_permanent", "poly_permanent", 0.0150, 0.1450),
    ("mk_mid_153_official", "official_pen", 0.0153, 0.1434),
    # thick bin 18-26 mm: radial scale on real meshes
    ("mk_thick_180_highlighter", "highlighter", 0.0180, 0.1450),
    ("mk_thick_200_permanent", "poly_permanent", 0.0200, 0.1450),
    ("mk_thick_210_highlighter", "highlighter", 0.0210, 0.1450),
    ("mk_thick_240_permanent", "poly_permanent", 0.0240, 0.1450),
    ("mk_thick_260_stabilo", "stabilo_boss", 0.0260, 0.1500),
]

BINS = {"thin 9-12": (0.009, 0.012), "mid 12-18": (0.012, 0.018), "thick 18-26": (0.018, 0.02601)}
DENSITY = 1050.0          # ABS/PP band (marker)
FILL = 0.535              # hollow-barrel fill, calibrated on edding 13.6 g (see params.py)
VERSION = "1"


# ---------------------------------------------------------------------------
def load_base(key):
    """Normalised base mesh + provenance (plain numpy + json only)."""
    v = np.load(os.path.join(NORM, key + ".v.npy"))
    f = np.load(os.path.join(NORM, key + ".f.npy"))
    with open(os.path.join(NORM, key + ".json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    return v, f, meta


def _bin(d):
    return next(k for k, (a, b) in BINS.items() if a <= d < b)


def _section_radius(v, f, x, tol=1e-12):
    """过平面 x 的截面最大半径（纯 numpy；粗网格中段没有顶点环也能量到，避免 profile 出现 0）。"""
    tri = v[f]                       # (m,3,3)
    d = tri[:, :, 0] - x             # 每个顶点的 x 差
    rmax = 0.0
    pts = []
    for e0, e1 in ((0, 1), (1, 2), (2, 0)):
        a = d[:, e0]
        b = d[:, e1]
        cross = (a * b) < 0
        if not cross.any():
            continue
        t = (a[cross] / (a[cross] - b[cross]))[:, None]
        p = tri[cross, e0, 1:] + t * (tri[cross, e1, 1:] - tri[cross, e0, 1:])
        pts.append(p)
    if not pts:
        return 0.0
    P = np.vstack(pts)
    return float(np.max(np.linalg.norm(P, axis=1)))


def build_variant(name, base_key, target_d, target_L):
    v0, f0, meta = load_base(base_key)
    v = np.array(v0, dtype=float)
    f = np.array(f0, dtype=np.int64)

    L0 = v[:, 0].max() - v[:, 0].min()
    d0 = 2.0 * np.median(np.linalg.norm(v[:, 1:], axis=1))
    sx = target_L / L0
    sr = target_d / d0                     # radial scale on the raw mesh
    v = v * np.array([sx, sr, sr])
    lo, hi = v.min(0), v.max(0)
    v = v - np.array([(lo[0] + hi[0]) / 2.0, 0.0, 0.0])

    # cap step from the radial profile along +x (larger radius near one end)
    proj = v[:, 0]
    rad = np.linalg.norm(v[:, 1:], axis=1)
    bins = 24
    edges = np.linspace(proj.min(), proj.max(), bins + 1)
    prof = np.zeros(bins)
    for i in range(bins):
        sel = (proj >= edges[i]) & (proj <= edges[i + 1])
        if sel.any():
            prof[i] = np.percentile(rad[sel], 80)
    body = np.median(prof[4:-4]) if len(prof) > 8 else np.median(prof)
    cap_step = float(prof.max() - body)

    vol = float(np.pi * (target_d / 2) ** 2 * target_L * FILL)
    mass = vol * DENSITY
    inertia = (np.diag([mass * target_d ** 2 / 8.0,
                        mass * (target_L ** 2 / 12.0 + target_d ** 2 / 16.0),
                        mass * (target_L ** 2 / 12.0 + target_d ** 2 / 16.0)])).tolist()

    tag = f"variant:marker_ready/{name}"
    com_xyz = [round(float(x), 9) for x in geom.surface_centroid(v, f)]
    # radius profile：**截面实测**，s 从 tail（-x）量、沿 +x 增；粗网格中段也不会是 0
    n_prof = 9
    profile = []
    for i in range(n_prof):
        s_ = target_L * (i + 0.5) / n_prof                      # 0..L，从 tail 量起
        x_ = float(v[:, 0].min()) + s_
        profile.append([round(s_, 6), round(_section_radius(v, f, x_), 6)])
    parts = {
        "pen": {
            "origin": {"kind": "point", "p": [0.0, 0.0, 0.0], "tol": None, "src": tag},
            "tip": {"kind": "point", "p": [target_L / 2, 0.0, 0.0],
                    "size": {"height": target_d * 0.25}, "up_axis": "x", "tol": None, "src": tag},
            "tail": {"kind": "point", "p": [-target_L / 2, 0.0, 0.0],
                     "size": {"height": target_d / 2}, "up_axis": "x", "tol": None, "src": tag},
            "body": {"kind": "axis", "p": [0.0, 0.0, 0.0], "axis": [1.0, 0.0, 0.0],
                     "size": {"length": target_L},
                     "span": [0.0, target_L], "profile": profile,
                     "grasp_s": 0.5, "tol": None, "src": tag},
            "com": {"kind": "point", "p": com_xyz, "com_from": "geometry",
                    "tol": None, "src": tag},
            "grasp": {"kind": "axis", "p": [0.0, 0.0, 0.0], "axis": [1.0, 0.0, 0.0],
                      "span": [-0.03, 0.03],
                      "section": {"shape": "circle", "r_out": target_d / 2},
                      "tol": None, "src": tag},
        }
    }
    ann = {
        "schema": "hq-shape-annot/1",
        "parts_schema": "hq-ctx-parts/3",
        "family": "marker",
        "name": name,
        "units": "m",
        "up_axis": "+z",
        "origin": "geometric centre; main axis +x (tip at +x), same as assets/Pen",
        "source": {
            "kind": "ready-made mesh, normalised (PCA + length + radial scale)",
            "base": meta["name"], "license": meta["license"], "author": meta["author"],
            "base_sha256": meta["source_sha256"],
            "base_file": meta["source_file"],
            **({"uid": meta["uid"], "url": meta["url"]} if "uid" in meta else {}),
            "radial_scale": round(float(sr), 6),
            "length_scale": round(float(sx), 6),
        },
        "params": {"d": target_d, "length": target_L, "density": DENSITY, "fill": FILL},
        "derived": {"diameter": target_d, "length": target_L,
                    "cap_step": round(cap_step, 5), "aspect": round(target_L / target_d, 3),
                    "bin": _bin(target_d)},
        "mass": {"volume_m3": vol, "mass_kg": mass, "com": com_xyz,
                 "inertia_kgm2": inertia,
                 "source": "hollow-barrel estimate (fill 0.535), not mesh volume"},
        "bbox": {"min": [-target_L / 2, -target_d / 2, -target_d / 2],
                 "max": [target_L / 2, target_d / 2, target_d / 2],
                 "extent": [target_L, target_d, target_d]},
        "placement": [{"name": "table_rest_lying", "point": [0.0, 0.0, -target_d / 2],
                       "axis": [0.0, 0.0, 1.0], "support": "line along +x"}],
        "parts": parts,
    }
    return v, f, ann, meta


def write_variant(idx, name, base, d, L, out_dir):
    """Write one variant (obj/ply/annotations/variant.yaml) + return its record."""
    v, f, ann, meta = build_variant(name, base, d, L)
    d_out = os.path.join(out_dir, name)
    os.makedirs(d_out, exist_ok=True)
    obj = os.path.join(d_out, name + ".obj")
    ply = os.path.join(d_out, name + ".ply")
    ajs = os.path.join(d_out, name + ".annotations.json")
    geom.write_obj(obj, v, f, header=f"shape_gen/marker_ready {name}")
    geom.write_ply(ply, v, f, header=f"shape_gen/marker_ready {name}")
    gen_sha = geom.sha256_file(os.path.abspath(__file__))
    ann["generator"] = {"file": "marker_ready.py", "version": VERSION, "sha256": gen_sha}
    ann["geometry"] = geom.check_faces(v, f)
    geom.write_json(ajs, ann)
    files = {os.path.basename(p): geom.sha256_file(p) for p in (obj, ply, ajs)}
    variant = {
        "name": name, "fpath": f"marker_ready/{name}", "body_scale": [1.0, 1.0, 1.0],
        "files": files,
        "source": {
            "generator": f"marker_ready.py@{gen_sha[:16]}",
            "base": meta["name"],
            "base_sha256": meta["source_sha256"],
            "license": meta["license"],
            **({"uid": meta["uid"], "url": meta["url"]} if "uid" in meta else {}),
            "params": {"d": d, "length": L},
            "seed": 0,
        },
        "functional_read": {"path": name + ".annotations.json", "key": "parts"},
        "status": "draft",
    }
    geom.write_yaml(os.path.join(d_out, name + ".variant.yaml"), variant)
    return {"index": idx, "variant": name, "d": d, "length": L,
            "bin": ann["derived"]["bin"], "mass_kg": ann["mass"]["mass_kg"],
            "base": meta["name"], "license": meta["license"], "author": meta["author"],
            "base_sha256": meta["source_sha256"], "files": files,
            "dir": d_out, "mesh": (v, f)}


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(IMAGES, exist_ok=True)
    gen_sha = geom.sha256_file(os.path.abspath(__file__))
    rows, tiles = [], []
    for idx, (name, base, d, L) in enumerate(VARIANTS, start=1):
        rec = write_variant(idx, name, base, d, L, OUT)
        v, f = rec.pop("mesh")
        print(f"{idx:02d} {name:26s} d={d*1e3:4.1f} L={L*1e3:5.1f} bin={rec['bin']:11s} "
              f"m={rec['mass_kg']*1e3:5.1f} g faces={len(f)}")
        eye, tgt, fov = render.frame(v, direction=(0.28, -0.86, 0.36), fill=1.02)
        img = render.render(v, f, size=380, eye=eye, target=tgt, fov=fov,
                            base_color=(0.30, 0.31, 0.35), floor_half=0.9, floor_tiles=8)
        cap = f"{idx:02d} D{d*1e3:.0f} L{L*1e3:.0f} {rec['bin'].split()[0]}"
        h, w, _ = img.shape
        tw, th = render.text_size(cap, scale=2)
        tile = np.full((h + th + 18, max(w, tw + 16), 3), 255, np.uint8)
        tile[8:8 + h, (tile.shape[1] - w) // 2:(tile.shape[1] - w) // 2 + w] = img
        render.draw_text(tile, (tile.shape[1] - tw) // 2, h + 10, cap, scale=2)
        tiles.append(tile)
        rows.append(rec)

    sheet = render.montage(tiles, cols=5, pad=8, backdrop=250)
    png = os.path.join(IMAGES, "马克笔_粗细分档_对比图.png")
    render.write_png(png, sheet)
    with open(os.path.join(OUT, "index.json"), "w", encoding="utf-8") as fh:
        json.dump({"bins": BINS, "generator_sha256": gen_sha, "rows": rows, "sheet": png},
                  fh, indent=1, ensure_ascii=False)
    print("sheet ->", png)
    return 0


if __name__ == "__main__":
    sys.exit(main())
