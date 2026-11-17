"""Generator: milk carton (family "milk", uid `milk`, 提篮 handle_basket) - simplified.

A rectangular carton with a gable top (the classic milk pack).  Simplified:
the gable is a plain triangular prism, no folded seam.

Frame: origin at the bottom centre, +z up, support plane z = 0.

Parameters / provenance: width 43…49.45 (reference value 54 is X only),
depth 45…51.75, height 98…112.7; mass 0.02…0.11 kg.
"""

import math
import os

import geom

FAMILY = "milk"
VERSION = "1"
DENSITY = 950.0
PAPER_D = 780.0


def default_params():
    return {
        "W": 0.046,          # 宽（43…49.45）
        "D": 0.048,          # 深（45…51.75）
        "H": 0.105,          # 总高（98…112.7）
        "top_mode": "gable", # gable（山墙顶）/ flat（平顶方盒）
        "gable_h": 0.018,    # 顶部山墙高（ENG；flat 时不用）
        "ridge_h": 0.004,    # flat 时顶部封口棱高
        "wall_t": 0.0012,
        "density": PAPER_D,
    }


def derive(p):
    if p.get("top_mode", "gable") == "flat":
        return {"body_h": p["H"] - p["ridge_h"],
                "top_z": p["H"], "apex_z": p["H"],
                "top_mode": "flat", "ridge_h": p["ridge_h"]}
    return {"body_h": p["H"] - p["gable_h"],
            "top_z": p["H"],
            "apex_z": p["H"], "top_mode": "gable"}


def check_relations(p):
    d = derive(p)
    out = []

    def req(ok, rid, text):
        out.append({"id": rid, "ok": bool(ok), "text": text})
        if not ok:
            raise ValueError(f"[milk] relation {rid} failed: {text}")

    req(0.043 <= p["W"] <= 0.0495 + 1e-9, "M1", f"W={p['W']*1e3:.1f} mm in [43, 49.5]")
    req(0.045 <= p["D"] <= 0.0518 + 1e-9, "M2", f"D={p['D']*1e3:.1f} mm in [45, 51.8]")
    req(0.098 <= p["H"] <= 0.1127 + 1e-9, "M3", f"H={p['H']*1e3:.1f} mm in [98, 112.7]")
    if p.get("top_mode", "gable") == "gable":
        req(0.008 <= p["gable_h"] <= 0.030, "M4", f"gable={p['gable_h']*1e3:.1f} mm in [8, 30]")
    else:
        req(0.002 <= p["ridge_h"] <= 0.006, "M4f", f"ridge={p['ridge_h']*1e3:.1f} mm in [2, 6]")
    req(d["body_h"] > 0.060, "M5", "body taller than 60 mm")
    return out


def build(p):
    W, D, H = p["W"], p["D"], p["H"]
    m = geom.Mesh()
    if p.get("top_mode", "gable") == "flat":
        rh = p["ridge_h"]
        body_h = H - rh
        geom.box(m, "body", -W / 2, W / 2, -D / 2, D / 2, 0.0, body_h)
        # 平顶封口棱：沿 x 的一道薄脊（平顶方盒的样子）
        geom.box(m, "seal_ridge", -W / 2, W / 2, -0.0012, 0.0012, body_h, H)
        return m, derive(p)
    gh = p["gable_h"]
    body_h = H - gh
    geom.box(m, "body", -W / 2, W / 2, -D / 2, D / 2, 0.0, body_h)
    # gable: a triangular prism (ridge along x)
    m.begin("gable")
    ridge_y = 0.0
    base_a = [-W / 2, -D / 2, body_h]
    base_b = [W / 2, -D / 2, body_h]
    base_c = [W / 2, D / 2, body_h]
    base_d = [-W / 2, D / 2, body_h]
    top_a = [-W / 2, ridge_y, H]
    top_b = [W / 2, ridge_y, H]
    b = m.add_vertices([base_a, base_b, base_c, base_d, top_a, top_b])
    # two slopes
    m.quad(b + 0, b + 1, b + 5, b + 4)      # front slope
    m.quad(b + 3, b + 4, b + 5, b + 2)      # back slope
    # two gable ends (triangles)
    m.add_face(b + 0, b + 4, b + 3)
    m.add_face(b + 1, b + 2, b + 5)
    # bottom of the gable (closes the prism against the body top)
    m.quad(b + 0, b + 3, b + 2, b + 1)
    m.end()
    return m, derive(p)


def parts_ctx(p, d, name, com=None):
    src = f"variant:milk/{name}"
    W, D, H = p["W"], p["D"], p["H"]
    return {"milk": {
        "origin": {"kind": "point", "p": [0.0, 0.0, 0.0], "tol": None, "src": src},
        "com": {"kind": "point", "p": list(com) if com is not None else [0.0, 0.0, 0.0],
                "com_from": "mass" if com is not None else "geometry", "tol": None, "src": src},
        "bottom": {"kind": "disk", "p": [0.0, 0.0, 0.0], "normal": [0.0, 0.0, -1.0],
                   "section": {"shape": "rect", "w": W, "d": D, "in_or_out": "out"},
                   "tol": None, "src": src},
        "top": {"kind": "disk", "p": [0.0, 0.0, H], "normal": [0.0, 0.0, 1.0],
                "section": {"shape": "rect", "w": W,
                            "d": (D if p.get("top_mode", "gable") == "flat" else 0.0),
                            "in_or_out": "out"},
                "top_mode": p.get("top_mode", "gable"),
                "gable_h": (p["gable_h"] if p.get("top_mode", "gable") == "gable" else 0.0),
                "tol": None, "src": src},
        "grasp": {"kind": "axis", "p": [0.0, 0.0, d["body_h"] / 2.0], "axis": [0.0, 0.0, 1.0],
                  "span": [0.004, d["body_h"] - 0.004],
                  "section": {"shape": "rect", "w": W, "d": D, "in_or_out": "out"},
                  "free_dirs": [[0.0, 2.0 * math.pi]], "tol": None, "src": src},
    }}


def generate(params, seed=0, out_dir=None):
    p = dict(params)
    rel = check_relations(p)
    m, d = build(p)
    verts, faces = m.welded_parts()
    props = geom.mass_properties(verts, faces, p["density"])
    lo, hi = verts.min(0), verts.max(0)
    name = p.get("name") or f"milk_{int(round(p['W']*1e3))}x{int(round(p['H']*1e3))}"
    ann = {
        "schema": "hq-shape-annot/1",
        "parts_schema": "hq-ctx-parts/3",
        "family": FAMILY,
        "name": name,
        "units": "m",
        "up_axis": "+z",
        "origin": "bottom centre; support plane z=0; +z up",
        "params": dict(p),
        "derived": d,
        "mass": props,
        "bbox": {"min": [float(x) for x in lo], "max": [float(x) for x in hi],
                 "extent": [float(x) for x in (hi - lo)]},
        "placement": [{"name": "table_rest", "point": [0.0, 0.0, 0.0], "axis": [0.0, 0.0, 1.0]}],
        "material": {"density": p["density"]},
        "simplified": True,
        "simplified_note": "方盒 + 三角山墙（不做折痕/封口）",
        "parts": parts_ctx(p, d, name, props["com"]),
        "relations": rel,
    }
    if out_dir is None:
        ann["geometry"] = geom.check_faces(verts, faces)
        return {"name": name, "annotations": ann, "mesh": (verts, faces)}
    return geom.emit_variant(__file__, FAMILY, name, verts, faces, ann, out_dir, seed, VERSION)
