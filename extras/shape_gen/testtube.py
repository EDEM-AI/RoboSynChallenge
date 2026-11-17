"""Generator: test tube (family "tube", uid `cube`, 上样 sample_loading).

One revolved solid: round (or flat) closed bottom, straight wall with an
optional taper, and a rolled rim at the mouth.  Pairs with the rack: the outer
diameter must pass the rack's cell with the sampled clearance (see
rack.tube_for), so the two are always drawn together.

Frame: origin at the bottom centre, +z up, support plane z = 0 (a tube lying on
the table rests on its side, but the simulation places it upright in the rack -
the parts are in the object frame either way).

Parameters / provenance: outer diameter 17 … 24.48 mm (P measured
16.78, S scale 1…1.44), length 195 mm nominal (the S whole-body scale is split
into diameter and length here); wall thickness not measured -> engineering.
"""

import math
import os

import geom

FAMILY = "tube"
VERSION = "1"
GLASS_D = 1200.0        # 硼硅玻璃 ~2230；塑料试管 ~1200（仿真资产材质未标）


def default_params():
    return {
        "d": 0.01678,           # 外径（P 实测 16.78）
        "L": 0.19503,           # 长（P 实测 195.03）
        "wall_t": 0.0010,       # 壁厚（未测 → ENG）
        "bottom": "round",      # round / flat
        "taper": 0.0,           # 口/底外径差（0 = 直筒），正数 = 口更粗
        "rim_h": 0.0020,        # 口沿翻边高度
        "density": GLASS_D,
    }


def derive(p):
    d, L = p["d"], p["L"]
    r = d / 2.0
    r_mouth = r * (1.0 + p["taper"])
    return {"r_body": r, "r_mouth": r_mouth,
            "r_in_body": max(r - p["wall_t"], 0.0005),
            "r_in_mouth": max(r_mouth - p["wall_t"], 0.0005),
            "h": L}


def check_relations(p):
    d = derive(p)
    out = []

    def req(ok, rid, text):
        out.append({"id": rid, "ok": bool(ok), "text": text})
        if not ok:
            raise ValueError(f"[tube] relation {rid} failed: {text}")

    # U1 外径在名义范围内
    req(0.0165 - 1e-9 <= p["d"] <= 0.0245 + 1e-9, "U1",
        f"d={p['d']*1e3:.2f} mm in [16.5, 24.5]（P 实测 16.78）")
    # U2 长度：名义 195，短档到 120（配额表：Ø20×150 / Ø24.5×120）
    req(0.115 <= p["L"] <= 0.200 + 1e-9, "U2", f"L={p['L']*1e3:.1f} mm in [115, 200]")
    # U3 壁厚 + 内腔
    req(p["wall_t"] >= 0.0004, "U3a", f"wall={p['wall_t']*1e3:.2f} mm >= 0.4")
    req(d["r_in_mouth"] >= 0.003, "U3b", f"inner mouth r={d['r_in_mouth']*1e3:.2f} mm >= 3")
    # U4 口沿翻边不超过内腔半径（不然口被自己堵死）
    req(p["rim_h"] < d["r_in_mouth"] * 2.0, "U4", "rim smaller than the bore")
    # U5 长径比：试管细长（>= 4），避免做成杯子
    req(p["L"] / p["d"] >= 4.0, "U5", f"aspect={p['L']/p['d']:.1f} >= 4")
    return out


def build(p):
    d = derive(p)
    L, rb, rm = p["L"], d["r_body"], d["r_mouth"]
    t = p["wall_t"]
    rim = p["rim_h"]
    if p["bottom"] == "round":
        # profile: bottom pole -> round bottom -> wall -> rim -> inner wall -> inner bottom
        prof = [(0.0, 0.0),
                (rb * 0.55, 0.0),
                (rb * 0.90, rb * 0.35),
                (rb, rb * 0.75),
                (rm, L - rim),
                (rm, L),
                (rm - t, L),
                (rm - t, L - rim),
                (rb - t, rb * 0.75),
                (max(rb - t - 0.001, 0.0005), rb * 0.40),
                (0.0, rb * 0.40)]
    else:
        prof = [(0.0, 0.0),
                (rb * 0.92, 0.0),
                (rb, 0.0015),
                (rm, L - rim),
                (rm, L),
                (rm - t, L),
                (rm - t, L - rim),
                (rb - t, 0.0020),
                (0.0, 0.0020)]
    m = geom.Mesh()
    geom.revolve_profile(m, "tube", prof, n=48)
    return m, d


def parts_ctx(p, d, name, com=None):
    src = f"variant:tube/{name}"
    L = p["L"]
    return {"cube": {
        "origin": {"kind": "point", "p": [0.0, 0.0, 0.0], "tol": None, "src": src},
        "com": {"kind": "point", "p": list(com) if com is not None else [0.0, 0.0, 0.0],
                "com_from": "mass" if com is not None else "geometry", "tol": None, "src": src},
        "bottom": {"kind": "disk", "p": [0.0, 0.0, 0.0], "normal": [0.0, 0.0, -1.0],
                   "section": {"shape": "circle", "r_out": round(d["r_body"], 6)},
                   "tol": None, "src": src},
        "mouth": {"kind": "disk", "p": [0.0, 0.0, L], "normal": [0.0, 0.0, 1.0],
                  "section": {"shape": "circle", "r_out": round(d["r_mouth"], 6),
                              "r_in": round(d["r_in_mouth"], 6)},
                  "tol": None, "src": src},
        "body": {"kind": "axis", "p": [0.0, 0.0, L / 2.0], "axis": [0.0, 0.0, 1.0],
                 "size": {"length": L}, "span": [0.0, L],
                 "section": {"shape": "circle", "r_out": round(d["r_body"], 6)},
                 "profile": [[0.0, round(d["r_body"], 6)],
                             [round(L - p["rim_h"], 6), round(d["r_mouth"], 6)],
                             [round(L, 6), round(d["r_mouth"], 6)]],
                 "grasp_s": 0.15, "tol": None, "src": src},
    }}


def generate(params, seed=0, out_dir=None):
    p = dict(params)
    rel = check_relations(p)
    d = derive(p)
    m, _ = build(p)
    verts, faces = m.welded(eps=1e-9)
    props = geom.mass_properties(verts, faces, p["density"])
    lo, hi = verts.min(0), verts.max(0)
    name = p.get("name") or f"tube_d{int(round(p['d']*1e4))}_l{int(round(p['L']*1e3))}"
    ann = {
        "schema": "hq-shape-annot/1",
        "parts_schema": "hq-ctx-parts/3",
        "family": FAMILY,
        "name": name,
        "units": "m",
        "up_axis": "+z",
        "origin": "bottom centre; support plane z=0; axis +z",
        "params": dict(p),
        "derived": d,
        "mass": props,
        "bbox": {"min": [float(x) for x in lo], "max": [float(x) for x in hi],
                 "extent": [float(x) for x in (hi - lo)]},
        "placement": [{"name": "table_rest_upright", "point": [0.0, 0.0, 0.0],
                       "axis": [0.0, 0.0, 1.0]}],
        "material": {"density": p["density"]},
        "parts": parts_ctx(p, d, name, props["com"]),
        "relations": rel,
    }
    if out_dir is None:
        ann["geometry"] = geom.check_faces(verts, faces)
        return {"name": name, "annotations": ann, "mesh": (verts, faces)}
    return geom.emit_variant(__file__, FAMILY, name, verts, faces, ann, out_dir, seed, VERSION)
