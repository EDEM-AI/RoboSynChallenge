"""Generator: pen holder (family "holder", uid `holder`, 递笔 items_handover).

A straight-walled cup with a solid bottom: the pen stands in it.  The mouth
bore is the part the handover judge reads (the "8 cm overlap" rule), and the
floor part gives the usable depth.

Frame: origin at the bottom centre, +z up, support plane z = 0.  The official
asset `PenHolder/pen_holder.obj` uses +y up; our variant declares +z (the
annotation says so, consumers read the declared up_axis).

Parameters / provenance: official asset measured in this session - outer
92.4 mm, height 96.35 mm, inner opening 85.6 mm, wall ~3.9 mm (with the config
body_scale 0.56/0.45/0.56 applied).  The narrow variant comes from the quota
table (draft: 内口 70 / 高 110).
"""

import math
import os

import geom

FAMILY = "holder"
VERSION = "1"
DENSITY = 1100.0        # 陶瓷 / 硬塑料杂档


def default_params():
    return {
        "mouth_r_in": 0.0428,     # 内口半径（官方 85.6 / 2）
        "H": 0.09635,             # 高（官方实测）
        "wall_t": 0.0039,         # 壁厚（官方 ~3.9）
        "bottom_t": 0.0060,       # 底厚（ENG）
        "taper": 0.0,             # 0 = 直壁；正数 = 口更大
        "density": DENSITY,
        "pen_d": 0.01533,         # 配对用的笔径（默认=仿真笔）
        "insert_gap": 0.0060,     # 与笔的净空（2 x items_handover contact_offset）
    }


def derive(p):
    r_in_mouth = p["mouth_r_in"]
    r_in_floor = r_in_mouth / (1.0 + p["taper"]) if p["taper"] > -0.5 else r_in_mouth
    return {"r_in_mouth": r_in_mouth, "r_in_floor": r_in_floor,
            "r_out_mouth": r_in_mouth + p["wall_t"],
            "r_out_floor": r_in_floor + p["wall_t"],
            "usable_depth": p["H"] - p["bottom_t"],
            "pen_clearance": (2 * r_in_mouth - p["pen_d"]) / 2.0}


def check_relations(p):
    d = derive(p)
    out = []

    def req(ok, rid, text):
        out.append({"id": rid, "ok": bool(ok), "text": text})
        if not ok:
            raise ValueError(f"[holder] relation {rid} failed: {text}")

    # H1 笔放得进去（直径 + 净空 <= 内口）
    req(p["pen_d"] + 2 * p["insert_gap"] <= 2 * d["r_in_mouth"] + 1e-12, "H1",
        f"pen {p['pen_d']*1e3:.2f} + 2x{p['insert_gap']*1e3:.1f} <= "
        f"bore {2*d['r_in_mouth']*1e3:.1f} mm")
    # H2 筒高 / 内口在包络内（官方 96.35 / 85.6；窄口档 110 / 70）
    req(0.085 <= p["H"] <= 0.115 + 1e-9, "H2", f"H={p['H']*1e3:.1f} mm in [85, 115]")
    req(0.030 <= p["mouth_r_in"] <= 0.045 + 1e-9, "H3",
        f"bore r={p['mouth_r_in']*1e3:.1f} mm in [30, 45]")
    # H4 壁厚 / 底厚
    req(p["wall_t"] >= 0.0015, "H4a", f"wall={p['wall_t']*1e3:.2f} mm >= 1.5")
    req(p["bottom_t"] >= 0.002, "H4b", f"bottom={p['bottom_t']*1e3:.2f} mm >= 2")
    # H5 可用深度够放笔（与 marker 的「露出 >=30 mm」互补）
    req(d["usable_depth"] >= 0.060, "H5", f"usable depth={d['usable_depth']*1e3:.1f} mm >= 60")
    return out


def build(p):
    d = derive(p)
    rm, rf = d["r_in_mouth"], d["r_in_floor"]
    w, H, bt = p["wall_t"], p["H"], p["bottom_t"]
    prof = [(0.0, 0.0),
            (rf + w, 0.0),
            (rm + w, H),
            (rm, H),
            (rf, bt),
            (0.0, bt)]
    m = geom.Mesh()
    geom.revolve_profile(m, "holder", prof, n=64)
    return m, d


def parts_ctx(p, d, name, com=None):
    src = f"variant:holder/{name}"
    H = p["H"]
    return {"holder": {
        "origin": {"kind": "point", "p": [0.0, 0.0, 0.0], "tol": None, "src": src},
        "com": {"kind": "point", "p": list(com) if com is not None else [0.0, 0.0, 0.0],
                "com_from": "mass" if com is not None else "geometry", "tol": None, "src": src},
        "mouth": {"kind": "disk", "p": [0.0, 0.0, H], "normal": [0.0, 0.0, 1.0],
                  "section": {"shape": "circle", "r_in": round(d["r_in_mouth"], 6),
                              "r_out": round(d["r_out_mouth"], 6)},
                  "tol": None, "src": src},
        "floor": {"kind": "disk", "p": [0.0, 0.0, p["bottom_t"]], "normal": [0.0, 0.0, 1.0],
                  "section": {"shape": "circle", "r_out": round(d["r_in_floor"], 6)},
                  "tol": None, "src": src},
        "bottom": {"kind": "disk", "p": [0.0, 0.0, 0.0], "normal": [0.0, 0.0, -1.0],
                   "section": {"shape": "circle", "r_out": round(d["r_out_floor"], 6)},
                   "tol": None, "src": src},
        "body": {"kind": "axis", "p": [0.0, 0.0, H / 2.0], "axis": [0.0, 0.0, 1.0],
                 "size": {"length": H}, "span": [0.0, H],
                 "section": {"shape": "circle", "r_out": round(d["r_out_mouth"], 6)},
                 "profile": [[0.0, round(d["r_out_floor"], 6)],
                             [round(H, 6), round(d["r_out_mouth"], 6)]],
                 "tol": None, "src": src},
    }}


def generate(params, seed=0, out_dir=None):
    p = dict(params)
    rel = check_relations(p)
    d = derive(p)
    m, _ = build(p)
    verts, faces = m.welded(eps=1e-9)
    props = geom.mass_properties(verts, faces, p["density"])
    lo, hi = verts.min(0), verts.max(0)
    name = p.get("name") or f"holder_bore{int(round(2*p['mouth_r_in']*1e3))}_h{int(round(p['H']*1e3))}"
    ann = {
        "schema": "hq-shape-annot/1",
        "parts_schema": "hq-ctx-parts/3",
        "family": FAMILY,
        "name": name,
        "units": "m",
        "up_axis": "+z",
        "origin": "bottom centre; support plane z=0; axis +z "
                  "(official pen_holder.obj is +y up - the annotation declares the frame)",
        "params": dict(p),
        "derived": d,
        "mass": props,
        "bbox": {"min": [float(x) for x in lo], "max": [float(x) for x in hi],
                 "extent": [float(x) for x in (hi - lo)]},
        "placement": [{"name": "table_rest", "point": [0.0, 0.0, 0.0], "axis": [0.0, 0.0, 1.0]}],
        "material": {"density": p["density"]},
        "parts": parts_ctx(p, d, name, props["com"]),
        "relations": rel,
    }
    if out_dir is None:
        ann["geometry"] = geom.check_faces(verts, faces)
        return {"name": name, "annotations": ann, "mesh": (verts, faces)}
    return geom.emit_variant(__file__, FAMILY, name, verts, faces, ann, out_dir, seed, VERSION)
