"""Generator: click bell (family "bell", 按铃 click_bell).

Two rigid parts in one mesh: the base drum and the moving cover, plus the
*declared* press joint (range / travel / axis).  The simulation asset is a URDF
with two links; here the geometry is generated and the joint is carried in
ctx.parts so the URDF builder can wire it up.

Frame: origin at the bottom centre of the base, +z up.  Press direction is -z
(the cover moves down), stroke 5-5.5 mm (URDF limit [-5, 0] mm).
"""

import math
import os

import geom

FAMILY = "bell"
VERSION = "1"
DENSITY = 1050.0        # ABS-ish shell


def default_params():
    return {
        "base_r": 0.0315,        # 外径 63 (58–63.8)
        "base_r_in": 0.0230,     # 座是环（盖要能下压进去）
        "base_h": 0.0205,        # 19–20.9
        "cover_r": 0.0220,       # 外径 44 (40–44)
        "cover_t": 0.0095,       # 9–9.9
        "cover_top_z": 0.0305,   # 盖顶偏移 29–31.9
        "button_r": 0.0165,      # 黄色按区半径（黄区 != 全盖）
        "stroke": 0.0050,        # 5–5.5 mm
        "density": DENSITY,
    }


def derive(p):
    return {
        "cover_bottom_z": p["cover_top_z"] - p["cover_t"],
        "press_room": (p["cover_top_z"] - p["cover_t"]) - 0.002,
        "button_r_max": p["cover_r"] - 0.0020,
    }


def check_relations(p):
    d = derive(p)
    out = []

    def req(ok, rid, text):
        out.append({"id": rid, "ok": bool(ok), "text": text})
        if not ok:
            raise ValueError(f"[bell] relation {rid} failed: {text}")

    # K1 the cover is smaller than the base opening so it can drop in
    req(p["cover_r"] <= p["base_r_in"] - 0.0005, "K1a",
        f"cover r={p['cover_r']*1e3:.1f} <= base bore r - 0.5 mm ({p['base_r_in']*1e3:.1f})")
    req(p["base_r_in"] < p["base_r"] - 0.002, "K1b", "base is a ring (has a bore)")
    # K2 行程 5–5.5 mm；压到底时盖沿离桌面还有 >= 2 mm
    req(0.005 - 1e-9 <= p["stroke"] <= 0.0055 + 1e-9, "K2a",
        f"stroke={p['stroke']*1e3:.2f} mm in [5, 5.5]")
    req(d["cover_bottom_z"] - p["stroke"] >= 0.002, "K2b",
        f"cover rim {d['cover_bottom_z']*1e3:.1f} mm - stroke = "
        f"{(d['cover_bottom_z']-p['stroke'])*1e3:.1f} mm >= 2 mm above the table")
    # K3 黄区在盖内，留一圈边
    req(p["button_r"] <= d["button_r_max"] + 1e-12, "K3",
        f"button r={p['button_r']*1e3:.1f} <= cover r - 2 mm")
    # K4 尺寸在名义包络里
    req(0.058 <= 2 * p["base_r"] <= 0.0638 + 1e-9, "K4a", f"base d={2*p['base_r']*1e3:.1f} mm")
    req(0.019 <= p["base_h"] <= 0.0209 + 1e-9, "K4b", f"base h={p['base_h']*1e3:.1f} mm")
    req(0.040 <= 2 * p["cover_r"] <= 0.044 + 1e-9, "K4c", f"cover d={2*p['cover_r']*1e3:.1f} mm")
    req(0.009 <= p["cover_t"] <= 0.0099 + 1e-9, "K4d", f"cover t={p['cover_t']*1e3:.1f} mm")
    req(0.029 <= p["cover_top_z"] <= 0.0319 + 1e-9, "K4e",
        f"cover top z={p['cover_top_z']*1e3:.1f} mm in [29, 31.9]")
    return out


def build(p):
    m = geom.Mesh()
    geom.revolve_profile(m, "base", [(p["base_r_in"], 0.0), (p["base_r"], 0.0),
                                     (p["base_r"], p["base_h"]),
                                     (p["base_r_in"], p["base_h"])], n=64)
    z0 = p["cover_top_z"] - p["cover_t"]
    geom.revolve_profile(m, "cover", [(0.0, z0), (p["cover_r"], z0),
                                      (p["cover_r"], p["cover_top_z"]),
                                      (0.0, p["cover_top_z"])], n=64)
    return m


def parts_ctx(p, name, com=None):
    src = f"variant:bell/{name}"
    return {"button": {
        "origin": {"kind": "point", "p": [0.0, 0.0, 0.0], "tol": None, "src": src},
        "press": {"kind": "joint", "q_name": "press", "axis": [0.0, 0.0, -1.0],
                  "range": [-p["stroke"], 0.0], "travel": p["stroke"], "unit": "m",
                  "link": "button_cover", "tol": None, "src": src},
        # 第 3 版：cover_top 的 section.r_out = 黄区半径（可按区）
        "cover_top": {"kind": "disk", "p": [0.0, 0.0, p["cover_top_z"]],
                      "normal": [0.0, 0.0, 1.0],
                      "section": {"shape": "circle", "r_out": p["button_r"]},
                      "size": {"diameter": 2 * p["cover_r"], "thickness": p["cover_t"]},
                      "link": "button_cover", "tol": None, "src": src},
        "base": {"kind": "disk", "p": [0.0, 0.0, 0.0], "normal": [0.0, 0.0, -1.0],
                 "section": {"shape": "circle", "r_out": p["base_r"]},
                 "tol": None, "src": src},
        "com": {"kind": "point", "p": list(com) if com is not None else [0.0, 0.0, 0.0],
                "com_from": "mass" if com is not None else "geometry",
                "tol": None, "src": src},
    }}


def generate(params, seed=0, out_dir=None):
    p = dict(params)
    rel = check_relations(p)
    d = derive(p)
    m = build(p)
    verts, faces = m.welded_parts()
    props = geom.mass_properties(verts, faces, p["density"])
    name = p.get("name") or f"bell_d{int(round(p['base_r']*2e3))}_s{int(round(p['stroke']*1e4))}"
    ann = {
        "schema": "hq-shape-annot/1",
        "parts_schema": "hq-ctx-parts/3",
        "family": FAMILY,
        "name": name,
        "units": "m",
        "up_axis": "+z",
        "origin": "bottom centre of the base; support plane z=0",
        "params": dict(p),
        "derived": d,
        "mass": props,
        "bbox": {"min": [-p["base_r"], -p["base_r"], 0.0],
                 "max": [p["base_r"], p["base_r"], p["cover_top_z"]],
                 "extent": [2 * p["base_r"], 2 * p["base_r"], p["cover_top_z"]]},
        "placement": [{"name": "table_rest", "point": [0.0, 0.0, 0.0],
                       "axis": [0.0, 0.0, 1.0]}],
        "material": {"density": p["density"]},
        "joint_simulated": False,
        "joint_note": "single rigid mesh; the press joint is declared in parts.press "
                      "for the URDF builder (2 links: base, button_cover)",
        "parts": parts_ctx(p, name, props["com"]),
        "relations": rel,
    }
    if out_dir is None:
        ann["geometry"] = geom.check_faces(verts, faces)
        return {"name": name, "annotations": ann, "mesh": (verts, faces)}
    return geom.emit_variant(__file__, FAMILY, name, verts, faces, ann, out_dir, seed, VERSION)
