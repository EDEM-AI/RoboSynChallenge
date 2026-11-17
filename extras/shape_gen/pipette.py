"""Generator: pipette (family "pipette", 移液 manipulate_pipette) - simplified.

No external model exists for this, so this is a *simplified*
parametric body: barrel + grip + hook + conical tip + plunger button.  It is
meant to carry the geometric and functional relations (grip width, button
stroke, tip position), not to reproduce the real ergonomics.

Frame: geometric centre, main axis +x, tip at -L/2 (P tip at
about -184 mm), button at +L/2.

Parameters / provenance: root envelope 255 x 28 x 65 mm, S scale
1..1.1 -> 255-280.5 / 28-30.8 / 65-71.5; button joint (URDF) range [-2, 4.67] mm,
span 6.67 mm; beaker 90 x 84 x 109 (P/S).
"""

import math
import os

import geom

FAMILY = "pipette"
VERSION = "1"
DENSITY = 1150.0        # ABS-ish body


def default_params():
    return {
        "L": 0.255,             # 根包络 255（S 上限 280.5）
        "body_w": 0.028,        # 两桶 28…30.8
        "body_h": 0.028,
        "hook_h": 0.065,        # 总高 65…71.5（挂钩顶）
        "grip_len": 0.090,      # 握段长（未实测，冻结）
        "tip_len": 0.018,       # 锥形尖段
        "tip_r": 0.0010,
        "button_r": 0.0060,     # 按钮顶直径 12（未实测，工程值）
        "button_len": 0.010,
        "stroke": 0.00667,      # 6.67 mm = URDF span [-2, 4.67]
        "density": DENSITY,
        "grip_shape": "round",  # round / ellipse（第 8 节：圆/椭圆握柄两族）
    }


def derive(p):
    bh = p["body_w"] * 0.78 if p.get("grip_shape") == "ellipse" else p["body_h"]
    return {
        "body_h_effective": bh,
        "grip_section": ({"shape": "ellipse", "a_out": p["body_w"] / 2.0, "b_out": bh / 2.0}
                         if p.get("grip_shape") == "ellipse"
                         else {"shape": "circle", "r_out": p["body_w"] / 2.0}),
        "height_semantics": "hook_h = total height from the bottom of the barrel to the top of the hook",
        "barrel_len": p["L"] - p["tip_len"] - p["button_len"],
        "tip_x": -p["L"] / 2.0,
        "button_top_x": p["L"] / 2.0,
        "stroke_mm": p["stroke"] * 1e3,
    }


def check_relations(p):
    d = derive(p)
    out = []

    def req(ok, rid, text):
        out.append({"id": rid, "ok": bool(ok), "text": text})
        if not ok:
            raise ValueError(f"[pipette] relation {rid} failed: {text}")

    # R1 握径进得了夹爪（<= 85 mm）；握径 = 杆体宽度（椭圆族跟着 body_w 走）
    req(p["body_w"] <= 0.085, "R1", f"grip w={p['body_w']*1e3:.1f} mm <= 85")
    # R2 根包络在名义范围内（255…280.5）
    req(0.253 <= p["L"] <= 0.2805 + 1e-9, "R2", f"L={p['L']*1e3:.1f} mm in [253, 280.5]")
    req(0.028 - 1e-9 <= p["body_w"] <= 0.0308 + 1e-9, "R2b",
        f"body width={p['body_w']*1e3:.1f} mm in [28, 30.8]")
    req(0.065 - 1e-9 <= p["hook_h"] <= 0.0715 + 1e-9, "R2c",
        f"hook height={p['hook_h']*1e3:.1f} mm in [65, 71.5]")
    # R3 按钮行程 = URDF span（6.67 mm），且按钮顶在杆端外
    req(abs(p["stroke"] - 0.00667) < 1e-6, "R3", f"stroke={p['stroke']*1e3:.2f} mm == 6.67 (URDF)")
    # R4 尖比杆细、且尖在杆端之外
    req(p["tip_r"] < p["body_w"] / 2.0, "R4", "tip radius < barrel radius")
    # R5 握段不压到尖端
    req(p["grip_len"] + p["tip_len"] < p["L"], "R5", "grip + tip shorter than the body")
    return out


def build(p):
    d = derive(p)
    L, bw = p["L"], p["body_w"]
    bh = bw * 0.78 if p.get("grip_shape") == "ellipse" else p["body_h"]
    # barrel: revolve along +z then rotate to +x (same trick as the marker)
    x0 = -L / 2.0 + p["tip_len"]
    x1 = L / 2.0 - p["button_len"]
    m = geom.Mesh()
    prof = [(0.0, 0.0), (bw / 2.0, 0.0), (bw / 2.0, x1 - x0), (0.0, x1 - x0)]
    geom.revolve_profile(m, "barrel", prof, n=48, scale=(bh / bw, 1.0, 1.0))
    R = [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
    m.transform("barrel", R=R, t=[x0, 0.0, 0.0])

    # conical tip pointing -x
    tip_prof = [(0.0, 0.0), (p["tip_r"], 0.0), (bw / 2.0, p["tip_len"]), (0.0, p["tip_len"])]
    geom.revolve_profile(m, "tip", tip_prof, n=40, scale=(bh / bw, 1.0, 1.0))
    m.transform("tip", R=R, t=[-L / 2.0, 0.0, 0.0])

    # plunger button (+x)
    btn_prof = [(0.0, 0.0), (p["button_r"], 0.0), (p["button_r"], p["button_len"]),
                (0.0, p["button_len"])]
    geom.revolve_profile(m, "button", btn_prof, n=32)
    m.transform("button", R=R, t=[L / 2.0 - p["button_len"], 0.0, 0.0])

    # hook: a small box on top near the button end (raises the asset to hook_h)
    # hook top = hook_h - bh/2 so the total height (bottom -bh/2 .. top) == hook_h
    z_top = p["hook_h"] - bh / 2.0
    geom.box(m, "hook", L / 2.0 - 0.055, L / 2.0 - 0.010, -0.004, 0.004, bh / 2.0 - 0.004, z_top)
    return m, d


def parts_ctx(p, d, name, com=None):
    src = f"variant:pipette/{name}"
    L = p["L"]
    return {"pipette": {
        "origin": {"kind": "point", "p": [0.0, 0.0, 0.0], "tol": None, "src": src},
        "com": {"kind": "point", "p": list(com) if com is not None else [0.0, 0.0, 0.0],
                "com_from": "mass" if com is not None else "geometry", "tol": None, "src": src},
        "tip": {"kind": "point", "p": [-L / 2.0, 0.0, 0.0], "tol": None, "src": src},
        # 按钮 = 棱柱关节 + 顶面（第 3 版「joint + disk」）
        "button": {"kind": "joint", "q_name": "press", "axis": [1.0, 0.0, 0.0],
                   "range": [-0.0020, 0.00467], "travel": p["stroke"], "unit": "m",
                   "p": [L / 2.0, 0.0, 0.0], "link": "pipette_button_link",
                   "section": {"shape": "circle", "r_out": p["button_r"]},
                   "tol": None, "src": src},
        "grasp": {"kind": "axis", "p": [0.0, 0.0, 0.0], "axis": [1.0, 0.0, 0.0],
                  "span": [-p["grip_len"] / 2.0, p["grip_len"] / 2.0],
                  "section": dict(d["grip_section"]),
                  "free_dirs": [[0.0, 2.0 * math.pi]], "tol": None, "src": src},
    }}


def generate(params, seed=0, out_dir=None):
    p = dict(params)
    rel = check_relations(p)
    d = derive(p)
    m, _ = build(p)
    verts, faces = m.welded_parts()
    props = geom.mass_properties(verts, faces, p["density"])
    lo, hi = verts.min(0), verts.max(0)
    name = p.get("name") or f"pipette_L{int(round(p['L']*1e3))}_g{int(round(p['body_w']*2e3))}"
    ann = {
        "schema": "hq-shape-annot/1",
        "parts_schema": "hq-ctx-parts/3",
        "family": FAMILY,
        "name": name,
        "units": "m",
        "up_axis": "+z",
        "origin": "geometric centre; main axis +x, tip at -L/2, button at +L/2",
        "params": dict(p),
        "derived": d,
        "mass": props,
        "bbox": {"min": [float(x) for x in lo], "max": [float(x) for x in hi],
                 "extent": [float(x) for x in (hi - lo)]},
        "placement": [{"name": "table_rest_lying", "point": [0.0, 0.0, float(lo[2])],
                       "axis": [0.0, 0.0, 1.0], "support": "line along +x"}],
        "material": {"density": p["density"]},
        "simplified": True,
        "simplified_note": "parametric simplified body (barrel + grip + hook + tip + button); "
                           "not a reproduction of the real ergonomics",
        "joint_simulated": False,
        "parts": parts_ctx(p, d, name, props["com"]),
        "relations": rel,
    }
    if out_dir is None:
        ann["geometry"] = geom.check_faces(verts, faces)
        return {"name": name, "annotations": ann, "mesh": (verts, faces)}
    return geom.emit_variant(__file__, FAMILY, name, verts, faces, ann, out_dir, seed, VERSION)
