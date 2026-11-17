"""Generator: bottle + cup (family "bottle" / "cup", 倒水 water_pouring).

Both are closed profiles revolved around +z, so the shell (wall thickness) is
part of the geometry and the vessels are watertight and hollow.

Frame: origin at the bottom centre, +z up, support plane z = 0 - the same
convention as the rack.  The official criterion "mouth = root + 0.236*y" holds
exactly when H = 236 mm; the mouth part carries the real height so the criterion
can be read instead of hard-coded.

Parameters and provenance: (P/S measured ranges, V for the shape
families).  Transparency (0..1) is a material parameter, not geometry: it is
recorded in the annotation so the render/材质 layer can read it.
"""

import json
import math
import os

import numpy as np

import geom

FAMILY = "bottle"
VERSION = "1"

PET_D = 1350.0          # kg/m3, PET
PP_D = 950.0
PAPER_D = 780.0         # paper cup


# ---------------------------------------------------------------------------
def default_bottle():
    return {
        "H": 0.236,             # 236 nominal, 221.84-240.72
        "body_rx": 0.0339,      # 67.8/2  (P: 宽 67.8)
        "body_ry": 0.0347,      # 69.4/2  (ellipse)
        "bottom_t": 0.0025,
        "wall_t": 0.0004,
        "shoulder_h": 0.040,    # 肩高（未实测，冻结）
        "neck_len": 0.021,
        "mouth_r_out": 0.0085,  # 口外径 17 mm
        "mouth_wall_t": 0.0010,
        "density": PET_D,
        "transmittance": 0.9,   # 0..1
        "shape": "straight_shoulder",
        "sides": 96,            # 96 = 圆；6/8/12 = 棱柱截面（绕 z 的低边数回转）
        "neck_offset": [0.0, 0.0],   # 偏心颈：颈部相对轴线的平移（m）
    }


def default_cup():
    return {
        "H": 0.088,             # 88 fixed
        "mouth_r_out": 0.0280,  # 外径 56 nominal
        "bottom_r_out": 0.0190,
        "wall_t": 0.0012,
        "bottom_t": 0.0016,
        "density": PAPER_D,
        "transmittance": 0.0,
        "shape": "tapered",
    }


# ---------------------------------------------------------------------------
def derive_bottle(p):
    ho = p["mouth_r_out"]
    hi = max(ho - p["mouth_wall_t"], 0.0005)
    z_neck = p["H"] - 0.004
    z_shoulder_end = z_neck - p["neck_len"]
    z_shoulder_start = z_shoulder_end - p["shoulder_h"]
    return {"mouth_r_out": ho, "mouth_r_in": hi,
            "z_shoulder_start": z_shoulder_start, "z_shoulder_end": z_shoulder_end,
            "z_neck": z_neck}


def check_relations_bottle(p):
    d = derive_bottle(p)
    out = []

    def req(ok, rid, text):
        out.append({"id": rid, "ok": bool(ok), "text": text})
        if not ok:
            raise ValueError(f"[bottle] relation {rid} failed: {text}")

    # B1 口要有洞、壁要有肉
    req(d["mouth_r_in"] >= 0.003, "B1a", f"mouth r_in={d['mouth_r_in']*1e3:.2f} mm >= 3")
    req(p["wall_t"] >= 0.0004, "B1b", f"wall={p['wall_t']*1e3:.2f} mm >= 0.4")
    # B2 可以抓：瓶身外径 <= 夹爪 85 mm
    req(2 * max(p["body_rx"], p["body_ry"]) <= 0.085, "B2",
        f"body {2*max(p['body_rx'],p['body_ry'])*1e3:.1f} mm <= 85 mm gripper")
    # B3 肩/颈不互相吃穿
    req(d["z_shoulder_start"] > p["bottom_t"] + 0.02, "B3",
        f"shoulder starts at {d['z_shoulder_start']*1e3:.1f} mm above the bottom")
    req(d["z_neck"] - d["z_shoulder_end"] >= 0.010, "B3b",
        f"neck length={p['neck_len']*1e3:.1f} mm >= 10")
    # B4 尺寸范围
    req(0.2218 - 1e-9 <= p["H"] <= 0.2408 + 1e-9, "B4",
        f"H={p['H']*1e3:.1f} mm in [221.8, 240.7]")
    req(0.0159 - 1e-9 <= 2 * p["mouth_r_out"] <= 0.0174 + 1e-9, "B5",
        f"mouth outer={2*p['mouth_r_out']*1e3:.2f} mm in [15.9, 17.4]")
    # B6 站得住：底面半径 > 0，质心投影落在底面上
    # B9 棱柱边数 / 偏心量有界
    req(int(p.get("sides", 96)) in (6, 8, 12, 96), "B9", f"sides={p.get('sides', 96)} in {{6,8,12,96}}")
    ox, oy = p.get("neck_offset", (0.0, 0.0))
    req(math.hypot(ox, oy) <= 0.20 * p["body_rx"], "B10",
        f"neck offset={math.hypot(ox,oy)*1e3:.1f} mm <= 20% of body radius")
    req(p["body_rx"] > 0 and p["body_ry"] > 0, "B6", "body radii positive")
    # B7 质量（PET 轻端 0.0065–0.016，整档 0.01–0.1）
    area = 2 * math.pi * 0.034 * (p["H"] - p["shoulder_h"]) + 2 * math.pi * 0.02 * p["shoulder_h"]
    mass = area * p["wall_t"] * p["density"] + math.pi * p["body_rx"] * p["body_ry"] * p["bottom_t"] * p["density"]
    req(0.0 <= p["transmittance"] <= 1.0, "B8",
        f"transmittance={p['transmittance']:.2f} in [0,1]")
    req(0.004 <= mass <= 0.12, "B7",
        f"mass≈{mass*1e3:.0f} g in [4,120] g (PET light end 6.5–16 g)")
    return out


def check_relations_cup(p):
    d = {"mouth_r_in": max(p["mouth_r_out"] - p["wall_t"], 0.0005)}
    out = []

    def req(ok, rid, text):
        out.append({"id": rid, "ok": bool(ok), "text": text})
        if not ok:
            raise ValueError(f"[cup] relation {rid} failed: {text}")

    req(2 * p["mouth_r_out"] >= 0.056 - 1e-9, "C1a",
        f"mouth outer={2*p['mouth_r_out']*1e3:.1f} mm >= 56")
    req(2 * p["mouth_r_out"] <= 0.07448 + 1e-9, "C1b",
        f"mouth outer={2*p['mouth_r_out']*1e3:.1f} mm <= 74.5")
    req(p["bottom_r_out"] < p["mouth_r_out"], "C2", "taper: bottom < mouth")
    req(d["mouth_r_in"] >= 0.004, "C3", f"mouth r_in={d['mouth_r_in']*1e3:.2f} mm >= 4")
    req(abs(p["H"] - 0.088) < 1e-9, "C4", f"H={p['H']*1e3:.1f} mm == 88 (fixed)")
    return out


# ---------------------------------------------------------------------------
def build_bottle(p):
    d = derive_bottle(p)
    H = p["H"]
    rx, ry = p["body_rx"], p["body_ry"]
    t = p["wall_t"]
    bt = p["bottom_t"]
    prof = [
        (0.0, 0.0),                                   # bottom outer centre
        (rx * 0.92, 0.0),                             # bottom outer edge
        (rx, 0.004),
        (rx, d["z_shoulder_start"]),
        (rx * 0.62, d["z_shoulder_end"]),             # shoulder
        (d["mouth_r_out"] * 1.25, d["z_shoulder_end"] + 0.006),
        (d["mouth_r_out"], d["z_neck"]),
        (d["mouth_r_out"], H),
        (d["mouth_r_in"], H),                         # rim
        (d["mouth_r_in"], d["z_neck"]),
        (d["mouth_r_out"] * 1.25 - t * 1.4, d["z_shoulder_end"] + 0.006),
        (rx * 0.62 - t * 1.4, d["z_shoulder_end"]),
        (rx - t, d["z_shoulder_start"]),
        (rx - t, 0.005),
        (rx - t - 0.001, 0.0 + bt),                   # inner bottom edge
        (0.0, bt),                                    # inner bottom centre
    ]
    m = geom.Mesh()
    geom.revolve_profile(m, "bottle", prof, n=int(p.get("sides", 96)),
                         scale=(1.0, ry / rx, 1.0))
    # 偏心颈：肩上以线性比例把顶点推向目标偏移（连续变形，保持水密）
    ox, oy = p.get("neck_offset", (0.0, 0.0))
    if abs(ox) > 1e-12 or abs(oy) > 1e-12:
        z0 = d["z_shoulder_start"]
        for i, v in enumerate(m.v):
            if v[2] > z0:
                f = (v[2] - z0) / max(H - z0, 1e-9)
                m.v[i][0] = v[0] + ox * f
                m.v[i][1] = v[1] + oy * f
    return m, d


def build_cup(p):
    H = p["H"]
    ro, bo, t, bt = p["mouth_r_out"], p["bottom_r_out"], p["wall_t"], p["bottom_t"]
    prof = [
        (0.0, 0.0),
        (bo, 0.0),
        (bo + t, bt),
        (ro, H),
        (ro - t, H),
        (bo, bt),
        (0.0, bt),
    ]
    m = geom.Mesh()
    geom.revolve_profile(m, "cup", prof, n=64)
    return m, {}


# ---------------------------------------------------------------------------
def parts_bottle(p, d, name, com=None):
    H = p["H"]
    src = f"variant:bottle/{name}"
    return {
        "bottle": {
            "origin": {"kind": "point", "p": [0.0, 0.0, 0.0], "tol": None, "src": src},
            "bottom": {"kind": "disk", "p": [0.0, 0.0, 0.0], "normal": [0.0, 0.0, -1.0],
                       "section": {"shape": "ellipse", "a_out": p["body_rx"], "b_out": p["body_ry"]},
                       "tol": None, "src": src},
            "com": {"kind": "point", "p": list(com) if com is not None else [0.0, 0.0, 0.0],
                    "com_from": "mass" if com is not None else "geometry",
                    "tol": None, "src": src},
            "mouth": {"kind": "disk",
                      "p": [round(float(p.get("neck_offset", (0, 0))[0]), 6),
                            round(float(p.get("neck_offset", (0, 0))[1]), 6), H],
                      "normal": [0.0, 0.0, 1.0],
                      "section": {"shape": "circle", "r_in": d["mouth_r_in"],
                                  "r_out": p["mouth_r_out"]},
                      "tol": None, "src": src},
            "body": {"kind": "axis", "p": [0.0, 0.0, H / 2], "axis": [0.0, 0.0, 1.0],
                     "span": [0.0, H],
                     "profile": [[0.0, p["body_rx"]],
                                 [d["z_shoulder_start"], p["body_rx"]],
                                 [d["z_shoulder_end"], p["body_rx"] * 0.62],
                                 [d["z_neck"], p["mouth_r_out"]],
                                 [H, p["mouth_r_out"]]],
                     "neck_offset": [float(p.get("neck_offset", (0.0, 0.0))[0]),
                                     float(p.get("neck_offset", (0.0, 0.0))[1])],
                     "profile_axis": "x (r_out = body_rx; the body is elliptical, "
                                     "y semi-axis = rx * body_ry/body_rx)",
                     "tol": None, "src": src},
            "grasp": {"kind": "axis", "p": [0.0, 0.0, H * 0.5], "axis": [0.0, 0.0, 1.0],
                      "span": [p["bottom_t"] + 0.01, float(d["z_shoulder_start"])],
                      "section": {"shape": "ellipse", "a_out": p["body_rx"], "b_out": p["body_ry"]},
                      "free_dirs": [[0.0, 2.0 * math.pi]], "tol": None, "src": src},
        }
    }


def parts_cup(p, name, com=None):
    H = p["H"]
    src = f"variant:cup/{name}"
    r_in = p["mouth_r_out"] - p["wall_t"]
    return {
        "cup": {
            "origin": {"kind": "point", "p": [0.0, 0.0, 0.0], "tol": None, "src": src},
            "bottom": {"kind": "disk", "p": [0.0, 0.0, 0.0], "normal": [0.0, 0.0, -1.0],
                       "section": {"shape": "circle", "r_out": p["bottom_r_out"]},
                       "tol": None, "src": src},
            "com": {"kind": "point", "p": list(com) if com is not None else [0.0, 0.0, 0.0],
                    "com_from": "mass" if com is not None else "geometry",
                    "tol": None, "src": src},
            "mouth": {"kind": "disk",
                      "p": [round(float(p.get("neck_offset", (0, 0))[0]), 6),
                            round(float(p.get("neck_offset", (0, 0))[1]), 6), H],
                      "normal": [0.0, 0.0, 1.0],
                      "section": {"shape": "circle", "r_in": r_in,
                                  "r_out": p["mouth_r_out"]},
                      "tol": None, "src": src},
            "grasp": {"kind": "axis", "p": [0.0, 0.0, H * 0.55], "axis": [0.0, 0.0, 1.0],
                      "span": [H * 0.25, H * 0.85],
                      "section": {"shape": "circle",
                                  "r_out": p["bottom_r_out"]
                                  + (p["mouth_r_out"] - p["bottom_r_out"]) * 0.55},
                      "free_dirs": [[0.0, 2.0 * math.pi]], "tol": None, "src": src},
        }
    }


# ---------------------------------------------------------------------------
def _annotate(kind, p, d, parts, props, ann_extra, name):
    return {
        "schema": "hq-shape-annot/1",
        "parts_schema": "hq-ctx-parts/3",
        "family": kind,
        "name": name,
        "units": "m",
        "up_axis": "+z",
        "origin": "bottom centre; support plane z=0",
        "params": dict(p),
        "derived": d,
        "mass": props,
        "bbox": ann_extra["bbox"],
        "placement": [{"name": "table_rest", "point": [0.0, 0.0, 0.0],
                       "axis": [0.0, 0.0, 1.0]}],
        "material": {"density": p["density"], "transmittance": p["transmittance"]},
        "parts": parts,
        **ann_extra.get("extra", {}),
    }


def generate_bottle(params, seed=0, out_dir=None):
    p = dict(params)
    rel = check_relations_bottle(p)
    m, d = build_bottle(p)
    verts, faces = m.welded(eps=1e-9)
    props = geom.mass_properties(verts, faces, p["density"])
    name = p.get("name") or f"bottle_h{int(round(p['H']*1e3))}_d{int(round(p['body_rx']*2e4))}"
    ann = _annotate("bottle", p, d, parts_bottle(p, d, name, props["com"]), props,
                    {"bbox": {"min": [-p["body_rx"], -p["body_ry"], 0.0],
                              "max": [p["body_rx"], p["body_ry"], p["H"]],
                              "extent": [2 * p["body_rx"], 2 * p["body_ry"], p["H"]]}}, name)
    ann["relations"] = rel
    if out_dir is None:
        ann["geometry"] = geom.check_faces(verts, faces)
        return {"name": name, "annotations": ann, "mesh": (verts, faces)}
    return geom.emit_variant(__file__, FAMILY, name, verts, faces, ann, out_dir, seed, VERSION)


def generate_cup(params, seed=0, out_dir=None):
    p = dict(params)
    rel = check_relations_cup(p)
    m, d = build_cup(p)
    verts, faces = m.welded(eps=1e-9)
    props = geom.mass_properties(verts, faces, p["density"])
    name = p.get("name") or f"cup_d{int(round(p['mouth_r_out']*2e3))}_h{int(round(p['H']*1e3))}"
    ann = _annotate("cup", p, d, parts_cup(p, name, props["com"]), props,
                    {"bbox": {"min": [-p["mouth_r_out"], -p["mouth_r_out"], 0.0],
                              "max": [p["mouth_r_out"], p["mouth_r_out"], p["H"]],
                              "extent": [2 * p["mouth_r_out"], 2 * p["mouth_r_out"], p["H"]]}}, name)
    ann["relations"] = rel
    if out_dir is None:
        ann["geometry"] = geom.check_faces(verts, faces)
        return {"name": name, "annotations": ann, "mesh": (verts, faces)}
    return geom.emit_variant(__file__, "cup", name, verts, faces, ann, out_dir, seed, VERSION)
