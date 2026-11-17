"""Generator: magnetic stirrer (family "mixer", 搅拌 mixer_operating) - simplified.

No usable external model exists, so this is a simplified body:
cabinet (250 x 85 x 301 mm, local +y up as the official asset), a round platform
on top with the beaker ring, and a lever switch on the front face that pivots.

Two uids are described in ctx.parts:
  * ``beaker_mixer`` - platform / ring (the beaker sits on the ring)
  * ``mixer_switch`` - flip joint (q_name, axis, range [0, theta_max], pivot) and
    push_point (the point the finger pushes, with the surface normal)

Frame: origin at the bottom centre, +y up (the mixer is y-up).

Parameters / provenance: envelope 250 x 85 x 301, platform ring
centre/radius not measured (frozen), switch as a curved paddle (弧片) with no
measured size -> engineering values, switch travel theta_max from the P001 plan.
"""

import math
import os

import geom

FAMILY = "mixer"
VERSION = "1"
DENSITY = 1050.0


def default_params():
    return {
        "W": 0.250,             # x
        "D": 0.085,             # z
        "H": 0.301,             # y（局部 +y 向上）
        "platform_r": 0.040,    # 台面外半径（未实测，工程值；不超过机身进深的一半）
        "platform_h": 0.010,
        "ring_r": [0.024, 0.028],   # 小圈同心圆半径（由小到大）
        "switch_x": -0.118,     # 开关在壳体上的局部位置（名义 −118,0,56）
        "switch_y": 0.150,
        "lever_len": 0.030,
        "lever_w": 0.012,
        "lever_t": 0.004,
        "theta_max": 0.60,      # rad，拨杆最大转角（P001 方案，工程值）
        "density": DENSITY,
    }


def derive(p):
    return {
        "platform_top_y": p["H"],
        "pivot": [p["switch_x"], p["switch_y"], p["D"] / 2.0],
        "push_point": [p["switch_x"], p["switch_y"], p["D"] / 2.0 + p["lever_len"]],
        "lever_com": [p["switch_x"], p["switch_y"], (p["D"] / 2.0 - 0.002 + p["D"] / 2.0 + p["lever_len"]) / 2.0],
    }


def check_relations(p):
    d = derive(p)
    out = []

    def req(ok, rid, text):
        out.append({"id": rid, "ok": bool(ok), "text": text})
        if not ok:
            raise ValueError(f"[mixer] relation {rid} failed: {text}")

    # M1 外壳在固定包络里
    req(abs(p["W"] - 0.250) < 1e-6 and abs(p["D"] - 0.085) < 1e-6, "M1a",
        f"cabinet {p['W']*1e3:.0f} x {p['D']*1e3:.0f} mm == 250 x 85")
    req(abs(p["H"] - 0.301) < 1e-6, "M1b", f"height {p['H']*1e3:.0f} mm == 301")
    # M2 小圈在台面内、比台面小
    req(max(p["ring_r"]) < p["platform_r"], "M2",
        f"ring {max(p['ring_r'])*1e3:.0f} < platform {p['platform_r']*1e3:.0f} mm")
    req(p["platform_r"] <= p["D"] / 2.0, "M2b",
        f"platform r={p['platform_r']*1e3:.0f} <= half depth={p['D']/2*1e3:.0f} mm")
    # M3 拨杆行程有限、力臂合理
    req(0.0 < p["theta_max"] <= math.pi / 2, "M3a", f"theta_max={p['theta_max']:.2f} rad in (0, pi/2]")
    req(p["lever_len"] >= 0.015, "M3b", f"lever len={p['lever_len']*1e3:.0f} mm >= 15")
    # M4 开关不伸到台面以上（不被烧杯挡住）
    req(p["switch_y"] < p["H"], "M4", "switch below the platform")
    # M5 拨杆在壳体正面之外（能碰到）
    req(p["D"] / 2.0 + p["lever_len"] > p["D"] / 2.0, "M5", "lever sticks out of the front face")
    return out


def build(p):
    m = geom.Mesh()
    geom.box(m, "cabinet", -p["W"] / 2, p["W"] / 2, 0.0, p["H"] - p["platform_h"],
             -p["D"] / 2, p["D"] / 2)
    # platform: a short cylinder on top (revolve, then move up)
    geom.revolve_profile(m, "platform", [(0.0, 0.0), (p["platform_r"], 0.0),
                                         (p["platform_r"], p["platform_h"]),
                                         (0.0, p["platform_h"])], n=48)
    # revolve builds along +z; rotate so the axis is +y and sit it on the cabinet
    R = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]
    m.transform("platform", R=R, t=[0.0, p["H"] - p["platform_h"], 0.0])
    # lever: a small paddle pivoting at (switch_x, switch_y, D/2), pointing +z
    geom.box(m, "lever", p["switch_x"] - p["lever_w"] / 2, p["switch_x"] + p["lever_w"] / 2,
             p["switch_y"] - p["lever_t"] / 2, p["switch_y"] + p["lever_t"] / 2,
             p["D"] / 2 - 0.002, p["D"] / 2 + p["lever_len"])
    return m, {}


def parts_ctx(p, d, name, com=None):
    src = f"variant:mixer/{name}"
    y_top = d["platform_top_y"]
    return {
        "beaker_mixer": {
            "origin": {"kind": "point", "p": [0.0, 0.0, 0.0], "tol": None, "src": src},
            "com": {"kind": "point", "p": list(com) if com is not None else [0.0, 0.0, 0.0],
                    "com_from": "mass" if com is not None else "geometry", "tol": None, "src": src},
            "platform": {"kind": "disk", "p": [0.0, y_top, 0.0], "normal": [0.0, 1.0, 0.0],
                         "section": {"shape": "circle", "r_out": p["platform_r"]},
                         "up_axis": "y", "tol": None, "src": src},
            "ring": {"kind": "disk", "p": [0.0, y_top, 0.0], "normal": [0.0, 1.0, 0.0],
                     "section": {"shape": "circle", "r_out": max(p["ring_r"])},
                     "rings": list(p["ring_r"]), "up_axis": "y", "tol": None, "src": src},
        },
        "mixer_switch": {
            "origin": {"kind": "point", "p": [0.0, 0.0, 0.0], "tol": None, "src": src},
            # 开关件的质心用它自己的（拨杆），不是整机
            "com": {"kind": "point", "p": list(d["lever_com"]), "com_from": "geometry",
                    "tol": None, "src": src},
            "origin_note": "same parent frame as beaker_mixer (single rigid body); "
                           "switch parts are expressed in the mixer frame",
            "flip": {"kind": "joint", "q_name": "flip", "axis": [1.0, 0.0, 0.0],
                     "range": [0.0, p["theta_max"]], "travel": p["theta_max"], "unit": "rad",
                     "pivot": d["pivot"], "link": "switch_lever", "tol": None, "src": src},
            "push_point": {"kind": "axis", "p": d["push_point"], "axis": [0.0, 0.0, 1.0],
                           "link": "switch_lever",
                           "section": {"shape": "rect", "w": p["lever_w"], "d": p["lever_t"],
                                       "in_or_out": "out"},
                           "lever": p["lever_len"], "tol": None, "src": src},
        },
    }


def generate(params, seed=0, out_dir=None):
    p = dict(params)
    rel = check_relations(p)
    d = derive(p)
    m, _ = build(p)
    verts, faces = m.welded_parts()
    props = geom.mass_properties(verts, faces, p["density"])
    lo, hi = verts.min(0), verts.max(0)
    name = p.get("name") or f"mixer_{int(round(p['W']*1e3))}x{int(round(p['D']*1e3))}"
    ann = {
        "schema": "hq-shape-annot/1",
        "parts_schema": "hq-ctx-parts/3",
        "family": FAMILY,
        "name": name,
        "units": "m",
        "up_axis": "+y",
        "origin": "bottom centre; +y up (official beaker_mixer asset is +y up)",
        "params": dict(p),
        "derived": d,
        "mass": props,
        "bbox": {"min": [float(x) for x in lo], "max": [float(x) for x in hi],
                 "extent": [float(x) for x in (hi - lo)]},
        "cabinet_extent": [p["W"], p["H"], p["D"]],
        "bbox_note": "bbox includes the switch lever, which protrudes in +z; "
                     "cabinet_extent is the official 250 x 301 x 85 envelope",
        "placement": [{"name": "table_rest", "point": [0.0, 0.0, 0.0], "axis": [0.0, 1.0, 0.0]}],
        "material": {"density": p["density"]},
        "simplified": True,
        "simplified_note": "parametric simplified cabinet + platform + lever; the real "
                           "switch is a curved paddle and the body is a full appliance shell",
        "joint_simulated": False,
        "parts": parts_ctx(p, d, name, props["com"]),
        "relations": rel,
    }
    if out_dir is None:
        ann["geometry"] = geom.check_faces(verts, faces)
        return {"name": name, "annotations": ann, "mesh": (verts, faces)}
    return geom.emit_variant(__file__, FAMILY, name, verts, faces, ann, out_dir, seed, VERSION)
