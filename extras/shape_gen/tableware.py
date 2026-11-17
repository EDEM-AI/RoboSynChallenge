"""Generator: tableware - plate / fork / spoon (family "plate" / "fork" / "spoon",
摆桌 table_rearrangement).

* plate : one revolved shell (dish with a rim).
* fork  : handle + head bar + 4 tines, each a closed extruded prism (multi-shell).
* spoon : handle prism + a revolved shallow bowl scaled to an ellipse.

Frame: plate origin = bottom centre, +z up.  Fork / spoon: geometric centre,
long axis +x, head at +L/2, tail at -L/2 (局部 +x 朝头).

Parameters / provenance: fork 203 long, head 25 wide, handle 3.7-4.5
wide; spoon 145 long, head 25, handle 4.0-5.3; plate 180 diameter, 34 high.
The measured "总高 18 / 14 mm" comes from the bent tines / bowl depth, which is
NOT modelled here (flat tines) - noted in the annotation.
"""

import math
import os

import geom

VERSION = "1"
STEEL_D = 7900.0        # 不锈钢参考档（不锈钢参考件 0.048–0.053 kg）


def default_plate():
    return {"r_out": 0.090, "H": 0.034, "floor_z": 0.006, "rim_w": 0.008,
            "wall_t": 0.004, "density": 1500.0, "material": "ceramic"}


def default_fork():
    return {"L": 0.203, "head_w": 0.025, "head_len": 0.045, "handle_w": 0.0042,
            "tine_n": 4, "tine_w": 0.0032, "gap_w": 0.0035, "thickness": 0.0022,
            "grasp_s": 0.30, "density": STEEL_D}


def default_spoon():
    return {"L": 0.145, "head_w": 0.025, "head_len": 0.052, "handle_w": 0.0048,
            "thickness": 0.0018, "bowl_depth": 0.006,
            "grasp_s": 0.475, "density": STEEL_D}


# ---------------------------------------------------------------------------
def check_plate(p):
    out = []

    def req(ok, rid, text):
        out.append({"id": rid, "ok": bool(ok), "text": text})
        if not ok:
            raise ValueError(f"[plate] relation {rid} failed: {text}")

    req(abs(2 * p["r_out"] - 0.180) < 1e-6, "P1", f"diameter={2*p['r_out']*1e3:.1f} mm == 180")
    req(abs(p["H"] - 0.034) < 1e-6, "P2", f"height={p['H']*1e3:.1f} mm == 34")
    req(p["floor_z"] > p["wall_t"], "P3", "floor above the wall thickness")
    req(p["rim_w"] < p["r_out"], "P4", "rim width smaller than the radius")
    return out


def check_fork(p):
    out = []

    def req(ok, rid, text):
        out.append({"id": rid, "ok": bool(ok), "text": text})
        if not ok:
            raise ValueError(f"[fork] relation {rid} failed: {text}")

    req(abs(p["L"] - 0.203) < 1e-6, "F1", f"L={p['L']*1e3:.1f} mm == 203")
    req(0.0037 - 1e-9 <= p["handle_w"] <= 0.0045 + 1e-9, "F2",
        f"handle width={p['handle_w']*1e3:.2f} mm in [3.7, 4.5]")
    req(abs(p["head_w"] - 0.025) < 1e-6, "F3", f"head width={p['head_w']*1e3:.1f} mm == 25")
    req(p["tine_n"] >= 3, "F4", f"tines={p['tine_n']} >= 3")
    span = p["tine_n"] * p["tine_w"] + (p["tine_n"] - 1) * p["gap_w"]
    req(span <= p["head_w"] + 1e-9, "F5", f"tines span={span*1e3:.1f} <= head {p['head_w']*1e3:.1f}")
    req(p["gap_w"] >= 0.002, "F6", f"tine gap={p['gap_w']*1e3:.1f} mm >= 2")
    # 握点在柄上、不在头/齿上
    gx = -p["L"] / 2 + p["grasp_s"] * p["L"]
    req(gx < p["L"] / 2 - p["head_len"], "F7",
        f"grasp x={gx*1e3:.1f} mm is on the handle (head starts at "
        f"{(p['L']/2-p['head_len'])*1e3:.1f})")
    req(p["handle_w"] <= 0.085, "F8", "handle fits the gripper")
    return out


def check_spoon(p):
    out = []

    def req(ok, rid, text):
        out.append({"id": rid, "ok": bool(ok), "text": text})
        if not ok:
            raise ValueError(f"[spoon] relation {rid} failed: {text}")

    req(abs(p["L"] - 0.145) < 1e-6, "S1", f"L={p['L']*1e3:.1f} mm == 145")
    req(0.0040 - 1e-9 <= p["handle_w"] <= 0.0053 + 1e-9, "S2",
        f"handle width={p['handle_w']*1e3:.2f} mm in [4.0, 5.3]")
    req(abs(p["head_w"] - 0.025) < 1e-6, "S3", f"bowl width={p['head_w']*1e3:.1f} mm == 25")
    gx = -p["L"] / 2 + p["grasp_s"] * p["L"]
    req(gx < p["L"] / 2 - p["head_len"], "S4",
        f"grasp x={gx*1e3:.1f} mm is on the handle")
    req(0.45 <= p["grasp_s"] <= 0.50, "S5", f"grasp_s={p['grasp_s']} in [0.45, 0.50]")
    return out


# ---------------------------------------------------------------------------
def build_plate(p):
    r, H, fz, rw = p["r_out"], p["H"], p["floor_z"], p["rim_w"]
    # radial section, CCW: bottom -> foot -> underside -> rim -> inner bowl -> floor
    prof = [(0.0, 0.0),
            (r - 0.020, 0.0),
            (r - 0.002, H - 0.012),
            (r, H),
            (r - 0.008, H),
            (r - 0.015, H - 0.010),
            (r * 0.45, fz),
            (0.0, fz)]
    m = geom.Mesh()
    geom.revolve_profile(m, "plate", prof, n=72)
    # sit on z = 0 (the underside of the foot is the lowest point)
    zmin = min(v[2] for v in m.v)
    m.transform("plate", t=[0.0, 0.0, -zmin])
    return m, {}


def _tapered_prism(mesh, name, x0, x1, w0, w1, t):
    z0, z1 = 0.0, t
    pts = [(x0, -w0 / 2), (x1, -w1 / 2), (x1, w1 / 2), (x0, w0 / 2)]
    geom.extrude_polygon(mesh, name, pts, z0, z1)


def build_fork(p):
    L, hw, hl, gw = p["L"], p["head_w"], p["head_len"], p["handle_w"]
    t = p["thickness"]
    head_x0 = L / 2 - hl
    tine_len = hl * 0.42
    head_bar_x1 = L / 2 - tine_len
    m = geom.Mesh()
    _tapered_prism(m, "handle", -L / 2, head_x0, gw, gw * 1.8, t)
    _tapered_prism(m, "head", head_x0, head_bar_x1, hw * 0.75, hw, t)
    span = p["tine_n"] * p["tine_w"] + (p["tine_n"] - 1) * p["gap_w"]
    x = -span / 2
    for k in range(p["tine_n"]):
        y = x + p["tine_w"] / 2
        _tapered_prism(m, f"tine_{k}", head_bar_x1 - 0.002, L / 2, p["tine_w"] * 1.0,
                       p["tine_w"] * 0.85, t)
        # shift in y
        idx = [i for i, _ in enumerate(m.v)][-8:]
        for i in idx:
            m.v[i][1] += y
        x += p["tine_w"] + p["gap_w"]
    return m, {}


def build_spoon(p):
    L, bw, hl, gw, t = p["L"], p["head_w"], p["head_len"], p["handle_w"], p["thickness"]
    bowl_cx = L / 2 - hl / 2
    a = hl / 2
    b = bw / 2
    m = geom.Mesh()
    _tapered_prism(m, "handle", -L / 2, bowl_cx - a * 0.75, gw, gw * 1.7, t)
    prof = [(0.0, t * 0.6), (b * 0.55, t * 0.6), (b * 0.95, p["bowl_depth"] * 0.5),
            (b, p["bowl_depth"]), (b * 0.94, p["bowl_depth"] * 0.9),
            (b * 0.5, p["bowl_depth"] * 0.35), (0.0, 0.0)]
    geom.revolve_profile(m, "bowl", prof, n=48, scale=(a / b, 1.0, 1.0))
    # move the bowl to its place on the +x side
    idx = [i for i, _ in enumerate(m.v)]
    start = 0
    for nm, s, e in m.part_ranges():
        if nm == "bowl":
            start = s
            break
    # translate bowl vertices (they are the last ones appended)
    for i in range(len(m.v)):
        pass
    m.transform("bowl", t=[bowl_cx, 0.0, 0.0])
    return m, {}


# ---------------------------------------------------------------------------
def parts_plate(p, name, com=None):
    src = f"variant:plate/{name}"
    r, H = p["r_out"], p["H"]
    rim = [{"id": k, "p": [r * math.cos(a), r * math.sin(a), H],
            "normal": [math.cos(a), math.sin(a), 0.0], "src": src, "tol": None}
           for k, a in enumerate([0.0, math.pi / 2, math.pi, 3 * math.pi / 2])]
    return {"plate": {
        "origin": {"kind": "point", "p": [0.0, 0.0, 0.0], "tol": None, "src": src},
        "com": {"kind": "point", "p": list(com) if com is not None else [0.0, 0.0, 0.0],
                "com_from": "mass" if com is not None else "geometry",
                "tol": None, "src": src},
        "center": {"kind": "point", "p": [0.0, 0.0, H], "tol": None, "src": src},
        "bottom": {"kind": "disk", "p": [0.0, 0.0, 0.0], "normal": [0.0, 0.0, -1.0],
                   "section": {"shape": "circle", "r_out": r}, "tol": None, "src": src},
        "rim": {"kind": "list", "items": rim, "tol": None, "src": src},
    }}


def handle_samples(kind, p, src=None):
    """柄中心线的采样点（第 3 版 fork/spoon.handle 是 list）。"""
    L = p["L"]
    hl = p.get("head_len", 0.045)
    x_tail, x_neck = -L / 2, L / 2 - hl
    n = 5
    items = []
    for k in range(n):
        f = k / (n - 1)
        x = x_tail + f * (x_neck - x_tail)
        w = p["handle_w"] + f * (p.get("head_w", 0.025) * 0.45 - p["handle_w"])
        items.append({"id": k, "p": [x, 0.0, p["thickness"] / 2], "axis": [1.0, 0.0, 0.0],
                      # 截面走 section.rect（第 3 版：横截面量放 section），厚度另在 size
                      "section": {"shape": "rect", "w": round(w, 6),
                                  "d": round(p["thickness"], 6), "in_or_out": "out"},
                      "size": {"thickness": round(p["thickness"], 6)},
                      "src": src, "tol": None})
    return items


def parts_tool(kind, p, name, com=None):
    src = f"variant:{kind}/{name}"
    L = p["L"]
    hl = p.get("head_len", 0.045)
    return {kind: {
        "origin": {"kind": "point", "p": [0.0, 0.0, 0.0], "tol": None, "src": src},
        "head": {"kind": "point", "p": [L / 2, 0.0, 0.0], "tol": None, "src": src},
        "tail": {"kind": "point", "p": [-L / 2, 0.0, 0.0], "tol": None, "src": src},
        # 第 3 版长条读法：span / profile 的 s 从 tail 量起（s=0 在尾部）
        "body": {"kind": "axis", "p": [0.0, 0.0, p["thickness"] / 2], "axis": [1.0, 0.0, 0.0],
                 "size": {"length": L}, "span": [0.0, L],
                 "profile": [[0.0, round(p["handle_w"] / 2, 6)],
                             [round(L - hl, 6), round(p.get("head_w", 0.025) / 2 * 0.75, 6)],
                             [round(L, 6), round(p.get("head_w", 0.025) / 2, 6)]],
                 "grasp_s": p["grasp_s"], "tol": None, "src": src},
        "handle": {"kind": "list", "items": handle_samples(kind, p, f"variant:{kind}/{name}"),
                   "grasp_s": p["grasp_s"], "tol": None, "src": src},
        "com": {"kind": "point", "p": list(com) if com is not None else [0.0, 0.0, 0.0],
                "com_from": "mass" if com is not None else "geometry",
                "tol": None, "src": src},
    }}


def _ann(family, p, name, props, parts, verts):
    return {
        "schema": "hq-shape-annot/1",
        "parts_schema": "hq-ctx-parts/3",
        "family": family,
        "name": name,
        "units": "m",
        "up_axis": "+z",
        "origin": ("bottom centre; support plane z=0" if family == "plate"
                   else "geometric centre; long axis +x, head at +L/2"),
        "params": dict(p),
        "derived": {"L": p.get("L"), "grasp_s": p.get("grasp_s")},
        "mass": props,
        "bbox": {"min": [float(x) for x in verts.min(0)],
                 "max": [float(x) for x in verts.max(0)],
                 "extent": [float(x) for x in (verts.max(0) - verts.min(0))]},
        "placement": [{"name": "table_rest", "point": [0.0, 0.0, 0.0], "axis": [0.0, 0.0, 1.0]}],
        "material": {"density": p["density"], "name": p.get("material", "steel")},
        "flat_model_note": ("tines / bowl are modelled flat; the measured 总高 18/14 mm "
                            "comes from the tine bend and bowl depth, not modelled") if family != "plate" else None,
        "parts": parts,
    }


def generate_plate(params, seed=0, out_dir=None):
    p = dict(params)
    rel = check_plate(p)
    m, d = build_plate(p)
    verts, faces = m.welded(eps=1e-9)
    props = geom.mass_properties(verts, faces, p["density"])
    name = p.get("name") or f"plate_d{int(round(p['r_out']*2e3))}_h{int(round(p['H']*1e3))}"
    ann = _ann("plate", p, name, props, parts_plate(p, name, props["com"]), verts)
    ann["relations"] = rel
    if out_dir is None:
        ann["geometry"] = geom.check_faces(verts, faces)
        return {"name": name, "annotations": ann, "mesh": (verts, faces)}
    return geom.emit_variant(__file__, "plate", name, verts, faces, ann, out_dir, seed, VERSION)


def generate_fork(params, seed=0, out_dir=None):
    p = dict(params)
    rel = check_fork(p)
    m, d = build_fork(p)
    verts, faces = m.welded_parts()
    props = geom.mass_properties(verts, faces, p["density"])
    name = p.get("name") or f"fork_L{int(round(p['L']*1e3))}_w{int(round(p['handle_w']*1e4))}"
    ann = _ann("fork", p, name, props, parts_tool("fork", p, name, props["com"]), verts)
    ann["relations"] = rel
    if out_dir is None:
        ann["geometry"] = geom.check_faces(verts, faces)
        return {"name": name, "annotations": ann, "mesh": (verts, faces)}
    return geom.emit_variant(__file__, "fork", name, verts, faces, ann, out_dir, seed, VERSION)


def generate_spoon(params, seed=0, out_dir=None):
    p = dict(params)
    rel = check_spoon(p)
    m, d = build_spoon(p)
    verts, faces = m.welded_parts()
    props = geom.mass_properties(verts, faces, p["density"])
    name = p.get("name") or f"spoon_L{int(round(p['L']*1e3))}_w{int(round(p['handle_w']*1e4))}"
    ann = _ann("spoon", p, name, props, parts_tool("spoon", p, name, props["com"]), verts)
    ann["relations"] = rel
    if out_dir is None:
        ann["geometry"] = geom.check_faces(verts, faces)
        return {"name": name, "annotations": ann, "mesh": (verts, faces)}
    return geom.emit_variant(__file__, "spoon", name, verts, faces, ann, out_dir, seed, VERSION)
