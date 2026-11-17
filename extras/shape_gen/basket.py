"""Generator: basket (family "basket", uid `basket`, 提篮 handle_basket) - simplified.

An open container: bottom + four side walls + two fixed handles.  Simplified on
purpose (板条箱式，不做软编织), because a deformable basket would change the
dynamics of the task.

Frame: origin at the bottom centre, +z up, support plane z = 0.

Parameters / provenance: outer envelope 330…344 wide, 256…267 deep,
177.6…185 high; handle rod 11.9…17.6 diameter; cavity dimensions not measured
(frozen to the envelope minus the wall).
"""

import math
import os

import numpy as np

import geom

FAMILY = "basket"
VERSION = "1"
DENSITY = 950.0         # PP


def default_params():
    return {
        "L": 0.344,             # 外包络
        "W": 0.267,
        "H": 0.185,
        "wall_t": 0.0030,
        "bottom_t": 0.0040,
        "handle_mode": "two_side",  # two_side（两侧各一）/ single_top（圆形单提把）
        "handle_d": 0.0140,     # 提手杆直径（11.9…17.6）
        "handle_h": 0.040,      # 提手高出上沿（ENG）
        "handle_inset": 0.060,  # 提手离两端的距离
        "density": DENSITY,
    }


def derive(p):
    return {"cavity": [p["L"] - 2 * p["wall_t"], p["W"] - 2 * p["wall_t"],
                       p["H"] - p["bottom_t"]],
            "mouth_z": p["H"],
            "handle_top_z": p["H"] + p["handle_h"]}


def check_relations(p):
    d = derive(p)
    out = []

    def req(ok, rid, text):
        out.append({"id": rid, "ok": bool(ok), "text": text})
        if not ok:
            raise ValueError(f"[basket] relation {rid} failed: {text}")

    req(0.256 <= p["W"] <= 0.345 + 1e-9, "K1a", f"W={p['W']*1e3:.1f} mm in [256, 345]")
    req(0.150 <= p["H"] <= 0.200 + 1e-9, "K1b", f"H={p['H']*1e3:.1f} mm in [150, 200]")
    req(0.0119 <= p["handle_d"] <= 0.0176 + 1e-9, "K2", f"handle d={p['handle_d']*1e3:.1f} mm")
    req(p["wall_t"] >= 0.0015, "K3", "wall >= 1.5 mm")
    req(d["cavity"][0] > 0.10 and d["cavity"][1] > 0.10, "K4", "cavity big enough for the milk box")
    req(p["handle_h"] >= 0.030, "K5", f"handle clear height={p['handle_h']*1e3:.0f} mm >= 30")
    return out


def _round_tube(m, name, path, radius, n=10):
    """圆截面杆沿 3D 折线（这里用来做圆形单提把的拱）。"""
    path = [np.asarray(p, float) for p in path]
    rings = []
    for i, p in enumerate(path):
        if i == 0:
            t = path[1] - path[0]
        elif i == len(path) - 1:
            t = path[-1] - path[-2]
        else:
            t = path[i + 1] - path[i - 1]
        t = t / np.linalg.norm(t)
        a = np.array([0.0, 0.0, 1.0]) if abs(t[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        u = np.cross(t, a)
        u = u / np.linalg.norm(u)
        v = np.cross(t, u)
        ang = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        rings.append(np.array([p + radius * (np.cos(k) * u + np.sin(k) * v) for k in ang]))
    m.begin(name)
    idx = [m.add_vertices(r) for r in rings]
    for s_ in range(len(rings) - 1):
        a0, b0 = idx[s_], idx[s_ + 1]
        for k in range(n):
            k2 = (k + 1) % n
            m.add_face(a0 + k, a0 + k2, b0 + k2)
            m.add_face(a0 + k, b0 + k2, b0 + k)
    c0 = m.add_vertices([rings[0].mean(0)])
    c1 = m.add_vertices([rings[-1].mean(0)])
    for k in range(n):
        k2 = (k + 1) % n
        m.add_face(c0, idx[0] + k2, idx[0] + k)
        m.add_face(c1, idx[-1] + k, idx[-1] + k2)
    m.end()


def _box(m, name, cx, cy, cz, sx, sy, sz):
    geom.box(m, name, cx - sx / 2, cx + sx / 2, cy - sy / 2, cy + sy / 2,
             cz - sz / 2, cz + sz / 2)


def build(p):
    L, W, H, t = p["L"], p["W"], p["H"], p["wall_t"]
    m = geom.Mesh()
    # bottom
    _box(m, "bottom", 0, 0, p["bottom_t"] / 2, L, W, p["bottom_t"])
    # four walls
    _box(m, "wall_x0", -L / 2 + t / 2, 0, H / 2, t, W, H)
    _box(m, "wall_x1", L / 2 - t / 2, 0, H / 2, t, W, H)
    _box(m, "wall_y0", 0, -W / 2 + t / 2, H / 2, L - 2 * t, t, H)
    _box(m, "wall_y1", 0, W / 2 - t / 2, H / 2, L - 2 * t, t, H)
    hd = p["handle_d"]
    if p.get("handle_mode") == "single_top":
        # 圆截面单提把：跨短边（y）的一道拱，顶点 H+handle_h
        a = W / 2.0 - t / 2.0
        half = (H + p["handle_h"]) - H
        n_pts = 17
        ys = np.linspace(-a, a, n_pts)
        drop = 0.012   # 两端往墙里插 12 mm，避免悬空
        path = [(0.0, float(y),
                 float(H - drop + (p["handle_h"] + drop) * np.sqrt(max(0.0, 1.0 - (y / a) ** 2))))
                for y in ys]
        _round_tube(m, "handle_single", path, hd / 2.0, n=10)
        return m, derive(p)
    # two fixed handles: an upside-down U on each of the two short sides
    for sx, tag in ((-1, "left"), (1, "right")):
        x = sx * (L / 2 - p["handle_inset"])
        # two uprights + a top bar
        for sy, nm in ((-1, "front"), (1, "back")):
            y = sy * (W / 2 - hd)
            _box(m, f"handle_{tag}_{nm}", x, y, H + p["handle_h"] / 2, hd, hd, p["handle_h"])
        _box(m, f"handle_{tag}_top", x, 0, H + p["handle_h"], hd, W - 2 * hd, hd)
    return m, derive(p)


def parts_ctx(p, d, name, com=None):
    src = f"variant:basket/{name}"
    L, W, H = p["L"], p["W"], p["H"]
    handles = []
    if p.get("handle_mode") == "single_top":
        handles.append({
            "id": 0, "kind": "axis",
            "p": [0.0, 0.0, H + p["handle_h"]],
            "axis": [0.0, 1.0, 0.0],
            "size": {"gap": p["handle_h"], "thickness": p["handle_d"]},
            "shape": "round",
            "tol": None, "src": src,
        })
        return {"basket": {
            "origin": {"kind": "point", "p": [0.0, 0.0, 0.0], "tol": None, "src": src},
            "com": {"kind": "point", "p": list(com) if com is not None else [0.0, 0.0, 0.0],
                    "com_from": "mass" if com is not None else "geometry", "tol": None, "src": src},
            "bottom": {"kind": "disk", "p": [0.0, 0.0, 0.0], "normal": [0.0, 0.0, -1.0],
                       "section": {"shape": "rect", "w": L, "d": W, "in_or_out": "out"},
                       "tol": None, "src": src},
            "mouth": {"kind": "disk", "p": [0.0, 0.0, H], "normal": [0.0, 0.0, 1.0],
                      "section": {"shape": "rect", "w": d["cavity"][0], "d": d["cavity"][1],
                                  "in_or_out": "in"},
                      "rim_z": H, "tol": None, "src": src},
            "handles": {"kind": "list", "items": handles, "tol": None, "src": src},
        }}
    for k, sx in enumerate((-1, 1)):
        handles.append({
            "id": k, "kind": "axis",
            "p": [sx * (L / 2 - p["handle_inset"]), 0.0, H + p["handle_h"]],
            "axis": [0.0, 1.0, 0.0],                     # 抓取杆的切向（沿 y）
            "size": {"gap": p["handle_h"], "thickness": p["handle_d"]},
            "tol": None, "src": src,
        })
    return {"basket": {
        "origin": {"kind": "point", "p": [0.0, 0.0, 0.0], "tol": None, "src": src},
        "com": {"kind": "point", "p": list(com) if com is not None else [0.0, 0.0, 0.0],
                "com_from": "mass" if com is not None else "geometry", "tol": None, "src": src},
        "bottom": {"kind": "disk", "p": [0.0, 0.0, 0.0], "normal": [0.0, 0.0, -1.0],
                   "section": {"shape": "rect", "w": L, "d": W, "in_or_out": "out"},
                   "tol": None, "src": src},
        "mouth": {"kind": "disk", "p": [0.0, 0.0, H], "normal": [0.0, 0.0, 1.0],
                  "section": {"shape": "rect", "w": d["cavity"][0], "d": d["cavity"][1],
                              "in_or_out": "in"},
                  "tol": None, "src": src},
        "handles": {"kind": "list", "items": handles, "tol": None, "src": src},
    }}


def generate(params, seed=0, out_dir=None):
    p = dict(params)
    rel = check_relations(p)
    m, d = build(p)
    verts, faces = m.welded_parts()
    props = geom.mass_properties(verts, faces, p["density"])
    lo, hi = verts.min(0), verts.max(0)
    name = p.get("name") or f"basket_{int(round(p['L']*1e3))}x{int(round(p['W']*1e3))}"
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
        "simplified_note": "板条箱式刚体篮（底 + 四壁 + 两个固定提手），不做软编织/可变形",
        "parts": parts_ctx(p, d, name, props["com"]),
        "relations": rel,
    }
    if out_dir is None:
        ann["geometry"] = geom.check_faces(verts, faces)
        return {"name": name, "annotations": ann, "mesh": (verts, faces)}
    return geom.emit_variant(__file__, FAMILY, name, verts, faces, ann, out_dir, seed, VERSION)
