"""Generator: the two assembly tubes (family "tube2", 组装 item_assembly).

Each tube is one revolved solid: a coarse head with a blind socket, a tapered
body, and a thin tail that acts as the pin.  The pair is generated from one
parameter set so the socket/pin fit is guaranteed by construction.

Frame: geometric centre, main axis +x, head at -L/2, tail at +L/2 (matches
: local x, both ends are geometry ends; semantics are on end_head /
end_tail).  Built along +z then rotated so the axis is +x.

Parameters / provenance: (length 202, widths 39/46, end radii 20/14 and
23/17, head coarse-short / tail thin-long).
"""

import math
import os

import numpy as np

import geom

FAMILY = "tube2"
VERSION = "1"
SILICONE_D = 1200.0     # kg/m3 (silicone; do not use a PLA density)


def default_pair():
    return {
        # tube A: head 20 mm, tail 14 mm (diameters)
        "L_a": 0.202, "head_r_a": 0.0200, "tail_r_a": 0.0140,
        "head_len_a": 0.028, "tail_len_a": 0.045, "socket_depth_a": 0.020,
        # tube B: head 23 mm, tail 17 mm
        "L_b": 0.2025, "head_r_b": 0.0230, "tail_r_b": 0.0170,
        "head_len_b": 0.030, "tail_len_b": 0.048, "socket_depth_b": 0.022,
        # fit
        "fit_gap": 0.0005,     # radial clearance socket - pin (工程：对齐 + 碰撞余量；参考件是硅胶弹性配合)
        "wall_t": 0.0020,
        "density": SILICONE_D,
    }


def derive(p):
    """几何关系：B 的尾（pin）插进 A 的头（socket），反之亦然。"""
    pin_a = p["tail_r_a"]
    pin_b = p["tail_r_b"]
    # the socket is sized from the *other* tube's pin + fit gap (by construction)
    soc_a = pin_b + p["fit_gap"]
    soc_b = pin_a + p["fit_gap"]
    return {
        "socket_r_a": soc_a, "socket_r_b": soc_b,
        "pin_r_a": pin_a, "pin_r_b": pin_b,
        "clearance_a_b": soc_a - pin_b,      # B 尾插 A 头
        "clearance_b_a": soc_b - pin_a,      # A 尾插 B 头
        "insert_a_b": min(p["socket_depth_a"], p["tail_len_b"] * 0.8),
        "insert_b_a": min(p["socket_depth_b"], p["tail_len_a"] * 0.8),
    }


def check_relations(p):
    d = derive(p)
    out = []

    def req(ok, rid, text):
        out.append({"id": rid, "ok": bool(ok), "text": text})
        if not ok:
            raise ValueError(f"[tube2] relation {rid} failed: {text}")

    # T1 配合间隙：正值，且覆盖对齐误差（不靠穿模）
    req(d["clearance_a_b"] >= p["fit_gap"] - 1e-9, "T1a",
        f"B tail -> A socket clearance={d['clearance_a_b']*1e3:.2f} mm >= {p['fit_gap']*1e3:.2f}")
    req(d["clearance_b_a"] >= p["fit_gap"] - 1e-9, "T1b",
        f"A tail -> B socket clearance={d['clearance_b_a']*1e3:.2f} mm >= {p['fit_gap']*1e3:.2f}")
    req(d["clearance_a_b"] <= 0.003 and d["clearance_b_a"] <= 0.003, "T1c",
        "clearance <= 3 mm (else the joint is loose)")
    # T1d 硅胶壁厚还得剩
    req(p["head_r_a"] - d["socket_r_a"] >= 0.0006, "T1d",
        f"A head wall={((p['head_r_a']-d['socket_r_a'])*1e3):.2f} mm >= 0.6 (silicone)")
    req(p["head_r_b"] - d["socket_r_b"] >= 0.0006, "T1e",
        f"B head wall={((p['head_r_b']-d['socket_r_b'])*1e3):.2f} mm >= 0.6 (silicone)")
    # T2 插深
    req(0.010 <= d["insert_a_b"] <= p["socket_depth_a"] + 1e-9, "T2a",
        f"insert B->A={d['insert_a_b']*1e3:.1f} mm within [10, socket depth]")
    req(0.010 <= d["insert_b_a"] <= p["socket_depth_b"] + 1e-9, "T2b",
        f"insert A->B={d['insert_b_a']*1e3:.1f} mm within [10, socket depth]")
    # T3 端部尺寸
    req(0.039 <= 2 * p["head_r_a"] <= 0.0554, "T3a", f"A head d={2*p['head_r_a']*1e3:.1f} mm")
    req(0.046 <= 2 * p["head_r_b"] <= 0.0654, "T3b", f"B head d={2*p['head_r_b']*1e3:.1f} mm")
    req(p["tail_r_a"] < p["head_r_a"] and p["tail_r_b"] < p["head_r_b"], "T3c",
        "tail thinner than head (head coarse-short / tail thin-long)")
    # T4 长度（202 … 286.84 = 202 x 1.42）
    for tag in ("a", "b"):
        L = p["L_" + tag]
        req(0.200 <= L <= 0.288, "T4" + tag, f"L_{tag}={L*1e3:.1f} mm in [200, 288]")
    # T5 插深不能吃掉整根尾
    req(p["tail_len_a"] > d["insert_b_a"] + 0.01, "T5a", "A tail longer than the insert depth")
    req(p["tail_len_b"] > d["insert_a_b"] + 0.01, "T5b", "B tail longer than the insert depth")
    # T6 几何质量粗估（实心包络；官方 config 里的 5–40 g 是另一档）
    vol_a = math.pi * p["head_r_a"] ** 2 * p["L_a"] * 0.6
    vol_b = math.pi * p["head_r_b"] ** 2 * p["L_b"] * 0.6
    req(0.001 <= vol_a * p["density"] <= 0.25, "T6a", "tube A solid estimate in [1,250] g")
    req(0.001 <= vol_b * p["density"] <= 0.25, "T6b", "tube B solid estimate in [1,250] g")
    return out


# ---------------------------------------------------------------------------
def _profile(head_r, tail_r, L, head_len, tail_len, socket_r, socket_depth, wall):
    """Closed (r,z) profile: blind socket at z=0 (head), thin pin at z=L."""
    r_mid = 0.5 * (head_r + tail_r)
    return [
        (0.0, socket_depth),            # socket bottom centre
        (socket_r, socket_depth),
        (socket_r, 0.0),                # socket mouth
        (head_r, 0.0),                  # head end face
        (head_r, head_len),
        (r_mid, head_len + 0.02),
        (tail_r, L - tail_len),
        (tail_r, L),                    # pin end
        (0.0, L),                       # tail centre (pole)
    ]


def build_tube(key, p):
    if key == "a":
        prof = _profile(p["head_r_a"], p["tail_r_a"], p["L_a"], p["head_len_a"],
                        p["tail_len_a"], derive(p)["socket_r_a"], p["socket_depth_a"],
                        p["wall_t"])
    else:
        prof = _profile(p["head_r_b"], p["tail_r_b"], p["L_b"], p["head_len_b"],
                        p["tail_len_b"], derive(p)["socket_r_b"], p["socket_depth_b"],
                        p["wall_t"])
    m = geom.Mesh()
    geom.revolve_profile(m, f"tube_{key}", prof, n=56)
    # axis +z -> +x, centre at 0
    L = p["L_" + key]
    R = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    m.transform(f"tube_{key}", R=R, t=[-L / 2.0, 0.0, 0.0])
    return m


def parts_tube(key, p, d, name, com=None):
    """ctx.parts for uid guijiao1 (tube a) / guijiao2 (tube b)."""
    src = f"variant:tube2/{name}"
    head_r = p["head_r_" + key]
    tail_r = p["tail_r_" + key]
    L = p["L_" + key]
    socket_r = d["socket_r_" + key]
    socket_depth = p["socket_depth_" + key]
    uid = "guijiao1" if key == "a" else "guijiao2"
    return {
        uid: {
            "origin": {"kind": "point", "p": [0.0, 0.0, 0.0], "tol": None, "src": src},
            # 第 3 版组装行：两端面 = disk（面心 + 朝外法向）
            "end_head": {"kind": "disk", "p": [-L / 2.0, 0.0, 0.0], "normal": [-1.0, 0.0, 0.0],
                         "section": {"shape": "circle", "r_out": head_r, "r_in": socket_r},
                         "tol": None, "src": src},
            "end_tail": {"kind": "disk", "p": [L / 2.0, 0.0, 0.0], "normal": [1.0, 0.0, 0.0],
                         "section": {"shape": "circle", "r_out": tail_r},
                         "tol": None, "src": src},
            "com": {"kind": "point", "p": list(com) if com is not None else [0.0, 0.0, 0.0],
                    "com_from": "mass" if com is not None else "geometry",
                    "tol": None, "src": src},
            "socket": {"kind": "disk", "p": [-L / 2.0, 0.0, 0.0], "normal": [-1.0, 0.0, 0.0],
                       "section": {"shape": "circle", "r_in": socket_r},
                       "size": {"depth": socket_depth}, "tol": None, "src": src},
            "pin": {"kind": "disk", "p": [L / 2.0, 0.0, 0.0], "normal": [1.0, 0.0, 0.0],
                    "section": {"shape": "circle", "r_out": tail_r},
                    "size": {"depth": p["tail_len_" + key]}, "tol": None, "src": src},
            # 第 3 版长条物体的读法：顶层 tube_axis / body / grasp_span，
            # span 与 profile 的 s 一律从 tail（+L/2 端）量起
            "tube_axis": {"kind": "axis", "p": [0.0, 0.0, 0.0], "axis": [-1.0, 0.0, 0.0],
                          "size": {"length": L}, "span": [0.0, L],
                          "profile": [[0.0, round(tail_r, 6)],
                                      [round(p["tail_len_" + key], 6), round(tail_r, 6)],
                                      [round(L - p["head_len_" + key], 6), round(head_r, 6)],
                                      [round(L, 6), round(head_r, 6)]],
                          "tol": None, "src": src},
            "body": {"kind": "axis", "p": [0.0, 0.0, 0.0], "axis": [-1.0, 0.0, 0.0],
                     "size": {"length": L}, "span": [0.0, L],
                     "section": {"shape": "circle", "r_out": round(max(head_r, tail_r), 6)},
                     "profile": [[0.0, round(tail_r, 6)],
                                 [round(p["tail_len_" + key], 6), round(tail_r, 6)],
                                 [round(L - p["head_len_" + key], 6), round(head_r, 6)],
                                 [round(L, 6), round(head_r, 6)]],
                     "grasp_s": 0.5, "tol": None, "src": src},
            "grasp_span": {"kind": "axis", "p": [0.0, 0.0, 0.0], "axis": [-1.0, 0.0, 0.0],
                           "span": [round(p["tail_len_" + key], 6),
                                    round(L - p["head_len_" + key], 6)],
                           "section": {"shape": "circle",
                                       "r_out": round(0.5 * (head_r + tail_r), 6)},
                           "tol": None, "src": src},
            "grasp": {"kind": "axis", "p": [0.0, 0.0, 0.0], "axis": [-1.0, 0.0, 0.0],
                      "span": [round(p["tail_len_" + key], 6),
                               round(L - p["head_len_" + key], 6)],
                      "section": {"shape": "circle",
                                  "r_out": 0.5 * (head_r + tail_r)},
                      "tol": None, "src": src},
        }
    }


def _annotate(key, p, d, props, name):
    L = p["L_" + key]
    head_r = p["head_r_" + key]
    return {
        "schema": "hq-shape-annot/1",
        "parts_schema": "hq-ctx-parts/3",
        "family": FAMILY,
        "name": name,
        "units": "m",
        "up_axis": "+z",
        "origin": "geometric centre; main axis +x (head at -L/2, tail/pin at +L/2)",
        "params": dict(p),
        "derived": d,
        "mass": props,
        "bbox": {"min": [-L / 2, -head_r, -head_r], "max": [L / 2, head_r, head_r],
                 "extent": [L, 2 * head_r, 2 * head_r]},
        "placement": [{"name": "table_rest_lying", "point": [0.0, 0.0, -head_r],
                       "axis": [0.0, 0.0, 1.0], "support": "line along +x"}],
        "material": {"density": p["density"]},
        "parts": parts_tube(key, p, d, name, props["com"]),
    }


# ---------------------------------------------------------------------------
# paired sampling: the two tubes are always drawn together
# ---------------------------------------------------------------------------
PAIR_RANGES = {
    "L": (0.202, 0.2868),          #: 202 .. 202*1.42
    "head_r": (0.0195, 0.0277),    # 宽 39..55.38 mm
    "tail_r": (0.0140, 0.0170),    # 端部半径 14..17 mm（头/尾比例由 head_r 与 tail_r 一起定）
    "head_len": (0.028, 0.040),
    "tail_len": (0.040, 0.060),
    "socket_depth": (0.018, 0.026),
}


def sample_pair(rng, n=1, strict=True):
    """Draw n *pairs* of tube parameter sets.

    The fit is guaranteed by construction: the socket radius of each tube is
    derived from the **other** tube's pin radius + fit gap (see derive()), so the
    two are never sampled independently.  Configurations that would leave less
    than 0.6 mm of silicone wall on a head are rejected and redrawn.
    """
    out = []
    tries = 0
    while len(out) < n and tries < 10000:
        tries += 1
        r = {k: float(rng.uniform(*v)) for k, v in PAIR_RANGES.items()}
        if r["tail_r"] >= r["head_r"] * 0.9:      # keep a visible head/tail step
            continue
        gap = float(rng.uniform(0.0003, 0.0010))
        p = {
            "L_a": r["L"], "head_r_a": r["head_r"], "tail_r_a": r["tail_r"],
            "head_len_a": r["head_len"], "tail_len_a": r["tail_len"],
            "socket_depth_a": r["socket_depth"],
            "L_b": r["L"] * float(rng.uniform(0.98, 1.02)),
            "head_r_b": r["head_r"] * float(rng.uniform(1.02, 1.15)),
            "tail_r_b": r["tail_r"] * float(rng.uniform(0.98, 1.05)),
            "head_len_b": r["head_len"] * float(rng.uniform(0.95, 1.15)),
            "tail_len_b": r["tail_len"] * float(rng.uniform(0.95, 1.15)),
            "socket_depth_b": r["socket_depth"] * float(rng.uniform(0.95, 1.10)),
            "fit_gap": gap, "wall_t": 0.0020, "density": SILICONE_D,
        }
        d = derive(p)
        if (p["head_r_a"] - d["socket_r_a"] < 0.0006
                or p["head_r_b"] - d["socket_r_b"] < 0.0006):
            continue
        try:
            check_relations(p)
        except ValueError:
            continue
        out.append(p)
    if strict and len(out) < n:
        raise RuntimeError(f"sample_pair: only {len(out)}/{n} valid pairs after {tries} tries")
    return out


def generate(params, seed=0, out_dir=None, key="a"):
    p = dict(params)
    rel = check_relations(p)
    d = derive(p)
    m = build_tube(key, p)
    verts, faces = m.welded(eps=1e-9)
    props = geom.mass_properties(verts, faces, p["density"])
    uid = "guijiao1" if key == "a" else "guijiao2"
    name = p.get("name") or f"{uid}_L{int(round(p['L_' + key] * 1e3))}_h{int(round(p['head_r_' + key] * 2e4))}"
    ann = _annotate(key, p, d, props, name)
    ann["relations"] = rel
    if out_dir is None:
        ann["geometry"] = geom.check_faces(verts, faces)
        return {"name": name, "annotations": ann, "mesh": (verts, faces)}
    return geom.emit_variant(__file__, FAMILY, name, verts, faces, ann, out_dir, seed, VERSION)
