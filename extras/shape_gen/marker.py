"""Generator: marker / pen (family "marker").

Covers the span from the thin pen that ships with the simulation (9.58 mm) to a
thick permanent marker (23.25 mm, S radial 1..1.55 on the sim pen), with an
optional cap and a round or slightly faceted cross-section.

Frame: origin at the geometric centre, main axis along +x, tip at +L/2,
tail at -L/2, +z up.  A marker lying on the table therefore needs no rotation
and its support point is (0, 0, -r).

Interface:  generate(params, seed, out_dir) -> variant.yaml content
"""

import math
import os

import numpy as np

import geom
import params as P

FAMILY = "marker"
VERSION = "1"


def default_params():
    return {
        "d": P.MARKER["d"]["nominal"],
        "length": P.MARKER["length"]["nominal"],
        "cap_len": P.MARKER["cap_len"]["nominal"],
        "cap_clearance": P.MARKER["cap_clearance"]["nominal"],
        "tip_len": P.MARKER["tip_len"]["nominal"],
        "sides": P.MARKER["sides"]["nominal"],
        "density": P.MARKER["density"]["nominal"],
        "fill": P.MARKER["fill"]["nominal"],
        "holder_inner_d": P.MARKER["holder_inner_d"]["nominal"],
        "holder_height": P.MARKER["holder_height"]["nominal"],
        "insert_gap": P.MARKER["insert_gap"]["nominal"],
    }


# ---------------------------------------------------------------------------
# relations
# ---------------------------------------------------------------------------
def check_relations(p):
    out = []

    def req(ok, rid, text):
        out.append({"id": rid, "ok": bool(ok), "text": text})
        if not ok:
            raise ValueError(f"[{FAMILY}] relation {rid} failed: {text}")

    d, L = p["d"], p["length"]
    d_cap = d + 2 * p["cap_clearance"] if p["cap_len"] > 0 else d

    # M1 fits into the pen holder with clearance
    req(max(d, d_cap) + 2 * p["insert_gap"] <= p["holder_inner_d"] + 1e-12, "M1",
        f"max(d, d_cap)={max(d, d_cap)*1e3:.2f} + 2*{p['insert_gap']*1e3:.1f} "
        f"<= holder_inner_d={p['holder_inner_d']*1e3:.2f} mm")
    # M2 length / holder-height ratio inside the observed band
    lo, hi = P.LIMITS["marker_len_over_holder"]
    ratio = L / p["holder_height"]
    req(lo - 1e-9 <= ratio <= hi + 1e-9, "M2",
        f"L/H_holder={ratio:.3f} in [{lo},{hi}] (sim 1.488, edding 1.42)")
    # M3 enough of the marker sticks out to be grasped
    exposed = L - p["holder_height"]
    req(exposed >= P.LIMITS["marker_min_exposed_m"] - 1e-12, "M3",
        f"exposed={exposed*1e3:.1f} mm >= {P.LIMITS['marker_min_exposed_m']*1e3:.0f} mm")
    # M4 cap / tip segment sane
    req(p["cap_len"] <= 0.5 * L, "M4a", f"cap_len={p['cap_len']*1e3:.1f} <= L/2")
    req(p["tip_len"] < L - p["cap_len"], "M4b", "tip segment shorter than the body")
    # M5 diameter / length inside the declared envelope
    dd = P.MARKER["d"]["range"]
    ll = P.MARKER["length"]["range"]
    req(dd[0] - 1e-12 <= d <= dd[1] + 1e-12, "M5a", f"d={d*1e3:.2f} mm in [{dd[0]*1e3:.2f},{dd[1]*1e3:.2f}]")
    req(ll[0] - 1e-12 <= L <= ll[1] + 1e-12, "M5b", f"L={L*1e3:.2f} mm in [{ll[0]*1e3:.2f},{ll[1]*1e3:.2f}]")
    # M6 mass must land in the pen / marker band (0.004..0.020 kg,
    #    13.6 g edding nominal)
    vol = _volume(p)
    mass = vol * p["density"] * p["fill"]
    lo_m, hi_m = P.LIMITS["marker_mass_kg"]
    req(lo_m - 1e-12 <= mass <= hi_m + 1e-12, "M6",
        f"mass={mass*1e3:.2f} g in [{lo_m*1e3:.0f},{hi_m*1e3:.0f}] g "
        f"(仿真笔档 4–20 g; edding 13.6 g; 同填充率外推到 23 mm 约 34 g)")
    return out


def _volume(p):
    """Closed-form volume of the generated marker."""
    r = p["d"] / 2.0
    L, cap = p["length"], p["cap_len"]
    tip = p["tip_len"]
    r_cap = r + p["cap_clearance"]
    r_tip = 0.45 * r
    n = p["sides"]
    poly = (n / (2.0 * math.pi)) * math.sin(2.0 * math.pi / n)   # polygon/circle area ratio
    v = 0.0
    v += (math.pi * r_cap ** 2 * cap) if cap > 0 else 0.0
    v += math.pi * r ** 2 * (L - cap - tip)
    v += math.pi * tip / 3.0 * (r * r + r * r_tip + r_tip * r_tip)
    return v * poly


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def build(p, seed=0):
    r = p["d"] / 2.0
    L, cap, tip = p["length"], p["cap_len"], p["tip_len"]
    r_cap = r + p["cap_clearance"]
    r_tip = 0.45 * r
    n = int(p["sides"])
    phase = (int(seed) % 8) * (2.0 * math.pi / (8.0 * n))   # deterministic micro-rotation

    zs, rings = [], []

    def ring(rad):
        return geom.ring_polygon(rad, n, phase=phase)

    if cap > 0:
        zs += [0.0, cap, cap]
        rings += [ring(r_cap), ring(r_cap), ring(r)]
    else:
        zs += [0.0]
        rings += [ring(r)]
    zs += [L - tip, L]
    rings += [ring(r), ring(r_tip)]

    m = geom.Mesh()
    geom.loft(m, "body", rings, zs, cap_start=True, cap_end=True)
    # z -> x, keep y, flip the third axis so the frame stays right handed
    R = np.array([[0.0, 0.0, 1.0],
                  [0.0, 1.0, 0.0],
                  [-1.0, 0.0, 0.0]])
    m.transform("body", R=R, t=[-L / 2.0, 0.0, 0.0])
    return m


# ---------------------------------------------------------------------------
# ctx.parts block (parts schema v1)
# ---------------------------------------------------------------------------
def parts_ctx(p, name, profile=None, com=None):
    """`ctx.parts["pen"]` per parts schema v3:
    tip / tail（point）、body（axis，中点 + 全长 + 半径剖面 + grasp_s）、com。
    """
    r = p["d"] / 2.0
    L = p["length"]
    r_cap = r + (p["cap_clearance"] if p["cap_len"] > 0 else 0.0)
    src = f"variant:marker/{name}"
    if profile is None:
        cap, tip = p["cap_len"], p["tip_len"]
        profile = [[0.0, r_cap], [cap, r_cap], [cap, r], [L - tip, r], [L, 0.45 * r]]
    return {
        "pen": {
            "origin": {"kind": "point", "p": [0.0, 0.0, 0.0], "tol": None, "src": src},
            "tip": {"kind": "point", "p": [L / 2.0, 0.0, 0.0],
                    "size": {"height": r}, "up_axis": "x", "tol": None, "src": src},
            "tail": {"kind": "point", "p": [-L / 2.0, 0.0, 0.0],
                     "size": {"height": r_cap}, "up_axis": "x", "tol": None, "src": src},
            "body": {"kind": "axis", "p": [0.0, 0.0, 0.0], "axis": [1.0, 0.0, 0.0],
                     "size": {"length": L}, "span": [0.0, L],
                     "profile": [[round(float(a), 6), round(float(b), 6)] for a, b in profile],
                     "grasp_s": 0.5, "tol": None, "src": src},
            "com": {"kind": "point", "p": list(com) if com is not None else [0.0, 0.0, 0.0],
                    "com_from": "mass" if com is not None else "geometry",
                    "tol": None, "src": src},
            "grasp": {"kind": "axis", "p": [0.0, 0.0, 0.0], "axis": [1.0, 0.0, 0.0],
                      "span": [-0.03, 0.03], "section": {"shape": "circle", "r_out": r},
                      "tol": None, "src": src},
        }
    }


# ---------------------------------------------------------------------------
# annotation
# ---------------------------------------------------------------------------
def annotate(p, mesh, props, checks, name="TBD"):
    r = p["d"] / 2.0
    L, cap, tip = p["length"], p["cap_len"], p["tip_len"]
    r_cap = r + p["cap_clearance"] if cap > 0 else r

    functional = [
        {"name": "tip", "kind": "tip", "point": [L / 2.0, 0.0, 0.0],
         "axis": [1.0, 0.0, 0.0], "radius": r * 0.45, "segment_len": tip},
        {"name": "tail", "kind": "tail", "point": [-L / 2.0, 0.0, 0.0],
         "axis": [-1.0, 0.0, 0.0], "radius": r_cap},
        {"name": "axis", "kind": "axis_line",
         "point": [0.0, 0.0, 0.0], "axis": [1.0, 0.0, 0.0], "length": L},
    ]
    if cap > 0:
        functional.append({
            "name": "cap_seam", "kind": "seam",
            "point": [-L / 2.0 + cap, 0.0, 0.0], "axis": [-1.0, 0.0, 0.0],
            "radius": r_cap, "note": "cap and body are one rigid body; no detach",
        })

    # two grip bands, one on each side of the centre (U: |s_L-s_R| >= 60 mm)
    grasps = []
    for s in (-0.03, 0.03):
        grasps.append({
            "name": f"grasp_{'tail' if s < 0 else 'tip'}_side",
            "point": [s, 0.0, 0.0],
            "axis": [1.0, 0.0, 0.0],          # pen grasp axis = pen axis
            "close_axis": [0.0, 0.0, 1.0],    # fingers close across the barrel
            "approach": [0.0, 0.0, -1.0],
            "width_m": p["d"],                # gripper opening needed
            "span_m": 0.02,                   # recommended axial grip length
        })

    ann = {
        "schema": "hq-shape-annot/1",
        "parts_schema": "hq-ctx-parts/3",
        "schema_ref": "RoboTwin-OD key point-axis: placement / functional / grasp (文献库 P40_robotwin2.md:131)",
        "family": FAMILY,
        "name": None,
        "units": "m",
        "up_axis": "+z",
        "origin": "geometric centre; main axis +x, tip at +L/2 (same as assets/Pen/the_pen.obj)",
        "seed": None,
        "params": dict(p),
        "derived": {"length": L, "d": p["d"], "cap_len": cap,
                    "r_cap": r_cap, "tip_len": tip, "sides": int(p["sides"]),
                    "cap_position": "tail (-x); the tip at +x always stays exposed"},
        "geometry": {
            "watertight": checks["watertight"],
            "orientation_consistent": checks["orientation_consistent"],
            "n_vertices": checks["n_vertices"],
            "n_faces": checks["n_faces"],
            "volume_m3": checks["volume_m3"],
            "n_parts": len(mesh.parts),
        },
        "mass": props,
        "bbox": {
            "min": [-L / 2.0, -r_cap, -r_cap],
            "max": [L / 2.0, r_cap, r_cap],
            "extent": [L, 2 * r_cap, 2 * r_cap],
        },
        "placement": [{
            "name": "table_rest_lying",
            "point": [0.0, 0.0, -r_cap],
            "axis": [0.0, 0.0, 1.0],
            "support": "line along +x, length L",
        }],
        "functional": functional,
        "grasp": grasps,
        "parts": parts_ctx(p, name, com=props["com"]),
        "mesh_parts": [{"name": nm, "face_start": s, "face_end": e}
                  for nm, s, e in mesh.part_ranges()],
    }
    return ann


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
GEN_FILE = os.path.abspath(__file__)


def generate(params, seed=0, out_dir=None):
    p = dict(params)
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

    relations = check_relations(p)
    mesh = build(p, seed=seed)
    verts, faces = mesh.welded(eps=1e-8)
    checks = geom.check_faces(verts, faces)
    if not (checks["watertight"] and checks["orientation_consistent"] and checks["volume_positive"]):
        raise AssertionError(f"mesh check failed: {checks}")
    v_closed = _volume(p)
    rel = abs(checks["volume_m3"] - v_closed) / v_closed
    if rel > 0.02:
        raise AssertionError(f"volume mismatch: mesh {checks['volume_m3']:.6e} vs closed form {v_closed:.6e}")

    props = geom.mass_properties(verts, faces, p["density"], fill=p["fill"])
    name = p.get("name") or _auto_name(p)
    ann = annotate(p, mesh, props, checks, name=name)
    ann["name"] = name
    ann["seed"] = int(seed)
    ann["generator"] = {"file": "marker.py", "version": VERSION,
                        "sha256": geom.sha256_file(GEN_FILE)}
    ann["relations"] = relations
    ann["volume_closed_form_m3"] = v_closed
    ann["volume_rel_error"] = rel

    if out_dir is None:
        return {"name": name, "annotations": ann, "mesh": (verts, faces)}

    obj_path = os.path.join(out_dir, name + ".obj")
    ply_path = os.path.join(out_dir, name + ".ply")
    ann_path = os.path.join(out_dir, name + ".annotations.json")
    var_path = os.path.join(out_dir, name + ".variant.yaml")

    geom.write_obj(obj_path, verts, faces, header=f"shape_gen/{FAMILY} {name} seed={seed}")
    geom.write_ply(ply_path, verts, faces, header=f"shape_gen/{FAMILY} {name} seed={seed}")
    geom.write_json(ann_path, ann)

    files = {}
    for fn in (name + ".obj", name + ".ply", name + ".annotations.json"):
        files[fn] = geom.sha256_file(os.path.join(out_dir, fn))

    variant = {
        "name": name,
        "fpath": f"{FAMILY}/{name}",
        "body_scale": [1.0, 1.0, 1.0],
        "files": files,
        "source": {
            "generator": f"marker.py@{ann['generator']['sha256'][:16]}",
            "params": {k: v for k, v in p.items() if k != "name"},
            "seed": int(seed),
        },
        "grasp": [{"name": g["name"], "point": g["point"], "axis": g["axis"]}
                  for g in ann["grasp"]],
        "functional_read": {"path": name + ".annotations.json", "key": "parts"},
        "status": "draft",
    }
    geom.write_yaml(var_path, variant)
    return variant


def _auto_name(p):
    cap = f"cap{int(round(p['cap_len'] * 1e3))}" if p["cap_len"] > 0 else "nocap"
    shape = "circle" if int(p["sides"]) >= 32 else f"n{int(p['sides'])}"
    return (f"marker_d{int(round(p['d'] * 1e4))}_l{int(round(p['length'] * 1e3))}"
            f"_{cap}_{shape}")
