"""One-off measurement of the official assets -> nominal_measured.json.

Reads (read only):
    assets/test_tube_rack_standard.ply
    assets/test_tube_standard.ply
    assets/Pen/the_pen.obj
    assets/PenHolder/pen_holder.obj
    configs/sample_loading/random/gym_config.json
    configs/items_handover/random/gym_config.json

and writes the numbers that params.py quotes, together with the sha256 of every
source file, so a reviewer can re-run this and diff.

    python3 measure_nominal.py [--repo DIR] [--out FILE]
"""

import argparse
import json
import os

import numpy as np

import geom

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REPO = os.environ.get("ROBOSYN_REPO", "RoboSynChallenge")  # path of an official-repo checkout
DEFAULT_OUT = os.path.join(HERE, "nominal_measured.json")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def load(path):
    import trimesh
    return trimesh.load(path, force="mesh", process=False)


def _bary_prep(tris2d, za, zb, zc):
    a = tris2d[:, 0]
    b = tris2d[:, 1]
    c = tris2d[:, 2]
    den = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    ok = np.abs(den) > 1e-12
    return a[ok], b[ok], c[ok], den[ok], za[ok], zb[ok], zc[ok]


def occupancy(T, zmin, zmax, xs, ys):
    """Footprint of all triangles whose 3 vertices sit inside [zmin, zmax]."""
    sel = np.all((T[:, :, 2] > zmin) & (T[:, :, 2] < zmax), axis=1)
    Tt = T[sel]
    if len(Tt) == 0:
        return np.zeros((len(ys), len(xs)), bool)
    a, b, c, den, za, zb, zc = _bary_prep(Tt[:, :, :2], Tt[:, 0, 2], Tt[:, 1, 2], Tt[:, 2, 2])
    X, Y = np.meshgrid(xs, ys)
    P = np.stack([X.ravel(), Y.ravel()], 1)
    out = np.zeros(len(P), bool)
    for s in range(0, len(P), 500):
        Q = P[s:s + 500]
        v2x = Q[:, 0][:, None] - a[:, 0][None, :]
        v2y = Q[:, 1][:, None] - a[:, 1][None, :]
        u = (v2x * (c[:, 1] - a[:, 1])[None, :] - v2y * (c[:, 0] - a[:, 0])[None, :]) / den[None, :]
        v = ((b[:, 0] - a[:, 0])[None, :] * v2y - (b[:, 1] - a[:, 1])[None, :] * v2x) / den[None, :]
        out[s:s + 500] = ((u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9)).any(axis=1)
    return out.reshape(X.shape)


def runs(occupied, coords):
    """(solid?, start, end) runs of a 1-D occupancy line."""
    out = []
    cur, start = bool(occupied[0]), coords[0]
    for v, x in zip(occupied[1:], coords[1:]):
        if bool(v) != cur:
            out.append((cur, float(start), float(x)))
            cur, start = bool(v), float(x)
    out.append((cur, float(start), float(coords[-1])))
    return out


def z_levels(V, bins=60):
    h, e = np.histogram(V[:, 2], bins=bins)
    return [(float(e[i]), int(h[i])) for i in range(len(h)) if h[i] > 0]


# ---------------------------------------------------------------------------
# per-asset measurements
# ---------------------------------------------------------------------------
def measure_rack(path):
    m = load(path)
    V, F = m.vertices, m.faces
    T = V[F]
    lo, hi = V.min(0), V.max(0)
    extent = hi - lo

    lv = z_levels(V)
    dense = [z for z, c in lv if c > 300]
    dense.sort()
    upper = [z for z in dense if z > hi[2] * 0.5]
    lower = [z for z in dense if z < 0.0]
    top_face = upper[-1]
    plate_lo = upper[-2] if len(upper) > 1 else top_face - 0.0036
    base_lo = max(lower) if lower else lo[2]
    plate_t = round(float(top_face - plate_lo), 5)
    base_t = round(float(base_lo - lo[2]), 5)

    # footprint of the top plate only (between its own two z levels)
    band = (plate_lo + 0.0002, top_face - 0.0002)

    xs = np.arange(lo[0] + 0.0008, hi[0] - 0.0008, 0.0004)
    occ_x = occupancy(T, *band, xs, np.array([0.0]))[0]
    r_x = runs(occ_x, xs)
    gap_x = [(s, e) for solid, s, e in r_x if not solid and (e - s) > 0.005]
    wall_x = [(s, e) for solid, s, e in r_x if solid and 0.0005 < (e - s) < 0.008]

    ys = np.arange(lo[1] + 0.0008, hi[1] - 0.0008, 0.0004)
    occ_y = occupancy(T, *band, np.array([0.0101]), ys)[:, 0]
    r_y = runs(occ_y, ys)
    gap_y = [(s, e) for solid, s, e in r_y if not solid and (e - s) > 0.005]
    wall_y = [(s, e) for solid, s, e in r_y if solid and 0.0015 < (e - s) < 0.008]

    def pitch(gaps):
        if len(gaps) < 2:
            return None
        c = [0.5 * (s + e) for s, e in gaps]
        return round(float(np.mean(np.diff(c))), 5)

    return {
        "extent_raw": [round(float(x), 5) for x in extent],
        "bounds_raw": [[round(float(x), 5) for x in lo], [round(float(x), 5) for x in hi]],
        "n_faces": int(len(F)),
        "watertight": bool(m.is_watertight),
        "cell_openings_x": [round(e - s, 5) for s, e in gap_x],
        "cell_openings_y": [round(e - s, 5) for s, e in gap_y],
        "walls_x": sorted({round(e - s, 4) for s, e in wall_x}),
        "walls_y": sorted({round(e - s, 4) for s, e in wall_y}),
        "n_cells_x": len(gap_x),
        "n_cells_y": len(gap_y),
        "pitch_x": pitch(gap_x),
        "pitch_y": pitch(gap_y),
        "edge_margin_x": [round(float(gap_x[0][0] - lo[0]), 5), round(float(hi[0] - gap_x[-1][1]), 5)] if gap_x else None,
        "edge_margin_y": [round(float(gap_y[0][0] - lo[1]), 5), round(float(hi[1] - gap_y[-1][1]), 5)] if gap_y else None,
        "top_plate_thickness": plate_t,
        "base_thickness": base_t,
        "z_levels_dense": [round(float(z), 5) for z in dense],
    }


def measure_box_asset(path):
    m = load(path)
    lo, hi = m.vertices.min(0), m.vertices.max(0)
    return {
        "bounds_raw": [[round(float(x), 5) for x in lo], [round(float(x), 5) for x in hi]],
        "extent_raw": [round(float(x), 5) for x in (hi - lo)],
        "n_faces": int(len(m.faces)),
    }


def measure_holder_opening(path):
    """Inner clear width of the pen holder, by scanning along x at the rim."""
    m = load(path)
    V, F = m.vertices, m.faces
    T = V[F]
    y_rim = V[:, 1].max() - 0.004
    xs = np.arange(V[:, 0].min() + 0.001, V[:, 0].max() - 0.001, 0.0004)
    sel = np.all(np.abs(T[:, :, 1] - y_rim) < 0.02, axis=1)
    Tt = T[sel]
    a, b, c, den, _, _, _ = _bary_prep(Tt[:, :, [0, 2]], Tt[:, 0, 1], Tt[:, 1, 1], Tt[:, 2, 1])
    occ = np.zeros(len(xs), bool)
    for s in range(0, len(xs), 400):
        P = xs[s:s + 400]
        v2x = P[:, None] - a[:, 0][None, :]
        v2y = np.full_like(v2x, 0.0) + (0.0 - a[:, 1][None, :])
        u = (v2x * (c[:, 1] - a[:, 1])[None, :] - v2y * (c[:, 0] - a[:, 0])[None, :]) / den[None, :]
        v = ((b[:, 0] - a[:, 0])[None, :] * v2y - (b[:, 1] - a[:, 1])[None, :] * v2x) / den[None, :]
        occ[s:s + 400] = ((u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9)).any(axis=1)
    walls = [(s, e) for solid, s, e in runs(occ, xs) if solid]
    if len(walls) >= 2:
        inner = walls[-1][0] - walls[0][1]
    else:
        inner = None
    return {"inner_opening_x_raw": round(float(inner), 5) if inner else None,
            "wall_runs": [[round(s, 5), round(e, 5)] for s, e in walls]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    A = os.path.join(args.repo, "assets")
    C = os.path.join(args.repo, "configs")
    out = {"schema": "hq-nominal-measured/1", "repo": args.repo, "assets": {}, "configs": {}}

    rack = measure_rack(os.path.join(A, "test_tube_rack_standard.ply"))
    tube = measure_box_asset(os.path.join(A, "test_tube_standard.ply"))
    pen = measure_box_asset(os.path.join(A, "Pen", "the_pen.obj"))
    holder = measure_box_asset(os.path.join(A, "PenHolder", "pen_holder.obj"))
    holder.update(measure_holder_opening(os.path.join(A, "PenHolder", "pen_holder.obj")))

    scales = {}
    for cfg_name, uid in (("sample_loading", "rack"), ("sample_loading", "cube"),
                          ("items_handover", "pen"), ("items_handover", "holder")):
        p = os.path.join(C, cfg_name, "random", "gym_config.json")
        d = json.load(open(p))
        for o in d.get("rigid_object", []):
            if o.get("uid") == uid:
                scales[f"{cfg_name}:{uid}"] = {
                    "body_scale": o.get("body_scale"),
                    "mass": (o.get("attrs") or {}).get("mass"),
                    "contact_offset": (o.get("attrs") or {}).get("contact_offset"),
                    "rest_offset": (o.get("attrs") or {}).get("rest_offset"),
                    "max_convex_hull_num": o.get("max_convex_hull_num"),
                    "sdf_resolution": o.get("sdf_resolution"),
                    "fpath": o.get("shape", {}).get("fpath"),
                }

    files = [os.path.join(A, "test_tube_rack_standard.ply"),
             os.path.join(A, "test_tube_standard.ply"),
             os.path.join(A, "Pen", "the_pen.obj"),
             os.path.join(A, "PenHolder", "pen_holder.obj")]

    out["assets"]["test_tube_rack_standard.ply"] = dict(rack, sha256=geom.sha256_file(files[0]))
    out["assets"]["test_tube_standard.ply"] = dict(tube, sha256=geom.sha256_file(files[1]))
    out["assets"]["Pen/the_pen.obj"] = dict(pen, sha256=geom.sha256_file(files[2]))
    out["assets"]["PenHolder/pen_holder.obj"] = dict(holder, sha256=geom.sha256_file(files[3]))
    out["configs"]["rigid_objects"] = scales

    # scaled values (config body_scale)
    def scaled(asset, key):
        b = out["assets"][asset]
        s = scales[key]["body_scale"]
        return [round(float(e * k), 5) for e, k in zip(b["extent_raw"], s)]

    out["scaled"] = {
        "rack": scaled("test_tube_rack_standard.ply", "sample_loading:rack"),
        "tube": scaled("test_tube_standard.ply", "sample_loading:cube"),
        "pen": scaled("Pen/the_pen.obj", "items_handover:pen"),
        "holder": scaled("PenHolder/pen_holder.obj", "items_handover:holder"),
        "holder_inner_opening_x": round(float(holder["inner_opening_x_raw"] * 0.56), 5),
    }

    geom.write_json(args.out, out)
    print(json.dumps({"out": args.out,
                      "rack_extent": out["scaled"]["rack"],
                      "tube": out["scaled"]["tube"],
                      "pen": out["scaled"]["pen"],
                      "holder_inner": out["scaled"]["holder_inner_opening_x"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
