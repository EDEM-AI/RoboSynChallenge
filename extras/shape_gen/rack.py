"""Generator: test-tube rack (family "rack") - v2, square-cell lattice.

Rebuilt after review: a reference rack and a simulated rack are both a
stack of square-cell grid plates inside a frame with openings in the side walls,
not a round-hole plate.  Structure measured on the official asset
(`test_tube_rack_standard.ply`, see nominal_measured.json):

    3 horizontal plates  (bottom solid 3.6 mm, two grid plates 3.6 mm each)
    12 x 5 square cells  (17.0 mm inside, pitch 20.2 / 19.0, grid wall 3.2 / 2.1)
    cells of the two grid plates aligned vertically
    side walls with big rectangular openings (4 posts, 3 windows on a long side)
    outer 250.2 x 105.6 x 72.1 mm (x1.1 in the task config -> 275.2 x 116.2 x 79.3)

Frame: origin at the bottom centre, +z up, support plane z = 0.  The official
asset is z-centred, so converting is the single translation +H/2 (recorded as
``legacy_root_offset_m``).

Interface:  generate(params, seed, out_dir) -> variant.yaml content
"""

import math
import os

import numpy as np

import geom
import params as P

FAMILY = "rack"
VERSION = "2"


# ---------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------
def default_params():
    return {
        "rows": 4,              # 行
        "cols": 10,             # 列（参考架 4x10；仿真资产 12x5）
        "cell": 0.0215,         # 格子边长（方形格子）
        "pitch_x": 0.0245,      # 相邻格中心距（x）
        "pitch_y": 0.0245,
        "layers": 3,            # 格子板层数 2-3
        "gap": 0.028,           # 层间距（相邻格子板的净距）
        "base_t": 0.0036,       # 底板厚
        "plate_t": 0.0036,      # 每层格子板厚
        "side_wall_t": 0.0030,  # 侧壁厚
        "openings_long": 3,     # 长边窗口数
        "post_w": 0.0120,       # 窗口之间的立柱宽
        "rail_h": 0.0060,       # 侧壁上下横档高
        "margin_x": 0.0050,
        "margin_y": 0.0050,
        "density": P.RACK["density"]["nominal"],
        "material": "pmma",          # pmma / pp / abs / petg（决定密度默认值）
        "color": [0.78, 0.82, 0.88],  # 视觉基色（RGB 0-1）
        "tube_d": 0.020,
        "tube_len": P.MEAS["test_tube_standard.ply"]["tube_length_m"],
    }


MATERIALS = {"pmma": 1190.0, "pp": 900.0, "abs": 1050.0, "petg": 1270.0}


def density_of(p):
    if p.get("density"):
        return float(p["density"])
    return MATERIALS.get(str(p.get("material", "pmma")).lower(), 1190.0)


def derive(p):
    rows, cols = int(p["rows"]), int(p["cols"])
    cell, px, py = p["cell"], p["pitch_x"], p["pitch_y"]
    margin_x, margin_y, sw = p["margin_x"], p["margin_y"], p["side_wall_t"]
    inner_x = (cols - 1) * px + cell
    inner_y = (rows - 1) * py + cell
    Lx = 2 * (sw + margin_x) + inner_x
    Wy = 2 * (sw + margin_y) + inner_y
    layers = int(p["layers"])
    H = p["base_t"] + layers * p["plate_t"] + (layers - 1) * p["gap"]
    x0 = -Lx / 2 + sw + margin_x
    y0 = -Wy / 2 + sw + margin_y
    xs = [x0 + j * px for j in range(cols)]
    ys = [y0 + i * py for i in range(rows)]
    plate_z = [p["base_t"] + k * (p["plate_t"] + p["gap"]) for k in range(layers)]
    return {"Lx": Lx, "Wy": Wy, "H": H, "rows": rows, "cols": cols,
            "xs": xs, "ys": ys, "plate_z": plate_z, "layers": layers,
            "inner_x": inner_x, "inner_y": inner_y,
            "n_cells": rows * cols}


# ---------------------------------------------------------------------------
# relations
# ---------------------------------------------------------------------------
def check_relations(p):
    d = derive(p)
    cell = p["cell"]
    c = (cell - p["tube_d"]) / 2.0
    tan_lim = math.tan(math.radians(P.LIMITS["rack_vertical_deg"]))
    out = []

    def req(ok, rid, text):
        out.append({"id": rid, "ok": bool(ok), "text": text})
        if not ok:
            raise ValueError(f"[{FAMILY}] relation {rid} failed: {text}")

    # S1 格子边长 > 试管外径 + 2*间隙
    req(c >= P.RACK["hole_clearance_c"]["range"][0] - 1e-12, "S1a",
        f"cell clearance c={c*1e3:.2f} mm >= {P.RACK['hole_clearance_c']['range'][0]*1e3:.2f}")
    req(c <= P.RACK["hole_clearance_c"]["range"][1] + 1e-12, "S1b",
        f"cell clearance c={c*1e3:.2f} mm <= {P.RACK['hole_clearance_c']['range'][1]*1e3:.2f}")
    # S2 层数 2-3（至少两层，才能定位 + 托住）
    req(int(p["layers"]) in (2, 3), "S2",
        f"layers={int(p['layers'])} in {{2,3}} (≥2 plates grip the tube)")
    # S3 层间距：两片格子板夹住，歪角不超过官方 5°
    lean = math.atan2(2 * c, max(p["gap"], 1e-9))
    req(lean <= math.radians(P.LIMITS["rack_vertical_deg"]) + 1e-12, "S3",
        f"lean=atan(2c/gap)={math.degrees(lean):.2f} deg <= {P.LIMITS['rack_vertical_deg']} deg")
    # S4 整体在桌面可达范围
    reach = P.LIMITS["rack_reach_m"]
    req(reach[0] - 1e-12 <= d["Lx"] <= reach[1] + 1e-12, "S4a",
        f"Lx={d['Lx']*1e3:.1f} mm in [{reach[0]*1e3:.0f},{reach[1]*1e3:.0f}]")
    req(reach[0] - 1e-12 <= d["Wy"] <= reach[1] + 1e-12, "S4b",
        f"Wy={d['Wy']*1e3:.1f} mm in [{reach[0]*1e3:.0f},{reach[1]*1e3:.0f}]")
    # S5 格壁（pitch - cell）要有肉
    req(p["pitch_x"] - cell >= P.LIMITS["rack_min_grid_wall_m"] - 1e-12, "S5a",
        f"grid wall x={(p['pitch_x']-cell)*1e3:.2f} mm >= {P.LIMITS['rack_min_grid_wall_m']*1e3:.1f}")
    req(p["pitch_y"] - cell >= P.LIMITS["rack_min_grid_wall_m"] - 1e-12, "S5b",
        f"grid wall y={(p['pitch_y']-cell)*1e3:.2f} mm >= {P.LIMITS['rack_min_grid_wall_m']*1e3:.1f}")
    # S6 管要露出架子足够长度才夹得到
    exposed = p["tube_len"] - (d["H"] - p["base_t"])
    req(exposed >= P.LIMITS["rack_min_exposed_m"] - 1e-12, "S6",
        f"exposed tube above the bottom={exposed*1e3:.1f} mm >= {P.LIMITS['rack_min_exposed_m']*1e3:.0f}")
    # S7 侧壁开口：至少一个窗口，立柱够宽、窗口够大
    n = int(p["openings_long"])
    req(n >= 1, "S7a", f"openings_long={n} >= 1")
    span = d["Lx"] - 2 * p["side_wall_t"]
    opening_w = (span - (n + 1) * p["post_w"]) / n
    req(p["post_w"] >= P.LIMITS["rack_min_post_m"] - 1e-12, "S7b",
        f"post_w={p['post_w']*1e3:.1f} mm >= {P.LIMITS['rack_min_post_m']*1e3:.1f}")
    req(opening_w >= P.LIMITS["rack_min_opening_m"] - 1e-12, "S7c",
        f"window width={opening_w*1e3:.1f} mm >= {P.LIMITS['rack_min_opening_m']*1e3:.0f}")
    # S8 架高
    hh = P.LIMITS["rack_height_m"]
    req(hh[0] - 1e-12 <= d["H"] <= hh[1] + 1e-12, "S8",
        f"H={d['H']*1e3:.1f} mm in [{hh[0]*1e3:.0f},{hh[1]*1e3:.0f}]")
    return out


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def _edges(start, n, pitch, cell, total_end):
    e = [start]
    for j in range(n):
        e.append(start + j * pitch)
        e.append(start + j * pitch + cell)
    e.append(total_end)
    out = [e[0]]
    for v in e[1:]:
        if abs(v - out[-1]) > 1e-12:
            out.append(v)
    return out


def square_cell_plate(mesh, name, xs, ys, cell, z0, z1, x0, x1, y0, y1):
    """Slab [x0,x1]x[y0,y1] pierced by square cells (one closed shell)."""
    mesh.begin(name)
    xe = _dedup([x0] + [v for x in xs for v in (x, x + cell)] + [x1])
    ye = _dedup([y0] + [v for y in ys for v in (y, y + cell)] + [y1])
    nu, nv = len(xe) - 1, len(ye) - 1

    hole = np.zeros((nu, nv), dtype=bool)
    for a in range(nu):
        cu = 0.5 * (xe[a] + xe[a + 1])
        for b in range(nv):
            cv = 0.5 * (ye[b] + ye[b + 1])
            for x in xs:
                if x < cu < x + cell:
                    for y in ys:
                        if y < cv < y + cell:
                            hole[a, b] = True
                            break
                    break

    base = len(mesh.v)
    for a in range(nu + 1):
        for b in range(nv + 1):
            mesh.v.append([xe[a], ye[b], z0])
            mesh.v.append([xe[a], ye[b], z1])

    def vid(a, b, top):
        return base + (a * (nv + 1) + b) * 2 + (1 if top else 0)

    for a in range(nu):
        for b in range(nv):
            if hole[a, b]:
                continue
            v00, v10 = vid(a, b, 0), vid(a + 1, b, 0)
            v01, v11 = vid(a, b + 1, 0), vid(a + 1, b + 1, 0)
            t00, t10 = vid(a, b, 1), vid(a + 1, b, 1)
            t01, t11 = vid(a, b + 1, 1), vid(a + 1, b + 1, 1)
            mesh.add_face(t00, t10, t11)          # top (+z)
            mesh.add_face(t00, t11, t01)
            mesh.add_face(v00, v11, v10)          # bottom (-z)
            mesh.add_face(v00, v01, v11)
            if a == 0 or hole[a - 1, b]:
                mesh.quad(t00, t01, v01, v00)     # -u
            if a == nu - 1 or hole[a + 1, b]:
                mesh.quad(t10, v10, v11, t11)     # +u
            if b == 0 or hole[a, b - 1]:
                mesh.quad(t00, v00, v10, t10)     # -v
            if b == nv - 1 or hole[a, b + 1]:
                mesh.quad(t01, t11, v11, v01)     # +v
    mesh.end()


def _dedup(seq, tol=1e-12):
    out = [seq[0]]
    for v in seq[1:]:
        if abs(v - out[-1]) > tol:
            out.append(v)
    return out


def build(p):
    d = derive(p)
    Lx, Wy, H = d["Lx"], d["Wy"], d["H"]
    sw, t, bt = p["side_wall_t"], p["plate_t"], p["base_t"]
    layers = d["layers"]
    cell = p["cell"]

    m = geom.Mesh()
    # bottom plate (solid, the tube rests on it)
    geom.box(m, "base", -Lx / 2, Lx / 2, -Wy / 2, Wy / 2, 0.0, bt)

    # grid plates
    for k, z in enumerate(d["plate_z"]):
        square_cell_plate(m, f"grid_{k}", d["xs"], d["ys"], cell, z, z + t,
                          -Lx / 2 + sw, Lx / 2 - sw, -Wy / 2 + sw, Wy / 2 - sw)

    # side walls: bottom rail + top rail + posts, windows between the posts
    zr0, zr1 = bt, H - p["rail_h"]
    # long sides (y = +-Wy/2), spanning x between the end walls
    xa, xb = -Lx / 2 + sw, Lx / 2 - sw
    n = int(p["openings_long"])
    span = xb - xa
    ow = (span - (n + 1) * p["post_w"]) / n
    for sy, tag in ((-1, "s"), (1, "n")):
        y0 = -Wy / 2 if sy < 0 else Wy / 2 - sw
        geom.box(m, f"rail_low_{tag}", xa, xb, y0, y0 + sw, bt, bt + p["rail_h"])
        geom.box(m, f"rail_top_{tag}", xa, xb, y0, y0 + sw, H - p["rail_h"], H)
        for k in range(n + 1):
            px0 = xa + k * (p["post_w"] + ow)
            geom.box(m, f"post_{tag}{k}", px0, px0 + p["post_w"], y0, y0 + sw,
                     bt + p["rail_h"], H - p["rail_h"])
    # end walls (x = +-Lx/2), one window each, spanning the full depth
    for sx, tag in ((-1, "w"), (1, "e")):
        x0 = -Lx / 2 if sx < 0 else Lx / 2 - sw
        geom.box(m, f"rail_low_{tag}", x0, x0 + sw, -Wy / 2, Wy / 2, bt, bt + p["rail_h"])
        geom.box(m, f"rail_top_{tag}", x0, x0 + sw, -Wy / 2, Wy / 2, H - p["rail_h"], H)
        geom.box(m, f"post_{tag}0", x0, x0 + sw, -Wy / 2, -Wy / 2 + p["post_w"],
                 bt + p["rail_h"], H - p["rail_h"])
        geom.box(m, f"post_{tag}1", x0, x0 + sw, Wy / 2 - p["post_w"], Wy / 2,
                 bt + p["rail_h"], H - p["rail_h"])
    return m, d


def analytic_volume(p, d):
    Lx, Wy, H = d["Lx"], d["Wy"], d["H"]
    sw, t, bt = p["side_wall_t"], p["plate_t"], p["base_t"]
    cell = p["cell"]
    n = int(p["openings_long"])
    v = Lx * Wy * bt
    plate_w, plate_h = Lx - 2 * sw, Wy - 2 * sw
    grid_area = plate_w * plate_h - d["n_cells"] * cell * cell
    v += d["layers"] * grid_area * t
    xa, xb = -Lx / 2 + sw, Lx / 2 - sw
    span = xb - xa
    ow = (span - (n + 1) * p["post_w"]) / n
    rail_h = p["rail_h"]
    v += 2 * span * sw * 2 * rail_h                       # long side rails (2 sides x 2 rails)
    v += 2 * (n + 1) * p["post_w"] * sw * (H - bt - 2 * rail_h)
    v += 2 * Wy * sw * 2 * rail_h                         # end wall rails
    v += 2 * 2 * p["post_w"] * sw * (H - bt - 2 * rail_h)  # end wall posts
    return v


# ---------------------------------------------------------------------------
# ctx.parts block (parts schema v1)
# ---------------------------------------------------------------------------
def parts_ctx(p, d, name, props=None):
    """`ctx.parts["rack"]` per parts schema v3.

    holes  : list, id 从 0 起（按局部 x 再 y 排序），p=孔口中心，axis=+z，
             section=rect（方格，in_or_out=in），size.depth（孔底 = p - depth*axis）
    plates : list, 每层板面 id / p / normal（顶板 = plates[-1]）
    """
    H = d["H"]
    src = f"variant:rack/{name}"
    cell = p["cell"]

    holes = []
    for i in range(d["rows"]):
        for j in range(d["cols"]):
            holes.append({
                "p": [d["xs"][j] + cell / 2.0, d["ys"][i] + cell / 2.0, H],
                "axis": [0.0, 0.0, 1.0],
                "section": {"shape": "rect", "w": cell, "d": cell, "in_or_out": "in"},
                "size": {"depth": H - p["base_t"]},
                "tol": None,
                "src": src,
            })
    holes.sort(key=lambda h: (round(h["p"][0], 9), round(h["p"][1], 9)))
    for k, h in enumerate(holes):
        h["id"] = k
        h["kind"] = "axis"

    plates = []
    n_plates = len(d["plate_z"])
    roles = (["bottom", "top"] if n_plates == 2 else ["bottom", "middle", "top"][:n_plates])
    for k, z in enumerate(d["plate_z"]):
        plates.append({
            "id": k,
            "role": roles[k],
            "kind": "disk",
            "p": [0.0, 0.0, z + p["plate_t"] / 2.0],
            "normal": [0.0, 0.0, 1.0],
            "section": {"shape": "rect",
                        "w": d["Lx"] - 2 * p["side_wall_t"],
                        "d": d["Wy"] - 2 * p["side_wall_t"],
                        "in_or_out": "out"},
            "tol": None,
            "src": src,
        })

    return {
        "rack": {
            "origin": {"kind": "point", "p": [0.0, 0.0, 0.0], "tol": None, "src": src},
            "holes": {"kind": "list", "items": holes, "tol": None, "src": src},
            "plates": {"kind": "list", "items": plates, "tol": None, "src": src},
            "com": {"kind": "point", "p": list(props["com"]), "com_from": "mass",
                    "tol": None, "src": src},

        }
    }


# ---------------------------------------------------------------------------
# annotation
# ---------------------------------------------------------------------------
def annotate(p, d, mesh, props, checks, name="TBD"):
    Lx, Wy, H = d["Lx"], d["Wy"], d["H"]
    cell = p["cell"]
    c = (cell - p["tube_d"]) / 2.0
    parts = parts_ctx(p, d, name, props)

    holes = []
    for it in parts["rack"]["holes"]["items"]:
        holes.append({
            "name": f"hole_{it['id']:02d}",
            "kind": "hole",
            "point": list(it["p"]),
            "axis": [0.0, 0.0, -1.0],          # insertion direction
            "cell": cell,
            "radius_equiv": cell / 2.0,
            "depth_to_support": it["size"]["depth"],
            "gap": c,
        })

    # grasp: the posts of the long side walls (12 mm wide, nearly full height)
    grasps = []
    span = Lx - 2 * p["side_wall_t"]
    n = int(p["openings_long"])
    ow = (span - (n + 1) * p["post_w"]) / n
    for sx, tag in ((-1, "left"), (1, "right")):
        px = (-Lx / 2 + p["side_wall_t"] + p["post_w"] / 2) if sx < 0 else \
             (Lx / 2 - p["side_wall_t"] - p["post_w"] / 2)
        grasps.append({
            "name": f"grasp_post_{tag}",
            "point": [px, Wy / 2 - p["side_wall_t"] / 2, H / 2],
            "axis": [0.0, 0.0, 1.0],           # fingers close vertically around the post? no:
            "close_axis": [1.0, 0.0, 0.0],
            "approach": [0.0, 1.0, 0.0],
            "width_m": p["post_w"],
            "span_m": H - p["base_t"] - 2 * p["rail_h"],
        })

    ann = {
        "schema": "hq-shape-annot/1",
        "parts_schema": "hq-ctx-parts/3",
        "schema_ref": "RoboTwin-OD key point-axis: placement / functional / grasp (文献库 P40_robotwin2.md:131)",
        "family": FAMILY,
        "name": name,
        "units": "m",
        "up_axis": "+z",
        "origin": "bottom centre of the bounding box; support plane z=0",
        "seed": None,
        "params": dict(p),
        "material": {"name": p.get("material", "pmma"), "density": density_of(p),
                     "color": list(p.get("color", [0.78, 0.82, 0.88]))},
        "derived": {
            "outer_m": {"Lx": Lx, "Wy": Wy, "H": H},
            "rows": d["rows"], "cols": d["cols"], "n_cells": d["n_cells"],
            "layers": d["layers"],
            "grid_plate_z": d["plate_z"],
            "cell": cell, "grid_wall": {"x": p["pitch_x"] - cell, "y": p["pitch_y"] - cell},
            "side_wall_t": p["side_wall_t"], "openings_long": int(p["openings_long"]),
        },
        "geometry": {
            "watertight": checks["watertight"],
            "orientation_consistent": checks["orientation_consistent"],
            "n_vertices": checks["n_vertices"],
            "n_faces": checks["n_faces"],
            "volume_m3": checks["volume_m3"],
            "n_parts": len(mesh.parts),
        },
        "mass": props,
        "bbox": {"min": [-Lx / 2, -Wy / 2, 0.0], "max": [Lx / 2, Wy / 2, H],
                 "extent": [Lx, Wy, H]},
        "legacy_root_offset_m": [0.0, 0.0, H / 2.0],
        "placement": [{
            "name": "table_rest", "point": [0.0, 0.0, 0.0], "axis": [0.0, 0.0, 1.0],
            "support_polygon": [[-Lx / 2, -Wy / 2, 0.0], [Lx / 2, -Wy / 2, 0.0],
                                [Lx / 2, Wy / 2, 0.0], [-Lx / 2, Wy / 2, 0.0]],
        }],
        "functional": holes,
        "grasp": grasps,
        "parts": parts,
        "notes": {"plates": "the rack top face is plates[-1] (top grid plate); "
                            "no separate 'top' part (v3 rack rows are holes / plates)"},
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
    mesh, d = build(p)
    verts, faces = mesh.welded_parts()
    checks = geom.check_faces(verts, faces)
    v_analytic = analytic_volume(p, d)
    rel_err = abs(checks["volume_m3"] - v_analytic) / v_analytic
    if rel_err > 0.02:
        raise AssertionError(
            f"volume mismatch: mesh {checks['volume_m3']:.6e} vs analytic {v_analytic:.6e}")
    if not (checks["watertight"] and checks["orientation_consistent"] and checks["volume_positive"]):
        raise AssertionError(f"mesh check failed: {checks}")

    props = geom.mass_properties(verts, faces, density_of(p))
    name = p.get("name") or _auto_name(p, d)
    ann = annotate(p, d, mesh, props, checks, name=name)
    ann["seed"] = int(seed)
    ann["generator"] = {"file": "rack.py", "version": VERSION,
                        "sha256": geom.sha256_file(GEN_FILE)}
    ann["relations"] = relations
    ann["volume_analytic_m3"] = v_analytic
    ann["volume_rel_error"] = rel_err

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
            "generator": f"rack.py@{ann['generator']['sha256'][:16]}",
            "params": {k: v for k, v in p.items() if k != "name"},
            "seed": int(seed),
        },
        "grasp": [{"name": g["name"], "point": g["point"], "axis": g["axis"]}
                  for g in ann["grasp"][:2]],
        # registry pointer: consumers read the parts block here
        "functional_read": {"path": name + ".annotations.json", "key": "parts"},
        "material": p.get("material", "pmma"),
        "color": list(p.get("color", [0.78, 0.82, 0.88])),
        "status": "draft",
    }
    geom.write_yaml(var_path, variant)
    return variant


def tube_for(rack_params, rng=None, clearance=None):
    """配套的试管参数：孔径/格边由架子定，管径 = 格边 - 2c（立得住的约束由 S3 反推 gap 上限）。

    抽样时**先抽架子、再派生管**，不是各抽各的。
    """
    p = dict(rack_params)
    d = derive(p)
    cell = p["cell"]
    # 立得住：atan(2c/gap) <= 5 deg  ->  c <= gap*tan(5)/2
    c_max = 0.5 * p["gap"] * math.tan(math.radians(P.LIMITS["rack_vertical_deg"]))
    c_min = P.RACK["hole_clearance_c"]["range"][0]
    if clearance is None:
        clearance = (c_min + min(c_max, P.RACK["hole_clearance_c"]["range"][1])) / 2.0
    else:
        clearance = float(np.clip(clearance, c_min, min(c_max, P.RACK["hole_clearance_c"]["range"][1])))
    d_tube = cell - 2.0 * clearance
    if d_tube <= 0.001:
        raise ValueError(f"tube_for: cell {cell*1e3:.1f} mm - 2c {2*clearance*1e3:.1f} mm "
                         f"<= 1 mm, no usable tube")
    return {"tube_d": d_tube, "tube_len": p.get("tube_len", 0.19503),
            "hole_clearance_c": clearance, "rack_cell": cell, "rack_gap": p["gap"]}


def _auto_name(p, d):
    return (f"rack_{d['rows']}x{d['cols']}_c{int(round(p['cell'] * 1e4))}"
            f"_L{int(p['layers'])}_h{int(round(d['H'] * 1e3))}")
