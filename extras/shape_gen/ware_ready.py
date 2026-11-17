"""Ready-made library variants: bell / plate / fork / spoon.

Reads the normalised base meshes written by ``ware_import.py`` (plain numpy, no
trimesh), scales each one to a reference size, and emits, per class, 5-8
variants in the same format as ``marker_ready.py``:

    out/libimport/<class>/<name>/<name>.obj / .ply / .annotations.json / .variant.yaml
    out/libimport/<class>/index.json
    images/<class>_library_comparison.png

Annotations follow ctx.parts 第 3 版.  Imported meshes are scanned or artist
meshes: most are not watertight, so ``geometry.watertight`` is reported as
measured and the mass is an explicit estimate (``mass.source``).

    python3 ware_ready.py            # build every variant + sheets
    python3 ware_ready.py bell       # one class
"""

import json
import os
import sys

import numpy as np

import geom
import render

HERE = os.path.dirname(os.path.abspath(__file__))
NORM = os.path.expanduser("~/objaverse_ware/normalised")
OUT = os.path.join(HERE, "out", "libimport")
IMAGES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")

VERSION = "1"
# effective density: imported meshes are shells, not solid bodies, so the mass is
# an estimate (recorded in mass.source) - sim-side mass stays the scene's business
DENSITY = {"bell": 1050.0, "plate": 2400.0, "fork": 7900.0, "spoon": 7900.0}
MATERIAL = {"bell": "abs_brass_shell", "plate": "porcelain", "fork": "steel", "spoon": "steel"}

# reference size envelope (m) and the size that is set per variant
ENVELOPE = {
    "bell": {"base_d": (0.070, 0.110), "h_over_d": (0.40, 1.00), "round": 0.25,
             "mass": (0.005, 0.600), "contact": 0.70, "top_r": 0.40},
    "plate": {"diameter": (0.150, 0.260), "h_over_d": (0.04, 0.32), "round": 0.18,
              "mass": (0.050, 1.200)},
    "fork": {"length": (0.170, 0.225), "head_w": (0.018, 0.042), "handle_w": (0.0020, 0.018),
             "mass": (0.005, 0.180), "grasp_to_com": 0.25},
    "spoon": {"length": (0.130, 0.180), "head_w": (0.022, 0.045), "handle_w": (0.0020, 0.018),
              "mass": (0.005, 0.180), "grasp_to_com": 0.25},
}

# (class, variant name, base key, target size in m); written by ware_select.py
#   bell  : target = base diameter
#   plate : target = rim diameter
#   fork  : target = length
#   spoon : target = length
SEL = os.path.join(HERE, "selection.json")


def load_variants():
    if not os.path.exists(SEL):
        raise SystemExit("run ware_select.py first")
    with open(SEL, encoding="utf-8") as fh:
        rows = json.load(fh)
    return [(r["cls"], r["name"], r["key"], r["target_m"]) for r in rows]


VARIANTS = load_variants()


# ---------------------------------------------------------------------------
def load_base(key):
    v = np.load(os.path.join(NORM, key + ".v.npy"))
    f = np.load(os.path.join(NORM, key + ".f.npy"))
    cp = os.path.join(NORM, key + ".c.npy")
    fc = np.load(cp) if os.path.exists(cp) else np.zeros(len(f), dtype=np.int64)
    with open(os.path.join(NORM, key + ".json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    return v, f, meta, fc


MAX_PARTS = 64


def mesh_parts(fc):
    """Contiguous runs of the per-face component id -> collision chunks.

    Scanned meshes can break into thousands of shells; more than MAX_PARTS runs
    are useless for CoACD, so then it is one full-range part plus an explicit
    ``mesh_parts_note`` (same convention as the other generators).
    """
    out, s = [], 0
    for i in range(1, len(fc) + 1):
        if i == len(fc) or fc[i] != fc[s]:
            out.append({"name": f"comp{int(fc[s])}_run{len(out)}",
                        "face_start": int(s), "face_end": int(i)})
            s = i
    if len(out) > MAX_PARTS:
        return ([{"name": "all", "face_start": 0, "face_end": int(len(fc))}],
                f"not decomposed: {len(out)} component runs > {MAX_PARTS}, "
                f"one full-range part instead")
    return out, None


def profile(w, axis=2, n=30):
    """(t, radius) along the up axis; radius = 85th percentile of the ring."""
    t = w[:, axis]
    lo, hi = float(t.min()), float(t.max())
    rad = np.linalg.norm(w[:, [i for i in range(3) if i != axis]], axis=1)
    out = []
    for i in range(n):
        a = lo + (hi - lo) * i / n
        b = lo + (hi - lo) * (i + 1) / n
        sel = (t >= a) & (t < b)
        if sel.any():
            out.append((0.5 * (a + b), float(np.percentile(rad[sel], 85))))
    return out


def _rad_at(prof, t, k=2):
    i = min(range(len(prof)), key=lambda j: abs(prof[j][0] - t))
    lo, hi = max(0, i - k), min(len(prof), i + k + 1)
    return max(p[1] for p in prof[lo:hi])


def scale_to(v, s):
    return v * float(s)


def _inertia_ok(mass, inertia, com, lo, hi):
    """正定 + 三角不等式 + 质心在包围盒内（扫描件的质量积分会给出负值）。"""
    if mass <= 0:
        return False
    I = np.asarray(inertia, dtype=float)
    d = np.diag(I)
    if (d <= 0).any():
        return False
    if not (d[0] + d[1] >= d[2] and d[0] + d[2] >= d[1] and d[1] + d[2] >= d[0]):
        return False
    c = np.asarray(com, dtype=float)
    return bool(np.all(c >= lo - 1e-9) and np.all(c <= hi + 1e-9))


def mass_est(v, f, cls):
    chk = geom.check_faces(v, f)
    lo, hi = v.min(0), v.max(0)
    if chk["watertight"] and chk["volume_positive"]:
        mp = geom.mass_properties(v, f, DENSITY[cls])
        if _inertia_ok(mp["mass_kg"], mp["inertia_kgm2"], mp["com"], lo, hi):
            return {"volume_m3": mp["volume_m3"], "mass_kg": mp["mass_kg"],
                    "com": mp["com"], "inertia_kgm2": mp["inertia_kgm2"],
                    "source": "mesh volume x density (mesh is watertight)"}
    # not watertight: the signed volume of the surface is still the best single
    # estimate for these single-shell imports; clamp it to the bbox volume.
    vol = abs(float(chk["volume_m3"]))
    ext = v.max(0) - v.min(0)
    bbox_vol = float(ext[0] * ext[1] * ext[2])
    if not (0.0 < vol <= bbox_vol):
        vol = 0.35 * bbox_vol
    mass = vol * DENSITY[cls]
    com = geom.surface_centroid(v, f)
    inertia = mass / 12.0 * np.array([
        ext[1] ** 2 + ext[2] ** 2, ext[0] ** 2 + ext[2] ** 2, ext[0] ** 2 + ext[1] ** 2])
    why = ("mesh mass properties failed a sanity check (non-positive inertia or "
           "centre outside the bbox); estimated instead"
           if (chk["watertight"] and chk["volume_positive"]) else
           "mesh is not watertight (shell)")
    return {"volume_m3": float(vol), "mass_kg": float(mass), "com": com.tolist(),
            "inertia_kgm2": (np.diag(inertia)).tolist(),
            "source": (f"signed mesh volume (estimate); inertia = uniform bbox; {why}")}


# ---------------------------------------------------------------------------
def parts_bell(w, name, com, m):
    src = f"variant:bell/{name}"
    H, overall_r, cover_r = m["H"], m["overall_r"], m["cover_r"]
    contact_r, button_r = m["base_r"], m["button_r"]
    cover_h = max(0.004, m["cover_top_z"] - m["cover_bottom_z"])
    return {"button": {
        "origin": {"kind": "point", "p": [0.0, 0.0, 0.0], "tol": None, "src": src},
        "com": {"kind": "point", "p": [round(float(x), 6) for x in com],
                "com_from": "geometry", "tol": None, "src": src},
        "press": {"kind": "joint", "q_name": "press", "axis": [0.0, 0.0, -1.0],
                  "range": [-m["stroke"], 0.0], "travel": m["stroke"], "unit": "m",
                  "link": "button_cover", "tol": None, "src": src},
        "cover_top": {"kind": "disk", "p": [0.0, 0.0, round(H, 6)],
                      "normal": [0.0, 0.0, 1.0],
                      "section": {"shape": "circle", "r_out": round(button_r, 6)},
                      "size": {"diameter": round(2 * cover_r, 6),
                               "cover_height": round(cover_h, 6),
                               "overall_diameter": round(2 * overall_r, 6)},
                      "link": "button_cover", "provisional": True, "tol": None, "src": src},
        "base": {"kind": "disk", "p": [0.0, 0.0, 0.0], "normal": [0.0, 0.0, -1.0],
                 "section": {"shape": "circle", "r_out": round(contact_r, 6)},
                 "size": {"overall_diameter": round(2 * overall_r, 6)},
                 "provisional": True, "tol": None, "src": src},
    }}


def parts_plate(w, name, com, m):
    src = f"variant:plate/{name}"
    rim_r, rim_z = m["rim_r"], m["rim_z"]
    rim = []
    for k, ang in enumerate([0.0, np.pi / 2, np.pi, 3 * np.pi / 2]):
        rim.append({"id": k, "p": [round(rim_r * np.cos(ang), 6), round(rim_r * np.sin(ang), 6),
                                   round(rim_z, 6)],
                    "normal": [round(np.cos(ang), 6), round(np.sin(ang), 6), 0.0],
                    "size": {"thickness": round(m["rim_t"], 6)},
                    "tol": None, "src": src})
    return {"plate": {
        "origin": {"kind": "point", "p": [0.0, 0.0, 0.0], "tol": None, "src": src},
        "com": {"kind": "point", "p": [round(float(x), 6) for x in com],
                "com_from": "geometry", "tol": None, "src": src},
        "center": {"kind": "disk", "p": [0.0, 0.0, round(rim_z, 6)],
                   "normal": [0.0, 0.0, 1.0],
                   "section": {"shape": "circle", "r_out": round(rim_r, 6)},
                   "tol": None, "src": src},
        "bottom": {"kind": "disk", "p": [0.0, 0.0, 0.0], "normal": [0.0, 0.0, -1.0],
                   "section": {"shape": "circle", "r_out": round(m["foot_r"], 6)},
                   "placement_surface": True, "tol": None, "src": src},
        "rim": {"kind": "list", "items": rim, "tol": None, "src": src},
    }}


def tool_geometry(w, f, m):
    """Neck position / widths / thickness for a fork or spoon."""
    L = m["L"]
    x_neck = m["neck_x"]
    prof = m["profile_x_w"]                      # [(x, width, thickness)]
    widths = [p[1] for p in prof]
    body_w = float(np.median([p[1] for p in prof[: max(2, len(prof) // 2)]]))

    def thickness_at(x0, x1):
        sel = (w[:, 0] >= x0) & (w[:, 0] <= x1)
        return float(w[sel, 2].max() - w[sel, 2].min()) if sel.any() else 0.0

    def width_at(x0, x1):
        """真实外宽：该 x 切片内网格 y 方向的跨度（不再用线性插值）。"""
        sel = (w[:, 0] >= x0) & (w[:, 0] <= x1)
        return float(w[sel, 1].max() - w[sel, 1].min()) if sel.any() else 0.0

    head_w = float(np.percentile(widths[-4:], 85)) if len(widths) >= 4 else max(widths)
    handle_w = body_w
    com = geom.surface_centroid(w, f)
    x_grasp = min(com[0], x_neck - 0.02 * L)
    x_grasp = max(x_grasp, w[:, 0].min() + 0.25 * L)
    return {"L": L, "x_neck": x_neck, "head_w": head_w, "handle_w": handle_w,
            "com": com, "x_grasp": x_grasp,
            "thickness_at": thickness_at, "width_at": width_at}


def parts_tool(kind, w, f, name, m, com=None):
    g = tool_geometry(w, f, m)
    src = f"variant:{kind}/{name}"
    L = g["L"]
    prof = m["profile_x_w"]
    x0 = w[:, 0].min()
    body_profile = [[round(float(x - x0), 6), round(float(wd / 2.0), 6)] for x, wd, _t in prof]
    items = []
    x_tail, x_neck = x0, g["x_neck"]
    for k in range(5):
        fr = k / 4.0
        x = x_tail + fr * (x_neck - x_tail)
        wid = round(g["width_at"](x - 0.01, x + 0.01), 6)
        if wid <= 0.0:
            wid = float(m["body_w"] + fr * (g["head_w"] * 0.45 - m["body_w"]))
        thick = round(g["thickness_at"](x - 0.01, x + 0.01), 6)
        items.append({"id": k, "p": [round(x, 6), 0.0, 0.0], "axis": [1.0, 0.0, 0.0],
                      # 字段表第 3 版：柄中心线每点给 section rect（w = 真实宽、d = 厚）；
                      # 写法与 tableware.py 一致：rect + in_or_out
                      # + size.thickness + src，不再另给 size.width
                      "section": {"shape": "rect", "w": round(wid, 6), "d": thick,
                                  "in_or_out": "out"},
                      "size": {"thickness": thick},
                      "tol": None, "src": src})
    return {kind: {
        "origin": {"kind": "point", "p": [0.0, 0.0, 0.0], "tol": None, "src": src},
        "head": {"kind": "point", "p": [round(L / 2, 6), 0.0, 0.0], "tol": None, "src": src},
        "tail": {"kind": "point", "p": [round(x0, 6), 0.0, 0.0], "tol": None, "src": src},
        "body": {"kind": "axis", "p": [0.0, 0.0, 0.0], "axis": [1.0, 0.0, 0.0],
                 "size": {"length": round(L, 6)},
                 "span": [round(x0, 6), round(x0 + L, 6)], "profile": body_profile,
                 "grasp_s": round(float((g["x_grasp"] - x0) / L), 4), "tol": None, "src": src},
        "handle": {"kind": "list", "items": items,
                   "grasp_s": round(float((g["x_grasp"] - x0) / L), 4),
                   "tol": None, "src": src},
        "com": {"kind": "point",
                "p": [round(float(x), 6) for x in (com if com is not None else g["com"])],
                "com_from": "mass" if com is not None else "geometry",
                "tol": None, "src": src},
    }}


# ---------------------------------------------------------------------------
def relations_bell(m, parts):
    env = ENVELOPE["bell"]
    out = []

    def req(ok, rid, text):
        out.append({"id": rid, "ok": bool(ok), "text": text})
        return ok

    p = parts["button"]
    overall_r = p["cover_top"]["size"]["overall_diameter"] / 2.0
    contact_r = p["base"]["section"]["r_out"]
    cover_r = p["cover_top"]["size"]["diameter"] / 2.0
    button_r = p["cover_top"]["section"]["r_out"]
    ok = True
    ok &= req(env["base_d"][0] <= 2 * overall_r <= env["base_d"][1], "BK1",
              f"overall d={2*overall_r*1e3:.1f} mm in [{env['base_d'][0]*1e3:.0f}, "
              f"{env['base_d'][1]*1e3:.0f}] (real bell is clearly bigger than sim 63 mm)")
    ok &= req(env["h_over_d"][0] <= m["H"] / (2 * overall_r) <= env["h_over_d"][1], "BK2",
              f"H/overall_d={m['H']/(2*overall_r):.2f} in {env['h_over_d']}")
    ok &= req(m["roundness"] <= env["round"], "BK3",
              f"roundness |dx-dy|/max={m['roundness']:.3f} <= {env['round']} (a bell is round)")
    ok &= req(contact_r >= 0.30 * overall_r, "BK4",
              f"contact r={contact_r*1e3:.1f} >= 0.30 x overall r={overall_r*1e3:.1f}")
    ok &= req(abs(m["stroke"] - 0.005) < 1e-9, "BK5",
              f"stroke={m['stroke']*1e3:.1f} mm == 5 (declared joint)")
    ok &= req(0.15 * cover_r <= button_r <= 0.95 * cover_r, "BK6",
              f"button r={button_r*1e3:.1f} in [0.15, 0.95] x cover r (top press area)")
    ok &= req(m["contact_frac"] >= env["contact"], "BK7",
              f"contact/overall={m['contact_frac']:.2f} >= {env['contact']} "
              f"(flat seat; a hanging bell has almost no contact ring)")
    ok &= req(m["top_r_frac"] <= env["top_r"], "BK8",
              f"top band r/overall={m['top_r_frac']:.2f} <= {env['top_r']} "
              f"(button on top, not a carry handle)")
    return out, bool(ok)


def relations_plate(m, parts):
    env = ENVELOPE["plate"]
    out = []

    def req(ok, rid, text):
        out.append({"id": rid, "ok": bool(ok), "text": text})
        return ok

    p = parts["plate"]
    rim_r = p["center"]["section"]["r_out"]
    ok = True
    ok &= req(env["diameter"][0] <= 2 * rim_r <= env["diameter"][1], "PL1",
              f"diameter={2*rim_r*1e3:.1f} mm in [{env['diameter'][0]*1e3:.0f}, "
              f"{env['diameter'][1]*1e3:.0f}]")
    ok &= req(env["h_over_d"][0] <= m["H"] / (2 * rim_r) <= env["h_over_d"][1], "PL2",
              f"H/d={m['H']/(2*rim_r):.3f} in {env['h_over_d']}")
    ok &= req(m["roundness"] <= env["round"], "PL3",
              f"roundness={m['roundness']:.3f} <= {env['round']}")
    ok &= req(p["bottom"]["section"]["r_out"] <= 1.001 * rim_r, "PL4",
              f"foot r={p['bottom']['section']['r_out']*1e3:.1f} <= rim r={rim_r*1e3:.1f}")
    ok &= req(m["rim_t"] > 1e-4, "PL5", f"rim thickness={m['rim_t']*1e3:.2f} mm > 0")
    return out, bool(ok)


def relations_tool(kind, m, parts):
    env = ENVELOPE[kind]
    out = []

    def req(ok, rid, text):
        out.append({"id": rid, "ok": bool(ok), "text": text})
        return ok

    g = parts[kind]
    L = g["body"]["size"]["length"]
    head_w = m["head_w"]
    handle_w = m["handle_w"]
    grasp_s = g["handle"]["grasp_s"]
    x0 = g["tail"]["p"][0]
    x_grasp = x0 + grasp_s * L
    com_x = g["com"]["p"][0]
    prefix = "FK" if kind == "fork" else "SP"
    ok = True
    ok &= req(env["length"][0] <= L <= env["length"][1], prefix + "1",
              f"L={L*1e3:.1f} mm in [{env['length'][0]*1e3:.0f}, {env['length'][1]*1e3:.0f}]")
    ok &= req(env["head_w"][0] <= head_w <= env["head_w"][1], prefix + "2",
              f"head w={head_w*1e3:.1f} mm in {env['head_w']}")
    ok &= req(env["handle_w"][0] <= handle_w <= env["handle_w"][1], prefix + "3",
              f"handle w={handle_w*1e3:.1f} mm in {env['handle_w']}")
    ok &= req(x_grasp <= m["neck_x"] - 0.001, prefix + "4",
              f"grasp x={x_grasp*1e3:.1f} <= neck x={m['neck_x']*1e3:.1f} (grip on the handle)")
    ok &= req(abs(x_grasp - com_x) <= env["grasp_to_com"] * L, prefix + "5",
              f"|grasp - com|={abs(x_grasp-com_x)*1e3:.1f} mm <= "
              f"{env['grasp_to_com']*L*1e3:.0f} mm (grip near the centre of mass)")
    ok &= req(handle_w <= 0.085, prefix + "6", "handle fits the 100 mm gripper")
    ok &= req(mass_in(m, kind), prefix + "7",
              f"mass={m['mass_kg']*1e3:.1f} g in {env['mass']}")
    return out, bool(ok)


def mass_in(m, kind):
    lo, hi = ENVELOPE[kind]["mass"]
    return lo <= m["mass_kg"] <= hi


# ---------------------------------------------------------------------------
def build(cls, key, target, name):
    v0, f, meta, fc = load_base(key)
    v = np.array(v0, dtype=float)
    f = np.array(f, dtype=np.int64)
    prof = profile(v)
    if cls == "bell":
        base_r0 = max(p[1] for p in prof)
        native = 2.0 * base_r0
    elif cls == "plate":
        native = 2.0 * max(p[1] for p in prof)
    else:
        native = float(meta["L"])
    s = float(target) / native
    v = v * s
    ext = v.max(0) - v.min(0)
    if cls in ("bell", "plate"):
        v[:, 0] -= (v[:, 0].min() + v[:, 0].max()) / 2.0
        v[:, 1] -= (v[:, 1].min() + v[:, 1].max()) / 2.0
        v[:, 2] -= v[:, 2].min()
    else:
        v[:, 0] -= (v[:, 0].min() + v[:, 0].max()) / 2.0
        v[:, 1] -= (v[:, 1].min() + v[:, 1].max()) / 2.0
        v[:, 2] -= (v[:, 2].min() + v[:, 2].max()) / 2.0

    mass = mass_est(v, f, cls)
    prof = profile(v)
    m = {"extent": (v.max(0) - v.min(0)).tolist(), "mass_kg": mass["mass_kg"],
         "roundness": float(abs(ext[0] - ext[1]) / max(ext[0], ext[1])),
         "stroke": 0.005}
    if cls == "bell":
        H = float(v[:, 2].max())
        r_max = max(p[1] for p in prof)
        z_max = max(prof, key=lambda p: p[1])[0]
        above = [p for p in prof if p[0] > z_max + 0.02 * H]
        cover_r = max((p[1] for p in above), default=r_max * 0.7)
        # base slab top: first z above the widest ring where the radius drops
        z_cut = z_max + 0.02 * H
        for p in sorted(above, key=lambda p: p[0]):
            if p[1] < 0.80 * r_max:
                z_cut = p[0]
                break
        overall_r = max(p[1] for p in prof)
        contact_r = _rad_at(prof, 0.02 * H)
        cover_r = max((p[1] for p in prof if p[0] >= 0.20 * H), default=0.9 * overall_r)
        r_all = np.linalg.norm(v[:, :2], axis=1)
        top_band = r_all[v[:, 2] > 0.90 * H]
        m["contact_frac"] = float(contact_r / overall_r)
        m["top_r_frac"] = float(np.percentile(top_band, 90) / overall_r) if len(top_band) else 1.0
        m.update({"H": H, "base_r": contact_r, "overall_r": overall_r, "cover_r": cover_r,
                  "cover_bottom_z": 0.20 * H, "cover_top_z": H,
                  "button_r": float(np.clip(_rad_at(prof, H - 0.10 * H), 0.35 * cover_r,
                                            0.85 * cover_r))})
        m["cover_top_z"] = H
    elif cls == "plate":
        H = float(v[:, 2].max())
        rim_r = max(p[1] for p in prof)
        rim_z = max(prof, key=lambda p: p[1])[0]
        foot_r = _rad_at(prof, 0.004 * H)
        m.update({"H": H, "rim_r": rim_r, "rim_z": rim_z, "foot_r": foot_r,
                  "rim_t": float(max(0.0005, 0.35 * H - 0.6 * (rim_r - foot_r)))})
    else:
        m.update({"L": float(meta["L"]) * s, "W": float(meta["W"]) * s,
                  "neck_x": float(meta["neck_x"]) * s,
                  "body_w": float(meta["body_w"]) * s,
                  "head_w": float(meta["head_w"]) * s})
        m["profile_x_w"] = [(x * s, wd * s, t * s) for x, wd, t in meta["profile_x_w"]]
        x0 = -m["L"] / 2.0
        hw = [wd for x, wd, _t in m["profile_x_w"] if x0 <= x <= m["neck_x"]]
        m["handle_w"] = float(np.percentile(hw, 25)) if hw else float(meta["body_w"]) * s
    if cls == "bell":
        parts = parts_bell(v, name, mass["com"], m)
        rel, ok = relations_bell(m, parts)
    elif cls == "plate":
        parts = parts_plate(v, name, mass["com"], m)
        rel, ok = relations_plate(m, parts)
    else:
        parts = parts_tool(cls, v, f, name, m, mass["com"])
        rel, ok = relations_tool(cls, m, parts)
    ann = {
        "schema": "hq-shape-annot/1",
        "parts_schema": "hq-ctx-parts/3",
        "family": cls,
        "name": name,
        "units": "m",
        "up_axis": "+z",
        "origin": ("bottom centre; support plane z=0" if cls in ("bell", "plate")
                   else "geometric centre; long axis +x, head at +L/2"),
        "source": {
            "kind": "ready-made mesh, normalised (PCA + up sign + uniform scale)",
            "base": meta["name"], "license": meta["license"], "author": meta["author"],
            "base_sha256": meta["source_sha256"],
            "base_file": meta.get("source_file", ""),
            "url": meta.get("url", ""),
            "source_kind": meta.get("source_kind", "objaverse_lvis"),
            "import_sha256": geom.sha256_file(os.path.join(HERE, "ware_import.py")),
            "native_extent_m": [round(x, 6) for x in meta["extent"]],
            "target_size_m": round(float(target), 6), "scale": round(float(s), 6),
        },
        "params": {"target_m": float(target), "density": DENSITY[cls],
                   "material": MATERIAL[cls], "stroke": m["stroke"]},
        "derived": {k: (round(v_, 6) if isinstance(v_, float) else v_)
                    for k, v_ in m.items() if k not in ("profile_x_w", "mass_kg")},
        "mass": mass,
        "bbox": {"min": [round(float(x), 6) for x in v.min(0)],
                 "max": [round(float(x), 6) for x in v.max(0)],
                 "extent": [round(float(x), 6) for x in (v.max(0) - v.min(0))]},
        "placement": ([{"name": "table_rest", "point": [0.0, 0.0, 0.0],
                        "axis": [0.0, 0.0, 1.0]}]
                      if cls in ("bell", "plate") else
                      [{"name": "table_rest_lying", "point": [0.0, 0.0, round(float(v[:, 2].min()), 6)],
                        "axis": [0.0, 0.0, 1.0], "support": "flat side"}]),
        "material": {"density": DENSITY[cls], "name": MATERIAL[cls],
                     "transmittance": 0.0},
        "joint_simulated": False if cls == "bell" else None,
        "joint_note": ("single rigid mesh; the press joint is declared in parts.press for the URDF "
                       "builder (2 links: base, button_cover)") if cls == "bell" else None,
        "parts": parts,
        "relations": rel,
    }
    runs, note = mesh_parts(fc)
    ann["mesh_parts"] = runs
    if note:
        ann["mesh_parts_note"] = note
    if not ok:
        bad = [r["id"] + ": " + r["text"] for r in rel if not r["ok"]]
        raise ValueError(f"[{cls}/{name}] relations failed: " + " | ".join(bad))
    return v, f, ann, meta


def write_variant(cls, idx, name, key, target, out_dir):
    v, f, ann, meta = build(cls, key, target, name)
    d_out = os.path.join(out_dir, name)
    os.makedirs(d_out, exist_ok=True)
    obj = os.path.join(d_out, name + ".obj")
    ply = os.path.join(d_out, name + ".ply")
    ajs = os.path.join(d_out, name + ".annotations.json")
    geom.write_obj(obj, v, f, header=f"shape_gen/ware_ready {cls} {name}")
    geom.write_ply(ply, v, f, header=f"shape_gen/ware_ready {cls} {name}")
    gen_sha = geom.sha256_file(os.path.abspath(__file__))
    ann["generator"] = {"file": "ware_ready.py", "version": VERSION, "sha256": gen_sha}
    ann["geometry"] = geom.check_faces(v, f)
    geom.write_json(ajs, ann)
    files = {os.path.basename(p): geom.sha256_file(p) for p in (obj, ply, ajs)}
    variant = {
        "name": name, "fpath": f"libimport/{cls}/{name}", "body_scale": [1.0, 1.0, 1.0],
        "files": files,
        "source": {
            "generator": f"ware_ready.py@{gen_sha[:16]}",
            "base": meta["name"], "base_sha256": meta["source_sha256"],
            "license": meta["license"], "author": meta["author"],
            "url": meta.get("url", ""),
            "source_kind": meta.get("source_kind", "objaverse_lvis"),
            "import_sha": geom.sha256_file(os.path.join(HERE, "ware_import.py"))[:16],
            "params": {"target_m": target, "density": DENSITY[cls]},
            "seed": 0,
        },
        "functional_read": {"path": name + ".annotations.json", "key": "parts"},
        "material": ann["material"],
        "status": "draft",
    }
    geom.write_yaml(os.path.join(d_out, name + ".variant.yaml"), variant)
    return {"index": idx, "variant": name, "class": cls, "target_m": target,
            "mass_kg": ann["mass"]["mass_kg"], "base": meta["name"],
            "license": meta["license"], "author": meta["author"],
            "base_sha256": meta["source_sha256"], "files": files, "dir": d_out,
            "mesh": (v, f), "bbox": ann["bbox"]["extent"]}


def main(argv):
    which = argv[1] if len(argv) > 1 else "all"
    classes = sorted({v[0] for v in VARIANTS}) if which == "all" else [which]
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(IMAGES, exist_ok=True)
    gen_sha = geom.sha256_file(os.path.abspath(__file__))
    title = {"bell": "铃", "plate": "盘", "fork": "叉", "spoon": "勺"}
    ok_all = True
    for cls in classes:
        rows, tiles, out_dir = [], [], os.path.join(OUT, cls)
        os.makedirs(out_dir, exist_ok=True)
        items = [(n, k, t) for c, n, k, t in VARIANTS if c == cls]
        for idx, (name, key, target) in enumerate(items, start=1):
            try:
                rec = write_variant(cls, idx, name, key, target, out_dir)
            except ValueError as e:
                print("skip:", e)
                ok_all = False
                continue
            v, f = rec.pop("mesh")
            ext = rec["bbox"]
            print(f"{idx:02d} {name:30s} target={target*1e3:5.1f} mm  "
                  f"ext={ext[0]*1e3:5.1f}x{ext[1]*1e3:5.1f}x{ext[2]*1e3:5.1f}  "
                  f"m={rec['mass_kg']*1e3:6.1f} g  faces={len(f)}")
            eye, tgt, fov = render.frame(v, direction=(0.32, -0.82, 0.45), fill=1.04)
            img = render.render(v, f, size=360, eye=eye, target=tgt, fov=fov,
                                base_color=(0.62, 0.63, 0.66), floor_half=0.5, floor_tiles=8)
            cap = f"{idx:02d} {name.split('_',1)[-1][:22]}"
            h, wd, _ = img.shape
            tw, th = render.text_size(cap, scale=2)
            tile = np.full((h + th + 18, max(wd, tw + 16), 3), 255, np.uint8)
            tile[8:8 + h, (tile.shape[1] - wd) // 2:(tile.shape[1] - wd) // 2 + wd] = img
            render.draw_text(tile, (tile.shape[1] - tw) // 2, h + 10, cap, scale=2)
            tiles.append(tile)
            rows.append(rec)
        if not tiles:
            print(f"{cls}: nothing built")
            ok_all = False
            continue
        cols = min(4, len(tiles))
        sheet = render.montage(tiles, cols=cols, pad=8, backdrop=250)
        png = os.path.join(IMAGES, f"{title[cls]}_库_对比图.png")
        render.write_png(png, sheet)
        with open(os.path.join(out_dir, "index.json"), "w", encoding="utf-8") as fh:
            json.dump({"class": cls, "generator_sha256": gen_sha, "rows": rows,
                       "sheet": png}, fh, indent=1, ensure_ascii=False)
        print("sheet ->", png)
    if not ok_all:
        print("RESULT: at least one requested class produced nothing or a variant was rejected")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
