"""Build the six trial variants per family + their comparison sheets.

    python3 make_samples.py [--assets DIR] [--images DIR]

Deterministic: same params + same seed produce byte-identical files (the sha256
of every mesh/annotation file is recorded in out/manifest.json).
"""

import argparse
import json
import os
import sys

import numpy as np

import basket
import bell
import bottle
import holder
import milk
import mixer
import pipette
import testtube
import geom
import marker
import rack
import render
import tableware
import tube2

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ASSETS = os.path.join(HERE, "out")
DEFAULT_IMAGES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")


# ---------------------------------------------------------------------------
# sample grids (nominal parameters are sample 01 of each family)
# ---------------------------------------------------------------------------
RACK_SAMPLES = [
    # -- two main sizes (the bulk of variants stay around two main sizes) --------------------
    ("rack_4x10_main", {
        "rows": 4, "cols": 10, "cell": 0.0215, "pitch_x": 0.0245, "pitch_y": 0.0245,
        "tube_d": 0.020, "layers": 3, "gap": 0.028,
    }),
    ("rack_4x12_main", {
        "rows": 4, "cols": 12, "cell": 0.0215, "pitch_x": 0.0245, "pitch_y": 0.0245,
        "tube_d": 0.020, "layers": 3, "gap": 0.030,
    }),
    # -- same two sizes, other row/col counts and structural knobs --------
    ("rack_3x8_compact", {
        "rows": 3, "cols": 8, "cell": 0.0215, "pitch_x": 0.0245, "pitch_y": 0.0245,
        "tube_d": 0.020, "layers": 2, "gap": 0.050,
    }),
    ("rack_5x10_tall", {
        "rows": 5, "cols": 10, "cell": 0.0215, "pitch_x": 0.0245, "pitch_y": 0.0245,
        "tube_d": 0.020, "layers": 3, "gap": 0.030,
    }),
    ("rack_4x10_thickwall", {
        "rows": 4, "cols": 10, "cell": 0.0215, "pitch_x": 0.0275, "pitch_y": 0.0275,
        "tube_d": 0.020, "layers": 3, "gap": 0.028,
    }),
    ("rack_4x12_small_cell", {
        "rows": 4, "cols": 12, "cell": 0.0185, "pitch_x": 0.0215, "pitch_y": 0.0215,
        "tube_d": 0.01678, "layers": 3, "gap": 0.028,
    }),
    ("rack_4x10_colored_pp", {
        "rows": 4, "cols": 10, "cell": 0.0215, "pitch_x": 0.0245, "pitch_y": 0.0245,
        "tube_d": 0.020, "layers": 3, "gap": 0.028,
        "material": "pp", "color": [0.86, 0.42, 0.30],
    }),
    ("rack_12x5_simsize", {
        "rows": 5, "cols": 12, "cell": 0.0169, "pitch_x": 0.0202, "pitch_y": 0.0190,
        "tube_d": 0.01678, "layers": 3, "gap": 0.0288,
        "side_wall_t": 0.0022, "margin_x": 0.00335, "margin_y": 0.00415,
        "plate_t": 0.0036, "base_t": 0.0036,
        "material": "abs", "color": [0.55, 0.60, 0.68],
    }),
]

MARKER_SAMPLES = [
    ("marker_nominal_sim", {"d": 0.01533, "length": 0.14340, "cap_len": 0.0}),
    ("marker_thin", {"d": 0.00958, "length": 0.14340, "cap_len": 0.016}),
    ("marker_edding", {"d": 0.01500, "length": 0.13700, "cap_len": 0.016}),
    ("marker_thick", {"d": 0.02325, "length": 0.14340, "cap_len": 0.020}),
    ("marker_faceted", {"d": 0.01900, "length": 0.14000, "cap_len": 0.016, "sides": 12}),
    ("marker_long_cap", {"d": 0.01700, "length": 0.15000, "cap_len": 0.025}),
]


# ---------------------------------------------------------------------------
# rendering helpers
# ---------------------------------------------------------------------------
def rack_face_colors(mesh, ranges, base=(0.78, 0.82, 0.88)):
    colors = np.tile(np.array([0.78, 0.80, 0.84]), (len(mesh.f[0]) if False else 0, 1))
    n = 0
    for _, s, e in ranges:
        n = max(n, e)
    colors = np.zeros((n, 3))
    import numpy as _np
    b = _np.asarray(base, dtype=float)
    palette = {
        "base": tuple(b * 0.55),
        "grid": tuple(_np.clip(b * 1.06, 0, 1)),
        "plate": tuple(_np.clip(b * 0.95, 0, 1)),
        "post": tuple(b * 0.72),
        "rail": tuple(_np.clip(b * 0.60, 0, 1)),
    }
    for nm, s, e in ranges:
        key = nm.split("_")[0]
        colors[s:e] = palette.get(key, (0.78, 0.80, 0.84))
    return colors


def marker_face_colors(verts, faces, cap_len, length):
    cen = verts[faces].mean(axis=1)
    col = np.tile(np.array([0.26, 0.27, 0.31]), (len(faces), 1))
    if cap_len > 0:
        tail = -length / 2.0 + cap_len
        is_cap = cen[:, 0] <= tail
        col[is_cap] = (0.72, 0.22, 0.18)
    return col


def tile(img, caption, pad=10, bg=(255, 255, 255), scale=2):
    h, w, _ = img.shape
    tw, th = render.text_size(caption, scale=scale)
    out_h = h + th + 2 * pad
    out_w = max(w, tw + 2 * pad)
    out = np.full((out_h, out_w, 3), bg, dtype=np.uint8)
    out[pad:pad + h, (out_w - w) // 2:(out_w - w) // 2 + w] = img
    render.draw_text(out, (out_w - tw) // 2, h + pad + 2, caption, scale=scale)
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def build_family(family, module, samples, assets_root, images_dir, tag):
    fam_dir = os.path.join(assets_root, family)
    os.makedirs(fam_dir, exist_ok=True)
    base = module.default_params()
    rows = []
    tiles = []
    for idx, (label, over) in enumerate(samples, start=1):
        p = dict(base)
        p.update(over)
        p["name"] = label
        seed = 0
        out = os.path.join(fam_dir, label)
        variant = module.generate(p, seed=seed, out_dir=out)

        name = variant["name"]
        verts, faces = _load_obj(os.path.join(out, name + ".obj"))
        if family == "rack":
            mesh = module.build(p)[0]
            ranges = mesh.part_ranges()
            face_colors = rack_face_colors(mesh, ranges, p.get("color", (0.78, 0.82, 0.88)))
            eye, target, fov = render.frame(verts, direction=(0.55, -0.72, 0.42), fill=0.95)
            extent = float(np.max(verts.max(0) - verts.min(0)))
        else:
            face_colors = marker_face_colors(verts, faces, p["cap_len"], p["length"])
            eye, target, fov = render.frame(verts, direction=(0.16, -0.92, 0.36), fill=1.02)
            extent = float(np.max(verts.max(0) - verts.min(0)))

        img = render.render(verts, faces, size=440, eye=eye, target=target, fov=fov,
                            face_colors=face_colors,
                            floor_half=max(0.32, 1.4 * extent),
                            floor_tiles=10, floor_tint=(0.60, 0.61, 0.64))
        cap = f"{idx:02d} {_short_caption(family, p)}"
        shot = tile(img, cap)
        png = os.path.join(images_dir, f"{tag}_{idx:02d}_{label}.png")
        render.write_png(png, shot)
        tiles.append(shot)
        rows.append({
            "index": idx,
            "variant": name,
            "dir": os.path.join(out),
            "params": {k: v for k, v in p.items() if k != "name"},
            "seed": seed,
            "files": variant["files"],
            "image": png,
            "mass_kg": json.load(open(os.path.join(out, name + ".annotations.json")))["mass"]["mass_kg"],
            "caption": cap,
        })
    sheet = render.montage(tiles, cols=3, pad=8, backdrop=250)
    sheet_path = os.path.join(images_dir, f"{tag}_对比图.png")
    render.write_png(sheet_path, sheet)
    return rows, sheet_path


def _short_caption(family, p):
    if family == "rack":
        return (f"{int(p['rows'])}X{int(p['cols'])} CELL{int(round(p['cell']*1e3*10))} "
                f"L{int(p['layers'])} {str(p.get('material','pmma')).upper()}")
    cap = "CAP" if p["cap_len"] > 0 else "NOCAP"
    shape = "RND" if int(p["sides"]) >= 32 else f"{int(p['sides'])}GON"
    return (f"D{int(round(p['d']*1e3*10))} L{int(round(p['length']*1e3))} "
            f"{cap} {shape}")


def _load_obj(path):
    verts, faces = [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                faces.append([int(x.split("/")[0]) - 1 for x in line.split()[1:4]])
    return np.asarray(verts), np.asarray(faces, dtype=np.int64)


# ---------------------------------------------------------------------------
# second batch: bottle / cup / tube2 / bell / plate / fork / spoon
# ---------------------------------------------------------------------------
EXTRA = {
    "bottle": [
        ("bottle_236_transparent", bottle.default_bottle, bottle.generate_bottle, {}),
        ("bottle_222_slim", bottle.default_bottle, bottle.generate_bottle,
         {"H": 0.2218, "body_rx": 0.0320, "body_ry": 0.0315, "transmittance": 0.6}),
        ("bottle_241_wide", bottle.default_bottle, bottle.generate_bottle,
         {"H": 0.2408, "body_rx": 0.0347, "body_ry": 0.0342, "shape": "tapered_shoulder"}),
        ("bottle_prism8", bottle.default_bottle, bottle.generate_bottle,
         {"sides": 8, "H": 0.2340}),
        ("bottle_neckoff6", bottle.default_bottle, bottle.generate_bottle,
         {"neck_offset": [0.006, 0.0], "H": 0.2340}),
    ],
    "cup": [
        ("cup_56", bottle.default_cup, bottle.generate_cup, {}),
        ("cup_74", bottle.default_cup, bottle.generate_cup,
         {"mouth_r_out": 0.0372, "bottom_r_out": 0.0250}),
        ("cup_deep_60", bottle.default_cup, bottle.generate_cup,
         {"mouth_r_out": 0.0300, "bottom_r_out": 0.0220}),
    ],
    "tube2": [
        ("tube2_a", tube2.default_pair, lambda p, seed=0, out_dir=None: tube2.generate(p, seed, out_dir, key="a"), {}),
        ("tube2_b", tube2.default_pair, lambda p, seed=0, out_dir=None: tube2.generate(p, seed, out_dir, key="b"), {}),
    ],
    "bell": [
        ("bell_63_stroke5", bell.default_params, bell.generate, {}),
        ("bell_58_stroke55", bell.default_params, bell.generate,
         {"base_r": 0.029, "cover_r": 0.020, "cover_top_z": 0.0290, "stroke": 0.0055}),
    ],
    "plate": [
        ("plate_180", tableware.default_plate, tableware.generate_plate, {}),
        ("plate_180_deep", tableware.default_plate, tableware.generate_plate,
         {"floor_z": 0.014, "rim_w": 0.012}),
    ],
    "fork": [
        ("fork_203_hand42", tableware.default_fork, tableware.generate_fork, {}),
        ("fork_203_hand37", tableware.default_fork, tableware.generate_fork,
         {"handle_w": 0.0037, "tine_w": 0.0030, "gap_w": 0.0040}),
    ],
    "spoon": [
        ("spoon_145_hand48", tableware.default_spoon, tableware.generate_spoon, {}),
        ("spoon_145_hand53", tableware.default_spoon, tableware.generate_spoon,
         {"handle_w": 0.0053}),
    ],
    "pipette": [
        ("pipette_255_round", pipette.default_params, pipette.generate, {}),
        ("pipette_280_ellipse", pipette.default_params, pipette.generate,
         {"L": 0.2805, "body_w": 0.0308, "body_h": 0.0308, "grip_shape": "ellipse",
          "button_r": 0.0070, "hook_h": 0.0715}),
    ],
    "tube": [
        ("tube_168x195_sim", testtube.default_params, testtube.generate, {}),
        ("tube_200x150", testtube.default_params, testtube.generate,
         {"d": 0.0200, "L": 0.150}),
        ("tube_245x120", testtube.default_params, testtube.generate,
         {"d": 0.0245, "L": 0.120, "bottom": "flat"}),
    ],
    "holder": [
        ("holder_bore86_h96", holder.default_params, holder.generate, {}),
        ("holder_bore70_h110", holder.default_params, holder.generate,
         {"mouth_r_in": 0.035, "H": 0.110, "wall_t": 0.0035}),
    ],
    "basket": [
        ("basket_344x267", basket.default_params, basket.generate, {}),
        ("basket_330x256", basket.default_params, basket.generate,
         {"L": 0.330, "W": 0.256, "H": 0.178}),
        ("basket_337_tall_handle", basket.default_params, basket.generate,
         {"L": 0.337, "W": 0.262, "H": 0.182, "handle_d": 0.0176, "handle_h": 0.055}),
        ("basket_344_double_handle", basket.default_params, basket.generate,
         {"handle_inset": 0.100, "handle_d": 0.0165, "handle_h": 0.045}),
    ],
    "milk": [
        ("milk_46x105", milk.default_params, milk.generate, {}),
        ("milk_50x113", milk.default_params, milk.generate,
         {"W": 0.0495, "D": 0.0518, "H": 0.1127}),
        ("milk_43x98", milk.default_params, milk.generate,
         {"W": 0.043, "D": 0.045, "H": 0.098}),
        ("milk_45x108_slim", milk.default_params, milk.generate,
         {"W": 0.045, "D": 0.046, "H": 0.108, "gable_h": 0.016}),
    ],
    "mixer": [
        ("mixer_250_std", mixer.default_params, mixer.generate, {}),
        ("mixer_250_bigpaddle", mixer.default_params, mixer.generate,
         {"lever_len": 0.040, "lever_w": 0.018, "theta_max": 0.50}),
    ],
}

CAM = {
    "bottle": ((0.42, -0.78, 0.30), 1.02),
    "cup": ((0.42, -0.78, 0.30), 1.02),
    "tube2": ((0.20, -0.94, 0.26), 1.02),
    "bell": ((0.45, -0.72, 0.42), 1.00),
    "plate": ((0.40, -0.76, 0.46), 1.08),
    "fork": ((0.30, -0.86, 0.52), 1.18),
    "spoon": ((0.30, -0.86, 0.52), 1.18),
    "pipette": ((0.30, -0.86, 0.40), 1.05),
    "tube": ((0.35, -0.78, 0.36), 1.05),
    "holder": ((0.40, -0.76, 0.42), 1.02),
    "mixer": ((0.55, -0.70, 0.45), 1.02),   # 已经转成 z 朝上再渲
    "basket": ((0.55, -0.72, 0.40), 1.05),
    "milk": ((0.50, -0.72, 0.42), 1.02),
}


def build_extra(assets_root, images_dir):
    rows, sheets = [], {}
    for family, items in EXTRA.items():
        fam_dir = os.path.join(assets_root, family)
        os.makedirs(fam_dir, exist_ok=True)
        tiles = []
        for idx, (label, dflt, gen, over) in enumerate(items, start=1):
            p = dict(dflt())
            p.update(over)
            p["name"] = label
            v = gen(p, 0, fam_dir)
            out = os.path.join(fam_dir, label)
            verts, faces = _load_obj(os.path.join(out, label + ".obj"))
            if family == "mixer":        # 搅拌器局部 +y 朝上：渲图时转成 z 朝上（只影响看图）
                verts = verts @ np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]).T
            direction, fill = CAM[family]
            eye, tgt, fov = render.frame(verts, direction=direction, fill=fill)
            img = render.render(verts, faces, size=420, eye=eye, target=tgt, fov=fov,
                                base_color=(0.62, 0.66, 0.72), floor_half=0.35, floor_tiles=10)
            cap = f"{idx:02d} " + label.replace("_", " ").upper()[:26]
            h, w, _ = img.shape
            tw, th = render.text_size(cap, scale=2)
            tile = np.full((h + th + 18, max(w, tw + 16), 3), 255, np.uint8)
            tile[8:8 + h, (tile.shape[1] - w) // 2:(tile.shape[1] - w) // 2 + w] = img
            render.draw_text(tile, (tile.shape[1] - tw) // 2, h + 10, cap, scale=2)
            tiles.append(tile)
            png = os.path.join(images_dir, f"{_CN[family]}_{idx:02d}_{label}.png")
            render.write_png(png, tile)
            rows.append({"family": family, "index": idx, "variant": v["name"],
                         "dir": out, "files": v["files"], "image": png,
                         "caption": cap})
        sheet = render.montage(tiles, cols=min(3, len(tiles)), pad=8, backdrop=250)
        sp = os.path.join(images_dir, f"{_CN[family]}_对比图.png")
        render.write_png(sp, sheet)
        sheets[family] = sp
        print(f"{family}: {len(items)} variants -> {sp}")
    return rows, sheets


_CN = {"bottle": "瓶", "cup": "杯", "tube2": "两管", "bell": "铃",
       "plate": "盘", "fork": "叉", "spoon": "勺", "pipette": "移液器", "mixer": "搅拌器",
       "tube": "试管", "holder": "笔筒", "basket": "篮子", "milk": "奶盒"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default=DEFAULT_ASSETS)
    ap.add_argument("--images", default=DEFAULT_IMAGES)
    args = ap.parse_args(argv)

    os.makedirs(args.images, exist_ok=True)
    all_rows = {}
    sheets = {}
    for family, module, samples, tag in (
        ("rack", rack, RACK_SAMPLES, "试管架"),
        ("marker", marker, MARKER_SAMPLES, "马克笔"),
    ):
        rows, sheet = build_family(family, module, samples, args.assets, args.images, tag)
        all_rows[family] = rows
        sheets[family] = sheet

    extra_rows, extra_sheets = build_extra(args.assets, args.images)

    manifest = {
        "schema": "hq-shape-samples/1",
        "extra_rows": extra_rows,
        "extra_sheets": extra_sheets,
        "generator_version": {"rack": rack.VERSION, "marker": marker.VERSION},
        "assets_root": args.assets,
        "images_dir": args.images,
        "families": all_rows,
        "sheets": sheets,
    }
    geom.write_json(os.path.join(args.assets, "manifest.json"), manifest)

    readme = _readme(manifest)
    with open(os.path.join(args.images, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(readme)
    with open(os.path.join(args.assets, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(readme)
    print(json.dumps({k: [r["variant"] for r in v] for k, v in all_rows.items()}, indent=1))
    return 0


def _readme(manifest):
    lines = ["# Shape samples (rack + marker)", "",
             "由 `final/shape_gen/make_samples.py` 生成；同一参数 + 同一 seed 逐字节同输出。", "",
             "每个样本目录含 `<name>.obj` / `<name>.ply` / `<name>.annotations.json` / `<name>.variant.yaml`。", ""]
    for fam, rows in manifest["families"].items():
        lines.append(f"## {fam}（对比图：{os.path.basename(manifest['sheets'][fam])}，3 列 × 2 行）")
        lines.append("")
        lines.append("| 序 | 变体 | 图 | 关键参数 | 质量 kg |")
        lines.append("|---|---|---|---|---:|")
        for r in rows:
            p = r["params"]
            if fam == "rack":
                key = (f"{p['rows']}×{p['cols']}, 格 {p['cell']*1e3:.1f} mm, "
                       f"格距 {p['pitch_x']*1e3:.1f}/{p['pitch_y']*1e3:.1f}, "
                       f"{int(p['layers'])} 层, {p.get('material','pmma')}")
            else:
                key = (f"Ø{p['d']*1e3:.2f} mm, 长 {p['length']*1e3:.0f} mm, "
                       f"帽 {p['cap_len']*1e3:.0f} mm, {int(p['sides'])} 边形, 填充 {p['fill']:.3f}")
            lines.append(f"| {r['index']:02d} | {r['variant']} | {os.path.basename(r['image'])} | {key} | {r['mass_kg']:.4f} |")
        lines.append("")
    if manifest.get("extra_rows"):
        lines.append("## 第二批（瓶/杯、两管、铃、盘/叉/勺）")
        lines.append("")
        lines.append("| 族 | 序 | 变体 | 图 |")
        lines.append("|---|---|---|---|")
        for r in manifest["extra_rows"]:
            lines.append(f"| {r['family']} | {r['index']:02d} | {r['variant']} | {os.path.basename(r['image'])} |")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
