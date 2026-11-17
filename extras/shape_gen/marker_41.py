"""从 marker_ready 的四基件取「粗 / 中」两档（只在已有基件上缩放，不改来源）。

四基件 = 库里的四支（不含官方仿真圆珠笔）：
    poly_permanent / sharpie_thin / stabilo_boss / highlighter
每支出两档：中 Ø15.0 mm、粗 Ø20.0 mm（同档同径，专家只看形状差别）。

避过细仿真笔（official_pen Ø9.6）；官方圆珠笔不进 FAQ 默认名单。

    python3 marker_41.py
"""

import json
import os

import numpy as np

import geom
import render
import marker_ready as MR

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "marker_41")
IMAGES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")

BASES = [
    ("poly", "poly_permanent", "Permanent Marker from Poly by Google"),
    ("sharpie", "sharpie_thin", "Sharpie Marker (thin)"),
    ("stabilo", "stabilo_boss", "Stabilo Boss Highlighter"),
    ("highlighter", "highlighter", "Highlighter"),
]
TIERS = [("mid", 0.0150, 0.1450), ("thick", 0.0200, 0.1450)]


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(IMAGES, exist_ok=True)
    rows, tiles = [], []
    idx = 0
    for short, key, cn in BASES:
        for tier, d, L in TIERS:
            idx += 1
            name = f"mk41_{short}_{tier}_{int(round(d*1e3))}"
            rec = MR.write_variant(idx, name, key, d, L, OUT)
            v, f = rec.pop("mesh")
            rec["tier"] = tier
            rec["base_cn"] = cn
            rows.append(rec)
            eye, tgt, fov = render.frame(v, direction=(0.28, -0.86, 0.36), fill=1.05)
            tiles.append(render.render(v, f, size=300, eye=eye, target=tgt, fov=fov,
                                       base_color=(0.30, 0.31, 0.35),
                                       floor_half=0.9, floor_tiles=8))
            print(f"{idx:02d} {name:30s} d={d*1e3:4.1f} L={L*1e3:5.1f} "
                  f"{cn:34s} {rec['bin']}")

    # 长条对比图
    n = len(tiles)
    h = max(t.shape[0] for t in tiles)
    w = max(t.shape[1] for t in tiles)
    cols = 4
    rows_n = (n + cols - 1) // cols
    sheet = np.full((rows_n * (h + 26), cols * w, 3), 250, np.uint8)
    for k, im in enumerate(tiles):
        r, c = divmod(k, cols)
        y = r * (h + 26)
        sheet[y:y + im.shape[0], c * w:c * w + im.shape[1]] = im
        cap = f"{rows[k]['variant']} D{rows[k]['d']*1e3:.0f}"
        render.draw_text(sheet, c * w + 8, y + h + 6, cap, scale=1,
                         color=(40, 44, 52))
    p = os.path.join(IMAGES, "马克笔_41_四基件_中粗.png")
    render.write_png(p, sheet)
    print("sheet ->", p)

    handoff = {
        "schema": "hq-marker-handoff/2",
        "from": "marker_ready", "to": "grasp-point load check",
        "rule": "四基件各自出中 Ø15.0 / 粗 Ø20.0 两档；来源沿用原 base（license/base_sha256 不变）；"
                "细仿真笔（official_pen Ø9.6）不取；官方圆珠笔不进 FAQ 默认名单",
        "excluded": {"thin_sim_pen": "mk_thin_096_official (official_pen Ø9.6)",
                     "official_ballpoint": "官方 assets/Pen/the_pen.obj 只作参照，不列 FAQ 默认"},
        "warning": "缩放按「径向中位数」到标称直径；网格上更粗的部位（夹/帽）会超出标称包络 —— 见 real_extent_mm",
        "items": [],
    }
    for r in rows:
        annp = os.path.join(r["dir"], r["variant"] + ".annotations.json")
        ann = json.load(open(annp, encoding="utf-8"))
        pk = list(ann["parts"])[0]
        g = ann["parts"][pk]["body"]
        v = np.asarray(__import__("trimesh").load(
            os.path.join(r["dir"], r["variant"] + ".obj"), force="mesh", process=False).vertices, float)
        ext = v.max(0) - v.min(0)
        rad = np.linalg.norm(v[:, 1:], axis=1)
        # 握点：span 中点处的径向半宽（网格实测分位数）
        x = v[:, 0] - v[:, 0].min()
        xm = 0.5 * g["size"]["length"]
        sel = np.abs(x - xm) < 0.004
        r_mid = float(np.percentile(rad[sel], 95)) if sel.any() else float("nan")
        handoff["items"].append({
            "variant": r["variant"], "tier": r["tier"],
            "d_nominal_mm": round(r["d"] * 1e3, 2), "L_mm": round(r["length"] * 1e3, 2),
            "real_extent_mm": [round(float(e) * 1e3, 2) for e in ext],
            "real_max_d_mm": round(float(rad.max()) * 2e3, 2),
            "grip_point_radius95_mm": round(r_mid * 1e3, 2) if r_mid == r_mid else None,
            "mass_kg": round(r["mass_kg"], 6),
            "obj": os.path.join(r["dir"], r["variant"] + ".obj"),
            "obj_sha256": r["files"][r["variant"] + ".obj"],
            "annotations": annp,
            "base": r["base"], "base_cn": r["base_cn"],
            "license": r["license"], "author": r["author"],
            "base_sha256": r["base_sha256"],
            "grip": g,
            "ask": "核握点是否落在真实可夹柱身上、加载后网格/质量/尺度是否与标注一致；"
                   "real_max_d 大于 d_nominal 的件要决定「留夹改标注」还是「去夹」",
        })
    hp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "marker_handoff.json")
    json.dump(handoff, open(hp, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("handoff ->", hp, len(handoff["items"]), "items")


if __name__ == "__main__":
    main()
