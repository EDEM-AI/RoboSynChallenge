"""移液「局部改型」——沿官方各段**独立**改长度 / 渐缩 / 粗细，
按钮保持刚性、轴向与碰撞配套跟随；整体 scale 只留给已有三档。

官方 pipette_good 的 tip 网格沿 +x 分三段（按原始 85.46 mm）：
    针     0.00 – 21.4 mm   r ≈ 1.0
    锥    21.4 – 43.0 mm   r 1.8 → 3.7
    桶    43.0 – 85.5 mm   r ≈ 4.3（后端有指托法兰）

三个变体：
    reshape_long_needle    针长 ×1.6、针径 ×0.9，其余不变
    reshape_short_barrel   桶长 ×0.8、桶径 ×1.15
    reshape_thick_barrel   桶径 ×1.30、锥径 ×1.10、针长 ×1.2

    python3 pipette_reshape.py
"""

import json
import os

import numpy as np
import trimesh

import geom

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(os.environ.get("ROBOSYN_REPO", "RoboSynChallenge"), "assets", "pipette_good", "meshes")
OUT = os.path.join(HERE, "out", "pipette_reshape")

BODY_SRC = os.path.join(ASSETS, "pipette_tip.obj")
BTN_SRC = os.path.join(ASSETS, "pipette_button.obj")
L_REF = 0.08546
SEG = [("needle", 0.0, 0.214), ("cone", 0.214, 0.430), ("barrel", 0.430, 1.0)]
BLEND = 0.03          # 段边界 3% 长度内平滑过渡


def _seg_of(f):
    for name, a, b in SEG:
        if a <= f <= b:
            return name
    return SEG[-1][0]


def _w(xn, lmul, rmul):
    """xn 是 0..1 的归一化 x；返回 (新 xn, 径向倍率)。段内线性，边界 BLEND 平滑。"""
    # 长度：分段累加（不归一 —— 总长会随各段伸长/缩短）
    xs = [0.0]
    for name, a, b in SEG:
        xs.append(xs[-1] + (b - a) * lmul[name])
    for i, (name, a, b) in enumerate(SEG):
        if a <= xn <= b:
            t = (xn - a) / max(b - a, 1e-12)
            x_new = xs[i] + t * (xs[i + 1] - xs[i])
            break
    else:
        x_new = xs[-1]
    # 径向：段内常数，边界线性混合
    r = rmul[_seg_of(xn)]
    for i in range(len(SEG) - 1):
        xb = SEG[i][2]
        if abs(xn - xb) < BLEND / 2:
            t = (xn - (xb - BLEND / 2)) / BLEND
            r = (1 - t) * rmul[SEG[i][0]] + t * rmul[SEG[i + 1][0]]
    return x_new, r


def reshape(v, f, lmul, rmul):
    x0, x1 = float(v[:, 0].min()), float(v[:, 0].max())
    span = x1 - x0
    xn = (v[:, 0] - x0) / span
    nv = np.array(v, dtype=float)
    for k in range(len(v)):
        x_new, r = _w(float(xn[k]), lmul, rmul)
        nv[k, 0] = x0 + x_new * span
        nv[k, 1:] *= r
    return nv, f


VARIANTS = [
    ("reshape_long_needle", {"needle": 1.6, "cone": 1.0, "barrel": 1.0},
     {"needle": 0.9, "cone": 1.0, "barrel": 1.0}, "针长 +60%、针径 −10%，便于夹细端"),
    ("reshape_short_barrel", {"needle": 1.0, "cone": 1.0, "barrel": 0.8},
     {"needle": 1.0, "cone": 1.0, "barrel": 1.15}, "桶短 20%、桶粗 +15%"),
    ("reshape_thick_barrel", {"needle": 1.2, "cone": 1.0, "barrel": 1.0},
     {"needle": 1.0, "cone": 1.10, "barrel": 1.30}, "桶粗 +30%、锥粗 +10%"),
]


def main():
    os.makedirs(OUT, exist_ok=True)
    body = trimesh.load(BODY_SRC, force="mesh", process=False)
    bv = np.asarray(body.vertices, float)
    bf = np.asarray(body.faces, np.int64)
    btn = trimesh.load(BTN_SRC, force="mesh", process=False)
    mv = np.asarray(btn.vertices, float)
    mf = np.asarray(btn.faces, np.int64)
    base_len = float(bv[:, 0].max() - bv[:, 0].min())
    # 官方装配：button 网格整体放在 x = +27.6701 mm（scale 1）处
    BTN_PLACE = 0.0276701

    made = []
    for name, lmul, rmul, desc in VARIANTS:
        nv, nf = reshape(bv, bf, lmul, rmul)
        L_body = float(nv[:, 0].max() - nv[:, 0].min())
        # 按钮刚性跟随：保持官方装配关系（网格放在 +27.6701 mm），只随桶后端平移
        delta = float(nv[:, 0].max()) - float(bv[:, 0].max())
        btn_off = BTN_PLACE + delta
        mv2 = mv + np.array([btn_off, 0.0, 0.0])
        v = np.vstack([nv, mv2])
        f = np.vstack([nf, mf + len(nv)])
        v = v - np.array([(v[:, 0].min() + v[:, 0].max()) / 2.0, 0.0, 0.0])

        # 质量：按表面积比（官方壳不水密，体积不可靠）
        def area(vv, ff):
            a, b, c = vv[ff[:, 0]], vv[ff[:, 1]], vv[ff[:, 2]]
            return 0.5 * float(np.linalg.norm(np.cross(b - a, c - a), axis=1).sum())
        a_new, a_old = area(v, f), area(np.vstack([bv, mv]), np.vstack([bf, mf + len(bv)]))
        mass = 0.013 * a_new / max(a_old, 1e-12)

        # 各段实测
        segs = []
        xb = [0.0]
        for sname, a, b in SEG:
            xb.append(xb[-1] + (b - a) * base_len * lmul[sname])
        for i, (sname, a, b) in enumerate(SEG):
            m = (nv[:, 0] - float(nv[:, 0].min()) >= xb[i] - 1e-6) & \
                (nv[:, 0] - float(nv[:, 0].min()) <= xb[i + 1] + 1e-6)
            r95 = float(np.percentile(np.linalg.norm(nv[m][:, 1:], axis=1), 95)) if m.any() else 0.0
            segs.append({"name": sname, "length_mm": round((b - a) * base_len * lmul[sname] * 1e3, 2),
                         "r95_mm": round(r95 * 1e3, 2)})

        btn_center = v[len(nv):].mean(0)
        ann = {
            "schema": "hq-shape-annot/1",
            "parts_schema": "hq-ctx-parts/3",
            "family": "pipette", "name": name, "units": "m", "up_axis": "+z",
            "origin": "official pipette_good, reshaped per segment (not a uniform scale)",
            "params": {"reshape": {"len": lmul, "rad": rmul}, "desc": desc,
                       "source": "assets/pipette_good/meshes/{pipette_tip,pipette_button}.obj"},
            "mass": {"volume_m3": None, "mass_kg": round(mass, 6),
                     "com": [round(float(x), 6) for x in geom.surface_centroid(v, f)],
                     "inertia_kgm2": None,
                     "source": "0.013 kg x 表面积比（官方壳不水密，体积不可靠）"},
            "bbox": {"min": [float(x) for x in v.min(0)], "max": [float(x) for x in v.max(0)],
                     "extent": [float(x) for x in (v.max(0) - v.min(0))]},
            "segments": segs,
            "button": {"offset_x": round(btn_off, 6),
                       "centre": [round(float(x), 6) for x in btn_center],
                       "axis": [1.0, 0.0, 0.0],
                       "stroke_m": 0.00667, "limit": [-0.002, 0.00467],
                       "collision": "官方 pipette_button_acd.obj（刚性，随按钮平移）"},
            "parts": {"pipette": {
                "origin": {"kind": "point", "p": [0.0, 0.0, 0.0], "tol": None, "src": f"variant:{name}"},
                "body": {"kind": "axis", "p": [0.0, 0.0, 0.0], "axis": [1.0, 0.0, 0.0],
                         "size": {"length": L_body}, "span": [0.0, L_body],
                         "section": {"shape": "circle", "r_out": round(max(s["r95_mm"] for s in segs) * 1e-3, 6)},
                         "grasp_s": 0.72, "tol": None, "src": f"variant:{name}"},
                "tip": {"kind": "point", "p": [float(v[:, 0].min()), 0.0, 0.0],
                        "normal": [-1.0, 0.0, 0.0], "tol": None, "src": f"variant:{name}"},
                "button": {"kind": "joint", "q_name": "press",
                           "axis": [0.0, 0.0, -1.0], "range": [-0.002, 0.00467],
                           "travel": 0.00667, "unit": "m",
                           "collision": "official ACD (rigid)", "tol": None, "src": f"variant:{name}"},
            }},
        }
        geom.emit_variant(__file__, "pipette", name, v, f, ann, OUT, 0, "1", allow_open=True)
        made.append({"name": name, "desc": desc, "body_len_mm": round(L_body * 1e3, 2),
                     "mass_kg": round(mass, 5), "segments": segs,
                     "extent_mm": [round(x * 1e3, 1) for x in (v.max(0) - v.min(0))]})
        print(f"{name:22s} body {L_body*1e3:6.1f} mm  mass {mass:7.4f} kg  ext "
              f"{[round(x*1e3,0) for x in (v.max(0)-v.min(0))]}")

    json.dump({"schema": "hq-pipette-reshape/1", "items": made},
              open(os.path.join(OUT, "index.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("->", OUT)


if __name__ == "__main__":
    main()
