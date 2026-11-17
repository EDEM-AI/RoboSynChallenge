"""Sharpie 两件按「光柱」重标 —— 只改径向缩放口径，不造新形状。

Measured: the clean shaft 24.7 / 32.9 mm，与标称 15 / 20 不符；右握 x≈45 mm 落在笔帽/夹带。
本脚本按**干净光柱段的 r50** 重标，产出 *_fix 两件；**原件（9dce6bc9 / 61161f54）不动**。

    python3 marker_fix.py
"""

import json
import os

import numpy as np
import trimesh

import geom

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "out", "marker_41")
OUT = os.path.join(HERE, "out", "marker_41_fix")
PAIRS = [("mk41_sharpie_mid_15", 0.0150), ("mk41_sharpie_thick_20", 0.0200)]


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for name, target in PAIRS:
        m = trimesh.load(os.path.join(SRC, name, name + ".obj"), force="mesh", process=False)
        v = np.asarray(m.vertices, float)
        f = np.asarray(m.faces, np.int64)
        L = float(v[:, 0].max() - v[:, 0].min())
        x0 = float(v[:, 0].min())
        xn = (v[:, 0] - x0) / L
        r = np.linalg.norm(v[:, 1:], axis=1)
        band = (xn > 0.09) & (xn < 0.17)          # 干净光柱段（-60..-48 mm）
        r_bar = float(np.median(r[band]))
        k = target / (2 * r_bar)
        nv = np.array(v, float)
        nv[:, 1:] *= k

        st = []
        for i in range(12):
            lo, hi = i / 12, (i + 1) / 12
            s = (xn >= lo) & (xn < hi + 1e-9)
            if s.any():
                st.append({"x_mm": [round((lo * L + x0) * 1e3, 1), round((hi * L + x0) * 1e3, 1)],
                           "r50_mm": round(float(np.median(r[s])) * k * 1e3, 2),
                           "r95_mm": round(float(np.percentile(r[s], 95)) * k * 1e3, 2)})

        new_name = name + "_fix"
        gf = (0.45, 0.70)                      # 握区占全长比例（避开 −36..−24 夹带 与 +48..+72 帽）
        gx = [round((x0 + gf[0] * L) * 1e3, 1), round((x0 + gf[1] * L) * 1e3, 1)]
        gs = [round(gf[0] * L, 6), round(gf[1] * L, 6)]
        ann = {
            "schema": "hq-shape-annot/1",
            "parts_schema": "hq-ctx-parts/3",
            "family": "marker", "name": new_name, "units": "m", "up_axis": "+z",
            "origin": "marker_41 原件的按光柱重标版本（形状未变，只改径向缩放口径）",
            "params": {"d": target, "length": L, "density": 1050.0, "fill": 0.535,
                       "fix": "缩放口径 由「全网格径向中位数」改为「干净光柱段 r50」；原件不动",
                       "source_item": name, "radial_scale_fix": round(k, 6)},
            "mass": {"volume_m3": None,
                     "mass_kg": round(float(np.pi * (target / 2) ** 2 * L * 0.535 * 1050.0), 6),
                     "com": [round(float(x), 6) for x in geom.surface_centroid(nv, f)],
                     "inertia_kgm2": None,
                     "source": "圆柱近似 x fill 0.535（与原件同法）"},
            "bbox": {"min": [float(x) for x in nv.min(0)], "max": [float(x) for x in nv.max(0)],
                     "extent": [float(x) for x in (nv.max(0) - nv.min(0))]},
            "stations_mm": st,
            "grip_band": {"x_mm": gx,
                          "why": "全长 45%–70% 段（x≈−7…+29 mm）：避开 −36…−24 的夹带与 +48…+72 的笔帽；"
                                 "The right-hand grasp x≈45 mm 正是帽/夹带，不能作握点"},
            "parts": {"pen": {
                "origin": {"kind": "point", "p": [0.0, 0.0, 0.0], "tol": None,
                           "src": f"variant:{new_name}"},
                "body": {"kind": "axis", "p": [0.0, 0.0, 0.0], "axis": [1.0, 0.0, 0.0],
                         "size": {"length": L}, "span": [0.0, L],
                         "section": {"shape": "circle", "r_out": round(target / 2, 6)},
                         "grasp_s": [0.45, 0.70], "tol": None, "src": f"variant:{new_name}"},
                "grip": {"kind": "axis", "p": [0.0, 0.0, 0.0], "axis": [1.0, 0.0, 0.0],
                         "span": gs, "section": {"shape": "circle", "r_out": round(target / 2, 6)},
                         "tol": None, "src": f"variant:{new_name}"},
            }},
        }
        geom.emit_variant(os.path.abspath(__file__), "marker", new_name, nv, f, ann, OUT, 0, "1",
                          allow_open=True)
        rows.append({"from": name, "to": new_name, "barrel_r_before_mm": round(r_bar * 1e3, 2),
                     "k": round(k, 4),
                     "new_extent_mm": [round(x * 1e3, 1) for x in (nv.max(0) - nv.min(0))],
                     "stations": st, "grip_band_mm": gx})
        print(f"{new_name:28s} k={k:.4f} ext={[round(x*1e3,1) for x in (nv.max(0)-nv.min(0))]}")

    json.dump({"schema": "hq-marker-fix/1",
               "note": "Re-measured: shaft 24.7/32.9 与标称 15/20 不符；这里按光柱重标，原件不动",
               "items": rows},
              open(os.path.join(OUT, "index.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("->", OUT)


if __name__ == "__main__":
    main()
