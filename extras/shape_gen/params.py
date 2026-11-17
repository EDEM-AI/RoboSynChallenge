"""Parameter ranges + provenance for the two trial shape families.

Every number below is either
  * MEAS  - measured on an existing official asset in this session (see
            measure_nominal.py / nominal_measured.json), or
  * CAT   - taken from a product catalogue cited in the nominal-dimension notes (B = BRAND /
            BRANDTECH catalogues; E = edding 3000 data sheet), or
  * CFG   - read from the official task config / task code, or
  * ENG   - engineering envelope derived from the numbers above; explicitly
            NOT a physical constant and NOT calibrated on the real robot.

Units are metres / kilograms.  The tables in the task log are generated from
this file, so the log and the code cannot drift apart.
"""

UNITS = "m"

# ---------------------------------------------------------------------------
# fixed reference measurements (official assets, this session)
# ---------------------------------------------------------------------------
MEAS = {
    # asset, raw extent (unscaled), config body_scale, scaled extent
    "test_tube_rack_standard.ply": {
        "raw_extent": [0.25021, 0.10559, 0.07208],
        "body_scale": [1.1, 1.1, 1.1],
        "scaled_extent": [0.27523, 0.11615, 0.07929],
        "cells": "12 (x) x 5 (y) square cells (measure_nominal.py)",
        "cell_opening": [0.0170, 0.0169],
        "wall": [0.0030, 0.0022],
        "pitch": [0.02015, 0.01895],
        "top_plate_thickness": 0.0036,
        "base_thickness": 0.0060,
        "edge_margin": [0.0060, 0.0064],
        "note": "12x5 lattice of square cells; not a drilled round-hole plate",
    },
    "test_tube_standard.ply": {
        "raw_extent": [0.01525, 0.01508, 0.15002],
        "body_scale": [1.1, 1.1, 1.3],
        "scaled_extent": [0.01678, 0.01659, 0.19503],
        "tube_outer_d_m": 0.01678,
        "tube_length_m": 0.19503,
    },
    "Pen/the_pen.obj": {
        "raw_extent": [0.14340, 0.00958, 0.00958],
        "body_scale": [1.0, 1.6, 1.6],
        "scaled_extent": [0.14340, 0.01533, 0.01533],
        "pen_d_m": 0.01533,
        "pen_length_m": 0.14340,
    },
    "PenHolder/pen_holder.obj": {
        "raw_extent": [0.16497, 0.21412, 0.16497],
        "body_scale": [0.56, 0.45, 0.56],
        "scaled_extent": [0.09238, 0.09635, 0.09238],
        "inner_opening_x_m": 0.08557,
        "wall_x_m": 0.00386,
        "note": "inner opening measured by ray casting through the cup axis",
    },
}

# official task config / code values used by the relations
CFG = {
    "sample_loading.contact_offset_m": 0.0002,
    "sample_loading.rest_offset_m": 0.00005,
    "sample_loading.success_vertical_angle_rad": 0.087266,   # 5 deg, sample_loading.py:160
    "sample_loading.success_z_thresh_m": 0.07,
    "items_handover.contact_offset_m": 0.003,
    "items_handover.rest_offset_m": 0.001,
    "items_handover.pen_init_grasp_offset_m": 0.02,
}

# ---------------------------------------------------------------------------
# rack
# ---------------------------------------------------------------------------
RACK = {
    "rows": {
        "nominal": 4,
        "range": [4],
        "unit": "count",
        "src": "V reference racks (4x10 / 4x12); simulated asset has 5 rows, see MEAS note",
        "tag": "V",
    },
    "cols": {
        "nominal": 10,
        "range": [10, 12],
        "unit": "count",
        "src": "V reference racks 4x10, 4x12; B 40-hole 4x10",
        "tag": "V",
    },
    "hole_d": {
        "nominal": 0.0205,
        "range": [0.0175, 0.0265],
        "unit": "m",
        "src": "B 适配管径 20 mm（目录）；关系 D = d_tube + 2c，c∈[0.5,2.0] mm",
        "tag": "B+ENG",
    },
    "tube_d": {
        "nominal": 0.01678,
        "range": [0.01678, 0.02448],
        "unit": "m",
        "src": "P 17x17x195（MEAS 16.78）；S 径向 1…1.44 → 24.48",
        "tag": "MEAS+S",
    },
    "hole_clearance_c": {
        "nominal": 0.00075,
        "range": [0.00005, 0.0020],
        "unit": "m",
        "src": "名义 0.75 mm（参考架按 20 mm 管、22 mm 格）；下限 0.05 = 仿真资产实测 0.06 mm "
               "（17.0 格 vs 16.78 管，比 contact_offset 0.2 还紧）；上限为工程包络",
        "tag": "MEAS+ENG",
    },
    "pitch_x": {
        "nominal": 0.02587,
        "range": [0.0200, 0.0340],
        "unit": "m",
        "src": "B 外框 265 mm / 10 孔 + 关系 pitch >= hole_d + wall",
        "tag": "B+ENG",
    },
    "pitch_y": {
        "nominal": 0.03117,
        "range": [0.0200, 0.0340],
        "unit": "m",
        "src": "B 外框 126 mm / 4 行 + 关系 pitch >= hole_d + wall",
        "tag": "B+ENG",
    },
    "wall_t": {
        "nominal": 0.0035,
        "range": [0.0022, 0.0060],
        "unit": "m",
        "src": "仿真资产实测壁厚 3.2/2.2 mm(×1.1 = 3.5/2.4)",
        "tag": "MEAS",
    },
    "plate_t": {
        "nominal": 0.0036,
        "range": [0.0030, 0.0060],
        "unit": "m",
        "src": "仿真资产实测顶板 3.6 mm",
        "tag": "MEAS",
    },
    "base_t": {
        "nominal": 0.0060,
        "range": [0.0050, 0.0080],
        "unit": "m",
        "src": "仿真资产实测底板 6.0 mm",
        "tag": "MEAS",
    },
    "height": {
        "nominal": 0.0750,
        "range": [0.0600, 0.1138],
        "unit": "m",
        "src": "B 外框 75 mm；P 79.3（MEAS）；S×1.44 上限 113.8",
        "tag": "B+MEAS",
    },
    "tiers": {
        "nominal": 1,
        "range": [1, 2],
        "unit": "count",
        "src": "1 = 只有顶板+底板；2 = 加一层中间板（工程族）",
        "tag": "ENG",
    },
    "tier_gap": {
        "nominal": 0.0250,
        "range": [0.0200, 0.0400],
        "unit": "m",
        "src": "工程：层间距须大于孔内管的可插入长度余量",
        "tag": "ENG",
    },
    "margin_x": {"nominal": 0.0060, "range": [0.0050, 0.0080], "unit": "m",
                 "src": "仿真资产实测边距 5.5/6.4 mm", "tag": "MEAS"},
    "margin_y": {"nominal": 0.0060, "range": [0.0050, 0.0080], "unit": "m",
                 "src": "仿真资产实测边距 5.5/6.4 mm", "tag": "MEAS"},
    "density": {"nominal": 1190.0, "range": [1150.0, 1250.0], "unit": "kg/m3",
                "src": "PMMA/亚克力 1150–1250（常用试管架材料）；仿真 config 写死 5 kg 属另一物理档",
                "tag": "ENG"},
    "envelope_L": {"nominal": 0.265, "range": [0.265, 0.396], "unit": "m",
                   "src": "候选包络（B 265 与 P/S 并集）", "tag": "B+P"},
    "envelope_W": {"nominal": 0.126, "range": [0.116, 0.16704], "unit": "m",
                   "src": "候选包络", "tag": "B+P"},
}

# ---------------------------------------------------------------------------
# marker
# ---------------------------------------------------------------------------
MARKER = {
    "d": {
        "nominal": 0.01533,
        "range": [0.00958, 0.02325],
        "unit": "m",
        "src": "仿真细笔实测 9.58 mm；仿真当前笔 15.33 mm（9.58×1.6）；"
               "S 径向 1…1.55 × 15 = 23.25 上限",
        "tag": "MEAS+S",
    },
    "length": {
        "nominal": 0.14340,
        "range": [0.1300, 0.1500],
        "unit": "m",
        "src": "仿真笔实测 143.4 mm；E 单件包装 137 mm",
        "tag": "MEAS+E",
    },
    "cap_len": {
        "nominal": 0.0160,
        "range": [0.0, 0.0250],
        "unit": "m",
        "src": "0 表示无帽；帽段长度按 E 图形比例拟合（0.11L–0.17L），未核验，列 X",
        "tag": "E(X)",
    },
    "cap_clearance": {
        "nominal": 0.0006,
        "range": [0.0004, 0.0010],
        "unit": "m",
        "src": "工程：帽外径略大于杆径，差值即帽壁",
        "tag": "ENG",
    },
    "tip_len": {
        "nominal": 0.0080,
        "range": [0.0040, 0.0120],
        "unit": "m",
        "src": "工程：锥形尖段（marker 尖），名义 8 mm",
        "tag": "ENG",
    },
    "sides": {
        "nominal": 48,
        "range": [12, 64],
        "unit": "count",
        "src": "12 = 略带棱截面；≥32 近似圆截面",
        "tag": "ENG",
    },
    "density": {
        "nominal": 1050.0,
        "range": [900.0, 1400.0],
        "unit": "kg/m3",
        "src": "ABS/PP/PE 塑料密度带（900–1400）",
        "tag": "ENG",
    },
    "fill": {
        "nominal": 0.535,
        "range": [0.45, 0.60],
        "unit": "ratio",
        "src": "E 净重 13.6 g（137×15）反标定：13.6 g / (1050 × 实心体积 2.42e-5) = 0.535",
        "tag": "E",
    },
    "holder_inner_d": {
        "nominal": 0.08560,
        "range": [0.0800, 0.0856],
        "unit": "m",
        "src": "笔筒资产实测内口 85.6 mm（0.56 缩放后；measure_nominal.py）",
        "tag": "MEAS",
    },
    "holder_height": {
        "nominal": 0.09635,
        "range": [0.09635, 0.09635],
        "unit": "m",
        "src": "笔筒资产实测 96.35 mm（0.45 缩放后）",
        "tag": "MEAS",
    },
    "insert_gap": {
        "nominal": 0.0060,
        "range": [0.0020, 0.0100],
        "unit": "m",
        "src": "2×items_handover contact_offset(3 mm)",
        "tag": "CFG",
    },
}

# ---------------------------------------------------------------------------
# relation thresholds (all in metres unless stated)
# ---------------------------------------------------------------------------
LIMITS = {
    # rack
    "rack_vertical_deg": 5.0,        # official success criterion, sample_loading.py:160
    "rack_min_exposed_m": 0.030,     # ENG: tube must stick out far enough to grasp
    "rack_reach_m": [0.06, 0.45],    # table-reachable outer size
    "rack_height_m": [0.055, 0.125],
    "rack_min_grid_wall_m": 0.0015,  # grid wall between two cells
    "rack_min_post_m": 0.008,        # side-wall post width
    "rack_min_opening_m": 0.020,     # window width
    # marker
    "marker_len_over_holder": [1.20, 1.75],   # ENG, nominal 1.488 measured
    "marker_min_exposed_m": 0.030,
    "marker_mass_kg": [0.003, 0.045],
}
