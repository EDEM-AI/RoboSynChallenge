#!/usr/bin/env python
"""Measure the simple functional parts used by faq_checks straight from the official asset files.

Covers the entries of ``faq_checks/parts_default.json`` that are plain geometric measurements of one mesh:

  items_handover      pen.tip / pen.tail        extreme points along the pen's long axis (local x)
                      holder.mouth              rim centre, up axis (local y), inner / outer rim radius
  item_assembly       guijiao1 / guijiao2 end_head   the +x end of each tube mesh
  drawer_open_place   drawer.floor / cavity     flat top face of the inner box's bottom plate and the height above it
                      duck.center               bounding-box centre of the tomato mesh

Usage (repository root, needs numpy and trimesh):
    python scripts/measure_faq_parts.py [--drawer-dir <dir containing InnerBox.stl>]

The other parts in the table (basket, milk, cup, bottle, beaker, rack, test tube, bell, ...) were measured at several body scales
with a longer mesh-feature routine and de-scaled to body_scale = 1; see the ``_meta`` entry of the JSON file.
The drawer meshes ship with EmbodiChain's downloaded asset pack, not with this repository.
"""
import argparse
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(path):
    import trimesh
    return np.asarray(trimesh.load(path, force="mesh", process=False).vertices, float)


def _end(V, axis, sign, frac):
    """Extreme point along ``axis``: mean coordinate of the vertices within ``frac`` of the extent at that end, centred on the other axes."""
    lo, hi = V[:, axis].min(), V[:, axis].max()
    span = hi - lo
    sel = V[:, axis] >= hi - frac * span if sign > 0 else V[:, axis] <= lo + frac * span
    p = 0.5 * (V.min(0) + V.max(0))
    p[axis] = V[sel, axis].mean()
    return p


def measure(drawer_dir=None):
    out = {}
    V = _load(os.path.join(ROOT, "assets/Pen/the_pen.obj"))
    out["items_handover.pen"] = {"tip": _end(V, 0, 1, 0.02).tolist(), "tail": _end(V, 0, -1, 0.02).tolist()}
    V = _load(os.path.join(ROOT, "assets/PenHolder/pen_holder.obj"))
    lo, hi = V[:, 1].min(), V[:, 1].max()
    c = 0.5 * (V.min(0) + V.max(0))
    rim = V[V[:, 1] >= hi - 0.02 * (hi - lo)]
    r = np.hypot(rim[:, 0] - c[0], rim[:, 2] - c[2])
    out["items_handover.holder.mouth"] = {"p": [c[0], hi, c[2]], "normal": [0, 1, 0], "r_in": float(r.min()), "r_out": float(r.max())}
    out["item_assembly.guijiao1.end_head"] = _end(_load(os.path.join(ROOT, "assets/guijiao_standard/2guijiao_standard.ply")), 0, 1, 0.03).tolist()
    out["item_assembly.guijiao2.end_head"] = _end(_load(os.path.join(ROOT, "assets/guijiao_standard/1guijiao_standard.ply")), 0, 1, 0.03).tolist()
    V = _load(os.path.join(ROOT, "assets/tomato/tomato_asset_simready/asset_simready.obj"))
    out["drawer_open_place.duck.center"] = (0.5 * (V.min(0) + V.max(0))).tolist()
    if drawer_dir:
        V = _load(os.path.join(drawer_dir, "InnerBox.stl"))
        z_floor = float(np.unique(np.round(V[:, 2], 4))[1])           # second lowest plane = top face of the bottom plate
        top = V[np.isclose(V[:, 2], z_floor, atol=1e-4)]
        out["drawer_open_place.drawer.floor"] = {"z": z_floor, "x": [float(top[:, 0].min()), float(top[:, 0].max())], "y": [float(top[:, 1].min()), float(top[:, 1].max())]}
        out["drawer_open_place.drawer.cavity.height"] = float(V[:, 2].max() - z_floor)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--drawer-dir", default=None)
    a = ap.parse_args()
    print(json.dumps(measure(a.drawer_dir), indent=1))
