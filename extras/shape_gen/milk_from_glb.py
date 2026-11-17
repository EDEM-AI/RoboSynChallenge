"""Derive the two milk-carton meshes (`milk_flat_cap`, `milk_gable_nocap`) from the original Sketchfab GLB files.

    python3 milk_from_glb.py --flat <ee5edb94...glb> --gable <71364315...glb> --out <dir>

Source works (CC BY 4.0, see the dataset ATTRIBUTION.md):
  flat   "DAY 14 #3 December2019- Search" by hardbottle  https://sketchfab.com/3d-models/ee5edb945f33416a9e63f3b4aa665d56
  gable  "Milk Carton" by cebbi                        https://sketchfab.com/3d-models/713643151cce41798f6dfb98bbff09cb

Steps: merge vertices, (gable only) keep the 220-face watertight carton body and drop the separate cap component,
rotate source Y-up to Z-up, scale anisotropically to 46 x 48 x 105 mm, put the origin at the bottom centre.
Needs numpy + trimesh (+ scipy for the connected-component search).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh

R_YUP_TO_ZUP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.0]])
TARGET_EXTENT = np.array([0.046, 0.048, 0.105])


def derive(glb_path, gable=False):
    scene = trimesh.load(glb_path)
    m = scene.to_mesh() if hasattr(scene, "to_mesh") else scene.dump(concatenate=True)
    m.merge_vertices(merge_tex=True, merge_norm=True)
    if gable:
        groups = trimesh.graph.connected_components(m.face_adjacency, nodes=np.arange(len(m.faces)), engine="scipy")
        g = next(g for g in groups if len(g) == 220)
        body = trimesh.Trimesh(vertices=m.vertices.copy(), faces=m.faces[g], process=False)
        body.remove_unreferenced_vertices()
        assert body.is_watertight
        m = body
    v = m.vertices @ R_YUP_TO_ZUP.T
    scale = TARGET_EXTENT / np.ptp(v, axis=0)
    offset = -v.min(0) * scale - np.array([TARGET_EXTENT[0] / 2, TARGET_EXTENT[1] / 2, 0.0])
    return trimesh.Trimesh(vertices=v * scale + offset, faces=m.faces.copy(), process=False)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--flat", required=True)
    ap.add_argument("--gable", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    derive(a.flat).export(out / "milk_flat_cap.ply")
    derive(a.gable, gable=True).export(out / "milk_gable_nocap.ply")
    print("wrote", out)


if __name__ == "__main__":
    main()
