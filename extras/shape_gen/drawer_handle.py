"""Drawer handle blocks for the upper drawer (two proposals: `shallow_arc`, `bridge`).

Only the handle geometry is generated here (convex blocks in the drawer inner-link frame, metres). The official
drawer meshes are NOT needed for that and are not part of this repository.

    python3 drawer_handle.py --out handles                      # writes handles/drawer_handle_<family>.obj
    python3 drawer_handle.py --out handles --official-drawer <dir with the official SlidingBoxDrawer files>
                                                                # additionally assembles full drawer packages

With --official-drawer the script keeps the official inner-box convex groups (all except the three official handle
groups geometry_5 / geometry_10 / geometry_11), appends the handle blocks, and writes `InnerBox_candidate.obj` plus a
`SlidingBoxDrawer.urdf` whose inner box uses it. The grasp frame (`fixed_handle` joint) is moved to the handle bar.
Needs numpy + trimesh.
"""
from __future__ import annotations

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh

FACE_X = 0.09114578          # outermost x of the official front panel (inner-link frame, m)
OFFICIAL_HANDLE_GROUPS = ("geometry_5", "geometry_10", "geometry_11")
GRASP_X = {"shallow_arc": 0.118, "bridge": 0.125}


def _box(lo, hi):
    lo = np.array(lo, float)
    hi = np.array(hi, float)
    m = trimesh.creation.box(extents=hi - lo)
    m.apply_translation((lo + hi) / 2)
    return m


def bridge_blocks():
    """Cross bar plus two feet (three overlapping boxes)."""
    f = FACE_X
    return [_box([f, -0.035, 0.095], [0.126, -0.025, 0.105]),
            _box([f, 0.025, 0.095], [0.126, 0.035, 0.105]),
            _box([0.122, -0.04, 0.091], [0.128, 0.04, 0.109])]


def shallow_arc_blocks():
    """16 convex segments approximating a shallow arc (width 80 mm) plus two feet."""
    f = FACE_X
    ys = np.linspace(-0.04, 0.04, 17)

    def x(y):
        return 0.108 + 0.010 * (1 - (y / 0.04) ** 2)
    out = []
    for a, b in zip(ys[:-1], ys[1:]):
        pts = [[x(y) + dx, y, z] for y in (a, b) for dx in (-0.003, 0.003) for z in (0.091, 0.109)]
        out.append(trimesh.convex.convex_hull(np.array(pts)))
    out += [_box([f, -0.035, 0.095], [0.113, -0.025, 0.105]), _box([f, 0.025, 0.095], [0.113, 0.035, 0.105])]
    return out


FAMILIES = {"shallow_arc": shallow_arc_blocks, "bridge": bridge_blocks}


def write_obj(path, meshes, prefix="part"):
    lines, off = [], 0
    for i, m in enumerate(meshes):
        lines.append(f"o {prefix}_{i}")
        lines.extend("v " + " ".join(f"{x:.10g}" for x in v) for v in m.vertices)
        lines.extend("f " + " ".join(str(int(x) + off + 1) for x in f) for f in m.faces)
        off += len(m.vertices)
    Path(path).write_text("\n".join(lines) + "\n")


def read_groups(path):
    """OBJ with `o` groups -> [(name, Trimesh)] (global or per-group vertex numbering both handled)."""
    vs, gs = [], []
    for line in Path(path).read_text().splitlines():
        a = line.split()
        if not a:
            continue
        if a[0] == "o":
            gs.append([a[1], [], len(vs)])
        elif a[0] == "v":
            vs.append([float(x) for x in a[1:4]])
        elif a[0] == "f":
            gs[-1][1].append([int(x.split("/")[0]) - 1 for x in a[1:]])
    out = []
    for name, fs, _start in gs:
        m = trimesh.Trimesh(vertices=vs, faces=fs, process=False)
        m.remove_unreferenced_vertices()
        out.append((name, m))
    return out


def assemble(official_dir, out_dir, family, handle):
    """Full drawer package from the official files + our handle blocks."""
    off = Path(official_dir)
    d = Path(out_dir) / family
    d.mkdir(parents=True, exist_ok=True)
    for p in off.iterdir():
        if p.is_file():
            shutil.copy2(p, d / p.name)
    parts = read_groups(off / "InnerBoxACD.obj")
    body = [m for n, m in parts if n not in OFFICIAL_HANDLE_GROUPS]
    write_obj(d / "InnerBox_candidate.obj", body + handle)
    root = ET.parse(off / "SlidingBoxDrawer.urdf").getroot()
    for tag in ("visual", "collision"):
        root.find(f"link[@name='inner_box']/{tag}/geometry/mesh").set("filename", "InnerBox_candidate.obj")
    root.find("joint[@name='fixed_handle']/origin").set("xyz", f"{GRASP_X[family]} 0 0.1")
    ET.ElementTree(root).write(d / "SlidingBoxDrawer.urdf", encoding="utf-8", xml_declaration=True)
    return d


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--official-drawer", help="directory with the official SlidingBoxDrawer files (optional)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for fam, fn in FAMILIES.items():
        blocks = fn()
        write_obj(out / f"drawer_handle_{fam}.obj", blocks, prefix="handle")
        print(f"{fam}: {len(blocks)} blocks -> {out / f'drawer_handle_{fam}.obj'}")
        if a.official_drawer:
            print("  assembled:", assemble(a.official_drawer, out, fam, blocks))


if __name__ == "__main__":
    main()
