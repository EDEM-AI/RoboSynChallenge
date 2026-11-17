# -*- coding: utf-8 -*-
"""P001 - split the official beaker_mixer.obj into "body" + "lever" and generate the mixer_switch assets.

Basis
-----
- Idea: keep the body as a rigid object, make the lever a separate revolute articulation.
- Cut box (mixer body frame, metres, already multiplied by body_scale 0.3):
  x < -0.110 and z > 0.038 and y in [-0.006, 0.007]  => lever, everything else => body.
  The base sits at z 25-40 mm, the lever sticks out above z 40 mm, outermost x = -121.6 mm;
  inside the box: 1611 faces / 1084 vertices, bbox x [-121.6, -109.1], y [-4.5, 6.3], z [37.5, 58.3] mm.
- Pivot: body y axis through (-0.105, 0, 0.038) m (top of the base).
- Mass: lever 10 g; inertia from the convex hull at uniform density (conservative; read the engine value back).

The outputs are derived from the official mesh and are NOT distributed with this repository;
run this script (or `P001_mixer_switch.ensure_assets()`) against your own checkout:

    python3 split_beaker_mixer.py --src <repo>/assets/beaker_mixer/beaker_mixer.obj \
        --out <dir>            # default: ../assets/mixer_switch (or $P001_ASSET_DIR via ensure_assets)

Deterministic: same input and same trimesh version give byte-identical output (face and vertex order are kept).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import shutil

import numpy as np

# ---------------------------------------------------------------- 常量（和补丁里的一致）

BODY_SCALE = 0.3                     # 官方 gym_config 里 beaker_mixer.body_scale
PIVOT_M = (-0.105, 0.0, 0.038)       # 转轴位置（搅拌器本体系、已乘 body_scale 的米制，和 button_offset 同系）
CUT_X_M = -0.110                     # 已乘 body_scale 的米制
CUT_Z_M = 0.038
CUT_Y_M = (-0.006, 0.007)
LEVER_MASS_KG = 0.010
THETA_MAX_DEG = 30.0                 # 缺省；随机 20–35° 由范围文件给
EFFORT_NM = 0.5
VELOCITY_RAD_S = 5.0
DAMPING = 0.005

# URDF 里的坐标一律用「原始网格系」（body_scale 之前）：关节体的 body_scale 和机身一样取 0.3，
# 由引擎把整份 URDF（网格 + 关节原点）一起缩放。抽屉的 body_scale=[1,1,0.35] 能正常动，
# 说明引擎缩放关节原点；这里照同一口径，并在 readback 里用拨杆世界位姿再核一次。
PIVOT_RAW = tuple(c / BODY_SCALE for c in PIVOT_M)


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- obj 读写

def read_obj(path: pathlib.Path):
    """返回 (v, vt, vn, faces)，faces 是 [(vi, ti, ni) 三个顶点] 的 list（0 基，缺项 None）。"""
    v, vt, vn, faces = [], [], [], []
    for line in open(path, "r", encoding="utf-8", errors="replace"):
        if line.startswith("v "):
            v.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("vt "):
            vt.append([float(x) for x in line.split()[1:3]])
        elif line.startswith("vn "):
            vn.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            tri = []
            for tok in line.split()[1:]:
                parts = tok.split("/")
                vi = int(parts[0]) - 1
                ti = int(parts[1]) - 1 if len(parts) > 1 and parts[1] else None
                ni = int(parts[2]) - 1 if len(parts) > 2 and parts[2] else None
                tri.append((vi, ti, ni))
            if len(tri) != 3:
                raise SystemExit(f"非三角面：{line[:80]!r}")
            faces.append(tri)
    return (np.asarray(v, dtype=np.float64), np.asarray(vt, dtype=np.float64) if vt else None,
            np.asarray(vn, dtype=np.float64) if vn else None, faces)


def write_obj(path: pathlib.Path, header: str, mtllib: str | None, v, vt, vn, faces, offset=(0.0, 0.0, 0.0)):
    """只写用到的顶点/贴图坐标/法向，索引重排。offset 平移到 link 局部系。"""
    used_v, used_t, used_n = {}, {}, {}
    out_f = []
    for tri in faces:
        idx = []
        for vi, ti, ni in tri:
            if vi not in used_v:
                used_v[vi] = len(used_v)
            a = used_v[vi]
            b = None
            c = None
            if ti is not None:
                if ti not in used_t:
                    used_t[ti] = len(used_t)
                b = used_t[ti]
            if ni is not None:
                if ni not in used_n:
                    used_n[ni] = len(used_n)
                c = used_n[ni]
            idx.append((a, b, c))
        out_f.append(idx)

    lines = [header.rstrip("\n")]
    if mtllib:
        lines.append(f"mtllib {mtllib}")
    lines.append("o obj1")
    order_v = sorted(used_v, key=used_v.get)
    for vi in order_v:
        p = v[vi] + np.asarray(offset, dtype=np.float64)
        lines.append("v %.5f %.5f %.5f" % (p[0], p[1], p[2]))
    if vt is not None and used_t:
        for ti in sorted(used_t, key=used_t.get):
            q = vt[ti]
            lines.append("vt %.6f %.6f" % (q[0], q[1]))
    if vn is not None and used_n:
        for ni in sorted(used_n, key=used_n.get):
            q = vn[ni]
            lines.append("vn %.4f %.4f %.4f" % (q[0], q[1], q[2]))
    lines.append("usemtl mat0")
    lines.append("s off")
    for tri in out_f:
        parts = []
        for a, b, c in tri:
            if b is None and c is None:
                parts.append(f"{a + 1}")
            elif c is None:
                parts.append(f"{a + 1}/{b + 1}")
            else:
                parts.append(f"{a + 1}/{b + 1}/{c + 1}")
        lines.append("f " + " ".join(parts))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- 主流程

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="官方 assets/beaker_mixer/beaker_mixer.obj")
    ap.add_argument("--out", default=str(pathlib.Path(__file__).resolve().parents[1] / "assets" / "mixer_switch"))
    a = ap.parse_args(argv)
    src = pathlib.Path(a.src).resolve()
    out = pathlib.Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    v, vt, vn, faces = read_obj(src)
    print(f"[split] 读入 {len(v)} 顶点 / {len(faces)} 面")
    sv = v * BODY_SCALE
    cent = np.asarray([sv[[p[0] for p in tri]].mean(0) for tri in faces])
    lever_mask = ((cent[:, 0] < CUT_X_M) & (cent[:, 2] > CUT_Z_M)
                  & (cent[:, 1] > CUT_Y_M[0]) & (cent[:, 1] < CUT_Y_M[1]))
    lever_faces = [f for f, m in zip(faces, lever_mask) if m]
    body_faces = [f for f, m in zip(faces, lever_mask) if not m]
    print(f"[split] 拨杆 {len(lever_faces)} 面 / 机身 {len(body_faces)} 面")
    if not lever_faces:
        raise SystemExit("⛔ 切分框里没有面，检查坐标口径")

    header = ("# P001 mixer_switch: 由官方 assets/beaker_mixer/beaker_mixer.obj 切分"
              f"（源 sha256 {sha256(src)[:16]}…）\n# 生成器 envpatch_p001/generators/split_beaker_mixer.py，勿手改")

    # 1) 机身（原坐标；材质照官方 mtl 一起拷）
    body_obj = out / "beaker_mixer_body.obj"
    write_obj(body_obj, header, "beaker_mixer.mtl", v, vt, vn, body_faces)
    shutil.copyfile(src.parent / "beaker_mixer.mtl", out / "beaker_mixer.mtl")
    for img in ("image0.png", "image1.png"):
        p = src.parent / img
        if p.exists():
            shutil.copyfile(p, out / img)

    # 2) 拨杆（link 局部系、原始网格坐标：平移到转轴）
    lever_obj = out / "mixer_switch_lever.obj"
    write_obj(lever_obj, header, "beaker_mixer.mtl", v, vt, vn, lever_faces,
              offset=(-PIVOT_RAW[0], -PIVOT_RAW[1], -PIVOT_RAW[2]))

    # 3) 拨杆碰撞：1 块凸包（STL，link 局部系、原始网格坐标）
    import trimesh

    pts = np.unique(np.asarray([v[[p[0] for p in tri]] for tri in lever_faces]).reshape(-1, 3), axis=0)
    pts_local = pts - np.asarray(PIVOT_RAW)
    hull = trimesh.convex.convex_hull(pts_local)
    hull_path = out / "mixer_switch_lever_hull.stl"
    hull.export(str(hull_path))
    vol = float(abs(hull.volume))
    com = np.asarray(hull.centroid, dtype=np.float64)
    inertia = np.asarray(hull.moment_inertia, dtype=np.float64) * (LEVER_MASS_KG / vol)
    bounds = np.asarray(hull.bounds, dtype=np.float64)
    s = BODY_SCALE

    # 4) URDF
    theta = math.radians(THETA_MAX_DEG)
    urdf = f"""<?xml version="1.0" encoding="utf-8"?>
<!-- P001 · 搅拌器两位拨动开关（机身留刚体、拨杆单独关节体）。
     由 envpatch_p001/generators/split_beaker_mixer.py 生成，勿手改。
     坐标：switch_base 原点 = 搅拌器刚体本体系原点；拨杆网格在 link 局部系（转轴为原点）。
     关节轴 = 本体 +y；往机身里推为正（θ 从 0 到 θmax）。 -->
<robot name="mixer_switch">
  <link name="switch_base">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="0.001"/>
      <inertia ixx="1e-9" ixy="0" ixz="0" iyy="1e-9" iyz="0" izz="1e-9"/>
    </inertial>
  </link>
  <link name="switch_lever">
    <inertial>
      <origin xyz="{com[0]:.6f} {com[1]:.6f} {com[2]:.6f}" rpy="0 0 0"/>
      <mass value="{LEVER_MASS_KG:.6f}"/>
      <inertia ixx="{inertia[0, 0]:.9e}" ixy="{inertia[0, 1]:.9e}" ixz="{inertia[0, 2]:.9e}"
               iyy="{inertia[1, 1]:.9e}" iyz="{inertia[1, 2]:.9e}" izz="{inertia[2, 2]:.9e}"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="mixer_switch_lever.obj"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="mixer_switch_lever_hull.stl"/>
      </geometry>
    </collision>
  </link>
  <joint name="flip" type="revolute">
    <origin xyz="{PIVOT_RAW[0]:.6f} {PIVOT_RAW[1]:.6f} {PIVOT_RAW[2]:.6f}" rpy="0 0 0"/>
    <parent link="switch_base"/>
    <child link="switch_lever"/>
    <axis xyz="0 1 0"/>
    <limit lower="0.0" upper="{theta:.6f}" effort="{EFFORT_NM}" velocity="{VELOCITY_RAD_S}"/>
    <dynamics damping="{DAMPING}" friction="0.0"/>
  </joint>
</robot>
"""
    urdf_path = out / "mixer_switch.urdf"
    urdf_path.write_text(urdf, encoding="utf-8")

    # 5) manifest
    files = {}
    for p in sorted(out.iterdir()):
        if p.is_file() and p.name != "manifest.yaml":
            files[p.name] = {"sha256": sha256(p), "bytes": p.stat().st_size}
    lever_bbox_raw_mm = (bounds * 1000.0).round(2).tolist()
    lever_bbox_scaled_mm = (bounds * s * 1000.0).round(2).tolist()
    manifest = {
        "patch": "P001",
        "generated_by": "envpatch_p001/generators/split_beaker_mixer.py",
        "source": {
            "obj": str(src),
            "obj_sha256": sha256(src),
            "body_scale": BODY_SCALE,
            "cut_region_m": {"x_lt": CUT_X_M, "z_gt": CUT_Z_M, "y": list(CUT_Y_M)},
            "note": "切分框是本体系已乘 body_scale 的米制；URDF 里的坐标是原始网格系（未乘），"
                    "关节体 body_scale 和机身一样取 0.3，由引擎统一缩放。",
        },
        "pivot_scaled_m": list(PIVOT_M),
        "pivot_raw_m": [round(float(x), 6) for x in PIVOT_RAW],
        "axis": [0, 1, 0],
        "theta_max_deg": THETA_MAX_DEG,
        "theta_max_rad": round(theta, 6),
        "lever": {
            "mass_kg": LEVER_MASS_KG,
            "frame": "raw (body_scale 之前)",
            "hull_volume_m3": vol,
            "hull_com_m": [round(float(x), 6) for x in com],
            "hull_bbox_local_mm": lever_bbox_raw_mm,
            "hull_bbox_local_mm_scaled": lever_bbox_scaled_mm,
            "inertia_kgm2": [[float(f"{x:.6e}") for x in row] for row in inertia],
            "inertia_kgm2_scaled": [[float(f"{x * s * s:.6e}") for x in row] for row in inertia],
            "n_faces": len(lever_faces),
            "n_verts": int(len(np.unique([p[0] for tri in lever_faces for p in tri]))),
        },
        "body": {"n_faces": len(body_faces), "max_convex_hull_num": 12,
                 "note": "源网格本身非封闭（67k 边界边），不做全局补洞；碰撞走官方 12 块凸分解，切面开口被拨杆本体遮住，需要渲染核对。"},
        "files": files,
    }
    (out / "manifest.yaml").write_text(
        "# 由 generators/split_beaker_mixer.py 生成；文件 sha 在下面\n"
        + json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("pivot_raw_m", "theta_max_rad")}, ensure_ascii=False))
    print(json.dumps(manifest["lever"], ensure_ascii=False, indent=1))
    print(f"[split] 产物 → {out}")


if __name__ == "__main__":
    main()
