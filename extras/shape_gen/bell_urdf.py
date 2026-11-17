"""Two-link URDF candidates for the library bells (the five bells need a two-link URDF).

The library bells are single rigid meshes.  This script splits each one at the
cover plane (the height where the button/top cap starts, taken from the bell's
own annotation `cover_top`) into:

    bell_<name>_base.obj    语气：下面那部分（座 + 钟壳）
    bell_<name>_cover.obj   顶盖 / 按钮那一段
    bell_<name>.urdf        base link + button_cover link + prismatic press joint

Joint: `press`, axis -z, limits [-0.005, 0] m (the official button.urdf also has a 5 mm stroke).

    python3 bell_urdf.py            # 全会库铃（out/libimport/bell/*/）
"""

import json
import os
import sys

import numpy as np

import geom

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "out", "libimport", "bell")
OUT = os.path.join(HERE, "out", "bell_urdf")

URDF = """<?xml version="1.0" encoding="UTF-8"?>
<!-- two-link bell candidate; source mesh：{src_name} (sha256={src_sha}) -->
<robot name="bell_{tag}">
  <link name="base">
    <inertial>
      <origin xyz="0 0 {base_com_z:.6f}" rpy="0 0 0"/>
      <mass value="{base_mass:.6f}"/>
      <inertia ixx="{ixx:.9f}" ixy="0." ixz="0." iyy="{iyy:.9f}" iyz="0." izz="{izz:.9f}"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="{base_mesh}"/></geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="{base_mesh}"/></geometry>
    </collision>
  </link>

  <link name="button_cover">
    <inertial>
      <origin xyz="0 0 {cover_com_z:.6f}" rpy="0 0 0"/>
      <mass value="{cover_mass:.6f}"/>
      <inertia ixx="{cixx:.9f}" ixy="0." ixz="0." iyy="{ciyy:.9f}" iyz="0." izz="{cizz:.9f}"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="{cover_mesh}"/></geometry>
    </visual>
    <collision>
      <origin xyz="0 0 {btn_z:.6f}" rpy="0 0 0"/>
      <geometry><cylinder radius="{btn_r:.6f}" length="{btn_len:.6f}"/></geometry>
    </collision>
  </link>

  <joint name="press" type="prismatic">
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <parent link="base"/>
    <child link="button_cover"/>
    <axis xyz="0 0 -1"/>
    <limit lower="-0.005" upper="0.0" effort="50." velocity="0.2"/>
  </joint>
  <gazebo reference="button_cover">
    <mu1>0.6</mu1><mu2>0.6</mu2><maxContacts>3</maxContacts>
  </gazebo>
</robot>
"""


def split_bell(path, z_split):
    import trimesh
    m = trimesh.load(path, force="mesh", process=False)
    v = np.asarray(m.vertices, float)
    f = np.asarray(m.faces, np.int64)
    cen = v[f].mean(axis=1)
    cover_mask = cen[:, 2] > z_split
    parts = {}
    for key, mask in (("base", ~cover_mask), ("cover", cover_mask)):
        ff = f[mask]
        if len(ff) == 0:
            return None
        idx = np.unique(ff)
        remap = np.full(len(v), -1, dtype=np.int64)
        remap[idx] = np.arange(len(idx))
        parts[key] = (v[idx], remap[ff])
    return parts


def box_inertia(mass, ext):
    ex, ey, ez = ext
    return (mass * (ey * ey + ez * ez) / 12.0,
            mass * (ex * ex + ez * ez) / 12.0,
            mass * (ex * ex + ey * ey) / 12.0)


def main():
    os.makedirs(OUT, exist_ok=True)
    made = []
    for d in sorted(os.listdir(SRC)):
        src_dir = os.path.join(SRC, d)
        ann_path = os.path.join(src_dir, d + ".annotations.json")
        obj_path = os.path.join(src_dir, d + ".obj")
        if not (os.path.isdir(src_dir) and os.path.exists(ann_path) and os.path.exists(obj_path)):
            continue
        ann = json.load(open(ann_path, encoding="utf-8"))
        bell = ann["parts"]["button"]
        z_top = float(bell["cover_top"]["p"][2])
        z_split = max(z_top - 0.012, 0.55 * z_top)     # 盖 = 顶面下 12 mm 那段（按面积分质量）
        parts = split_bell(obj_path, z_split)
        if parts is None:
            print(f"  skip {d}: split failed")
            continue
        name = d
        out_dir = os.path.join(OUT, name)
        os.makedirs(out_dir, exist_ok=True)
        meshes = {}
        for key, (v, f) in parts.items():
            p = os.path.join(out_dir, f"{name}_{key}.obj")
            geom.write_obj(p, v, f, header=f"bell {name} {key}")
            meshes[key] = p
        # 库铃是壳（不水密），质量按**表面积**比例分（壳的质量正比于面积）
        vm = {}
        for key, (v, f) in parts.items():
            a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
            vm[key] = 0.5 * float(np.linalg.norm(np.cross(b - a, c - a), axis=1).sum())
        tot = sum(vm.values()) or 1.0
        ann_mass = (ann.get("mass") or {}).get("mass_kg")
        mass_total = float(ann_mass) if ann_mass else 0.2
        mass_src = "annotation.mass" if ann_mass else "ENG 0.2 kg 默认（库铃无质量数据）"
        base_mass = mass_total * vm["base"] / tot
        cover_mass = mass_total * vm["cover"] / tot
        def shell_inertia(v, f, mass):
            """壳近似：每个三角按面积分配质量，算绕自身质心的惯量 + 质心。"""
            a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
            cr = np.cross(b - a, c - a)
            area = 0.5 * np.linalg.norm(cr, axis=1)
            tot = area.sum() or 1.0
            cen = (a + b + c) / 3.0
            com = (cen * area[:, None]).sum(axis=0) / tot
            m_i = mass * area / tot
            r = cen - com
            I = np.zeros((3, 3))
            for k in range(3):
                I[k, k] = np.sum(m_i * (np.sum(r * r, axis=1) - r[:, k] ** 2))
            for i in range(3):
                for j in range(i + 1, 3):
                    I[i, j] = I[j, i] = -np.sum(m_i * r[:, i] * r[:, j])
            return com, I

        bv, bf = parts["base"]
        cv, cf = parts["cover"]
        bcom, bI = shell_inertia(bv, bf, base_mass)
        ccom, cI = shell_inertia(cv, cf, cover_mass)
        bi = (bI[0, 0], bI[1, 1], bI[2, 2])
        ci = (cI[0, 0], cI[1, 1], cI[2, 2])
        # 按钮碰撞/触发：2/3 号按钮直径 ×1.3，5 mm 行程
        face_d = float((ann.get("measurements") or {}).get("button_press_face_diameter_m")
                       or ann["parts"]["button"]["cover_top"]["section"]["r_out"] * 2.0)
        btn_scale = 1.3 if name in ("bell_02_77f2ea_84", "bell_03_b739cb_93") else 1.0
        btn_d = face_d * btn_scale
        btn_len = max(z_top - z_split, 0.002)
        btn_z = z_split + btn_len / 2.0
        import hashlib
        src_sha = hashlib.sha256(open(obj_path, "rb").read()).hexdigest()
        urdf = URDF.format(
            tag=name, src_name=d, src_sha=src_sha,
            base_mesh=os.path.basename(meshes["base"]),     # 相对 URDF 所在目录
            cover_mesh=os.path.basename(meshes["cover"]),
            base_mass=base_mass, cover_mass=cover_mass,
            base_com_z=float(bcom[2]), cover_com_z=float(ccom[2]),
            ixx=bi[0], iyy=bi[1], izz=bi[2], cixx=ci[0], ciyy=ci[1], cizz=ci[2],
            btn_r=btn_d / 2.0, btn_len=btn_len, btn_z=btn_z)
        up = os.path.join(out_dir, f"{name}.urdf")
        open(up, "w", encoding="utf-8").write(urdf)
        made.append({"name": name, "urdf": up, "z_split": round(z_split, 6),
                     "base_mass": round(base_mass, 6), "cover_mass": round(cover_mass, 6),
                     "base_com_z": round(float(bcom[2]), 6), "cover_com_z": round(float(ccom[2]), 6),
                     "mass_source": mass_src, "stroke_m": 0.005,
                     "button_scale": btn_scale,
                     "button_face_d_before_m": round(face_d, 6),
                     "button_face_d_after_m": round(btn_d, 6),
                     "collision": {"link": "button_cover", "shape": "cylinder",
                                   "radius": round(btn_d / 2.0, 6), "length": round(btn_len, 6),
                                   "centre_z": round(btn_z, 6)},
                     "contact_trigger": {"kind": "contact", "link": "button_cover",
                                         "stroke_trigger_m": 0.005,
                                         "rule": "press joint 到 -0.005 m（5 mm）且接触力 > 0 即触发",
                                         "readout": "no simulator run; button-face cylinder is in the URDF"}})
        print(f"  {name}: split z={z_split:.4f}  base={base_mass*1e3:.0f} g / cover={cover_mass*1e3:.0f} g")
    with open(os.path.join(OUT, "index.json"), "w", encoding="utf-8") as fh:
        json.dump({"schema": "hq-bell-urdf/1",
                   "note": "两连杆候选：base + button_cover，press 棱柱 joint 行程 5 mm（-z）；源为库铃网格，按盖面切分。",
                   "items": made}, fh, indent=1, ensure_ascii=False)
    print(f"\n{len(made)} bell URDF -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
