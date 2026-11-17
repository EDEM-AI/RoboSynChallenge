"""把生成架（原点 = 包围盒底面中心）另存一份「官方原点版」：原点挪到包围盒中心（官方 test_tube_rack_standard 的约定，z 居中）。
只平移，不改形状：网格顶点 z −= H/2；标注里所有坐标点（p / point / com / bbox.min / bbox.max / support_polygon）同样平移；
legacy_root_offset_m 归零。用法：python3 recenter_rack.py <out/rack 目录> <输出目录> <变体名>...
输出 <输出目录>/<名>_zc/<名>_zc.{obj,ply,annotations.json} 和 index.json（每个文件 sha256）。"""
import sys, json, hashlib, struct
from pathlib import Path
import numpy as np

POINT_KEYS = {"p", "point", "com", "min", "max"}

def shift_ann(o, dz, key=None):
    if isinstance(o, dict):
        return {k: shift_ann(v, dz, k) for k, v in o.items()}
    if isinstance(o, list):
        if key in POINT_KEYS and len(o) == 3 and all(isinstance(x, (int, float)) for x in o):
            return [o[0], o[1], round(o[2] - dz, 9)]
        if key == "support_polygon":
            return [[q[0], q[1], round(q[2] - dz, 9)] for q in o]
        return [shift_ann(v, dz, key if key in ("items",) else None) for v in o]
    return o

def read_ply(p):
    raw = p.read_bytes(); i = raw.index(b"end_header\n") + len(b"end_header\n"); head = raw[:i].decode().splitlines()
    assert "format binary_little_endian 1.0" in head
    nv = int([l for l in head if l.startswith("element vertex")][0].split()[-1]); nf = int([l for l in head if l.startswith("element face")][0].split()[-1])
    props = [l for l in head if l.startswith("property")]
    assert props[:3] == ["property float x", "property float y", "property float z"] and len(props) == 4, props
    V = np.frombuffer(raw, "<f4", nv * 3, i).reshape(nv, 3).copy(); return raw[:i], V, raw[i + nv * 12:], nf

def main():
    src, dst, names = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]; index = []
    for n in names:
        a = json.loads((src / n / f"{n}.annotations.json").read_text(encoding="utf-8"))
        dz = float(a["legacy_root_offset_m"][2]); H = float(a["bbox"]["extent"][2]); assert abs(dz - H / 2) < 1e-9 and abs(a["bbox"]["min"][2]) < 1e-9
        nn = n + "_zc"; od = dst / nn; od.mkdir(parents=True, exist_ok=True)
        head, V, tail, nf = read_ply(src / n / f"{n}.ply")
        assert abs(float(V[:, 2].min())) < 1e-6 and abs(float(V[:, 2].max()) - H) < 1e-5, (V[:, 2].min(), V[:, 2].max(), H)
        V[:, 2] -= np.float32(dz)
        (od / f"{nn}.ply").write_bytes(head.replace(n.encode(), nn.encode()) + V.astype("<f4").tobytes() + tail)
        lines = []
        for l in (src / n / f"{n}.obj").read_text().splitlines():
            if l.startswith("v "):
                t = l.split(); l = f"v {t[1]} {t[2]} {float(t[3]) - dz:.9f}"
            lines.append(l)
        (od / f"{nn}.obj").write_text("\n".join(lines) + "\n")
        b = shift_ann(a, dz); b["name"] = nn; b["origin"] = "centre of the bounding box (official test_tube_rack_standard convention); support plane z = -H/2"
        b["legacy_root_offset_m"] = [0.0, 0.0, 0.0]; b["recentered_from"] = dict(name=n, dz_m=dz, by="recenter_rack.py")
        for it in b["parts"]["rack"]["holes"]["items"] + b["parts"]["rack"]["plates"]["items"]: it["src"] = f"variant:rack/{nn}"
        (od / f"{nn}.annotations.json").write_text(json.dumps(b, ensure_ascii=False, indent=1), encoding="utf-8")
        sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        index.append(dict(name=nn, from_name=n, dz_m=dz, extent_mm=[round(x * 1000, 1) for x in a["bbox"]["extent"]], layers=a["params"].get("layers"),
                          z_range_m=[round(float(V[:, 2].min()), 5), round(float(V[:, 2].max()), 5)], faces=nf,
                          hole_top_z_m=b["parts"]["rack"]["holes"]["items"][0]["p"][2],
                          files={f.name: sha(f) for f in sorted(od.iterdir())}))
    (dst / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    for r in index: print(r["name"], r["layers"], r["extent_mm"], "z", r["z_range_m"], "孔口z", r["hole_top_z_m"], r["files"][r["name"] + ".ply"][:8])

main()
