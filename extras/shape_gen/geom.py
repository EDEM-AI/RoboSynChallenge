"""Mesh construction / validation / export helpers for the shape generators.

Pure numpy: no trimesh, no boolean library, no simulator import.

Conventions
-----------
* Right handed, metres, "+z" is up for every exported asset (official assets
  use the same convention).
* Every mesh is closed and consistently oriented; ``check()`` proves it by
  edge counting (each undirected edge used by exactly two triangles, each
  directed edge used exactly once).
* Export is byte-deterministic: vertices/triangles keep insertion order and
  are formatted with fixed precision.

The building blocks are the ones the two families need:
    box()                solid slab / post
    perforated_plate()   slab with round through holes (rack top plate)
    loft()               rings -> closed surface (marker body, cap, tip)
"""

import hashlib
import json
import math
import os
import struct

import numpy as np

FLOAT_FMT = "%.9f"


# ----------------------------------------------------------------------------
# accumulator
# ----------------------------------------------------------------------------
class Mesh:
    def __init__(self):
        self.v = []
        self.f = []
        self.parts = []          # [{"name":..,"face_start":..,"face_end":..}]
        self._cur = None

    # -- vertices / faces ---------------------------------------------------
    def add_vertices(self, arr):
        base = len(self.v)
        arr = np.asarray(arr, dtype=float).reshape(-1, 3)
        self.v.extend(arr.tolist())
        return base

    def add_face(self, i, j, k):
        self.f.append((int(i), int(j), int(k)))

    def quad(self, a, b, c, d):
        """Two triangles for quad a-b-c-d (a,b,c,d in winding order)."""
        self.add_face(a, b, c)
        self.add_face(a, c, d)

    # -- part bookkeeping ---------------------------------------------------
    def part(self, name):
        self._cur = {"name": name, "face_start": len(self.f), "face_end": None}
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._cur["face_end"] = len(self.f)
        self.parts.append(self._cur)
        self._cur = None
        return False

    def begin(self, name):
        self._cur = {"name": name, "face_start": len(self.f), "face_end": None}

    def end(self):
        self._cur["face_end"] = len(self.f)
        self.parts.append(self._cur)
        self._cur = None

    # -- arrays -------------------------------------------------------------
    def arrays(self):
        return np.asarray(self.v, dtype=float), np.asarray(self.f, dtype=np.int64)

    def welded_parts(self, eps=1e-9):
        """Weld inside each part, keep parts as separate shells.

        Parts that touch face-to-face (posts on the base, rails against a grid
        plate) must NOT be welded across the seam, otherwise the shared edges
        end up with four faces and the mesh stops being edge-manifold.
        """
        V, F = self.arrays()
        out_v, out_f, offset = [], [], 0
        for _name, s, e in self.part_ranges():
            fs = F[s:e]
            idx = np.unique(fs)
            sub = V[idx]
            key = np.round(sub / eps).astype(np.int64)
            _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
            sub2 = sub[first]
            local = np.zeros(len(V), dtype=np.int64)
            local[idx] = inv.reshape(-1)
            f2 = local[fs]
            keep = (f2[:, 0] != f2[:, 1]) & (f2[:, 1] != f2[:, 2]) & (f2[:, 0] != f2[:, 2])
            out_v.append(sub2)
            out_f.append(f2[keep] + offset)
            offset += len(sub2)
        return np.vstack(out_v), np.vstack(out_f)

    def welded(self, eps=1e-9):
        """Merge coincident vertices and drop vertices no face references."""
        V, F = self.arrays()
        key = np.round(V / eps).astype(np.int64)
        _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
        V2 = V[first]
        F2 = inv.reshape(-1)[F]
        keep = ((F2[:, 0] != F2[:, 1]) & (F2[:, 1] != F2[:, 2]) & (F2[:, 0] != F2[:, 2]))
        F2 = F2[keep]
        used = np.unique(F2)
        remap = np.full(len(V2), -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        return V2[used], remap[F2]

    def part_ranges(self):
        return [(p["name"], p["face_start"], p["face_end"]) for p in self.parts]

    def transform(self, part_name, R=None, t=None):
        """Rotate/translate the vertices of one part."""
        V = np.asarray(self.v)
        _, s, e = [(n, s, e) for n, s, e in self.part_ranges() if n == part_name][0]
        # vertex indices used by this part's faces
        faces = np.asarray(self.f[s:e])
        idx = np.unique(faces)
        if R is not None:
            V[idx] = V[idx] @ np.asarray(R).T
        if t is not None:
            V[idx] = V[idx] + np.asarray(t)
        self.v = V.tolist()


# ----------------------------------------------------------------------------
# primitives (axis aligned)
# ----------------------------------------------------------------------------
def box(mesh, name, x0, x1, y0, y1, z0, z1):
    """Axis-aligned solid box with outward normals."""
    mesh.begin(name)
    b = mesh.add_vertices(np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ]))
    # bottom (-z)
    mesh.quad(b + 0, b + 3, b + 2, b + 1)
    # top (+z)
    mesh.quad(b + 4, b + 5, b + 6, b + 7)
    # -y
    mesh.quad(b + 0, b + 1, b + 5, b + 4)
    # +y
    mesh.quad(b + 3, b + 7, b + 6, b + 2)
    # -x
    mesh.quad(b + 0, b + 4, b + 7, b + 3)
    # +x
    mesh.quad(b + 1, b + 2, b + 6, b + 5)
    mesh.end()


# ----------------------------------------------------------------------------
# perforated plate:  rectangle minus round through holes, single shell
# ----------------------------------------------------------------------------
def _cell_rect_ring(x0, x1, y0, y1, k):
    """Rectangle boundary, CCW, k segments per side (4k points)."""
    pts = []
    for i in range(k):                        # bottom  y0, x0->x1
        pts.append((x0 + (x1 - x0) * i / k, y0))
    for i in range(k):                        # right   x1, y0->y1
        pts.append((x1, y0 + (y1 - y0) * i / k))
    for i in range(k):                        # top     y1, x1->x0
        pts.append((x1 - (x1 - x0) * i / k, y1))
    for i in range(k):                        # left    x0, y1->y0
        pts.append((x0, y1 - (y1 - y0) * i / k))
    return np.asarray(pts, dtype=float)


def perforated_plate(mesh, name, Lx, Wy, xs, ys, radius, z0, z1, k=8):
    """Slab [0,Lx]x[0,Wy]x[z0,z1] pierced by round holes at (xs[i], ys[j]).

    The plate is tiled into cells whose x/y edges sit at the midpoints between
    neighbouring holes (and at the plate border).  Top face, bottom face, hole
    walls and border walls are emitted for every cell; vertices are shared at
    the seams so the result is one closed, consistently oriented shell.
    """
    mesh.begin(name)
    xe = [0.0] + [(xs[i] + xs[i + 1]) * 0.5 for i in range(len(xs) - 1)] + [float(Lx)]
    ye = [0.0] + [(ys[j] + ys[j + 1]) * 0.5 for j in range(len(ys) - 1)] + [float(Wy)]
    n = 4 * k

    for j in range(len(ys)):
        for i in range(len(xs)):
            x0, x1 = xe[i], xe[i + 1]
            y0, y1 = ye[j], ye[j + 1]
            outer = _cell_rect_ring(x0, x1, y0, y1, k)
            cx, cy = xs[i], ys[j]
            ang = np.arctan2(outer[:, 1] - cy, outer[:, 0] - cx)
            inner = np.stack([cx + radius * np.cos(ang),
                              cy + radius * np.sin(ang)], axis=1)
            assert len(outer) == n

            ot = mesh.add_vertices(np.column_stack([outer, np.full(n, z1)]))
            ob = mesh.add_vertices(np.column_stack([outer, np.full(n, z0)]))
            it = mesh.add_vertices(np.column_stack([inner, np.full(n, z1)]))
            ib = mesh.add_vertices(np.column_stack([inner, np.full(n, z0)]))

            for a in range(n):
                b = (a + 1) % n
                # top face (+z)
                mesh.add_face(ot + a, ot + b, it + b)
                mesh.add_face(ot + a, it + b, it + a)
                # bottom face (-z)
                mesh.add_face(ob + a, ib + a, ib + b)
                mesh.add_face(ob + a, ib + b, ob + b)
                # hole wall (normal points towards the hole axis)
                mesh.quad(it + a, it + b, ib + b, ib + a)

            # border walls, only on sides that coincide with the plate border
            if i == 0:
                _wall(mesh, ob, ot, 3 * k, n, n)       # left edge side
            if i == len(xs) - 1:
                _wall(mesh, ob, ot, k, 2 * k, n)       # right edge side
            if j == 0:
                _wall(mesh, ob, ot, 0, k, n)           # bottom (y0) side
            if j == len(ys) - 1:
                _wall(mesh, ob, ot, 2 * k, 3 * k, n)   # top (y1) side
    mesh.end()


def _wall(mesh, ob, ot, a0, a1, n):
    """Border wall quad strip for ring indices [a0, a1) (outward normals)."""
    for a in range(a0, a1):
        b = (a + 1) % n
        mesh.quad(ot + a, ob + a, ob + b, ot + b)


# ----------------------------------------------------------------------------
# loft along +z (rings are 2D point arrays, CCW)
# ----------------------------------------------------------------------------
def loft(mesh, name, rings, zs, cap_start=True, cap_end=True):
    """Closed surface through rings at heights zs."""
    mesh.begin(name)
    rings = [np.asarray(r, dtype=float) for r in rings]
    n = len(rings[0])
    idx = []
    for r, z in zip(rings, zs):
        idx.append(mesh.add_vertices(np.column_stack([r, np.full(len(r), float(z))])))
    for s in range(len(rings) - 1):
        a0, b0 = idx[s], idx[s + 1]
        for a in range(n):
            b = (a + 1) % n
            # outward normals for CCW rings going up in +z
            mesh.add_face(a0 + a, a0 + b, b0 + b)
            mesh.add_face(a0 + a, b0 + b, b0 + a)
    if cap_end:
        c_end = mesh.add_vertices([[np.mean(rings[-1][:, 0]), np.mean(rings[-1][:, 1]), zs[-1]]])
        a0 = idx[-1]
        for a in range(n):
            b = (a + 1) % n
            mesh.add_face(c_end, a0 + a, a0 + b)          # +z normal
    if cap_start:
        c_st = mesh.add_vertices([[np.mean(rings[0][:, 0]), np.mean(rings[0][:, 1]), zs[0]]])
        a0 = idx[0]
        for a in range(n):
            b = (a + 1) % n
            mesh.add_face(c_st, a0 + b, a0 + a)           # -z normal
    mesh.end()


def ring_circle(radius, n, cx=0.0, cy=0.0, phase=0.0):
    a = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False) + phase
    return np.stack([cx + radius * np.cos(a), cy + radius * np.sin(a)], axis=1)


def ring_polygon(radius, n, cx=0.0, cy=0.0, phase=0.0, flatten=0.0):
    """Regular n-gon, optionally slightly faceted ("略带棱")."""
    a = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False) + phase
    r = np.full(n, radius)
    return np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], axis=1)


# ----------------------------------------------------------------------------
# validation
# ----------------------------------------------------------------------------
def check(mesh, verbose=False):
    """Return a dict of geometry checks (all must be True for a valid asset)."""
    V, F = mesh.arrays()
    return check_faces(V, F, verbose=verbose)


def check_faces(V, F, verbose=False):
    edges = {}
    directed = {}
    for tri in F:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            key = (a, b) if a < b else (b, a)
            edges[key] = edges.get(key, 0) + 1
            directed[(a, b)] = directed.get((a, b), 0) + 1
    two_manifold = all(c == 2 for c in edges.values())
    one_directed = all(c == 1 for c in directed.values())
    closed = two_manifold and one_directed

    # signed volume (divergence theorem)
    a = V[F[:, 0]]
    b = V[F[:, 1]]
    c = V[F[:, 2]]
    vol = float(np.sum(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6.0)

    # degenerate / duplicate faces
    degen = int(np.sum((np.linalg.norm(np.cross(b - a, c - a), axis=1) * 0.5) < 1e-15))
    uniq = len(set(map(tuple, np.sort(F, axis=1))))

    res = {
        "watertight": bool(closed),
        "edge_manifold": bool(two_manifold),
        "orientation_consistent": bool(one_directed),
        "volume_positive": bool(vol > 0),
        "volume_m3": vol,
        "n_vertices": int(len(V)),
        "n_faces": int(len(F)),
        "degenerate_faces": degen,
        "duplicate_faces": int(len(F) - uniq),
    }
    if verbose:
        print(json.dumps(res, indent=1))
    return res


def surface_points(verts, faces, samples_per_face=1):
    """Triangle centroids (used by tests to probe the surface)."""
    return verts[faces].mean(axis=1)


def surface_centroid(verts, faces):
    """Area-weighted centroid of the surface - a good COM proxy for a thin shell."""
    a = verts[faces[:, 0]]
    b = verts[faces[:, 1]]
    c = verts[faces[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    cen = (a + b + c) / 3.0
    w = area.sum()
    if w <= 0:
        return verts.mean(axis=0)
    return (cen * area[:, None]).sum(axis=0) / w


def distance_point_to_surface(p, verts, faces):
    """Exact point-triangle distance (small helper, brute force over faces)."""
    p = np.asarray(p, dtype=float)
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    e0 = v1 - v0
    e1 = v2 - v0
    d = v0 - p
    a = np.einsum("ij,ij->i", e0, e0)
    b = np.einsum("ij,ij->i", e0, e1)
    c = np.einsum("ij,ij->i", e1, e1)
    dd = np.einsum("ij,ij->i", e0, d)
    e = np.einsum("ij,ij->i", e1, d)
    det = a * c - b * b
    det = np.where(np.abs(det) < 1e-18, 1e-18, det)
    s = (b * e - c * dd) / det
    t = (b * dd - a * e) / det
    s = np.clip(s, 0.0, 1.0)
    t = np.clip(t, 0.0, 1.0)
    # re-project onto the triangle (simple clamp, good enough for probes)
    q = v0 + s[:, None] * e0 + t[:, None] * e1
    return float(np.min(np.linalg.norm(q - p, axis=1)))


def ray_hits(verts, faces, origin, direction, tmax=10.0):
    """Number of triangle crossings for a ray (brute force, no rtree)."""
    origin = np.asarray(origin, dtype=float)
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    e1 = v1 - v0
    e2 = v2 - v0
    h = np.cross(np.broadcast_to(d, e2.shape), e2)
    a = np.einsum("ij,ij->i", e1, h)
    ok = np.abs(a) > 1e-12
    f = np.zeros_like(a)
    f[ok] = 1.0 / a[ok]
    s = origin - v0
    u = f * np.einsum("ij,ij->i", s, h)
    q = np.cross(s, e1)
    v = f * (q @ d)
    t = f * np.einsum("ij,ij->i", e2, q)
    hit = ok & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (t > 1e-6) & (t < tmax)
    return int(np.sum(hit))


def ray_distance(verts, faces, origin, direction):
    """Distance to the first triangle hit along a ray (brute force)."""
    origin = np.asarray(origin, dtype=float)
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    e1 = v1 - v0
    e2 = v2 - v0
    h = np.cross(np.broadcast_to(d, e2.shape), e2)
    a = np.einsum("ij,ij->i", e1, h)
    ok = np.abs(a) > 1e-12
    f = np.zeros_like(a)
    f[ok] = 1.0 / a[ok]
    s = origin - v0
    u = f * np.einsum("ij,ij->i", s, h)
    q = np.cross(s, e1)
    v = f * (q @ d)
    t = f * np.einsum("ij,ij->i", e2, q)
    hit = ok & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (t > 1e-5)
    if not hit.any():
        return float("inf")
    return float(t[hit].min())


def inside(verts, faces, p):
    """Point-in-mesh by parity along +x."""
    return ray_hits(verts, faces, p, (1.0, 0.0, 0.0)) % 2 == 1


# ----------------------------------------------------------------------------
# mass properties (uniform density tetrahedron integration)
# ----------------------------------------------------------------------------
def mass_properties(verts, faces, density, fill=1.0):
    """Exact second moments by the tetra (origin, a, b, c) decomposition."""
    a = verts[faces[:, 0]]
    b = verts[faces[:, 1]]
    c = verts[faces[:, 2]]
    vol = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0

    # int x x^T dV over one tetra = V/10*(aa+bb+cc) + V/20*(all cross terms)
    def outer(x, y):
        return x[:, :, None] * y[:, None, :]

    M = (vol[:, None, None] / 10.0) * (outer(a, a) + outer(b, b) + outer(c, c))
    M += (vol[:, None, None] / 20.0) * (
        outer(a, b) + outer(b, a) + outer(a, c) + outer(c, a) + outer(b, c) + outer(c, b)
    )
    M = M.sum(axis=0)

    total = vol.sum()
    # centroid of a tetra is (a+b+c)/4, weighted by signed volume
    com = ((a + b + c) / 4.0 * vol[:, None]).sum(axis=0) / total

    I_o = np.trace(M) * np.eye(3) - M          # about the origin
    d = com
    shift = total * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    I_com = I_o - shift                        # about the centre of mass

    m = total * density * fill
    return {
        "volume_m3": float(total),
        "mass_kg": float(m),
        "density_effective_kgm3": float(density * fill),
        "com": com.tolist(),
        "inertia_kgm2": (I_com * density * fill).tolist(),
    }


# ----------------------------------------------------------------------------
# export
# ----------------------------------------------------------------------------
def write_obj(path, verts, faces, header=""):
    lines = [f"# {header}".rstrip()]
    for v in verts:
        lines.append("v " + " ".join(FLOAT_FMT % x for x in v))
    for f in faces:
        lines.append("f %d %d %d" % (f[0] + 1, f[1] + 1, f[2] + 1))
    data = ("\n".join(lines) + "\n").encode("utf-8")
    with open(path, "wb") as fh:
        fh.write(data)
    return data


def write_ply(path, verts, faces, header=""):
    """Binary little endian PLY (same layout official assets use)."""
    head = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"comment {header}".rstrip() + "\n"
        f"element vertex {len(verts)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        f"element face {len(faces)}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    ).encode("utf-8")
    body = verts.astype("<f4").tobytes()
    tri = np.empty(len(faces), dtype=[("n", "u1"), ("i", "<i4", (3,))])
    tri["n"] = 3
    tri["i"] = faces.astype("<i4")
    data = head + body + tri.tobytes()
    with open(path, "wb") as fh:
        fh.write(data)
    return data


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


# ----------------------------------------------------------------------------
# json / yaml writing (deterministic, no timestamps)
# ----------------------------------------------------------------------------
def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return round(float(o), 9)
    return o


def write_json(path, obj):
    data = json.dumps(_clean(obj), indent=2, ensure_ascii=False, sort_keys=False) + "\n"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(data)
    return data.encode("utf-8")


def _yaml_scalar(v, indent):
    pad = "  " * indent
    if isinstance(v, dict):
        out = []
        for k, val in v.items():
            if isinstance(val, (dict, list)):
                out.append(f"{pad}{k}:")
                out.append(_yaml_scalar(val, indent + 1))
            else:
                out.append(f"{pad}{k}: {_fmt(val)}")
        return "\n".join(out)
    if isinstance(v, list):
        out = []
        for val in v:
            if isinstance(val, (dict, list)):
                sub = _yaml_scalar(val, indent + 1).split("\n")
                out.append(f"{pad}- " + sub[0].strip())
                out.extend(sub[1:])
            else:
                out.append(f"{pad}- {_fmt(val)}")
        return "\n".join(out)
    return f"{pad}{_fmt(v)}"


def _fmt(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (np.floating, float)):
        return repr(round(float(v), 9))
    if isinstance(v, (np.integer, int)):
        return str(int(v))
    if v is None:
        return "null"
    return str(v)


def write_yaml(path, obj):
    data = _yaml_scalar(_clean(obj), 0) + "\n"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(data)
    return data.encode("utf-8")


# ----------------------------------------------------------------------------
# revolve a closed (r, z) profile around +z  ->  hollow bodies (bottles, cups,
# tubes, bells, plates).  Points with r ~ 0 become a single pole vertex.
# ----------------------------------------------------------------------------
def revolve_profile(mesh, name, pts, n=64, scale=(1.0, 1.0, 1.0), flip=False):
    """pts: closed list of (r, z) in CCW order (outer side first, then inner)."""
    mesh.begin(name)
    pts = [(float(r), float(z)) for r, z in pts]
    if abs(pts[0][0] - pts[-1][0]) < 1e-12 and abs(pts[0][1] - pts[-1][1]) < 1e-12:
        pts = pts[:-1]
    m = len(pts)
    sx, sy, sz = scale
    pole = [None] * m
    ring_base = len(mesh.v)

    def vid(k, i):
        r, z = pts[i]
        if r <= 1e-9:
            if pole[i] is None:
                pole[i] = mesh.add_vertices([[0.0, 0.0, z * sz]])
            return pole[i]
        return ring_base + k * m + i
    for k in range(n):
        a = 2.0 * math.pi * k / n
        for i in range(m):
            r, z = pts[i]
            mesh.v.append([r * math.cos(a) * sx, r * math.sin(a) * sy, z * sz])

    for k in range(n):
        k2 = (k + 1) % n
        for i in range(m):
            j = (i + 1) % m
            r0, r1 = pts[i][0], pts[j][0]
            if r0 <= 1e-9 and r1 <= 1e-9:
                continue
            v00, v01 = vid(k, i), vid(k, j)
            v10, v11 = vid(k2, i), vid(k2, j)
            if r0 <= 1e-9:                       # pole at pts[i]
                tri = (v00, v11, v01)            # P, (k2,j), (k,j)
                mesh.add_face(*tri[::-1] if flip else tri)
            elif r1 <= 1e-9:                     # pole at pts[j]
                tri = (v00, v10, v01)            # (k,i), (k2,i), P
                mesh.add_face(*tri[::-1] if flip else tri)
            else:
                if flip:
                    mesh.quad(v00, v01, v11, v10)
                else:
                    mesh.quad(v00, v10, v11, v01)
    mesh.end()


def extrude_polygon(mesh, name, pts, z0, z1):
    """Extrude a simple CCW polygon (list of (x, y)) between z0 and z1."""
    mesh.begin(name)
    pts = [(float(x), float(y)) for x, y in pts]
    n = len(pts)
    bot = mesh.add_vertices([[x, y, z0] for x, y in pts])
    top = mesh.add_vertices([[x, y, z1] for x, y in pts])
    for i in range(n):
        j = (i + 1) % n
        mesh.quad(bot + i, bot + j, top + j, top + i)
    for a, b, c in _triangulate(pts):
        mesh.add_face(bot + b, bot + a, bot + c)     # bottom (-z)
        mesh.add_face(top + a, top + b, top + c)     # top (+z)
    mesh.end()


def _point_in_tri(p, a, b, c):
    def cr(u, v, w):
        return (v[0] - u[0]) * (w[1] - u[1]) - (v[1] - u[1]) * (w[0] - u[0])
    d1, d2, d3 = cr(a, b, p), cr(b, c, p), cr(c, a, p)
    return (d1 >= -1e-12) and (d2 >= -1e-12) and (d3 >= -1e-12)


def _triangulate(pts):
    """Ear clipping for a simple CCW polygon; returns index triples."""
    n = len(pts)
    if n < 3:
        return []
    idx = list(range(n))
    tris = []
    guard = 0
    while len(idx) > 3 and guard < 100000:
        guard += 1
        cut = False
        for k in range(len(idx)):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            a, b, c = pts[i0], pts[i1], pts[i2]
            cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            if cross <= 1e-14:
                continue
            if any(_point_in_tri(pts[j], a, b, c) for j in idx if j not in (i0, i1, i2)):
                continue
            tris.append((i0, i1, i2))
            idx.pop(k)
            cut = True
            break
        if not cut:
            break
    if len(idx) == 3:
        tris.append(tuple(idx))
    return tris


# ----------------------------------------------------------------------------
# shared variant writer (obj + ply + annotations + variant.yaml)
# ----------------------------------------------------------------------------
def emit_variant(gen_file, family, name, verts, faces, ann, out_dir, seed=0, version="1",
                 extra_source=None, allow_open=False):
    checks = check_faces(verts, faces)
    ann["geometry"] = checks
    gen_sha = sha256_file(os.path.abspath(gen_file))
    ann["generator"] = {"file": os.path.basename(gen_file), "version": version, "sha256": gen_sha}
    ann["seed"] = int(seed)
    if not allow_open and not (checks["watertight"] and checks["orientation_consistent"]
                               and checks["volume_positive"]):
        raise AssertionError(f"[{family}/{name}] mesh check failed: {checks}")
    d = os.path.join(out_dir, name)
    os.makedirs(d, exist_ok=True)
    paths = {"obj": os.path.join(d, name + ".obj"),
             "ply": os.path.join(d, name + ".ply"),
             "ann": os.path.join(d, name + ".annotations.json")}
    write_obj(paths["obj"], verts, faces, header=f"shape_gen/{family} {name} seed={seed}")
    write_ply(paths["ply"], verts, faces, header=f"shape_gen/{family} {name} seed={seed}")
    write_json(paths["ann"], ann)
    files = {os.path.basename(p): sha256_file(p) for p in paths.values()}
    src = {"generator": f"{os.path.basename(gen_file)}@{gen_sha[:16]}",
           "params": {k: v for k, v in ann.get("params", {}).items()},
           "seed": int(seed)}
    if extra_source:
        src.update(extra_source)
    variant = {"name": name, "fpath": f"{family}/{name}", "body_scale": [1.0, 1.0, 1.0],
               "files": files, "source": src,
               "functional_read": {"path": name + ".annotations.json", "key": "parts"},
               "status": "draft"}
    for key in ("material",):
        if key in ann:
            variant[key] = ann[key]
    write_yaml(os.path.join(d, name + ".variant.yaml"), variant)
    return variant
