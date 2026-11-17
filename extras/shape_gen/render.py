"""Minimal offline software renderer (numpy only) + PNG writer.

Why: this machine has no matplotlib / PIL / pyglet / OpenGL, so trimesh's
save_image() cannot run.  This renders a triangle mesh with a z-buffer, a
Lambert-ish shading model and a checkerboard floor, and writes a PNG using
only the standard library (zlib).

Deterministic: no randomness, no timestamps.
"""

import struct
import zlib

import numpy as np


# ----------------------------------------------------------------------------
# PNG
# ----------------------------------------------------------------------------
def write_png(path, img):
    """img: uint8 array (H, W, 3).  Writes an 8-bit RGB PNG."""
    img = np.ascontiguousarray(img.astype(np.uint8))
    h, w, _ = img.shape
    raw = b"".join(b"\x00" + img[row].tobytes() for row in range(h))

    def chunk(tag, data):
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 6))
    png += chunk(b"IEND", b"")
    with open(path, "wb") as fh:
        fh.write(png)


# ----------------------------------------------------------------------------
# camera
# ----------------------------------------------------------------------------
def look_at(eye, target, up=(0.0, 0.0, 1.0)):
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    up = np.asarray(up, dtype=float)
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    n = np.linalg.norm(right)
    if n < 1e-9:
        right = np.array([1.0, 0.0, 0.0])
    else:
        right /= n
    up2 = np.cross(right, fwd)
    return eye, right, up2, fwd


def project(pts, cam, size, fov_deg):
    """World points -> (screen_x, screen_y, depth)."""
    eye, right, up2, fwd = cam
    rel = pts - eye
    x = rel @ right
    y = rel @ up2
    z = rel @ fwd
    f = 0.5 * size / np.tan(np.radians(fov_deg) * 0.5)
    zz = np.maximum(z, 1e-6)
    sx = f * x / zz + 0.5 * size
    sy = -f * y / zz + 0.5 * size
    return sx, sy, z


# ----------------------------------------------------------------------------
# rasteriser
# ----------------------------------------------------------------------------
class Canvas:
    def __init__(self, size, bg=(0.93, 0.94, 0.96)):
        self.size = size
        self.color = np.zeros((size, size, 3), dtype=float)
        self.color[:] = np.asarray(bg, dtype=float)
        # vertical background gradient
        g = np.linspace(1.0, 0.86, size)[:, None, None]
        self.color *= g
        self.depth = np.full((size, size), np.inf, dtype=float)

    def _fill(self, sx, sy, z, color):
        """Fill one triangle (already projected) with flat/gouraud shading."""
        x0 = int(np.floor(min(sx)))
        x1 = int(np.ceil(max(sx)))
        y0 = int(np.floor(min(sy)))
        y1 = int(np.ceil(max(sy)))
        s = self.size
        if x1 < 0 or y1 < 0 or x0 >= s or y0 >= s:
            return
        x0 = max(x0, 0)
        y0 = max(y0, 0)
        x1 = min(x1, s - 1)
        y1 = min(y1, s - 1)
        if x1 < x0 or y1 < y0:
            return

        ax, ay = sx[0], sy[0]
        bx, by = sx[1], sy[1]
        cx, cy = sx[2], sy[2]
        area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        if abs(area) < 1e-9:
            return

        xs = np.arange(x0, x1 + 1) + 0.5
        ys = np.arange(y0, y1 + 1) + 0.5
        X, Y = np.meshgrid(xs, ys)

        w0 = ((bx - X) * (cy - Y) - (by - Y) * (cx - X)) / area
        w1 = ((cx - X) * (ay - Y) - (cy - Y) * (ax - X)) / area
        w2 = 1.0 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            return

        # perspective-correct depth via 1/z interpolation
        invz = w0 / z[0] + w1 / z[1] + w2 / z[2]
        with np.errstate(divide="ignore"):
            depth = 1.0 / invz
        win = self.depth[y0 : y1 + 1, x0 : x1 + 1]
        upd = inside & (depth < win)
        if not upd.any():
            return
        win[upd] = depth[upd]
        region = self.color[y0 : y1 + 1, x0 : x1 + 1]
        region[upd] = color


def shade(normals, verts, cam):
    """Per-face Lambert = 0.30 + 0.70*|n.l1| + 0.15*|n.l2|, two lights."""
    eye = cam[0]
    fwd = cam[3]
    l1 = -fwd + np.array([0.45, 0.35, 0.55])
    l1 /= np.linalg.norm(l1)
    l2 = -fwd + np.array([-0.6, -0.3, 0.2])
    l2 /= np.linalg.norm(l2)
    n = normals
    nn = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.maximum(nn, 1e-12)
    i1 = np.abs(n @ l1)
    i2 = np.abs(n @ l2)
    return np.clip(0.30 + 0.62 * i1 + 0.18 * i2, 0.08, 1.0)


def face_normals(verts, faces):
    v = verts[faces]
    n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    return n


def render(verts, faces, size=520, eye=(0.42, -0.42, 0.34), target=(0.0, 0.0, 0.03),
           fov=30.0, base_color=(0.80, 0.82, 0.86), floor_z=0.0,
           floor_half=0.24, floor_tiles=12, bg=(0.93, 0.94, 0.96),
           floor_tint=(0.62, 0.62, 0.64), up=(0.0, 0.0, 1.0),
           face_colors=None):
    """Render mesh(es) into an uint8 RGB image."""
    canvas = Canvas(size, bg=bg)
    cam = look_at(eye, target, up=up)

    # -- floor checkerboard ------------------------------------------------
    if floor_half > 0:
        n = floor_tiles
        step = 2.0 * floor_half / n
        gx = np.linspace(-floor_half, floor_half, n + 1)
        for i in range(n):
            for j in range(n):
                c = np.array(floor_tint) * (1.0 if (i + j) % 2 == 0 else 0.86)
                q = np.array([
                    [gx[i], gx[j], floor_z],
                    [gx[i + 1], gx[j], floor_z],
                    [gx[i + 1], gx[j + 1], floor_z],
                    [gx[i], gx[j + 1], floor_z],
                ])
                for tri in ([0, 1, 2], [0, 2, 3]):
                    sx, sy, z = project(q[tri], cam, size, fov)
                    if min(z) <= 0.02:
                        continue
                    canvas._fill(sx, sy, z, c)

    # -- object ------------------------------------------------------------
    allv = np.asarray(verts, dtype=float)
    allf = np.asarray(faces, dtype=int)
    fn = face_normals(allv, allf)
    inten = shade(fn, allv, cam)
    sx, sy, sz = project(allv, cam, size, fov)
    base = np.asarray(base_color, dtype=float)
    for k in range(len(allf)):
        idx = allf[k]
        z = sz[idx]
        if min(z) <= 0.02:
            continue
        col = (base if face_colors is None else np.asarray(face_colors[k])) * inten[k]
        canvas._fill(sx[idx], sy[idx], z, col)

    return np.clip(canvas.color * 255.0, 0, 255).astype(np.uint8)


def frame(verts, direction=(0.72, -0.62, 0.42), size=520, fill=0.80, dist_factor=3.4):
    """Camera eye/target/fov that fits the mesh into the frame."""
    v = np.asarray(verts, dtype=float)
    lo, hi = v.min(axis=0), v.max(axis=0)
    center = (lo + hi) / 2.0
    radius = max(np.linalg.norm(hi - lo) / 2.0, 1e-6)
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    dist = dist_factor * radius
    eye = center + d * dist
    fov = 2.0 * np.degrees(np.arctan(fill * radius / dist))
    return eye, center, fov


# ----------------------------------------------------------------------------
# tiny 5x7 bitmap font (labels on the comparison sheets)
# ----------------------------------------------------------------------------
_FONT = {
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01110", "10001", "10000", "10000", "10000", "10001", "01110"],
    "D": ["11100", "10010", "10001", "10001", "10001", "10010", "11100"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01110", "10001", "10000", "10111", "10001", "10001", "01110"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
    "J": ["00111", "00010", "00010", "00010", "00010", "10010", "01100"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "11011", "10001"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00110", "01000", "10000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
    "6": ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00010", "01100"],
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
    ".": ["00000", "00000", "00000", "00000", "00000", "01100", "01100"],
    ",": ["00000", "00000", "00000", "00000", "01100", "00100", "01000"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    "_": ["00000", "00000", "00000", "00000", "00000", "00000", "11111"],
    "/": ["00001", "00010", "00010", "00100", "01000", "01000", "10000"],
    "(": ["00010", "00100", "01000", "01000", "01000", "00100", "00010"],
    ")": ["01000", "00100", "00010", "00010", "00010", "00100", "01000"],
    "=": ["00000", "00000", "11111", "00000", "11111", "00000", "00000"],
    ":": ["00000", "01100", "01100", "00000", "01100", "01100", "00000"],
    "+": ["00000", "00100", "00100", "11111", "00100", "00100", "00000"],
    "x": ["00000", "10001", "01010", "00100", "01010", "10001", "00000"],
    "?": ["01110", "10001", "00001", "00110", "00100", "00000", "00100"],
}


def text_size(text, scale=2, spacing=1):
    return len(text) * (5 + spacing) * scale - spacing * scale, 7 * scale


def draw_text(img, x, y, text, scale=2, color=(40, 44, 52), spacing=1):
    """Draw uppercase text into an uint8 RGB image (in place)."""
    h, w, _ = img.shape
    col = np.asarray(color, dtype=np.uint8)
    cx = x
    for ch in text.upper():
        glyph = _FONT.get(ch, _FONT["?"])
        for row, bits in enumerate(glyph):
            for c, bit in enumerate(bits):
                if bit == "1":
                    y0 = y + row * scale
                    x0 = cx + c * scale
                    if 0 <= y0 < h - scale + 1 and 0 <= x0 < w - scale + 1:
                        img[y0:y0 + scale, x0:x0 + scale] = col
        cx += (5 + spacing) * scale
    return img


def montage(images, cols, pad=6, backdrop=255):
    """Tile uint8 images into a grid."""
    h, w, _ = images[0].shape
    rows = (len(images) + cols - 1) // cols
    out = np.full((rows * h + (rows + 1) * pad, cols * w + (cols + 1) * pad, 3),
                  backdrop, dtype=np.uint8)
    for k, im in enumerate(images):
        r, c = divmod(k, cols)
        y = pad + r * (h + pad)
        x = pad + c * (w + pad)
        out[y : y + h, x : x + w] = im
    return out
