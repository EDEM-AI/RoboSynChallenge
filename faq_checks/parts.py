"""Functional-part tables (``parts``) for the FAQ criteria, and their scaling to one episode.

A *part* is a named geometric feature of an object in its own mesh-local frame: ``bottom``, ``mouth``, ``tip``, ``ring`` ...
Each is a small dict (``kind``: point / disk / axis / joint / ...; ``p``: position; ``normal`` / ``axis``; ``section``; ``size`` ...).
See ``parts_default.json`` for the data and its provenance.

The data file stores the parts of every task object **at body_scale = (1, 1, 1)** (metres, radians, mesh-local frame).
For one episode the object's actual body_scale ``s`` is applied by ``scale_parts``:

* points:            diag(s) p
* tangent axes:      normalize(diag(s) a)
* normals:           normalize(diag(1/s) n)
* lengths along an axis a:  length * |diag(s) a|
* sections:          rect ``w`` / ``d`` are scaled along their own axes; a circle becomes an ellipse when the two cross axes scale differently
* ``size.thickness`` / ``gap`` / ``depth``: along ``*_axis`` if given, else part-specific default, else ``up_axis``, ``normal``, ``axis``
* prismatic ``range`` / ``travel`` of a part with ``"scaled": false`` are NOT scaled (the engine's joint-limit reader ignores body_scale)

Only numpy and the standard library are needed.
"""
import json
import math
import os

import numpy as np

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parts_default.json")

_SCALAR_DEFAULT = {("handles", "gap"): "up", ("handles", "thickness"): "up",
                   ("plates", "thickness"): "up", ("holes", "depth"): "axis"}


def _r(x, n=6):
    if isinstance(x, (list, tuple)):
        return [_r(v, n) for v in x]
    if isinstance(x, np.ndarray):
        return [_r(v, n) for v in x.tolist()]
    if isinstance(x, (np.floating, float)):
        v = float(x)
        if not math.isfinite(v):
            return None
        v = round(v, n)
        return 0.0 if v == 0 else v
    if isinstance(x, (np.integer, int)) and not isinstance(x, bool):
        return int(x)
    return x


def _unit(v):
    v = np.asarray(v, float).reshape(3)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError("zero vector cannot be an axis")
    return v / n


def _perp_axes(axis):
    return tuple(i for i in range(3) if i != axis)


def _ax_vec(i):
    v = [0.0, 0.0, 0.0]
    v[int(i)] = 1.0
    return v


def _scalar_scale(p, k, s, up, part_name=None):
    """Factor for a directed scalar (thickness / gap / depth).  Returns (factor, source); (None, "missing") if it cannot be determined."""
    smax = float(np.max(s))
    uni = float(np.max(s) - np.min(s)) <= 1e-9 * max(smax, 1e-9)

    def along(vec):
        return float(np.linalg.norm(s * _unit(np.asarray(vec, float))))

    def up_val():
        ua = p.get("up_axis") or ("xyz"[up] if up is not None else None)
        if ua is None:
            return None
        try:
            return float(s["xyz".index(str(ua))])
        except (ValueError, IndexError):
            return None

    a = p.get(k + "_axis")
    if a is not None:
        return along(a), "explicit"
    dflt = _SCALAR_DEFAULT.get((str(part_name), k))
    if dflt == "up":
        v = up_val()
        if v is not None:
            return v, "default_up"
    if dflt == "axis" and p.get("axis") is not None:
        return along(p["axis"]), "default_axis"
    v = up_val()
    if v is not None:
        return v, "up_axis"
    for key in ("normal", "axis"):
        if p.get(key) is not None:
            return along(p[key]), key
    if uni:
        return smax, "uniform"
    return None, "missing"


def scale_parts(parts, s, up=None):
    """Apply body_scale ``s`` (3 numbers) to the unit-scale parts of one object.  ``up`` = index (0/1/2) of the object's up axis, or None."""
    s = np.asarray(s, float).reshape(3)
    out = {}

    def sp(p):
        return _r(np.asarray(p, float) * s)

    def saxis(a):
        return _r(_unit(np.asarray(a, float) * s))

    def snormal(n):
        return _r(_unit(np.asarray(n, float) / s))

    def sec(d, nrm, ax):
        d = dict(d or {})
        shape = d.get("shape")
        base = nrm if nrm is not None else ax
        o = _perp_axes(int(np.argmax(np.abs(np.asarray(base, float))))) if base is not None else (0, 1)
        if shape == "circle":
            sa, sb = float(s[o[0]]), float(s[o[1]])
            rin, rout = d.get("r_in"), d.get("r_out")
            if abs(sa - sb) <= 0.01 * max(sa, sb):
                d2 = dict(shape="circle")
                if rin is not None:
                    d2["r_in"] = _r(float(rin) * sa)
                if rout is not None:
                    d2["r_out"] = _r(float(rout) * sa)
                return d2
            d2 = dict(shape="ellipse")
            if rin is not None:
                d2["a_in"] = _r(float(rin) * sa)
                d2["b_in"] = _r(float(rin) * sb)
            if rout is not None:
                d2["a_out"] = _r(float(rout) * sa)
                d2["b_out"] = _r(float(rout) * sb)
            e = np.zeros(3)
            e[o[0]] = 1.0
            f = np.zeros(3)
            f[o[1]] = 1.0
            d2["axes"] = [_r(e), _r(f)]
            return d2
        if shape == "rect":
            d2 = dict(shape="rect", in_or_out=d.get("in_or_out"))
            axes = d.get("axes")
            if axes and len(axes) == 2:
                a0 = _unit(np.asarray(axes[0], float))
                a1 = _unit(np.asarray(axes[1], float))
                k0 = float(np.linalg.norm(s * a0))
                k1 = float(np.linalg.norm(s * a1))
                if d.get("w") is not None:
                    d2["w"] = _r(float(d["w"]) * k0)
                if d.get("d") is not None:
                    d2["d"] = _r(float(d["d"]) * k1)
                d2["axes"] = [_r(_unit(s * a0)), _r(_unit(s * a1))]
            else:
                if d.get("w") is not None:
                    d2["w"] = _r(float(d["w"]) * float(s[o[0]]))
                if d.get("d") is not None:
                    d2["d"] = _r(float(d["d"]) * float(s[o[1]]))
                d2["axes"] = [_ax_vec(o[0]), _ax_vec(o[1])]
            if d.get("r_in") is not None:
                if d2.get("w") is not None and d2.get("d") is not None:
                    d2["r_in"] = _r(0.5 * min(float(d2["w"]), float(d2["d"])))
                else:
                    d2["r_in"] = _r(float(d["r_in"]) * float(np.mean(s[[o[0], o[1]]])))
            return d2
        return d

    def one(p, label):
        p = dict(p or {})
        q = {"kind": p.get("kind")}
        ax = p.get("axis")
        nrm = p.get("normal")
        if p.get("p") is not None:
            q["p"] = sp(p["p"])
        if ax is not None:
            q["axis"] = saxis(ax)
        if nrm is not None:
            q["normal"] = snormal(nrm)
        if p.get("section") is not None:
            q["section"] = sec(p["section"], nrm, ax)
        size = dict(p.get("size") or {})
        if "length" in size:
            a = np.asarray(ax, float) if ax is not None else None
            size["length"] = _r(float(size["length"]) * (float(np.linalg.norm(s * a)) if a is not None else float(np.mean(s))))
        if "height" in size:
            ua = p.get("up_axis") or ("xyz"[up] if up is not None else "z")
            size["height"] = _r(float(size["height"]) * float(s["xyz".index(str(ua))]))
        for k in ("thickness", "gap", "depth"):
            if k in size and size[k] is not None:
                kk, _ = _scalar_scale(p, k, s, up, label)
                if kk is None:
                    size.pop(k, None)
                else:
                    size[k] = _r(float(size[k]) * kk)
        if size:
            q["size"] = size
        if p.get("span") is not None:
            k = float(np.linalg.norm(s * np.asarray(ax, float)))
            q["span"] = [_r(float(p["span"][0]) * k), _r(float(p["span"][1]) * k)]
        if p.get("range") is not None:
            rng = [float(p["range"][0]), float(p["range"][1])]
            if p.get("unit") != "rad" and ax is not None and p.get("scaled", True):
                k = float(np.linalg.norm(s * np.asarray(ax, float)))
                rng = [rng[0] * k, rng[1] * k]
            q["range"] = _r(rng)
            q["travel"] = _r(rng[1] - rng[0])
        if p.get("poly") is not None:
            q["poly"] = [sp(v) for v in p["poly"]]
        if p.get("aabb") is not None:
            q["aabb"] = [sp(v) for v in p["aabb"]]
        if p.get("items") is not None:
            items = []
            for it in p["items"]:
                it = dict(it)
                ii = {"id": it.get("id")}
                if it.get("p") is not None:
                    ii["p"] = sp(it["p"])
                if it.get("axis") is not None:
                    ii["axis"] = saxis(it["axis"])
                if it.get("normal") is not None:
                    ii["normal"] = snormal(it["normal"])
                if it.get("section") is not None:
                    ii["section"] = sec(it["section"], it.get("normal"), it.get("axis"))
                if it.get("size"):
                    sz = dict(it["size"])
                    for k in ("thickness", "gap", "depth"):
                        if sz.get(k) is not None:
                            kk, _ = _scalar_scale(it, k, s, up, label)
                            if kk is None:
                                sz.pop(k, None)
                            else:
                                sz[k] = _r(float(sz[k]) * kk)
                    if sz.get("height") is not None:
                        ua = it.get("up_axis") or ("xyz"[up] if up is not None else "z")
                        sz["height"] = _r(float(sz["height"]) * float(s["xyz".index(str(ua))]))
                    ii["size"] = sz
                for k in ("role", "kind", "thickness_axis", "gap_axis", "depth_axis", "up_axis"):
                    if it.get(k) is not None:
                        ii[k] = it[k]
                items.append(ii)
            q["items"] = items
        for k in ("range_src", "up_axis", "unit"):
            if k in p and p[k] is not None:
                q[k] = p[k]
        return q

    for name, p in parts.items():
        out[name] = one(p, name)
    return out


_TABLE = None


def load_table(path=None):
    """The unit-scale parts table: {"tasks": {task: {uid: {"up_axis": "z"|None, "parts": {...}}}}, ...}."""
    global _TABLE
    if path is None and _TABLE is not None:
        return _TABLE
    with open(path or DATA_FILE, "r", encoding="utf-8") as f:
        t = json.load(f)
    if path is None:
        _TABLE = t
    return t


def parts_for_episode(task, body_scale, table=None):
    """Parts of every object of ``task`` for one episode: {uid: {part: {...}}}, body_scale applied.

    ``body_scale`` = {uid: (sx, sy, sz) or a single number}; objects missing from it are taken as unit scale."""
    table = table or load_table()
    out = {}
    for uid, ent in table["tasks"][task].items():
        sc = body_scale.get(uid)
        sc = np.ones(3) if sc is None else np.broadcast_to(np.asarray(sc, float).reshape(-1), (3,))
        up = None if not ent.get("up_axis") else "xyz".index(ent["up_axis"])
        out[uid] = scale_parts(ent["parts"], sc, up)
    return out


def parts_of(v):
    """Accept a parts dict, or a dict wrapping it under ``"parts"``; returns the {uid: {...}} dict."""
    if isinstance(v, dict) and isinstance(v.get("parts"), dict):
        v = v["parts"]
    return v if isinstance(v, dict) else {}


class EpisodeContext:
    """Everything a task module needs to know about one episode besides the per-step observations.

    body_scale: {uid: (sx, sy, sz)}; parts: scaled parts table (computed from the default table if not given);
    table: {"top_z": ..., "xy_min": [x, y], "xy_max": [x, y]} or None (water_pouring); rate_hz: control-step frequency;
    limits: optional {"pipette_slide_lo": ...} read from the simulator."""

    def __init__(self, task, body_scale=None, parts=None, table=None, rate_hz=25.0, limits=None):
        self.task = task
        self.body_scale = dict(body_scale or {})
        self.parts = parts if parts is not None else parts_for_episode(task, self.body_scale)
        self.table = table
        self.rate_hz = float(rate_hz or 25.0)
        self.limits = dict(limits or {})
