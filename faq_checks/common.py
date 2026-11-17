"""Shared helpers for the optional FAQ-aligned success criteria.

Every task module in this package (``<task>_faq.py``) follows the same pattern:

    judge = Judge(parts, ...)          # per-episode geometry comes in once
    out = judge.step(obs)              # call once per env step
    out["done"]  - every FAQ sentence is satisfied on this step
    out["conds"] - one entry per FAQ sentence: True / False / None (None = not enough evidence)
    out["hold_n"] - number of consecutive ``done`` steps ending at this step

An episode counts as FAQ-successful when ``done`` holds on the last step and has held for at least
``HOLD_STEPS`` consecutive steps.

``obs`` is a plain dict of numpy values.  Task modules also provide ``obs_from_scene(scene)`` that
builds it from a ``Scene`` (see ``scene.py``), so the same code path serves the live simulator
and offline replays of recorded states.

Only numpy and the standard library are required here.
"""
import numpy as np

HOLD_STEPS = 20           # the task must stay completed for about 20 steps at the end
UP = np.array([0.0, 0.0, 1.0])


def T_from_pos_quat(pos, q_wxyz):
    """4x4 pose from a position and a (w, x, y, z) quaternion."""
    w, x, y, z = [float(v) for v in q_wxyz]
    n = (w * w + x * x + y * y + z * z) ** 0.5 or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    T = np.eye(4)
    T[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                 [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                 [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    T[:3, 3] = [float(v) for v in pos]
    return T


def unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n <= 1e-12:
        raise ValueError("invalid axis")
    return v / n


def world_point(T, p_local):
    T = np.asarray(T, float)
    return T[:3, 3] + T[:3, :3] @ np.asarray(p_local, float)


def world_dir(T, v_local):
    return np.asarray(T, float)[:3, :3] @ unit(v_local)


def tilt_deg(axis_world):
    """Angle between an axis and world-up in degrees: 0 = pointing up, 180 = pointing down."""
    return float(np.degrees(np.arccos(np.clip(unit(axis_world)[2], -1.0, 1.0))))


def angle_between_deg(a, b):
    return float(np.degrees(np.arccos(np.clip(float(unit(a) @ unit(b)), -1.0, 1.0))))


def disk_lowest_z(T, disk):
    """Lowest world height of a disk part (``p``, ``normal``, ``section.r_out``): centre height minus radius * sin(tilt)."""
    c = world_point(T, disk["p"])
    n = world_dir(T, disk["normal"])
    r = float(disk.get("section", {}).get("r_out") or disk.get("section", {}).get("r_in") or 0.0)
    return float(c[2] - r * (max(0.0, 1.0 - float(n[2]) ** 2)) ** 0.5)


class Tail:
    """Counts consecutive true steps.  A gap in step numbers resets the count."""

    def __init__(self):
        self.n, self.t = 0, None

    def push(self, ok, t=None):
        if t is not None and self.t is not None and t != self.t + 1:
            self.n = 0
        self.t = t if t is not None else self.t
        self.n = self.n + 1 if ok else 0
        return self.n


def tri(conds):
    """Three-valued conjunction: any False -> False; else any None -> None; else True."""
    v = list(conds.values())
    return False if any(x is False for x in v) else None if any(x is None for x in v) else True


def summarize(task, conds_last, tail_n, n_steps, measured, definition):
    """One result row for an episode: last-step conditions, hold bookkeeping and measured values."""
    last = tri(conds_last)
    return dict(task=task, faq=conds_last, faq_success=bool(tail_n >= HOLD_STEPS and last is True),
                hold=dict(tail=bool(tail_n >= HOLD_STEPS), last=bool(last is True), steps=int(tail_n)),
                n_steps=int(n_steps), measured=measured, definition=definition)
