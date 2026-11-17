"""一集的公共量：门七的 series（位姿 / 关节 / 法兰 / 夹爪 / 接触 / 速度）+ 指尖位置、接触查询、判官结果。"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

from .load import gate7

INF = float("inf")


def quat_R(q):
    q = np.asarray(q, float)
    n = np.linalg.norm(q)
    if not n or not np.isfinite(n):
        return np.full((3, 3), np.nan)
    w, x, y, z = q / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


class Ctx:
    def __init__(self, ep):
        self.G = gate7()
        self.ep = ep
        self.h = ep.get("header") or {}
        self.task = self.h.get("task")
        self.S = self.G.series(ep)
        self.n = self.S["n"]
        self.t = self.S["t"]
        d = np.diff(self.S["sim_t"]) if self.n > 1 else np.array([])
        d = d[np.isfinite(d) & (d > 0)]
        self.dt = float(np.median(d)) if len(d) else float(self.S["dt0"])
        self.rows = self.S["rows"]
        self._tips = None
        self._judge = "未算"
        self.has_contacts = self.S["contacts"] is not None
        # 接触列只在 rich 子集的集里有；有接触列的集里「某步没有这一对」才算没接触（缓冲只是最近一个物理子步，见 header.contact.substep）
        self.contact_note = ((self.h.get("contact") or {}).get("substep")) or ("没有接触列" if not self.has_contacts else "")

    # ---------------------------------------------------------------- 取列
    def col(self, k):
        return np.array([self.G._num((r.get("state") or {}).get(k)) for r in self.rows], float)

    def has(self, k):
        return any(k in (r.get("state") or {}) for r in self.rows[:3])

    def pose(self, pre):
        """pre = 'obj_duck' / 'art_drawer' / 'artlink_drawer_handle_xpos' → (T,7) x y z qw qx qy qz；没有 ⇒ None。"""
        if not self.has(f"{pre}_x"):
            return None
        return np.stack([self.col(f"{pre}_{a}") for a in ("x", "y", "z", "qw", "qx", "qy", "qz")], axis=1)

    # ---------------------------------------------------------------- 指尖
    def tips(self):
        """每步每只手的指尖点 (T, k, 3)：有手指连杆（link_<臂>_f1/f2）就用两指；否则法兰 ⊗ header.tcp.tip（和手指中点差 ≤ 5 mm）。
        src 说明用的是哪种。"""
        if self._tips is not None:
            return self._tips
        pts, src = [], None
        for hand in ("left", "right"):
            if self.has(f"link_{hand}_f1_x"):
                for f in ("f1", "f2"):
                    pts.append(np.stack([self.col(f"link_{hand}_{f}_{a}") for a in "xyz"], axis=1))
                src = "手指连杆 f1/f2"
            elif self.has(f"link_{hand}_eef_x"):
                P = self.pose(f"link_{hand}_eef")
                off = np.asarray(((self.h.get("tcp") or {}).get("tip")) or [0.0, -0.003, 0.141], float)
                tip = np.array([P[i, :3] + quat_R(P[i, 3:7]) @ off for i in range(self.n)])
                pts.append(tip)
                pts.append(P[:, :3])      # 法兰本身也算（前臂 / 腕碰的粗代理）
                src = src or "法兰 ⊗ tcp 指尖（代理）"
        self._tips = (np.stack(pts, axis=1) if pts else np.full((self.n, 0, 3), np.nan), src or "没有手的位姿")
        return self._tips

    def tip_dist(self, P):
        """每步两手指尖到点 P (T,3) 的最近距离；没有手 ⇒ inf。"""
        X, _ = self.tips()
        if X.shape[1] == 0:
            return np.full(self.n, INF)
        d = np.linalg.norm(X - P[:, None, :], axis=2)
        d = np.where(np.isfinite(d), d, INF)
        return d.min(axis=1)

    # ---------------------------------------------------------------- 接触
    def contacts_at(self, i):
        if not self.has_contacts:
            return None
        return self.S["contacts"][i] or []

    def robot_touch(self, uid, lo, hi, via_objects=True):
        """[lo, hi] 步内 uid（含 uid:连杆）有没有被机器人碰过；via_objects ⇒ 也算「同一步也碰着机器人的物体」碰 uid（手里拿着的东西去撞）。
        → (True/False/None, 第一步)。没有接触列 ⇒ None。"""
        if not self.has_contacts:
            return None, None
        lo, hi = max(0, int(lo)), min(self.n - 1, int(hi))
        for i in range(lo, hi + 1):
            prs = self.S["contacts"][i] or []
            held = set()
            for a, b, *_ in prs:
                if a.startswith("robot"):
                    held.add(b.split(":")[0])
                if b.startswith("robot"):
                    held.add(a.split(":")[0])
            for a, b, *_ in prs:
                ua, ub = a.split(":")[0], b.split(":")[0]
                if uid not in (ua, ub):
                    continue
                other = ub if ua == uid else ua
                if other == "robot" or (via_objects and other in held):
                    return True, i
        return False, None

    # ---------------------------------------------------------------- optional strict judge (external `judges` module, not part of this package)
    def judge(self):
        if self._judge != "未算":
            return self._judge
        self._judge = None
        try:
            from judges import SPECS, judge_rows   # noqa: E402  (optional; absent => CRIT detectors report unknown)
        except Exception:  # noqa: BLE001
            return None
        if self.task not in SPECS:
            return None
        rows, cur = [], {}
        for r in self.rows:
            cur.update(r.get("state") or {})
            rows.append({k: v for k, v in cur.items() if v is not None})
        sc = (((self.h.get("opening") or {}).get("readback") or {}).get("scales")) or {}
        try:
            self._judge = judge_rows(self.task, rows, dict(rigid=sc), float(self.h.get("rate_hz") or 25.0))
        except Exception as e:  # noqa: BLE001
            self._judge = dict(error=f"{type(e).__name__}: {e}"[:200])
        return self._judge

    def static_before(self, a, i, win_s=0.2, v=0.01):
        """位姿序列 a (T,7) 在第 i 步之前 win_s 内是不是静止（差分线速 < v）。"""
        w = max(1, int(round(win_s / self.dt)))
        lo = max(0, i - w)
        if i - lo < 1:
            return True
        sp = np.linalg.norm(np.diff(a[lo:i + 1, :3], axis=0), axis=1) / self.dt
        return bool(np.nanmax(sp) < v) if np.isfinite(sp).any() else False

    def steps(self, s):
        return max(1, int(math.ceil(s / self.dt - 1e-9)))
