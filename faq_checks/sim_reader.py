"""Reads one env step of a live EmbodiChain env into a ``FlatScene`` (see ``scene.py``).

Getters only (``get_local_pose``, ``get_qpos``, ``get_link_pose``, ``get_body_scale``, contact buffer reads); nothing here steps or
modifies the simulation, except one switch: contact reporting on the CPU physics scene is turned on once
(``enable_contact_data_update_on_cpu(True)``), because only two official tasks attach a ``ContactSensor`` and the criteria
need robot-vs-object contacts.  Contacts are read from the same buffer ``ContactSensor`` uses: it holds the contacts of the **last
physics sub-step** of the control step, not an aggregate of all sub-steps.  With GPU physics contacts are not available; the scene then
reports ``contacts=None`` and criteria that depend on contacts come out as ``None`` (not enough evidence) instead of guessing.

The code is written against EmbodiChain 0.2.4.post1 (``SimulationManager`` getters, ``RigidObject`` / ``Articulation`` / ``Robot`` classes) and has
been exercised offline against a mock of those classes only (tests/test_faq_checks_sim_reader.py).
"""
import math

import numpy as np

from .scene import FlatScene, POSE_AXES

EEF_LINKS = {"left": "left_link6", "right": "right_link6"}   # wrist flange of each arm; "eef" in scene keys


def _np(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _row0(x):
    a = _np(x)
    if a.ndim >= 2:
        a = a[0]
    return a.reshape(-1)


def _body_scale(obj, cfg_scale=None):
    """Actual body scale of a rigid object (getter) or articulation (getter if it has one, else the configured value)."""
    for getter in (lambda: obj.get_body_scale(), lambda: obj.entities[0].get_body_scale()):
        try:
            v = _row0(getter()).astype(float)
            if v.size >= 3 and np.all(np.isfinite(v)):
                return v[:3]
        except Exception:  # noqa: BLE001
            pass
    if cfg_scale is None:
        return np.ones(3)
    return np.broadcast_to(np.asarray(cfg_scale, float).reshape(-1), (3,)).copy()


class SimReader:
    """Per-env reader.  ``module`` is a task module (``<task>_faq``) that declares what to read: RIGID, ARTS, ART_LINKS, EEF, WELD_EVENT."""

    def __init__(self, env, module, gym_config=None):
        self.env = getattr(env, "unwrapped", env)
        self.m = module
        self.cfg = gym_config or {}
        self.sim = getattr(self.env, "sim", None)
        self.robot = getattr(self.env, "robot", None)
        self.ps = None                 # CPU physics scene (for contacts)
        self.contacts_ok = False
        self.ids = {}                  # user_id -> name
        self._weld_prev = False
        self.art_uids = None           # articulations present in the scene (None = unknown)
        self.errors = {}
        self._enable_contacts()

    # ---------------------------------------------------------------- setup
    def _enable_contacts(self):
        if not getattr(self.m, "NEEDS_CONTACTS", False):
            return
        try:
            import dexsim
            if getattr(dexsim.get_world_config(), "enable_gpu_sim", False):
                self.errors["contacts"] = "GPU physics: contact buffer not read"
                return
            self.ps = dexsim.default_world().get_physics_scene()
            self.ps.enable_contact_data_update_on_cpu(True)
            self.contacts_ok = True
        except Exception as e:  # noqa: BLE001
            self.errors["contacts"] = "%s: %s" % (type(e).__name__, e)

    def begin_episode(self):
        """Call after every reset: rebuilds the user-id -> name map and clears event memory."""
        self.ids = {}
        self._weld_prev = bool(self._weld_flag())
        sim = self.sim
        try:
            self.art_uids = set(sim.get_articulation_uid_list())
        except Exception:  # noqa: BLE001
            self.art_uids = None
        if sim is None or not self.contacts_ok:
            return
        try:
            for uid in list(sim.get_rigid_object_uid_list()):
                for i in _np(sim.get_rigid_object(uid).get_user_ids()).reshape(-1):
                    self.ids[int(i)] = uid
        except Exception as e:  # noqa: BLE001
            self.errors["ids_rigid"] = "%s: %s" % (type(e).__name__, e)
        arts = []
        try:
            arts = [(u, sim.get_articulation(u)) for u in list(sim.get_articulation_uid_list())]
        except Exception as e:  # noqa: BLE001
            self.errors["ids_art_list"] = "%s: %s" % (type(e).__name__, e)
        if self.robot is not None:
            arts.append(("robot", self.robot))
        for name, art in arts:
            for ln in list(getattr(art, "link_names", []) or []):
                try:
                    for i in _np(art.get_user_ids(ln)).reshape(-1):
                        self.ids[int(i)] = "%s:%s" % (name, ln)
                except Exception as e:  # noqa: BLE001
                    self.errors["ids_link"] = "%s: %s" % (type(e).__name__, e)

    def body_scales(self):
        """{uid: (sx, sy, sz)} of the task's objects, read from the live scene."""
        out = {}
        cfg_scale = {}
        for k in ("rigid_object", "articulation"):
            for r in self.cfg.get(k, []):
                if r.get("body_scale") is not None:
                    cfg_scale[r["uid"]] = r["body_scale"]
        for uid in tuple(self.m.RIGID) + tuple(self._arts()):
            try:
                obj = self.sim.get_rigid_object(uid) if uid in self.m.RIGID else self.sim.get_articulation(uid)
            except Exception:  # noqa: BLE001
                obj = None
            if obj is not None:
                out[uid] = _body_scale(obj, cfg_scale.get(uid))
        return out

    def _arts(self):
        """The task's articulations that exist in this scene (an optional one such as mixer_switch may be absent)."""
        if self.art_uids is None:
            return tuple(self.m.ARTS)
        return tuple(u for u in self.m.ARTS if u in self.art_uids)

    def table_extent(self):
        """Table top height and horizontal extent, from the ``table`` entry of the gym config's background list."""
        for b in self.cfg.get("background", []):
            if b.get("uid") == "table":
                size = np.asarray(b["shape"]["size"], float) * np.broadcast_to(np.asarray(b.get("body_scale", [1, 1, 1]), float), (3,))
                pos = np.asarray(b["init_pos"], float)
                return dict(top_z=float(pos[2] + size[2] / 2), xy_min=[float(v) for v in pos[:2] - size[:2] / 2], xy_max=[float(v) for v in pos[:2] + size[:2] / 2])
        return None

    def rate_hz(self):
        f = getattr(self.env, "control_freq", None)
        return float(f) if f else 25.0

    def limits(self):
        out = {}
        if "pipette" in getattr(self.m, "ARTS", ()):
            try:
                pip = self.sim.get_articulation("pipette")
                lim = getattr(pip, "qpos_limits", None)
                if lim is None and hasattr(pip, "body_data"):
                    lim = pip.body_data.qpos_limits
                if lim is not None:
                    out["pipette_slide_lo"] = float(_np(lim)[0, 0, 0])
            except Exception as e:  # noqa: BLE001
                self.errors["limits"] = "%s: %s" % (type(e).__name__, e)
        return out

    # ---------------------------------------------------------------- per step
    def _weld_flag(self):
        v = getattr(self.env, "_guijiao_attached", None)
        if v is None:
            return False
        try:
            return bool(_row0(v)[0])
        except Exception:  # noqa: BLE001
            return False

    def read(self, t):
        """One step -> FlatScene."""
        out = {}
        sim = self.sim

        def put_pose(prefix, obj):
            pose = _row0(obj.get_local_pose(to_matrix=False))
            for a, v in zip(POSE_AXES, pose):
                out["%s_%s" % (prefix, a)] = float(v)

        for uid in self.m.RIGID:
            try:
                put_pose("obj_" + uid, sim.get_rigid_object(uid))
            except Exception as e:  # noqa: BLE001
                self.errors["rigid." + uid] = "%s: %s" % (type(e).__name__, e)
        for uid in self._arts():
            try:
                art = sim.get_articulation(uid)
                if art is None:
                    continue
                put_pose("art_" + uid, art)
                q = _row0(art.get_qpos())
                for j in range(min(q.size, 8)):
                    out["art_%s_qpos_%d" % (uid, j)] = float(q[j])
                for ln in self.m.ART_LINKS.get(uid, ()):
                    lp = _row0(art.get_link_pose(ln, to_matrix=False))
                    for a, v in zip(POSE_AXES, lp):
                        out["artlink_%s_%s_%s" % (uid, ln, a)] = float(v)
            except Exception as e:  # noqa: BLE001
                self.errors["art." + uid] = "%s: %s" % (type(e).__name__, e)
        for side in self.m.EEF:
            try:
                pose = _row0(self.robot.get_link_pose(EEF_LINKS[side], to_matrix=False))
                for a, v in zip(POSE_AXES, pose):
                    out["link_%s_eef_%s" % (side, a)] = float(v)
            except Exception as e:  # noqa: BLE001
                self.errors["eef." + side] = "%s: %s" % (type(e).__name__, e)
        events = set()
        if getattr(self.m, "WELD_EVENT", False):
            now = self._weld_flag()
            if now and not self._weld_prev:
                events.add("weld")
            self._weld_prev = now
        return FlatScene(out, self._read_contacts(), events, t)

    def _read_contacts(self):
        """Contact rows [name_a, name_b, n_points, impulse_sum, min_distance], one per unordered pair; None if unavailable."""
        if not self.contacts_ok:
            return None
        try:
            data, uids = self.ps.get_cpu_contact_buffer()
            data, uids = _np(data), _np(uids)
        except Exception as e:  # noqa: BLE001
            self.errors["contact_read"] = "%s: %s" % (type(e).__name__, e)
            return None
        acc = {}
        for k in range(int(len(data))):
            a = self.ids.get(int(uids[k][0]), "?%d" % int(uids[k][0]))
            b = self.ids.get(int(uids[k][1]), "?%d" % int(uids[k][1]))
            key = (a, b) if a <= b else (b, a)
            imp = float(data[k][9]) if len(data[k]) > 9 else 0.0
            dist = float(data[k][10]) if len(data[k]) > 10 else math.nan
            c = acc.setdefault(key, [0, 0.0, math.inf])
            c[0] += 1
            c[1] += imp if math.isfinite(imp) else 0.0
            if math.isfinite(dist):
                c[2] = min(c[2], dist)
        return [[a, b, c[0], c[1], c[2] if math.isfinite(c[2]) else None] for (a, b), c in acc.items()]
