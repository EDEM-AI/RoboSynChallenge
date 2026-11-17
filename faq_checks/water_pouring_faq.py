"""water_pouring: step-wise FAQ criteria.

FAQ: "Pick up the cup and bottle. Perform a pouring motion from the bottle into the cup. Place both the bottle and cup stably on the table."

The simulator's built-in check (``water_pouring.py``, ``_evaluate_task_state``) is reused as far as possible, with its hard-coded geometry
replaced by this episode's; what the FAQ asks and the simulator does not check (cup picked up, both put back on the table) is defined here.

| FAQ sentence            | check                                                                                                              | value             |
| bottle lifted           | simulator "held": bottle origin < 0.30 m from the right wrist and > 3 cm above its start height (latched)          | 0.30 m, 3 cm      |
| cup lifted              | cup origin > 1 cm above its start height at some step                                                              | 1 cm              |
| poured                  | simulator "pouring_now", at some step: held and bottle axis 45-120 deg from vertical and bottle mouth legally above|                   |
|                         | the cup and cup not fallen (cup axis < 45 deg from vertical)                                                       |                   |
|                         | "legally above": the bottle mouth (parts: bottle.mouth; the simulator uses origin + 0.236 x bottle y) to the cup   |                   |
|                         | mouth centre (the simulator uses the cup origin) is < 8 cm horizontally; the bottle mouth is above the cup-mouth    | 8 cm, 0-26 cm,    |
|                         | plane by less than 26 cm (simulator: 4-30 cm above the cup origin of the default cup, whose mouth is 4.4 cm above  | -0.02             |
|                         | its origin); the horizontal part of the bottle axis points towards the cup (dot product > -0.02)                    |                   |
| bottle / cup set down   | on every step after the pour (each object separately): on the table AND upright AND still AND hand off             |                   |
|                         | on the table: lowest point of its bottom disk is <= its height at the start + 2 mm and the bottom centre is inside  | 2 mm              |
|                         | the table's extent                                                                                                 |                   |
|                         | upright: axis tilt <= its tilt at the start + 5 deg (a paper cup resting by itself is already about 6 deg off)     | 5 deg             |
|                         | still: compared with the previous step, origin moved <= 0.1 mm and axis turned <= 0.1 deg                          | 0.1 mm, 0.1 deg   |
|                         | hand off: no robot link touches it                                                                                 |                   |
| hold                    | all of the above (picked both, poured, and both set down on this step) for >= 20 consecutive steps to the last step | 20                |
"""
import numpy as np

from . import common as C
from . import parts as P

TASK = "water_pouring"
UIDS = ("bottle", "cup")
RIGID = ("bottle", "cup")
ARTS = ()
ART_LINKS = {}
EEF = ("right",)
NEEDS_CONTACTS = True
HOLD_STEPS = C.HOLD_STEPS
NEAR_EEF_M = 0.30         # simulator: bottle origin to right wrist
LIFT_BOTTLE_M = 0.03      # simulator: bottle above its start
LIFT_CUP_M = 0.01
POUR_DEG = (45.0, 120.0)  # simulator: bottle axis from vertical
MOUTH_XY_M = 0.08         # simulator
MOUTH_DZ_M = (0.0, 0.26)  # simulator: 4-30 cm relative to the cup origin, here relative to the cup-mouth plane
AXIS_DOT = -0.02          # simulator
FALL_DEG = 45.0           # simulator: cup fallen
ON_TABLE_M = 0.002
UPRIGHT_DEG = 5.0
STILL_M = 0.0001
STILL_DEG = 0.1

DEFINITION = ("bottle lifted = simulator's 'held' (bottle origin < 0.30 m from the right wrist and > 3 cm above its start; latched); cup lifted = > 1 cm above its start; "
              "poured = the simulator's 'pouring_now' with bottle and cup mouths taken from this episode's parts; bottle / cup set down = after the pour, each on the table (lowest point of its "
              "bottom <= start + 2 mm), upright (<= start tilt + 5 deg), still (per step <= 0.1 mm, 0.1 deg) and hand off (no robot contact); hold = all of them for >= 20 consecutive steps up to the last step")


class Judge:
    def __init__(self, parts, table_z, table_xy=None):
        """parts = episode parts table (bottle.mouth / bottle.bottom / cup.mouth / cup.bottom); table_z = table top height;
        table_xy = (xy_min, xy_max); if not given the table extent is not checked."""
        parts = P.parts_of(parts)
        self.p = {u: dict(mouth=parts[u]["mouth"], bottom=parts[u]["bottom"]) for u in UIDS}
        self.table_z = float(table_z)
        self.table_xy = None if table_xy is None else (np.asarray(table_xy[0], float), np.asarray(table_xy[1], float))
        self.open = None                      # state at the start: origin height, bottom height over the table, tilt, for each object
        self.prev = None
        self.picked = dict(bottle=False, cup=False)
        self.poured = False
        self.tail = C.Tail()
        self.max = dict(bottle_lift_m=0.0, cup_lift_m=0.0, pour_steps=0)

    # axis = normal of the mouth part (bottle: local y, cup: local z, the axes the simulator's check uses; the paper cup's mouth and bottom centres are not on one vertical line)
    def _axis(self, T, u):
        return C.world_dir(T, self.p[u]["mouth"]["normal"])

    def _geo(self, T, u):
        return dict(z=float(np.asarray(T, float)[2, 3]), gap=C.disk_lowest_z(T, self.p[u]["bottom"]) - self.table_z,
                    tilt=C.tilt_deg(self._axis(T, u)))

    def pour_now(self, Tb, Tc, eef_right, held):
        ax = self._axis(Tb, "bottle")
        ang = C.tilt_deg(ax)
        mouth = C.world_point(Tb, self.p["bottle"]["mouth"]["p"])
        cm = C.world_point(Tc, self.p["cup"]["mouth"]["p"])
        m2c = cm - mouth
        dxy = float(np.linalg.norm(m2c[:2]))
        dz = float(mouth[2] - cm[2])
        dot = float(ax[:2] @ m2c[:2])
        cup_tilt = C.tilt_deg(self._axis(Tc, "cup"))
        ok = bool(held and POUR_DEG[0] < ang < POUR_DEG[1] and dxy < MOUTH_XY_M and MOUTH_DZ_M[0] < dz < MOUTH_DZ_M[1]
                  and dot > AXIS_DOT and cup_tilt < FALL_DEG)
        return ok, dict(bottle_deg=ang, mouth_xy_mm=dxy * 1000, mouth_dz_mm=dz * 1000, axis_dot=dot, cup_deg=cup_tilt)

    def step(self, obs):
        """obs: T_bottle, T_cup (4x4), eef_right (right wrist position), free_bottle / free_cup (hand off this object: True / False / None), t (optional).
        The first call is taken as the start state (before the robot touches anything)."""
        T = dict(bottle=np.asarray(obs["T_bottle"], float), cup=np.asarray(obs["T_cup"], float))
        g = {u: self._geo(T[u], u) for u in UIDS}
        if self.open is None:
            self.open = g
        o = self.open
        lift = {u: g[u]["z"] - o[u]["z"] for u in UIDS}
        near = float(np.linalg.norm(T["bottle"][:3, 3] - np.asarray(obs["eef_right"], float))) < NEAR_EEF_M
        held = bool(near and (lift["bottle"] > LIFT_BOTTLE_M or self.picked["bottle"]))
        self.picked["bottle"] |= held
        self.picked["cup"] |= lift["cup"] > LIFT_CUP_M
        now, pm = self.pour_now(T["bottle"], T["cup"], obs["eef_right"], held)
        self.poured |= now
        self.max["pour_steps"] += int(now)
        for u in UIDS:
            self.max[u + "_lift_m"] = max(self.max[u + "_lift_m"], lift[u])
        placed, pl = {}, {}
        for u in UIDS:
            on = g[u]["gap"] <= o[u]["gap"] + ON_TABLE_M
            if self.table_xy is not None:
                c = C.world_point(T[u], self.p[u]["bottom"]["p"])[:2]
                on = on and bool(np.all(c >= self.table_xy[0]) and np.all(c <= self.table_xy[1]))
            up = g[u]["tilt"] <= o[u]["tilt"] + UPRIGHT_DEG
            if self.prev is None:
                dm, dd = float("nan"), float("nan")
                still = False
            else:
                dm = float(np.linalg.norm(T[u][:3, 3] - self.prev[u][:3, 3]))
                dd = C.angle_between_deg(self._axis(T[u], u), self._axis(self.prev[u], u))
                still = dm <= STILL_M and dd <= STILL_DEG
            free = obs.get("free_" + u)
            ok = None if free is None and on and up and still else bool(on and up and still and free)
            placed[u] = ok
            pl[u] = dict(on_table=bool(on), upright=bool(up), still=bool(still), hand_free=free,
                         gap_over_open_mm=(g[u]["gap"] - o[u]["gap"]) * 1000, tilt_over_open_deg=g[u]["tilt"] - o[u]["tilt"],
                         step_move_mm=dm * 1000, step_turn_deg=dd)
        self.prev = T
        after = self.poured
        conds = {"bottle_lifted": bool(self.picked["bottle"]), "cup_lifted": bool(self.picked["cup"]), "poured": bool(self.poured),
                 "bottle_set_down": (placed["bottle"] if after else False), "cup_set_down": (placed["cup"] if after else False)}
        done = C.tri(conds) is True
        n = self.tail.push(done, obs.get("t"))
        return dict(done=done, hold_n=n, conds=conds, pour_now=now, pour=pm, placed=pl, m=dict(bottle_max_lift_mm=self.max["bottle_lift_m"] * 1e3,
                    cup_max_lift_mm=self.max["cup_lift_m"] * 1e3, pour_steps=self.max["pour_steps"]))


def make_judge(ctx):
    tb = ctx.table or {}
    return Judge(ctx.parts, tb["top_z"], (tb["xy_min"], tb["xy_max"]) if "xy_min" in tb else None)


def obs_from_scene(scene, ctx=None):
    fb, fc = scene.robot_touching("bottle"), scene.robot_touching("cup")
    return dict(T_bottle=scene.obj_T("bottle"), T_cup=scene.obj_T("cup"), eef_right=scene.eef_pos("right"),
                free_bottle=None if fb is None else not fb, free_cup=None if fc is None else not fc, t=scene.t)
