"""item_assembly: step-wise FAQ criteria.

FAQ: "Pick up the two silicone tubes from the table. Connect the head of the right tube to the tail of the left tube. ...
Once inserted, the task ends."

Functional ends: ``guijiao1`` local +x end = functional tail; ``guijiao2`` local +x end = functional head.  (Both are called
``end_head`` in the parts table; that is only the name of the coordinate end.)

Thresholds are the two the simulator already uses (end distance < 20 mm, axis angle <= 15 deg) plus the 20-step hold; the
"lifted" height (20 mm) is the only number added.

| FAQ sentence      | check                                                                                                              | value  |
| both lifted       | both tubes got >= 20 mm above their start height at the same time, at some step                                    | 20 mm  |
| head joined tail  | on the insertion step (the step the attach / weld constraint was created) the functional head-to-tail distance     | 20 mm, |
|                   | is < 20 mm and the two tube axes make <= 15 deg                                                                    | 15 deg |
| inserted          | an insertion step happened and the two tubes were in contact on it                                                 |        |
| still joined      | on this step head and tail are still < 20 mm apart with axes <= 15 deg                                             |        |
| hold              | counted from the step after the insertion: every step still joined, >= 20 consecutive steps up to the last step    | 20     |
"""
import numpy as np

from . import common as C
from . import parts as P

TASK = "item_assembly"
UIDS = ("guijiao1", "guijiao2")
RIGID = ("guijiao1", "guijiao2")
ARTS = ()
ART_LINKS = {}
EEF = ()
NEEDS_CONTACTS = True
WELD_EVENT = True            # the env's own attach (weld) step is reported to the scene as the "weld" event
HOLD_STEPS = C.HOLD_STEPS
END_DIST_M = 0.020        # functional head - functional tail distance
AXIS_DEG = 15.0           # angle between the two tube axes
LIFT_M = 0.020            # "lifted": >= 20 mm above the start height (the FAQ gives no number)
FUNC_END_PART = {"guijiao1": "end_head", "guijiao2": "end_head"}   # which part of the parts table holds the functional tail / head of each tube
DEFINITION = ("both lifted = both tubes >= 20 mm above their start height at the same time; head joined tail = on the insertion (weld) step the functional head-to-tail distance is < 20 mm "
              "and the tube axes make <= 15 deg (functional ends: guijiao1 +x = tail, guijiao2 +x = head); inserted = a weld happened and the tubes were in contact on that step; "
              "hold = from the step after the insertion every step still joined, for >= 20 consecutive steps up to the last step")


def connect_state(T1, T2, e1_local, e2_local):
    """Is 'right-tube head joined to left-tube tail' true on this step?  T1 / T2 = guijiao1 / guijiao2 world poses (4x4);
    e*_local = each tube's functional end in its local frame (metres, body scale applied)."""
    T1, T2 = np.asarray(T1, float), np.asarray(T2, float)
    p1 = T1[:3, 3] + T1[:3, :3] @ np.asarray(e1_local, float)
    p2 = T2[:3, 3] + T2[:3, :3] @ np.asarray(e2_local, float)
    d = float(np.linalg.norm(p1 - p2))
    c = abs(float(T1[:3, 0] @ T2[:3, 0])) / (float(np.linalg.norm(T1[:3, 0])) * float(np.linalg.norm(T2[:3, 0])) or 1.0)
    ang = float(np.degrees(np.arccos(min(1.0, c))))
    return dict(ok=bool(d < END_DIST_M and ang <= AXIS_DEG), end_dist_mm=d * 1000.0, axis_deg=ang)


class Judge:
    def __init__(self, parts):
        parts = P.parts_of(parts)
        self.e1 = parts["guijiao1"][FUNC_END_PART["guijiao1"]]["p"]
        self.e2 = parts["guijiao2"][FUNC_END_PART["guijiao2"]]["p"]
        self.z0, self.lift, self.picked = None, 0.0, False
        self.t_weld, self.weld_ok, self.weld_touch, self.weld_m = None, False, None, None
        self.tail = C.Tail()
        self.i = -1

    def step(self, obs):
        """obs: T_guijiao1, T_guijiao2 (4x4), weld (the insertion / weld event happened on this step: True / False),
        touching (the two tubes are in contact on this step: True / False / None), t.  The first call is the start."""
        self.i += 1
        t = int(obs.get("t", self.i))
        T1, T2 = np.asarray(obs["T_guijiao1"], float), np.asarray(obs["T_guijiao2"], float)
        if self.z0 is None:
            self.z0 = (float(T1[2, 3]), float(T2[2, 3]))
        lift = min(float(T1[2, 3]) - self.z0[0], float(T2[2, 3]) - self.z0[1])
        self.lift = max(self.lift, lift)
        self.picked |= lift >= LIFT_M
        cs = connect_state(T1, T2, self.e1, self.e2)
        if obs.get("weld") and self.t_weld is None:
            self.t_weld, self.weld_ok, self.weld_touch, self.weld_m = t, bool(cs["ok"]), obs.get("touching"), cs
        ins = self.t_weld is not None
        conds = {"both_lifted": bool(self.picked), "head_joined_tail": bool(ins and self.weld_ok),
                 "inserted": (None if self.weld_touch is None else bool(self.weld_touch)) if ins else False, "still_joined": bool(cs["ok"])}
        done = C.tri(conds) is True
        n = self.tail.push(bool(done and t > self.t_weld) if ins else False, t)      # the hold is counted from the step after the insertion
        return dict(done=done, hold_n=n, conds=conds,
                    m=dict(max_joint_lift_mm=self.lift * 1e3, weld_step=self.t_weld, weld_end_dist_mm=None if self.weld_m is None else self.weld_m["end_dist_mm"],
                           weld_axis_deg=None if self.weld_m is None else self.weld_m["axis_deg"], end_dist_mm=cs["end_dist_mm"], axis_deg=cs["axis_deg"]))


def make_judge(ctx):
    return Judge(ctx.parts)


def obs_from_scene(scene, ctx=None):
    return dict(T_guijiao1=scene.obj_T("guijiao1"), T_guijiao2=scene.obj_T("guijiao2"), weld="weld" in scene.events,
                touching=scene.touching("guijiao1", "guijiao2"), t=scene.t)
