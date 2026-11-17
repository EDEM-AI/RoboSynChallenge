"""items_handover: step-wise FAQ criteria.

FAQ: "Pick up the marker on the right. Place it stably into the pen holder. The pen holder must not tip over."

The simulator's built-in check (``items_handover.py``): holder upright (axis < 75 deg from vertical) and pen upright-ish (long axis < 75 deg
from vertical) and pen origin within 3 cm (horizontally) of the holder origin and pen/holder bounding boxes overlapping in height by > 8 cm.
The semantics are kept, with the fixed 3 cm / 8 cm replaced by this episode's holder geometry:

| FAQ sentence   | check                                                                                                              | value       |
| pen lifted     | the pen origin got >= 5 cm above its start height at some step                                                     | 5 cm        |
| pen in holder  | pen origin within the holder mouth's inner radius of the holder axis; the pen's lower end below the mouth plane;   | 75 deg      |
|                | pen long axis < 75 deg from vertical                                                                               |             |
| pen stable     | no robot link touches the pen; compared with the previous step, pen origin moved <= 0.1 mm and its long axis      | 0.1 mm,     |
|                | turned <= 0.1 deg                                                                                                  | 0.1 deg     |
| holder upright | holder axis < 75 deg from vertical                                                                                 | 75 deg      |
| hold           | all of the above for >= 20 consecutive steps up to the last step                                                   | 20          |
"""
import numpy as np

from . import common as C
from . import parts as P

TASK = "items_handover"
UIDS = ("pen", "holder")
RIGID = ("pen", "holder")
ARTS = ()
ART_LINKS = {}
EEF = ()
NEEDS_CONTACTS = True
HOLD_STEPS = C.HOLD_STEPS
LIFT_M = 0.05
FALL_DEG = 75.0
STILL_M = 0.0001
STILL_DEG = 0.1
DEFINITION = ("pen lifted = pen origin >= 5 cm above its start; pen in holder = pen origin within the holder mouth's inner radius of the holder axis, pen's lower end below the mouth plane, "
              "pen long axis < 75 deg from vertical (the simulator's semantics, with 3 cm / 8 cm replaced by this episode's geometry); pen stable = no robot contact with the pen, "
              "per-step displacement <= 0.1 mm and long-axis turn <= 0.1 deg; holder upright = holder axis < 75 deg from vertical; hold = all of them for >= 20 consecutive steps up to the last step")


class Judge:
    def __init__(self, parts):
        parts = P.parts_of(parts)
        h, p = parts["holder"], parts["pen"]
        self.mouth, self.mouth_n = np.asarray(h["mouth"]["p"], float), C.unit(h["mouth"]["normal"])
        self.r_in = float(h["mouth"]["section"].get("r_in") or h["mouth"]["section"]["r_out"])
        self.ends = [np.asarray(p["tip"]["p"], float), np.asarray(p["tail"]["p"], float)]
        self.z0, self.lift, self.prev = None, 0.0, None
        self.tail = C.Tail()

    def step(self, obs):
        """obs: T_pen, T_holder (4x4), free_pen (no robot link touches the pen: True / False / None), t.  The first call is the start."""
        Tp, Th = np.asarray(obs["T_pen"], float), np.asarray(obs["T_holder"], float)
        if self.z0 is None:
            self.z0 = float(Tp[2, 3])
        self.lift = max(self.lift, float(Tp[2, 3]) - self.z0)
        up = C.world_dir(Th, self.mouth_n)
        m = C.world_point(Th, self.mouth)
        d = Tp[:3, 3] - m
        rad = float(np.linalg.norm(d - up * float(d @ up)))
        e = [C.world_point(Tp, x) for x in self.ends]
        low = min(float((x - m) @ up) for x in e)
        pen_tilt = min(C.tilt_deg(e[0] - e[1]), C.tilt_deg(e[1] - e[0]))
        holder_tilt = C.tilt_deg(up)
        inside = bool(rad <= self.r_in and low < 0.0 and pen_tilt < FALL_DEG)
        if self.prev is None:
            dm = dd = float("nan")
            still = False
        else:
            dm = float(np.linalg.norm(Tp[:3, 3] - self.prev[:3, 3]))
            dd = C.angle_between_deg(Tp[:3, :3] @ (self.ends[0] - self.ends[1]), self.prev[:3, :3] @ (self.ends[0] - self.ends[1]))
            still = dm <= STILL_M and dd <= STILL_DEG
        self.prev = Tp
        free = obs.get("free_pen")
        conds = {"pen_lifted": bool(self.lift >= LIFT_M), "pen_in_holder": inside,
                 "pen_stable": (None if free is None and still else bool(still and free)), "holder_upright": bool(holder_tilt < FALL_DEG)}
        done = C.tri(conds) is True
        n = self.tail.push(done, obs.get("t"))
        return dict(done=done, hold_n=n, conds=conds,
                    m=dict(pen_max_lift_mm=self.lift * 1e3, pen_to_holder_axis_mm=rad * 1e3, holder_inner_radius_mm=self.r_in * 1e3, pen_low_end_vs_mouth_mm=low * 1e3,
                           pen_tilt_deg=pen_tilt, holder_tilt_deg=holder_tilt, step_move_mm=dm * 1e3, step_turn_deg=dd, hand_off_pen=free))


def make_judge(ctx):
    return Judge(ctx.parts)


def obs_from_scene(scene, ctx=None):
    tch = scene.robot_touching("pen")
    return dict(T_pen=scene.obj_T("pen"), T_holder=scene.obj_T("holder"), free_pen=None if tch is None else not tch, t=scene.t)
