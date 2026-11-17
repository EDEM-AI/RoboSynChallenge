"""sample_loading: step-wise FAQ criteria.

FAQ: (1) pick up the test tube on the right, (2) insert it stably into the test-tube rack, (3) leaving it on the rack without it
falling is enough.

Four conditions are required together: the tube bottom is horizontally inside a hole of this episode's rack, the tube bottom is
<= 2 mm above the rack's floor plate, the tube is no more tilted than the official vertical threshold, and after letting go the tube
stays so for 20 steps and is still, upright and not fallen on the last step.

| FAQ sentence     | check                                                                                                              | value          |
| tube lifted      | the tube origin got >= 2 cm above its start height at some step                                                   | 2 cm           |
| tube in hole     | for some hole of this episode's rack: tube bottom below the hole-mouth plane, the bottom-to-mouth segment passes    |                |
|                  | through the hole-mouth section, the bottom's own horizontal position is inside the section, tube pointing up       |                |
| tube seated      | tube bottom <= 2 mm above the rack's floor plate (= the hole mouth minus one hole depth)                           | 2 mm           |
| tube vertical    | tube axis (bottom -> mouth) < 0.087266 rad (about 5 deg) from vertical                                             | 0.087266 rad   |
| hand off         | no robot link touches the tube                                                                                     |                |
| tube still       | tube origin speed < 0.05 m/s (step displacement / step duration; the simulator's VEL_THRESH; the simulator also    | 0.05 m/s       |
|                  | asks for 8 consecutive steps, which the 20-step hold covers)                                                       |                |
| hold             | all of the above (including "tube still") for >= 20 consecutive steps up to the last step                          | 20             |

The rack floor is taken from the hole depth in the parts table, so that table must hold the real hole depth.
"""
import numpy as np

from . import common as C
from . import parts as P

TASK = "sample_loading"
UIDS = ("cube", "rack")
RIGID = ("cube", "rack")
ARTS = ()
ART_LINKS = {}
EEF = ()
NEEDS_CONTACTS = True
HOLD_STEPS = C.HOLD_STEPS
LIFT_M = 0.02
BOTTOM_GAP_M = 0.002
VERTICAL_RAD = 0.087266   # official threshold
STILL_MPS = 0.05          # official VEL_THRESH
DEFINITION = ("tube lifted = tube origin >= 2 cm above its start; tube in hole = bottom below a hole-mouth plane, bottom-to-mouth segment crosses the hole-mouth section, "
              "bottom horizontally inside the section, tube pointing up; tube seated = bottom <= 2 mm above the rack floor plate (hole mouth minus one hole depth); "
              "tube vertical = axis < 0.087266 rad from vertical (the official threshold); hand off = no robot link touches the tube; "
              "tube still = speed < 0.05 m/s (the official threshold); hold = all of them for >= 20 consecutive steps up to the last step")


def _basis(h):
    n = C.unit(h["axis"])
    b = h["section"].get("axes")
    if b is None:
        b = np.eye(3)[np.argsort(np.abs(n))[:2]]
    return n, [C.unit(x) for x in np.asarray(b, float)]


def _in_section(sec, a, b):
    if sec.get("shape") == "rect":
        return abs(a) < float(sec["w"]) / 2 and abs(b) < float(sec["d"]) / 2
    if sec.get("shape") == "circle":
        return float(np.hypot(a, b)) < float(sec["r_in"])
    raise ValueError("unsupported hole section shape")


class Judge:
    def __init__(self, parts, rate_hz=25.0):
        """parts = episode parts table; rate_hz = control-step frequency."""
        self.dt = 1.0 / float(rate_hz or 25.0)
        parts = P.parts_of(parts)
        self.bottom, self.mouth = np.asarray(parts["cube"]["bottom"]["p"], float), np.asarray(parts["cube"]["mouth"]["p"], float)
        self.holes = parts["rack"]["holes"]["items"]
        if not self.holes:
            raise ValueError("rack.holes.items is empty")
        self.z0, self.lift, self.prev = None, 0.0, None
        self.tail = C.Tail()

    def step(self, obs):
        """obs: T_cube (the tube), T_rack (4x4), free_cube (no robot link touches the tube: True / False / None), t.  The first call is the start."""
        Tc, Tr = np.asarray(obs["T_cube"], float), np.asarray(obs["T_rack"], float)
        if self.z0 is None:
            self.z0 = float(Tc[2, 3])
        self.lift = max(self.lift, float(Tc[2, 3]) - self.z0)
        bot, mou = C.world_point(Tc, self.bottom), C.world_point(Tc, self.mouth)
        axis = mou - bot
        best = None
        for i, h in enumerate(self.holes):
            n_l, basis = _basis(h)
            center = C.world_point(Tr, h["p"])
            n = C.world_dir(Tr, n_l)
            if n[2] < 0:
                n = -n
            depth = float((center - bot) @ n)
            ad = float(axis @ n)
            k = depth / ad if abs(ad) > 1e-12 else float("nan")
            cross = bot + k * axis
            w = [C.world_dir(Tr, v) for v in basis]
            ok = bool(depth > 0 and 0 < k < 1 and ad > 0
                      and _in_section(h["section"], float((cross - center) @ w[0]), float((cross - center) @ w[1]))
                      and _in_section(h["section"], float((bot - center) @ w[0]), float((bot - center) @ w[1])))
            hd = (h.get("size") or {}).get("depth")
            gap = None if hd is None else float(hd) - depth
            cand = dict(ok=ok, hole=h.get("id", i), depth=depth, gap=gap)
            if best is None or (ok and not best["ok"]) or (ok == best["ok"] and abs(depth) < abs(best["depth"])):
                best = cand
        tilt = float(np.radians(C.tilt_deg(axis)))
        if self.prev is None:
            mv = float("nan")
            still = False
        else:
            mv = float(np.linalg.norm(Tc[:3, 3] - self.prev))
            still = mv / self.dt < STILL_MPS
        self.prev = Tc[:3, 3].copy()
        free = obs.get("free_cube")
        conds = {"tube_lifted": bool(self.lift >= LIFT_M), "tube_in_hole": bool(best["ok"]),
                 "tube_seated": (None if best["gap"] is None else bool(best["ok"] and best["gap"] <= BOTTOM_GAP_M)),
                 "tube_vertical": bool(tilt < VERTICAL_RAD), "hand_off": free, "tube_still": bool(still)}
        done = C.tri(conds) is True
        n_ = self.tail.push(done, obs.get("t"))
        return dict(done=done, hold_n=n_, conds=conds,
                    m=dict(tube_max_lift_mm=self.lift * 1e3, hole=best["hole"], insert_depth_mm=best["depth"] * 1e3, gap_to_floor_mm=None if best["gap"] is None else best["gap"] * 1e3,
                           tilt_deg=float(np.degrees(tilt)), step_move_mm=mv * 1e3, hand_off_tube=free))


def make_judge(ctx):
    return Judge(ctx.parts, ctx.rate_hz)


def obs_from_scene(scene, ctx=None):
    tch = scene.robot_touching("cube")
    return dict(T_cube=scene.obj_T("cube"), T_rack=scene.obj_T("rack"), free_cube=None if tch is None else not tch, t=scene.t)
