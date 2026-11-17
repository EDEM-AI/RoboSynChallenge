"""handle_basket: step-wise FAQ criteria.

FAQ, three sentences: (1) pick up the milk on the left and put it into the basket, (2) move the basket on the right toward the
centre of the view, holding it, with a clear displacement, (3) put the basket down stably.

The simulator's built-in check (``handle_basket.py``): milk in basket (milk-to-basket-origin horizontal distance < 10 cm and the
milk above the basket origin) and basket displaced > 15 cm along +y (and lifted > 1 cm or milk in basket), for 55 steps.  It does
not require putting the basket down.  Its semantics are kept, with the fixed 10 cm replaced by the basket's actual mouth;
"put down stably" has no simulator counterpart and is defined here:

| FAQ sentence       | check                                                                                                         | value       |
| milk lifted        | the milk origin got >= 5 cm above its start height at some step                                               | 5 cm        |
| milk in basket     | the milk's bottom centre, in the basket frame, is inside the mouth rectangle (inner frame) horizontally and    |             |
|                    | between the basket's inner floor and the mouth plane                                                          |             |
| basket carried     | at some step the basket was > 1 cm above its start height and displaced >= 15 cm along +y                      | 1 cm, 15 cm |
| basket at target   | on this step the basket's +y displacement is still >= 15 cm                                                    | 15 cm       |
| basket set down    | no robot link touches the basket or the milk; basket origin <= 1 cm above its start height; mouth normal      | 1 cm, 5 deg,|
|                    | within 5 degrees of its start tilt; basket and milk move <= 0.1 mm per step                                  | 0.1 mm      |
| hold               | all of the above for >= 20 consecutive steps up to the last step                                              | 20          |

Informational only (reported in ``measured``, never part of the verdict): the height and downward speed of the milk bottom at the
moment the hand last let go of it, and how far it kept falling afterwards.
"""
import numpy as np

from . import common as C
from . import parts as P

TASK = "handle_basket"
UIDS = ("milk", "basket")
RIGID = ("milk", "basket")
ARTS = ()
ART_LINKS = {}
EEF = ()
NEEDS_CONTACTS = True
HOLD_STEPS = C.HOLD_STEPS
MILK_LIFT_M = 0.05
MOVE_Y_M = 0.15
Z_LIFT_M = 0.01
UPRIGHT_DEG = 5.0
STILL_M = 0.0001
DEFINITION = ("milk lifted = milk origin >= 5 cm above its start; milk in basket = milk bottom centre inside the basket mouth (inner frame) and between the inner floor and the mouth plane; "
              "basket carried = at some step basket > 1 cm above its start and >= 15 cm along +y (the simulator's numbers); basket at target = displacement still >= 15 cm on this step; "
              "basket set down = no robot contact with basket or milk, basket origin <= 1 cm above its start, mouth normal within 5 deg of its start tilt, "
              "per-step displacement of basket and milk <= 0.1 mm; hold = all of them for >= 20 consecutive steps up to the last step")


class Judge:
    def __init__(self, parts):
        parts = P.parts_of(parts)
        b = parts["basket"]
        self.n = C.unit(b["mouth"]["normal"])
        self.mouth, self.floor = np.asarray(b["mouth"]["p"], float), np.asarray(b["floor"]["p"], float)
        sec = b["mouth"]["section"]
        self.axes = [C.unit(a) for a in sec["axes"]]
        self.half = (float(sec["w"]) / 2.0, float(sec["d"]) / 2.0)
        self.milk_bottom = np.asarray(parts["milk"]["bottom"]["p"], float)
        self.milk_up = C.unit(parts["milk"]["top"]["normal"])
        self.open = None
        self.milk_lift, self.carried, self.prev = 0.0, False, None
        self.touch_prev, self.rel = None, None      # rel = last release: step, milk-bottom height over the floor, downward speed, height, lowest height since
        self.bz_prev = None
        self.tail = C.Tail()

    def step(self, obs):
        """obs: T_milk, T_basket (4x4), free (no robot link touches basket or milk: True / False / None), t; for the informational release
        numbers also touch_milk (a robot link touches the milk on this step) and dt (step duration in s, default 0.04).  The first call is the start."""
        Tm, Tb = np.asarray(obs["T_milk"], float), np.asarray(obs["T_basket"], float)
        tilt = C.tilt_deg(C.world_dir(Tb, self.n))
        if self.open is None:
            self.open = dict(milk_z=float(Tm[2, 3]), basket=Tb[:3, 3].copy(), tilt=tilt)
        o = self.open
        self.milk_lift = max(self.milk_lift, float(Tm[2, 3]) - o["milk_z"])
        v = Tb[:3, :3].T @ (C.world_point(Tm, self.milk_bottom) - Tb[:3, 3])
        h = float((v - self.floor) @ self.n)
        top = float((self.mouth - self.floor) @ self.n)
        lat = [abs(float((v - self.mouth) @ a)) for a in self.axes]
        inside = bool(lat[0] < self.half[0] and lat[1] < self.half[1] and -0.005 <= h <= top)
        dy, dz = float(Tb[1, 3] - o["basket"][1]), float(Tb[2, 3] - o["basket"][2])
        self.carried |= dz > Z_LIFT_M and dy >= MOVE_Y_M
        if self.prev is None:
            mv = float("nan")
            still = False
        else:
            mv = max(float(np.linalg.norm(Tb[:3, 3] - self.prev[0])), float(np.linalg.norm(Tm[:3, 3] - self.prev[1])))
            still = mv <= STILL_M
        self.prev = (Tb[:3, 3].copy(), Tm[:3, 3].copy())
        free = obs.get("free")
        down = bool(dz <= Z_LIFT_M and tilt <= o["tilt"] + UPRIGHT_DEG and still)
        conds = {"milk_lifted": bool(self.milk_lift >= MILK_LIFT_M), "milk_in_basket": inside, "basket_carried": bool(self.carried),
                 "basket_at_target": bool(dy >= MOVE_Y_M), "basket_set_down": (None if free is None and down else bool(down and free))}
        done = C.tri(conds) is True
        n = self.tail.push(done, obs.get("t"))
        milk_tilt = C.tilt_deg(C.world_dir(Tm, self.milk_up))
        bz = float(C.world_point(Tm, self.milk_bottom)[2])
        tm = obs.get("touch_milk")
        if tm is not None:
            if self.touch_prev and not tm and self.milk_lift >= MILK_LIFT_M:           # the hand lets go on this step
                vz = (self.bz_prev - bz) / float(obs.get("dt") or 0.04)
                self.rel = dict(t=obs.get("t"), h=h, vz=vz, z=bz, zmin=bz)
            if self.rel is not None and not tm:
                self.rel["zmin"] = min(self.rel["zmin"], bz)
            self.touch_prev = bool(tm)
        self.bz_prev = bz
        drop = None if self.rel is None else self.rel["z"] - self.rel["zmin"]
        return dict(done=done, hold_n=n, conds=conds,
                    m=dict(release_step=None if self.rel is None else self.rel["t"], milk_bottom_over_floor_at_release_mm=None if self.rel is None else self.rel["h"] * 1e3,
                           milk_fall_speed_at_release_mm_s=None if self.rel is None else self.rel["vz"] * 1e3, milk_fell_after_release_mm=None if drop is None else drop * 1e3,
                           milk_tilt_deg=milk_tilt, milk_max_lift_mm=self.milk_lift * 1e3, milk_bottom_over_floor_mm=h * 1e3, milk_off_mouth_centre_mm=[x * 1e3 for x in lat],
                           mouth_half_extent_mm=[x * 1e3 for x in self.half], basket_dy_mm=dy * 1e3, basket_dz_mm=dz * 1e3, basket_tilt_over_start_deg=tilt - o["tilt"],
                           step_move_mm=mv * 1e3, robot_off_basket_and_milk=free))


def make_judge(ctx):
    return Judge(ctx.parts)


def obs_from_scene(scene, ctx=None):
    dt = 1.0 / float(ctx.rate_hz) if ctx is not None else 0.04
    a, b = scene.robot_touching("basket"), scene.robot_touching("milk")
    return dict(T_milk=scene.obj_T("milk"), T_basket=scene.obj_T("basket"), free=None if a is None else not (a or b), touch_milk=b, dt=dt, t=scene.t)
