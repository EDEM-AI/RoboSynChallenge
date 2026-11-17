"""Runs a task's FAQ criteria next to an env: one ``FaqChecker`` per env, ``FaqEnvProxy`` to hook reset / step."""
from . import common as C
from . import parts as P
from .sim_reader import SimReader
from . import task_module


class FaqChecker:
    """Judges the episodes of one env with ``<task>_faq``.

    ``start_episode()`` after every reset, ``observe()`` after every step (the reset state counts as step 0), ``result()`` any time.
    """

    def __init__(self, env, task, gym_config=None):
        self.task = task
        self.m = task_module(task)
        self.reader = SimReader(env, self.m, gym_config)
        self.judge = None
        self.last = None
        self.t = 0
        self.n_obs_missing = 0
        self.error = None

    def start_episode(self):
        self.reader.begin_episode()
        self.t, self.last, self.n_obs_missing, self.error = 0, None, 0, None
        try:
            ctx = P.EpisodeContext(self.task, self.reader.body_scales(), table=self.reader.table_extent(),
                                   rate_hz=self.reader.rate_hz(), limits=self.reader.limits())
            self.ctx = ctx
            self.judge = self.m.make_judge(ctx)
        except Exception as e:  # noqa: BLE001  never break the evaluation run because of an optional check
            self.judge, self.error = None, "setup: %s: %s" % (type(e).__name__, e)
        self.observe()

    def observe(self):
        if self.judge is None:
            return None
        try:
            scene = self.reader.read(self.t)
            obs = self.m.obs_from_scene(scene, self.ctx)
            if obs is None:
                self.n_obs_missing += 1
            else:
                self.last = self.judge.step(obs)
        except Exception as e:  # noqa: BLE001
            self.error = "step %d: %s: %s" % (self.t, type(e).__name__, e)
        self.t += 1
        return self.last

    @property
    def hold_reached(self):
        """True once the criteria have held for HOLD_STEPS consecutive steps."""
        return bool(self.last is not None and self.last["hold_n"] >= C.HOLD_STEPS)

    def result(self):
        """Episode row: faq_success, per-sentence flags on the last step, hold bookkeeping, errors."""
        if self.last is None:
            return dict(task=self.task, faq_success=False, faq={}, hold=dict(tail=False, last=False, steps=0), n_steps=self.t, error=self.error or "no observation",
                        contacts_available=self.reader.contacts_ok, reader_errors=dict(self.reader.errors))
        row = C.summarize(self.task, self.last["conds"], self.last["hold_n"], self.t, self.last.get("m", {}), self.m.DEFINITION)
        row["contacts_available"] = self.reader.contacts_ok
        if self.error:
            row["error"] = self.error
        if self.reader.errors:
            row["reader_errors"] = dict(self.reader.errors)
        return row


class FaqEnvProxy:
    """Wraps an env (or another proxy): after ``reset`` and every ``step`` the checker observes the new state.  Everything else is delegated."""

    def __init__(self, env, checker):
        self._env = env
        self.faq = checker
        self.active = False        # only judge the real episode, not e.g. the expert feasibility pre-check that shares this env

    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset(self, *args, **kwargs):
        out = self._env.reset(*args, **kwargs)
        if self.active:
            self.faq.start_episode()
        return out

    def step(self, action, *args, **kwargs):
        out = self._env.step(action, *args, **kwargs)
        if self.active:
            self.faq.observe()
        return out
