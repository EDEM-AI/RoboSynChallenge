"""Golden episodes: recorded env states are replayed through ``obs_from_scene`` + ``Judge``.

``fixtures/<task>.json.gz`` holds, per task, up to two recorded episodes (one the criteria judged successful, one not) with only the state
entries the task module reads and the contact rows involving the task's objects.  The expected per-step ``done`` / ``hold_n`` and the
last-step sentences are the output of the criteria code these modules were rewritten from, run on the same states and the same parts.
Any change of the criteria, of the parts table or of the scaling rules shows up here.
"""
import glob
import gzip
import json
import os
import unittest

from faq_checks import TASKS, common as C, parts as P, task_module
from faq_checks.scene import FlatScene

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def replay(task, ep):
    m = task_module(task)
    c = ep["context"]
    parts = c.get("parts_override") or P.parts_for_episode(task, c["body_scale"])
    parts = dict(parts, **c.get("parts_extra", {}))
    ctx = P.EpisodeContext(task, c["body_scale"], parts=parts, table=c["table"], rate_hz=c["rate_hz"], limits=c["limits"])
    judge = m.make_judge(ctx)
    welds = set(ep["weld_steps"])
    done, hold, last = [], [], None
    for st in ep["steps"]:
        scene = FlatScene(st["state"], st["contacts"], {"weld"} if st["t"] in welds else (), st["t"])
        obs = m.obs_from_scene(scene, ctx)
        if obs is None:
            continue
        last = judge.step(obs)
        done.append(int(last["done"]))
        hold.append(last["hold_n"])
    return last, done, hold


class TestGoldenEpisodes(unittest.TestCase):
    def test_fixtures_exist_for_every_task(self):
        have = {os.path.basename(p).split(".")[0] for p in glob.glob(os.path.join(FIXTURES, "*.json.gz"))}
        self.assertEqual(have, set(TASKS))

    def test_replays_match_expected(self):
        n = 0
        for path in sorted(glob.glob(os.path.join(FIXTURES, "*.json.gz"))):
            with gzip.open(path, "rt", encoding="utf-8") as f:
                data = json.load(f)
            task = data["task"]
            for ep in data["episodes"]:
                with self.subTest(episode=ep["name"]):
                    last, done, hold = replay(task, ep)
                    exp = ep["expected"]
                    self.assertEqual(done, exp["done"])
                    self.assertEqual(hold, exp["hold_n"])
                    self.assertEqual(last["conds"], exp["conds_last"])
                    row = C.summarize(task, last["conds"], last["hold_n"], len(done), {}, "")
                    self.assertEqual(row["faq_success"], exp["faq_success"])
                    n += 1
        self.assertGreaterEqual(n, len(TASKS))


if __name__ == "__main__":
    unittest.main()
