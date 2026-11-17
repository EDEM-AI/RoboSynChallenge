"""FlatScene / SimReader / FaqChecker against small mock objects that mimic the EmbodiChain getters used.

The mocks follow the API of EmbodiChain 0.2.4.post1 (``get_local_pose(to_matrix=False)`` -> (N, 7) wxyz poses, ``get_qpos()`` -> (N, J),
``get_link_pose(link, to_matrix=False)``, ``get_user_ids()``, physics-scene ``get_cpu_contact_buffer()`` -> (data (n, 11), user ids (n, 2))).
They test the reading code's own logic, not the simulator.
"""
import sys
import types
import unittest

import numpy as np

from faq_checks import common as C, task_module
from faq_checks.runtime import FaqChecker, FaqEnvProxy
from faq_checks.scene import FlatScene
from faq_checks.sim_reader import SimReader


class Obj:
    def __init__(self, pos, scale=(1, 1, 1), ids=(1,), qpos=None, links=None, link_names=None):
        self.pos, self.scale, self.ids = pos, scale, ids
        self.qpos = qpos
        self.links = links or {}
        self.link_names = link_names or list(self.links)

    def get_local_pose(self, to_matrix=False):
        return np.array([[*self.pos, 1, 0, 0, 0]], float)

    def get_body_scale(self):
        return np.array([self.scale], float)

    def get_user_ids(self, link=None):
        return np.array(self.ids, np.int32)

    def get_qpos(self):
        return np.array([self.qpos], float)

    def get_link_pose(self, link, to_matrix=False):
        return np.array([[*self.links[link], 1, 0, 0, 0]], float)


class Sim:
    def __init__(self, rigid, arts):
        self.rigid, self.arts = rigid, arts

    def get_rigid_object(self, uid):
        return self.rigid[uid]

    def get_articulation(self, uid):
        return self.arts[uid]

    def get_rigid_object_uid_list(self):
        return list(self.rigid)

    def get_articulation_uid_list(self):
        return list(self.arts)


class PhysScene:
    def __init__(self):
        self.buffer = (np.zeros((0, 11)), np.zeros((0, 2), np.int32))
        self.enabled = False

    def enable_contact_data_update_on_cpu(self, flag):
        self.enabled = flag

    def get_cpu_contact_buffer(self):
        return self.buffer


def install_fake_dexsim(ps, gpu=False):
    m = types.ModuleType("dexsim")
    m.get_world_config = lambda: types.SimpleNamespace(enable_gpu_sim=gpu)
    m.default_world = lambda: types.SimpleNamespace(get_physics_scene=lambda: ps)
    sys.modules["dexsim"] = m


class FakeEnv:
    control_freq = 25

    def __init__(self, sim, robot):
        self.sim, self.robot = sim, robot
        self.unwrapped = self
        self.steps = 0
        self._guijiao_attached = np.array([False])

    def reset(self, *a, **k):
        return {}, {}

    def step(self, action):
        self.steps += 1
        return {}, 0.0, False, False, {}


def make_env():
    rigid = {"guijiao1": Obj((0.0, 0, 0.9), (1.8, 1.7, 1.7), ids=(11,)), "guijiao2": Obj((0.1, 0, 0.9), (2.1, 2, 2), ids=(12,)),
             "table": Obj((0, 0, 0), ids=(99,))}
    robot = Obj((0, 0, 0), ids=(500, 501), links={"right_link6": (0.5, -0.2, 1.0), "left_link6": (0.5, 0.2, 1.0)}, link_names=["right_link6", "left_link6"])
    robot.get_user_ids = lambda link=None: np.array({"right_link6": [500], "left_link6": [501]}.get(link, [500, 501]), np.int32)
    return FakeEnv(Sim(rigid, {}), robot)


class TestSimReader(unittest.TestCase):
    def setUp(self):
        self.ps = PhysScene()
        install_fake_dexsim(self.ps)

    def tearDown(self):
        sys.modules.pop("dexsim", None)

    def test_reads_poses_scales_and_contacts(self):
        env = make_env()
        m = task_module("item_assembly")
        rd = SimReader(env, m, {})
        self.assertTrue(self.ps.enabled)                       # contact reporting was switched on once
        rd.begin_episode()
        self.assertEqual(rd.rate_hz(), 25.0)
        sc = rd.body_scales()
        self.assertTrue(np.allclose(sc["guijiao1"], [1.8, 1.7, 1.7]))
        # one contact point between guijiao1 and the robot's right_link6, one guijiao1-guijiao2
        data = np.zeros((2, 11)); data[:, 9] = [0.5, 0.2]; data[:, 10] = [-0.001, 0.002]
        self.ps.buffer = (data, np.array([[11, 500], [12, 11]], np.int32))
        scene = rd.read(7)
        self.assertEqual(scene.t, 7)
        self.assertTrue(np.allclose(scene.obj_T("guijiao1")[:3, 3], [0, 0, 0.9]))
        self.assertTrue(scene.robot_touching("guijiao1"))
        self.assertFalse(scene.robot_touching("guijiao2"))
        self.assertTrue(scene.touching("guijiao1", "guijiao2"))

    def test_weld_event_fires_once_on_the_rising_edge(self):
        env = make_env()
        rd = SimReader(env, task_module("item_assembly"), {})
        rd.begin_episode()
        self.assertEqual(rd.read(0).events, set())
        env._guijiao_attached = np.array([True])
        self.assertEqual(rd.read(1).events, {"weld"})
        self.assertEqual(rd.read(2).events, set())

    def test_gpu_physics_means_contacts_unknown(self):
        install_fake_dexsim(self.ps, gpu=True)
        rd = SimReader(make_env(), task_module("item_assembly"), {})
        rd.begin_episode()
        self.assertIsNone(rd.read(0).contacts)
        self.assertIn("contacts", rd.errors)

    def test_table_extent_from_gym_config(self):
        cfg = {"background": [{"uid": "table", "shape": {"size": [1.0, 1.0, 0.1]}, "init_pos": [0.725, 0.0, 0.775], "body_scale": [1, 1, 1]}]}
        rd = SimReader(make_env(), task_module("water_pouring"), cfg)
        t = rd.table_extent()
        self.assertAlmostEqual(t["top_z"], 0.825)
        self.assertTrue(np.allclose(t["xy_min"], [0.225, -0.5]))
        self.assertTrue(np.allclose(t["xy_max"], [1.225, 0.5]))

    def test_eef_pose_is_read_for_tasks_that_need_it(self):
        env = make_env()
        rd = SimReader(env, task_module("water_pouring"), {})
        env.sim.rigid.update({"bottle": Obj((0.5, 0, 0.9), ids=(21,)), "cup": Obj((0.6, 0, 0.9), ids=(22,))})
        rd.begin_episode()
        scene = rd.read(0)
        self.assertTrue(np.allclose(scene.eef_pos("right"), [0.5, -0.2, 1.0]))


class TestChecker(unittest.TestCase):
    def setUp(self):
        self.ps = PhysScene()
        install_fake_dexsim(self.ps)

    def tearDown(self):
        sys.modules.pop("dexsim", None)

    def test_proxy_only_observes_when_active(self):
        env = make_env()
        chk = FaqChecker(env, "item_assembly", {})
        px = FaqEnvProxy(env, chk)
        px.reset()
        px.step(None)
        self.assertEqual(chk.t, 0)                              # inactive: the feasibility pre-check must not be judged
        px.active = True
        px.reset()
        self.assertEqual(chk.t, 1)                              # the reset state counts as step 0
        px.step(None)
        self.assertEqual(chk.t, 2)
        self.assertEqual(env.steps, 2)                          # steps reach the wrapped env unchanged
        res = chk.result()
        self.assertFalse(res["faq_success"])
        self.assertIn("faq", res)

    def test_checker_never_raises_into_the_eval_loop(self):
        env = make_env()
        chk = FaqChecker(env, "item_assembly", {})
        del env.sim.rigid["guijiao2"]                           # reading will fail
        chk.start_episode()
        chk.observe()
        self.assertFalse(chk.result()["faq_success"])


if __name__ == "__main__":
    unittest.main()
