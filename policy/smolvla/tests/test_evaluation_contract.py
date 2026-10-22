import io
import json
import tempfile
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from policy.smolvla import deploy_policy as adapter
from policy.smolvla import smolvla_worker as worker


class Clock:
    def __init__(self):
        self.value = 0.0

    def now(self, device=None):
        return self.value

    def advance(self, seconds):
        self.value += seconds

    def finish(self, started_at, samples, device=None):
        samples.append(self.value - started_at)


class QueuedPolicy:
    def __init__(self, clock, legacy=False):
        self.clock = clock
        if legacy:
            self._action_queue = deque()
        else:
            self._queues = {"action": deque()}
        self.calls = 0
        self.seen = []

    @property
    def queue(self):
        return self._queues["action"] if hasattr(self, "_queues") else self._action_queue

    def reset(self):
        self.queue.clear()

    def select_action(self, obs):
        self.seen.append(obs)
        if not self.queue:
            self.clock.advance(5)
            self.calls += 1
            self.queue.extend(torch.full((1, 14), float(index)) for index in range(3))
        else:
            self.clock.advance(1)
        return self.queue.popleft()


class EvaluationContractTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.policy = QueuedPolicy(self.clock)
        self.obs = {"observation.state": np.zeros(14, dtype=np.float32),
                    "observation.images.cam_high": np.full((3, 2, 2), 255, dtype=np.uint8)}

    def preprocess(self, obs):
        self.clock.advance(2)
        return obs

    def postprocess(self, action):
        self.clock.advance(1)
        return action * 2

    def load(self, path):
        self.clock.advance(1000)  # File transport must not enter inference samples.
        return self.obs, "test instruction"

    def infer(self, count):
        with patch.object(worker, "_load_obs", side_effect=self.load), \
                patch.object(worker, "_clock", side_effect=self.clock.now):
            return worker._infer(self.policy, self.preprocess, self.postprocess,
                                 {"obs_path": "unused.npz", "n_action_steps": count}, "cpu")

    def test_counts_model_calls_not_cached_actions(self):
        result = self.infer(7)
        self.assertEqual(result["inference_action_indices"], [0, 3, 6])
        self.assertEqual(result["durations_s"], [8.0, 6.0, 6.0])
        self.assertEqual(self.policy.calls, 3)
        np.testing.assert_array_equal(result["actions"],
                                      np.array([[0] * 14, [2] * 14, [4] * 14] * 2 + [[0] * 14]))
        self.assertEqual(self.policy.seen[0]["observation.images.cam_high"].dtype, torch.float32)
        torch.testing.assert_close(self.policy.seen[0]["observation.images.cam_high"], torch.ones(3, 2, 2))
        self.assertEqual(self.policy.seen[0]["task"], "test instruction")

    def test_only_byte_images_are_scaled_to_unit_range(self):
        image = np.array([0, 64, 128, 255], dtype=np.uint8).reshape(1, 2, 2)
        self.obs["observation.images.cam_high"] = image
        self.obs["observation.images.cam_left_wrist"] = np.full((3, 2, 2), 0.25, dtype=np.float32)
        self.obs["diagnostic_byte"] = np.array([255], dtype=np.uint8)
        original_state = self.obs["observation.state"].copy()
        self.infer(1)
        seen = self.policy.seen[0]
        torch.testing.assert_close(seen["observation.images.cam_high"], torch.from_numpy(image).float() / 255)
        torch.testing.assert_close(seen["observation.images.cam_left_wrist"], torch.full((3, 2, 2), 0.25))
        self.assertEqual(seen["diagnostic_byte"].dtype, torch.uint8)
        np.testing.assert_array_equal(seen["observation.state"].numpy(), original_state)
        np.testing.assert_array_equal(self.obs["observation.images.cam_high"], image)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for the Byte resize regression")
    def test_byte_image_can_reach_cuda_bilinear_resize(self):
        class ResizePolicy:
            def select_action(self, obs):
                image = obs["observation.images.cam_high"].unsqueeze(0).to("cuda")
                resized = torch.nn.functional.interpolate(image, size=(4, 4), mode="bilinear", align_corners=False)
                torch.testing.assert_close(resized, torch.ones_like(resized))
                return torch.zeros(1, 14, device=image.device)

        with patch.object(worker, "_load_obs", return_value=(self.obs, "test instruction")):
            result = worker._infer(ResizePolicy(), lambda obs: obs, lambda action: action,
                                   {"obs_path": "unused", "n_action_steps": 1}, "cuda")
        self.assertEqual(len(result["actions"]), 1)
        self.assertEqual(len(result["durations_s"]), 1)
        self.assertGreater(result["durations_s"][0], 0)

    def test_queue_survives_requests_and_reset(self):
        self.infer(1)
        cached = self.infer(2)
        self.assertEqual(cached["durations_s"], [])
        self.assertEqual(cached["inference_action_indices"], [])
        fresh = self.infer(1)
        self.assertEqual(fresh["durations_s"], [8.0])
        self.policy.reset()
        self.assertEqual(self.infer(1)["durations_s"], [8.0])

    def test_legacy_queue_attribute_is_also_supported(self):
        self.policy = QueuedPolicy(self.clock, legacy=True)
        result = self.infer(4)
        self.assertEqual(result["inference_action_indices"], [0, 3])
        self.assertEqual(result["durations_s"], [8.0, 6.0])

    def test_cached_prefix_does_not_inflate_next_forward(self):
        self.infer(1)
        result = self.infer(4)
        self.assertEqual(result["inference_action_indices"], [2])
        self.assertEqual(result["durations_s"], [8.0])

    def test_backend_without_action_queue_counts_each_call(self):
        policy = SimpleNamespace(select_action=lambda obs: self.postprocess(torch.zeros(1, 14)))
        with patch.object(worker, "_load_obs", side_effect=self.load), \
                patch.object(worker, "_clock", side_effect=self.clock.now):
            result = worker._infer(policy, self.preprocess, self.postprocess,
                                   {"obs_path": "unused", "n_action_steps": 2}, "cpu")
        self.assertEqual(result["inference_action_indices"], [0, 1])
        self.assertEqual(result["durations_s"], [4.0, 2.0])

    def test_actions_match_original_worker_pipeline(self):
        original = QueuedPolicy(Clock())
        tensors = {key: torch.from_numpy(value) for key, value in self.obs.items()}
        tensors["task"] = "test instruction"
        expected = [(original.select_action(tensors) * 2).squeeze(0).cpu().numpy()
                    .astype(np.float32).tolist() for _ in range(8)]
        self.assertEqual(self.infer(8)["actions"], expected)

    def test_cuda_clock_synchronizes_requested_device(self):
        with patch.object(worker.torch.cuda, "synchronize") as sync, \
                patch.object(worker.time, "perf_counter", return_value=12.0):
            self.assertEqual(worker._clock("cuda:2"), 12.0)
            sync.assert_called_once_with("cuda:2")
            worker._clock("cpu")
            sync.assert_called_once()

    def make_client(self, response):
        client = adapter.SmolVLAWorkerClient.__new__(adapter.SmolVLAWorkerClient)
        client.smolvla_steps = 3
        client._rpc = lambda request: response
        return client

    def test_client_keeps_action_api_and_validates_timing(self):
        response = {"actions": [[0] * 14] * 3, "durations_s": [0.25],
                    "inference_action_indices": [0]}
        client = self.make_client(response)
        actions = client.infer(self.obs, "instruction")
        self.assertEqual(len(actions), 3)
        self.assertEqual(actions[0].dtype, np.float32)
        self.assertEqual(client.last_inference_times_s, [0.25])
        self.assertEqual(client.last_inference_action_indices, [0])
        client.reset()
        self.assertEqual(client.last_inference_times_s, [])
        self.assertEqual(client.last_inference_action_indices, [])

    def test_rejects_missing_malformed_or_fabricated_timing(self):
        cases = [{}, {"durations_s": [0], "inference_action_indices": [0]},
                 {"durations_s": [float("nan")], "inference_action_indices": [0]},
                 {"durations_s": [float("inf")], "inference_action_indices": [0]},
                 {"durations_s": [-1], "inference_action_indices": [0]},
                 {"durations_s": [True], "inference_action_indices": [0]},
                 {"durations_s": ["0.2"], "inference_action_indices": [0]},
                 {"durations_s": [0.2], "inference_action_indices": [3]},
                 {"durations_s": [0.2], "inference_action_indices": [True]},
                 {"durations_s": [0.2, 0.2], "inference_action_indices": [0, 0]},
                 {"durations_s": [0.2, 0.2], "inference_action_indices": [1, 0]},
                 {"durations_s": [0.2], "inference_action_indices": []}]
        for fields in cases:
            with self.subTest(fields=fields):
                client = self.make_client(dict(actions=[[0] * 14] * 3, **fields))
                with self.assertRaisesRegex(RuntimeError, "update both adapter and worker"):
                    client.infer(self.obs, "instruction")
                self.assertEqual(client.last_inference_times_s, [])

    def test_main_json_protocol_infer_reset_and_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "observation.npz"
            np.savez_compressed(path, **self.obs, task=np.asarray("test instruction"))
            requests = [{"cmd": "infer", "obs_path": str(path), "n_action_steps": 1},
                        {"cmd": "reset"},
                        {"cmd": "infer", "obs_path": str(path), "n_action_steps": 1},
                        {"cmd": "unknown"}]
            stdout = io.StringIO()
            with patch.object(worker, "_parse_args", return_value=SimpleNamespace(device="cpu")), \
                    patch.object(worker, "_load_runtime", return_value=(
                        self.policy, self.preprocess, self.postprocess)), \
                    patch.object(worker, "_clock", side_effect=self.clock.now), \
                    patch.object(worker.sys, "stdin", io.StringIO(
                        "\n".join(json.dumps(request) for request in requests))), \
                    patch.object(worker.sys, "stdout", stdout):
                self.assertEqual(worker.main(), 0)
            responses = [json.loads(line) for line in stdout.getvalue().splitlines()]
            self.assertEqual(responses[0]["durations_s"], [8.0])
            self.assertEqual(responses[1], {"ok": True})
            self.assertEqual(responses[2]["durations_s"], [8.0])
            self.assertFalse(responses[3]["ok"])

    def test_adapter_worker_round_trip_with_real_observation_and_action_encoding(self):
        client = self.make_client(None)
        client.smolvla_steps = 4
        client.image_keys = None
        client.rescale_gripper = True
        client.gripper_indices = (6, 13)
        client.gripper_source_range = (0, 1)
        client.maybe_print_action_debug = lambda actions: None
        client.maybe_print_env_action_debug = lambda action: None
        client._rpc = lambda request: worker._infer(
            self.policy, self.preprocess, self.postprocess, request, "cpu")
        observed = []
        obs = {"robot": {"qpos": np.zeros((1, 14), dtype=np.float32)},
               "sensor": {name: {"color": np.full((1, 2, 2, 3), 255, dtype=np.uint8)}
                          for name in ("cam_high", "cam_left_wrist", "cam_right_wrist")}}

        def step(action):
            observed.append(action.clone())
            self.clock.advance(1000)
            return obs, 0, False, False, {}

        env = SimpleNamespace(step=step, _current_instruction="test instruction",
            unwrapped=SimpleNamespace(device="cpu", single_action_space=SimpleNamespace(
                shape=(14,), low=np.full(14, -1), high=np.full(14, 1))))
        with patch.object(worker, "_clock", side_effect=self.clock.now), \
                patch.object(adapter, "start_inference", side_effect=self.clock.now), \
                patch.object(adapter, "finish_inference", side_effect=self.clock.finish):
            result = adapter.eval(env, client, obs)
        self.assertEqual(result[3], [8.0, 6.0])
        self.assertEqual(len(observed), 4)
        self.assertEqual(tuple(observed[0].shape), (1, 14))
        self.assertEqual(observed[0][0, 6].item(), -1.0)
        self.assertEqual(observed[1][0, 6].item(), 1.0)
        self.assertEqual(observed[2][0, 0].item(), 4.0)
        self.assertEqual(self.policy.seen[0]["observation.images.cam_high"].shape, (3, 2, 2))

    def test_eval_four_values_with_transport_and_environment_excluded(self):
        for timeout in (False, np.array([False]), torch.tensor([False]), True,
                        np.array([True]), torch.tensor([True])):
            with self.subTest(timeout=timeout):
                self.clock.value = 0
                actions = [np.zeros(14, dtype=np.float32)] * 3

                def infer(obs, task):
                    self.clock.advance(100)  # RPC, compression and file I/O excluded.
                    return actions

                def encode_obs(obs, image_keys):
                    self.clock.advance(2)
                    return self.obs

                def encode_action(action, env, **kwargs):
                    self.clock.advance(3)
                    return torch.as_tensor(action).unsqueeze(0)

                def step(action):
                    self.clock.advance(1000)
                    return "final", 0, False, timeout, {"step": True}

                model = SimpleNamespace(infer=infer, image_keys={},
                    last_inference_times_s=[8, 6], last_inference_action_indices=[0, 2],
                    rescale_gripper=True, gripper_indices=(6, 13), gripper_source_range=(0, 1),
                    maybe_print_action_debug=lambda actions: self.clock.advance(10),
                    maybe_print_env_action_debug=lambda action: self.clock.advance(10))
                env = SimpleNamespace(unwrapped=SimpleNamespace(device="cpu"), step=step)
                with patch.object(adapter, "encode_obs", side_effect=encode_obs), \
                        patch.object(adapter, "encode_action", side_effect=encode_action), \
                        patch.object(adapter, "start_inference", side_effect=self.clock.now), \
                        patch.object(adapter, "finish_inference", side_effect=self.clock.finish):
                    result = adapter.eval(env, model, "initial")
                self.assertEqual(len(result), 4)
                self.assertEqual(result[0], "final")
                self.assertEqual(result[1], {"step": True})
                self.assertIs(result[2], timeout)
                self.assertEqual(result[3], [13, 6] if bool(np.asarray(timeout).reshape(-1)[0])
                                 else [13, 9])


if __name__ == "__main__":
    unittest.main()
