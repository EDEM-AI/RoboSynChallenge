"""Pure helpers for aggregating and persisting policy evaluation metrics."""

from __future__ import annotations

import json
import os
from pathlib import Path


class EvaluationMetrics:
    """Collect episode metrics using the challenge's timeout convention."""

    def __init__(self, timeout_action_steps):
        self.timeout_action_steps = int(timeout_action_steps)
        if self.timeout_action_steps <= 0:
            raise ValueError("timeout_action_steps must be positive")
        self.episodes = []

    def record_episode(
        self,
        *,
        episode,
        seed,
        success,
        env_steps,
        inference_times_s,
        inference_timing_scope,
    ):
        """Record one episode, charging every failure the full timeout length."""
        success = bool(success)
        env_steps = int(env_steps)
        effective_steps = env_steps if success else self.timeout_action_steps
        inference_times = [float(value) for value in inference_times_s]
        total_inference_time = sum(inference_times)
        episode_result = {
            "episode": int(episode),
            "seed": int(seed),
            "success": success,
            "observed_env_steps": env_steps,
            "effective_action_steps": effective_steps,
            "action_steps_ratio": effective_steps / self.timeout_action_steps,
            "inference_call_count": len(inference_times),
            "total_inference_time_seconds": total_inference_time,
            "average_inference_time_seconds": (
                total_inference_time / len(inference_times)
                if inference_times
                else None
            ),
            "inference_timing_scope": inference_timing_scope,
        }
        self.episodes.append(episode_result)
        return episode_result

    def summary(self):
        episode_count = len(self.episodes)
        success_count = sum(result["success"] for result in self.episodes)
        effective_steps = [
            result["effective_action_steps"] for result in self.episodes
        ]
        inference_time_total = sum(
            (result["average_inference_time_seconds"] or 0.0)
            * result["inference_call_count"]
            for result in self.episodes
        )
        inference_call_count = sum(
            result["inference_call_count"] for result in self.episodes
        )
        average_action_steps = (
            sum(effective_steps) / episode_count if episode_count else None
        )
        return {
            "episode_count": episode_count,
            "success_count": success_count,
            "success_rate": success_count / episode_count if episode_count else None,
            "average_action_steps": average_action_steps,
            "average_action_steps_ratio": (
                average_action_steps / self.timeout_action_steps
                if average_action_steps is not None
                else None
            ),
            "inference_call_count": inference_call_count,
            "average_inference_calls_per_episode": (
                inference_call_count / episode_count if episode_count else None
            ),
            "average_inference_time_seconds": (
                inference_time_total / inference_call_count
                if inference_call_count
                else None
            ),
            "average_inference_time_per_episode_seconds": (
                inference_time_total / episode_count if episode_count else None
            ),
        }


def write_evaluation_result(path, payload):
    """Atomically write an evaluation result so interrupted writes stay readable."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as result_file:
        json.dump(payload, result_file, indent=2, ensure_ascii=False)
        result_file.write("\n")
    os.replace(temporary_path, path)
