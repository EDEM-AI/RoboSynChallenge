import json

import pytest

from scripts.eval_metrics import EvaluationMetrics, write_evaluation_result


def test_summary_counts_failures_as_timeout_and_weights_inference_calls():
    metrics = EvaluationMetrics(timeout_action_steps=100)
    metrics.record_episode(
        episode=1,
        seed=7,
        success=True,
        env_steps=40,
        inference_times_s=[0.1, 0.3],
        inference_timing_scope="model_forward",
    )
    failed = metrics.record_episode(
        episode=2,
        seed=8,
        success=False,
        env_steps=25,
        inference_times_s=[0.2],
        inference_timing_scope="model_forward",
    )

    assert failed["observed_env_steps"] == 25
    assert failed["effective_action_steps"] == 100
    assert metrics.summary() == {
        "episode_count": 2,
        "success_count": 1,
        "success_rate": 0.5,
        "average_action_steps": 70.0,
        "average_action_steps_ratio": 0.7,
        "inference_call_count": 3,
        "average_inference_calls_per_episode": 1.5,
        "average_inference_time_seconds": pytest.approx(0.2),
        "average_inference_time_per_episode_seconds": pytest.approx(0.3),
    }


def test_empty_summary_and_atomic_json_write(tmp_path):
    metrics = EvaluationMetrics(timeout_action_steps=10)
    assert metrics.summary()["average_action_steps"] is None

    result_path = tmp_path / "nested" / "evaluation_metrics.json"
    write_evaluation_result(result_path, {"summary": metrics.summary()})

    assert json.loads(result_path.read_text()) == {"summary": metrics.summary()}
    assert not result_path.with_suffix(".json.tmp").exists()
