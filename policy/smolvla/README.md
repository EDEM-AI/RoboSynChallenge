# SmolVLA

The root uv environment provides LeRobot v3.0 training and simulator evaluation. The evaluation worker uses the launcher’s Python executable and inherits its GPU selection.

See the [training and evaluation guide](../../docs/tutorials/policy/smolvla.md).

## Evaluation contract

The adapter returns `(obs, info, truncated, inference_times_s)`. Timing covers
observation processing, model inference and action conversion for newly inferred
chunks. Queued actions reuse a chunk and do not count as new inference calls.

`--smolvla_steps` controls how many actions are executed per query.
`--smolvla_rescale_gripper` defaults to `auto`, which uses the checkpoint
statistics to determine whether to map normalized gripper actions to the
environment range. The task deployment supplies the episode limit.
