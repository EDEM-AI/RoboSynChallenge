"""Run the locked LeRobot trainer with a local or published dataset."""

import os
import sys

from robosynchallenge.hub import DATASET_REVISION, dataset_source


def run_training(policy, dataset, output, options):
    repo_id, root = dataset_source(dataset)
    policy_type = [f"--policy.type={policy}"]
    if any(arg == "--policy.path" or arg.startswith("--policy.path=") for arg in options):
        policy_type = []
    dataset_options = [f"--dataset.repo_id={repo_id}", "--dataset.video_backend=pyav"]
    if root is not None:
        dataset_options.append(f"--dataset.root={root}")
    os.execv(
        sys.executable,
        [sys.executable, "-m", "lerobot.scripts.lerobot_train",
         *policy_type, "--policy.push_to_hub=false", *dataset_options,
         f"--output_dir={output}", *options, f"--dataset.revision={DATASET_REVISION}"],
    )


if __name__ == "__main__":
    run_training(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:])
