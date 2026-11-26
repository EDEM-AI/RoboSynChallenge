"""Collect a native Task Program episode from a specified reset seed."""

import argparse
from dataclasses import dataclass, replace

import gymnasium as gym

from embodichain.lab.gym.envs.demo import execute_demo_episode
from embodichain.lab.gym.utils import gym_utils
from embodichain.lab.gym.utils.registration import (
    discover_task_packages,
    execute_init_hooks,
)
from embodichain.lab.sim import SimulationManager
from embodichain.lab.sim.atomic_skills import AffordanceSamplingContext

from scripts.eval_policy import prepare_episode_reset


@dataclass(frozen=True)
class NativeAffordanceBranch:
    seed: int
    index: int

    def create_plan_transform(self, request, *, engine):
        return None

    def prepare_planning_context(self, request, context, *, engine):
        return replace(
            context,
            affordance_sampling=AffordanceSamplingContext(
                count=16, seed=self.seed, branch_override=self.index
            ),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    gym_utils.add_env_launcher_args_to_parser(parser, require_gym_config=True)
    parser.add_argument("--affordance-branch", type=int)
    parser.add_argument("--dataset-dir", dest="expansion_dataset_dir")
    args = parser.parse_args()
    if args.seed is None or args.seed < 0:
        parser.error("Seeded collection requires a non-negative --seed.")
    if args.affordance_branch is not None and args.affordance_branch < 0:
        parser.error("--affordance-branch must be non-negative.")
    if args.num_envs != 1:
        parser.error("Seeded collection accepts one environment.")
    if args.filter_dataset_saving and args.expansion_dataset_dir is not None:
        parser.error("--dataset-dir requires dataset recording to be enabled.")

    discover_task_packages()
    execute_init_hooks()
    cfg, selection, _ = gym_utils.build_env_cfg_from_args(args)
    env = gym.make(selection["id"], cfg=cfg)
    try:
        seed = prepare_episode_reset(env, args.seed)
        env.reset(seed=seed, options={"save_data": False})
        options = {}
        if args.affordance_branch is not None:
            options["plan_transform_factory"] = NativeAffordanceBranch(
                seed, args.affordance_branch
            )
        result = execute_demo_episode(env, **options)
        accepted = all(result.success) and bool(env.unwrapped.is_task_success().all())
        for segment in result.segments:
            print(f"{segment.name} success={segment.success}")
        print(f"Seed {seed} collected {result.length} actions. Success={accepted}")
        if accepted:
            rows = [] if args.filter_dataset_saving else [0]
            env.reset(options={"save_data": True, "commit_env_ids": rows})
        return 0 if accepted else 1
    finally:
        env.close()
        SimulationManager.flush_cleanup_queue()


if __name__ == "__main__":
    raise SystemExit(main())
