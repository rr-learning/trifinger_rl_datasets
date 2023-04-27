"""Evaluate a policy in simulation."""
import argparse
import importlib
import json
import logging
import pathlib
import sys
import typing

import gymnasium as gym

from trifinger_rl_datasets import Evaluation, PolicyBase, TriFingerDatasetEnv


def load_policy_class(policy_class_str: str) -> typing.Type[PolicyBase]:
    """Import the given policy class

    Args:
        The name of the policy class in the format "package.module.Class".

    Returns:
        The specified policy class.

    Raises:
        RuntimeError: If importing of the class fails.
    """
    try:
        module_name, class_name = policy_class_str.rsplit(".", 1)
        logging.info("import %s from %s" % (class_name, module_name))
        module = importlib.import_module(module_name)
        Policy = getattr(module, class_name)
    except Exception:
        raise RuntimeError(
            "Failed to import policy %s from module %s" % (class_name, module_name)
        )

    return Policy


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "task",
        type=str,
        choices=["push", "lift"],
        help="Which task to evaluate ('push' or 'lift').",
    )
    parser.add_argument(
        "policy_class",
        type=str,
        help="Name of the policy class (something like 'package.module.Class').",
    )
    parser.add_argument(
        "--visualization",
        "-v",
        action="store_true",
        help="Enable visualization of environment.",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=64,
        help="Number of episodes to run. Default: %(default)s",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        metavar="FILENAME",
        help="Save results to a JSON file.",
    )
    args = parser.parse_args()

    if args.task == "push":
        env_name = "trifinger-cube-push-sim-expert-v0"
    elif args.task == "lift":
        env_name = "trifinger-cube-lift-sim-expert-v0"
    else:
        print("Invalid task %s" % args.task)
        return 1

    Policy = load_policy_class(args.policy_class)

    policy_config = Policy.get_policy_config()

    if policy_config.flatten_obs:
        print("Using flattened observations")
    else:
        print("Using structured observations")

    env = typing.cast(
        TriFingerDatasetEnv,
        gym.make(
            env_name,
            disable_env_checker=True,
            visualization=args.visualization,
            flatten_obs=policy_config.flatten_obs,
            image_obs=policy_config.image_obs,
        ),
    )

    policy = Policy(env.action_space, env.observation_space, env.sim_env.episode_length)

    evaluation = Evaluation(env)
    eval_res = evaluation.evaluate(policy=policy, n_episodes=args.n_episodes)
    json_result = json.dumps(eval_res, indent=4)

    print("Evaluation result: ")
    print(json_result)

    if args.output:
        args.output.write_text(json_result)

    return 0


if __name__ == "__main__":
    sys.exit(main())
