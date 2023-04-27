"""Demo for doing a rollout in simulation."""


import argparse

import gymnasium as gym
import numpy as np

import trifinger_rl_datasets  # noqa


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "--env",
        type=str,
        default="trifinger-cube-push-sim-expert-v0",
        help="Name of dataset environment to load.",
    )
    argparser.add_argument(
        "--no-visualization",
        dest="visualization",
        action="store_false",
        help="Disables visualization, i.e., rendering of the environment in a GUI.",
    )
    args = argparser.parse_args()

    env = gym.make(args.env, disable_env_checker=True, visualization=args.visualization)
    obs, info = env.reset()
    truncated = False

    while not truncated:
        obs, rew, terminated, truncated, info = env.step(env.action_space.sample())
