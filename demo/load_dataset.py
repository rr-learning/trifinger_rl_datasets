"""Load a complete dataset into memory and perform a rollout."""

import argparse

import gymnasium as gym

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
        "--data-dir",
        type=str,
        default=None,
        help="Path to data directory.If not set, the default data directory '~/.trifinger_rl_datasets' is used.",
    )
    args = argparser.parse_args()

    env = gym.make(
        args.env,
        disable_env_checker=True,
        visualization=True,  # enable visualization
        data_dir=args.data_dir,
    )
    dataset = env.get_dataset()

    n_transitions = len(dataset["observations"])
    print("Number of transitions: ", n_transitions)

    assert dataset["actions"].shape[0] == n_transitions
    assert dataset["rewards"].shape[0] == n_transitions

    print("First observation: ", dataset["observations"][0])

    obs, info = env.reset()
    truncated = False
    terminated = False
    while not (truncated or terminated):
        obs, rew, terminated, truncated, info = env.step(env.action_space.sample())
