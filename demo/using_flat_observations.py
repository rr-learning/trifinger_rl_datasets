"""How to use the provided index ranges to work with flat observations."""

import argparse
import json

import gymnasium as gym

import trifinger_rl_datasets  # noqa


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "--env",
        type=str,
        default="trifinger-cube-push-real-expert-v0",
        help="Name of dataset environment to load.",
    )
    argparser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to data directory. If not set, the default data directory '~/.trifinger_rl_datasets' is used.",
    )
    args = argparser.parse_args()

    env = gym.make(args.env, disable_env_checker=True, data_dir=args.data_dir)

    # load only a subset of obervations, actions and rewards
    n_observations = 10
    dataset = env.get_dataset(rng=(0, 750))

    # get mapping from observation components to index ranges
    obs_indices, obs_shapes = env.get_obs_indices()

    print("Observation component indices: ", json.dumps(obs_indices, indent=4))
    print("Observation component shapes: ", json.dumps(obs_shapes, indent=4))

    # print cube position over time
    print("Cube position over time: ")
    for i in range(n_observations):
        index_range = obs_indices["camera_observation"]["object_position"]
        print(dataset["observations"][i][slice(*index_range)])
