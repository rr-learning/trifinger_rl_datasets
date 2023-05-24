"""Load part of a datset defined by a range of transitions."""


import argparse

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
        "--range",
        type=int,
        nargs=2,
        default=[1000, 2000],
        help="Range of timesteps to load image data for.",
    )
    argparser.add_argument(
        "--zarr-path", type=str, default=None, help="Path to Zarr file to load."
    )
    argparser.add_argument(
        "--flatten-obs", action="store_true", help="Flatten observations."
    )
    argparser.add_argument(
        "--data-dir", type=str, default=None, help="Path to data directory."
    )
    args = argparser.parse_args()

    env = gym.make(
        args.env,
        disable_env_checker=True,
        flatten_obs=args.flatten_obs,
        data_dir=args.data_dir,
    )

    # load only a subset of obervations, actions and rewards
    dataset = env.get_dataset(rng=tuple(args.range), zarr_path=args.zarr_path)

    n_observations = len(dataset["observations"])
    print("Number of observations: ", n_observations)

    assert dataset["actions"].shape[0] == n_observations
    assert dataset["rewards"].shape[0] == n_observations
