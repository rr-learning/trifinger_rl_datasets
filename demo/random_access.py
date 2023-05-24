"""Load small parts of dataset at random positions to test performance."""


import argparse
from time import time

import gymnasium as gym
import numpy as np

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
        "--n-parts",
        type=int,
        default=500,
        help="Number of contiguous parts to load from file.",
    )
    argparser.add_argument(
        "--part-size",
        type=int,
        default=10,
        help="Number of transitions to load per part.",
    )
    argparser.add_argument(
        "--zarr_path", type=str, default=None, help="Path to Zarr file to load."
    )
    argparser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to data directory.If not set, the default data directory '~/.trifinger_rl_datasets' is used.",
    )
    args = argparser.parse_args()

    # create environment
    env = gym.make(args.env, disable_env_checker=True, data_dir=args.data_dir)

    stats = env.get_dataset_stats(zarr_path=args.zarr_path)
    print("Number of timesteps in dataset: ", stats["n_timesteps"])

    # load subsets of the dataset at random positions
    indices = []
    for i in range(args.n_parts):
        start = np.random.randint(0, stats["n_timesteps"] - args.part_size)
        if args.part_size == 1:
            indices.append(start)
        else:
            indices.extend(range(start, start + args.part_size))
    indices = np.array(indices)
    t0 = time()
    part = env.get_dataset(indices=indices, zarr_path=args.zarr_path)
    t1 = time()
    print(f"Loaded {args.n_parts} parts of size {args.part_size} in {t1 - t0:.2f} s")

    print("Observation shape: ", part["observations"].shape)
    print("Action shape: ", part["actions"].shape)
