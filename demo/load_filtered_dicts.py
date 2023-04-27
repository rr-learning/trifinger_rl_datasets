import argparse
import gymnasium as gym

import trifinger_rl_datasets  # noqa


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demonstrate how to customize observation space by filtering."
    )
    parser.add_argument(
        "--env-name",
        type=str,
        default="trifinger-cube-push-sim-expert-v0",
        help="Name of the gym environment to load.",
    )
    parser.add_argument(
        "--do-not-filter-obs",
        action="store_true",
        help="Do not filter observations if this is set.",
    )
    parser.add_argument(
        "--flatten-obs",
        action="store_true",
        help="Flatten observations again after filtering if this is set.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to data directory.If not set, the default data directory '~/.trifinger_rl_datasets' is used.",
    )
    args = parser.parse_args()

    # Nested dictionary defines which observations to keep.
    # Everything that is not included or has value False
    # will be dropped.
    obs_to_keep = {
        "robot_observation": {
            "position": True,
            "velocity": True,
            "fingertip_force": False,
        },
        "camera_observation": {"object_keypoints": True},
    }
    env = gym.make(
        args.env_name,
        disable_env_checker=True,
        # enable visualization,
        visualization=True,
        # filter observations,
        obs_to_keep=None if args.do_not_filter_obs else obs_to_keep,
        # flatten observation
        flatten_obs=args.flatten_obs,
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
