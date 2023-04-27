"""Demo including camera images in the observation."""


import argparse

import cv2
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
        "--flatten-obs", action="store_true", help="Flattens observations if set."
    )
    argparser.add_argument(
        "--no-visualization",
        dest="visualization",
        action="store_false",
        help="Disables visualization, i.e., rendering of the environment in a GUI.",
    )
    argparser.add_argument(
        "--data-dir", type=str, default=None, help="Path to data directory."
    )
    args = argparser.parse_args()

    env = gym.make(
        args.env,
        disable_env_checker=True,
        visualization=args.visualization,
        # include camera images in the observation
        image_obs=True,
        flatten_obs=args.flatten_obs,
        data_dir=args.data_dir,
    )
    obs, info = env.reset()
    truncated = False
    terminated = False

    # do one step in environment to get observations
    obs, rew, terminated, truncated, info = env.step(env.action_space.sample())

    if args.flatten_obs:
        # obs is a tuple containing an array with all observations but the images
        # and an array containing the images
        other_obs, images = obs
        print("Shape of all observations except images: ", other_obs.shape)
        print("Shape of images: ", images.shape)
    else:
        # obs is a nested dictionary if flatten_obs is False
        images = obs["camera_observation"]["images"]
        print("Shape of images: ", images.shape)

    # change to (height, width, channels) format for cv2
    images = np.transpose(images, (0, 2, 3, 1))
    images = np.concatenate(images, axis=0)
    # convert RGB to BGR for cv2
    output_image = cv2.cvtColor(images, cv2.COLOR_RGB2BGR)
    # show images from last time step
    cv2.imshow("Camera images", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
