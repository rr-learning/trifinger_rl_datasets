"""Create video from camera images."""


import argparse

import cv2
import gymnasium as gym
import numpy as np

import trifinger_rl_datasets  # noqa


def create_video(
    env, output_path, camera_id, timestep_range, zarr_path, show_reward=True
):
    """Create video from camera images.

    Args:
        dataset (dict):  Dataset to load images from.
        output_path (str):  Output path for video file.
        camera_id (str):  ID of the camera for which to load images.
    """

    image_range = env.convert_timestep_to_image_index(np.array(timestep_range))
    # load relevant part of images in dataset
    images = env.get_image_data(
        # images from 3 cameras for each timestep
        rng=(image_range[0], image_range[1] + 3),
        zarr_path=zarr_path,
        timestep_dimension=True,
    )
    if show_reward:
        # load rewards for the specified timesteps
        image_indices = env.convert_timestep_to_image_index(
            np.arange(*tuple(timestep_range))
        )
        dataset = env.get_dataset(
            rng=(timestep_range[0], timestep_range[1] + 1), zarr_path=zarr_path
        )

    # select only images from the specified camera
    images = images[:, camera_id, ...]

    # create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 10
    video_writer = cv2.VideoWriter(
        output_path, fourcc, fps, (images.shape[-1], images.shape[-2])
    )

    max_bar_height = 50
    # loop over images
    for i, image in enumerate(images):
        # convert to channeel last format for cv2
        img = np.transpose(image, (1, 2, 0))
        if show_reward:
            # draw bar with height proportional to reward
            index = np.argmax(image_indices == i * 3 + image_range[0])
            reward = dataset["rewards"][index]
            img[img.shape[0] - max_bar_height :, 260:, :] = 150
            bar_height = int(reward * max_bar_height)
            img[img.shape[0] - bar_height :, 260:, 1] = 255
        # convert RGB to BGR for cv2
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # write image to video
        video_writer.write(img)

    # close video writer
    video_writer.release()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument("output_path", type=str, help="Path to output video file.")
    argparser.add_argument(
        "camera_id", type=int, help="ID of the camera for which to load images."
    )
    argparser.add_argument(
        "--env",
        type=str,
        default="trifinger-cube-push-real-expert-image-mini-v0",
        help="Name of dataset environment to load.",
    )
    argparser.add_argument(
        "--timestep-range",
        type=int,
        nargs=2,
        default=[0, 750],
        help="Range of timesteps (not camera timesteps) to load image data for.",
    )
    argparser.add_argument(
        "--zarr-path", type=str, default=None, help="Path to Zarr file to load."
    )
    argparser.add_argument(
        "--data-dir", type=str, default=None, help="Path to data directory."
    )
    argparser.add_argument(
        "--no-reward", action="store_true", help="Do not show reward bar. "
    )
    args = argparser.parse_args()

    env = gym.make(args.env, disable_env_checker=True, data_dir=args.data_dir)
    create_video(
        env,
        args.output_path,
        args.camera_id,
        args.timestep_range,
        args.zarr_path,
        not args.no_reward,
    )
