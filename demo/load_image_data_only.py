"""Load image data from Zarr file and display it."""


import argparse

import cv2
import gymnasium as gym
import numpy as np

import trifinger_rl_datasets  # noqa


def show_images(images, timestep_dimension):
    """Show loaded images.

    Args:
        images (np.ndarray):  Array containing the image data.
        no_timestep_dimension (bool):  If False, the first dimension of the
            image_data array is assumed to correspond to camera timesteps.
            Otherwise, the first dimension is assumed to correspond to
            images."""

    if timestep_dimension:
        n_timesteps, n_cameras, n_channels, height, width = images.shape
        output_image = np.zeros(
            (n_cameras * height, n_timesteps * width, n_channels), dtype=np.uint8
        )
    else:
        n_images, n_channels, height, width = images.shape
        output_image = np.zeros((height, n_images * width, n_channels), dtype=np.uint8)
    # loop over tuples containing images from all cameras at one timestep
    for i, image_s in enumerate(images):
        if timestep_dimension:
            # concatenate images from all cameras along the height axis
            image_s = np.concatenate(image_s, axis=1)
        # change to (height, width, channels) format for cv2
        image_s = np.transpose(image_s, (1, 2, 0))
        # copy column of camera images to output image
        output_image[:, i * width : (i + 1) * width, ...] = image_s
    # convert RGB to BGR for cv2
    output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

    if timestep_dimension:
        legend = "Each column corresponds to the camera images at one timestep."
    else:
        legend = "Camera images"
    print(legend)
    print("Press any key to close window.")
    cv2.imshow(legend, output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "--env",
        type=str,
        default="trifinger-cube-push-real-expert-image-mini-v0",
        help="Name of dataset environment to load.",
    )
    argparser.add_argument(
        "--n-timesteps",
        type=int,
        default=10,
        help="Number of camera timesteps to load image data for.",
    )
    argparser.add_argument(
        "--zarr-path", type=str, default=None, help="Path to Zarr file to load."
    )
    argparser.add_argument(
        "--do-not-show-images",
        action="store_true",
        help="Do not show images if this is set.",
    )
    argparser.add_argument(
        "--no-timestep-dimension",
        dest="timestep_dimension",
        action="store_false",
        help="Do not include the timestep dimension in the output array.",
    )
    argparser.add_argument(
        "--data-dir", type=str, default=None, help="Path to data directory."
    )
    args = argparser.parse_args()

    # create environment
    env = gym.make(args.env, disable_env_checker=True, data_dir=args.data_dir)

    # get information about image data
    image_stats = env.get_image_stats(zarr_path=args.zarr_path)
    print("Image dataset:")
    for key, value in image_stats.items():
        print(f"{key}: {value}")

    # load image data
    print(f"Loading {args.n_timesteps} timesteps of image data.")
    from time import time

    t0 = time()
    images = env.get_image_data(
        # images from 3 cameras for each timestep
        rng=(0, 3 * args.n_timesteps),
        zarr_path=args.zarr_path,
        timestep_dimension=args.timestep_dimension,
    )
    print(f"Loading took {time() - t0:.3f} seconds.")

    # show images
    if not args.do_not_show_images:
        show_images(images, args.timestep_dimension)
