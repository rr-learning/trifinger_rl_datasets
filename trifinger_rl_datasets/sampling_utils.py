"""Utils for sampling of cube pose."""

import numpy as np
from scipy.spatial.transform import Rotation

from trifinger_simulation.tasks.move_cube import (
    _CUBE_WIDTH,
    _ARENA_RADIUS,
    _base_orientations,
    Pose,
)


def random_yaw_orientation():
    # first "roll the die" to see which face is pointing upward
    up_face = np.random.choice(range(len(_base_orientations)))
    up_face_rot = _base_orientations[up_face]
    # then draw a random yaw rotation
    yaw_angle = np.random.uniform(0, 2 * np.pi)
    yaw_rot = Rotation.from_euler("z", yaw_angle)
    # and combine them
    orientation = yaw_rot * up_face_rot
    return yaw_angle, orientation.as_quat()


def random_xy(cube_yaw):
    """Sample an xy position for cube which maximally covers arena.

    In particular, the cube can touch the barrier for all yaw anels."""

    theta = np.random.uniform(0, 2 * np.pi)

    # Minimum distance of cube center from arena boundary
    min_dist = (
        _CUBE_WIDTH
        / np.sqrt(2)
        * max(
            abs(np.sin(0.25 * np.pi + cube_yaw - theta)),
            abs(np.cos(0.25 * np.pi + cube_yaw - theta)),
        )
    )

    # sample uniform position in circle
    # (https://stackoverflow.com/a/50746409)
    radius = (_ARENA_RADIUS - min_dist) * np.sqrt(np.random.random())

    # x,y-position of the cube
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    return x, y


def sample_initial_cube_pose():
    yaw_angle, orientation = random_yaw_orientation()
    x, y = random_xy(yaw_angle)
    z = _CUBE_WIDTH / 2
    goal = Pose()
    goal.position = np.array((x, y, z))
    goal.orientation = orientation
    return goal
