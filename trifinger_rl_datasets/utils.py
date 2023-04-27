"""Utility methods for working with object poses and keypoints."""

import numpy as np
import quaternion


def to_quat(x):
    return np.quaternion(x[3], x[0], x[1], x[2])


def to_world_space(x_local, pose):
    """Transform point from local object coordinate system to world space.

    Args:
        x_local: Coordinates of point in local frame.
        pose: Object pose containing position and orientation.
    Returns:
        The coordinates in world space.
    """
    q_rot = to_quat(pose.orientation)
    transl = pose.position
    q_local = np.quaternion(0.0, x_local[0], x_local[1], x_local[2])
    q_global = q_rot * q_local * q_rot.conjugate()
    return transl + np.array([q_global.x, q_global.y, q_global.z])


def get_keypoints_from_pose(pose, num_keypoints=8, dimensions=(0.065, 0.065, 0.065)):
    """Calculate keypoints (coordinates of the corners of the cube) from pose.

    Args:
        pose: Object pose containing position and orientation of cube.
        num_keypoints: Number of keypoints to generate.
        dimensions: Dimensions of the cube.
    Returns:
        Array containing the keypoints.
    """
    keypoints = []
    for i in range(num_keypoints):
        # convert to binary representation
        str_kp = "{:03b}".format(i)
        # set components of keypoints according to digits in binary representation
        loc_kp = [
            (1.0 if str_kp[i] == "0" else -1.0) * 0.5 * d
            for i, d in enumerate(dimensions)
        ][::-1]
        glob_kp = to_world_space(loc_kp, pose)
        keypoints.append(glob_kp)

    return np.array(keypoints, dtype=np.float32)


def get_pose_from_keypoints(keypoints, dimensions=(0.065, 0.065, 0.065)):
    """Calculate pose (position, orientation) from keypoints.

    Args:
        keypoints: At least three keypoints representing the pose.
        dimensions: Dimensions of the cube.
    Returns:
        Tuple containing the coordinates of the cube center and a
        quaternion representing the orientation.
    """
    center = np.mean(keypoints, axis=0)
    kp_centered = np.array(keypoints) - center
    kp_scaled = kp_centered / np.array(dimensions) * 2.0

    loc_kps = []
    for i in range(3):
        # convert to binary representation
        str_kp = "{:03b}".format(i)
        # set components of keypoints according to digits in binary representation
        loc_kp = [(1.0 if str_kp[i] == "0" else -1.0) for i in range(3)][::-1]
        loc_kps.append(loc_kp)
    K_loc = np.transpose(np.array(loc_kps))
    K_loc_inv = np.linalg.inv(K_loc)
    K_glob = np.transpose(kp_scaled[0:3])
    R = np.matmul(K_glob, K_loc_inv)
    quat = quaternion.from_rotation_matrix(R)

    return center, np.array([quat.x, quat.y, quat.z, quat.w])
