from pathlib import Path
from time import sleep, time
from typing import Tuple, Dict, Any, Optional
import logging

import cv2
import gymnasium as gym
import numpy as np
from pybullet import ER_TINY_RENDERER

import trifinger_simulation
import trifinger_simulation.visual_objects
from trifinger_simulation import trifingerpro_limits
import trifinger_simulation.tasks.move_cube as task

from .sampling_utils import sample_initial_cube_pose
from .utils import to_quat, get_keypoints_from_pose


class CameraWrapper:
    """Simple wrapper around camera array to change default renderer."""

    def __init__(self, camera):
        self.camera = camera

    def get_images(self, renderer=ER_TINY_RENDERER):
        return self.camera.get_images(renderer)


class SimTriFingerCubeEnv(gym.Env):
    """
    Gym environment for simulated manipulation of a cube with a TriFingerPro platform.
    """

    _initial_finger_position = [0.0, 0.9, -2.0] * 3
    _max_fingertip_vel = 5.0
    # parameters of reward function
    _kernel_reward_weight = 4.0
    _logkern_scale = 30
    _logkern_offset = 2
    # for how long to play the resetting trajectory
    _reset_trajectory_length = 18700
    # how many robot steps per environment step
    _step_size = 20

    def __init__(
        self,
        episode_length: int = 15,
        difficulty: int = 4,
        keypoint_obs: bool = True,
        obs_action_delay: int = 0,
        reward_type: str = "dense",
        visualization: bool = False,
        real_time: bool = True,
        image_obs: bool = False,
        camera_config_robot: int = 1,
    ):
        """
        Args:
            episode_length (int): How often step will run before done is True.
            keypoint_obs (bool): Whether to give keypoint observations for
                pose in addition to position and quaternion.
            obs_action_delay (int): Delay between arrival of an observation
                and application of the action computed from this
                observation in milliseconds.
            reward_type (str): Which reward to use. Can be 'dense' or 'sparse'.
            visualization (bool): If true, the PyBullet GUI is run for visualization.
            real_time (bool): If true, the environment is stepped in real
                time instead of as fast as possible (ignored if visualization is
                disabled).
            image_obs (bool): If true, the camera images are returned as part
                of the observation.
            camera_config_robot (int): ID of the robot to retrieve camera
                configs from. Only used if image_obs is True.
        """
        # Basic initialization
        # ====================

        self.logger = logging.getLogger("trifinger_rl_datasets.SimTriFingerCubeEnv")

        assert (
            obs_action_delay < self._step_size
        ), "Delay between retrieval of observation and sending of next \
            action has to be smaller than step size (20 ms)."

        # will be initialized in reset()
        self.platform: Optional[trifinger_simulation.TriFingerPlatform] = None

        self.episode_length = episode_length
        self.difficulty = difficulty
        self.keypoint_obs = keypoint_obs
        self.n_keypoints = 8
        self.obs_action_delay = obs_action_delay
        self.reward_type = reward_type
        self.visualization = visualization
        self.real_time = real_time
        self.image_obs = image_obs
        self.camera_config_robot = camera_config_robot

        # load trajectory that is played back for resetting the cube
        trajectory_file_path = (
            Path(__file__).resolve().parent
            / "data"
            / "trifingerpro_shuffle_cube_trajectory_fast.npy"
        )
        with open(trajectory_file_path, "rb") as f:
            self._cube_reset_traj = np.load(f)

        # simulated robot has robot ID 0
        self.robot_id = 0

        if image_obs:
            # create camera object
            camera_config_dir = Path(__file__).resolve().parent / "data"
            calib_filename_pattern = f"r{self.camera_config_robot}_" + "camera{id}.yml"
            self.camera = trifinger_simulation.camera.create_trifinger_camera_array_from_config(
                camera_config_dir, calib_filename_pattern=calib_filename_pattern
            )
        else:
            self.camera = None

        # Create the action and observation spaces
        # ========================================

        robot_torque_space = gym.spaces.Box(
            low=trifingerpro_limits.robot_torque.low,
            high=trifingerpro_limits.robot_torque.high,
        )
        robot_position_space = gym.spaces.Box(
            low=trifingerpro_limits.robot_position.low,
            high=trifingerpro_limits.robot_position.high,
            dtype=np.float32,
        )
        robot_velocity_space = gym.spaces.Box(
            low=trifingerpro_limits.robot_velocity.low,
            high=trifingerpro_limits.robot_velocity.high,
            dtype=np.float32,
        )
        robot_fingertip_force_space = gym.spaces.Box(
            low=np.zeros(trifingerpro_limits.n_fingers),
            high=np.ones(trifingerpro_limits.n_fingers),
            dtype=np.float32,
        )
        robot_fingertip_pos_space = gym.spaces.Box(
            low=np.array([[-0.6, -0.6, 0.0]] * trifingerpro_limits.n_fingers),
            high=np.array([[0.6, 0.6, 0.6]] * trifingerpro_limits.n_fingers),
            dtype=np.float32,
        )
        robot_fingertip_vel_space = gym.spaces.Box(
            low=np.array(
                [[-self._max_fingertip_vel] * 3] * trifingerpro_limits.n_fingers
            ),
            high=np.array(
                [[self._max_fingertip_vel] * 3] * trifingerpro_limits.n_fingers
            ),
            dtype=np.float32,
        )
        robot_id_space = gym.spaces.Box(low=0, high=20, shape=(1,), dtype=np.int_)

        # camera observation space
        camera_obs_space_dict: Dict[str, gym.Space] = {
            "object_position": gym.spaces.Box(
                low=trifingerpro_limits.object_position.low,
                high=trifingerpro_limits.object_position.high,
                dtype=np.float32,
            ),
            "object_orientation": gym.spaces.Box(
                low=trifingerpro_limits.object_orientation.low,
                high=trifingerpro_limits.object_orientation.high,
                dtype=np.float32,
            ),
            "delay": gym.spaces.Box(low=0.0, high=0.30, shape=(1,), dtype=np.float32),
            "confidence": gym.spaces.Box(
                low=0.0, high=1.0, shape=(1,), dtype=np.float32
            ),
        }
        if self.keypoint_obs:
            camera_obs_space_dict["object_keypoints"] = gym.spaces.Box(
                low=np.array([[-0.6, -0.6, 0.0]] * self.n_keypoints),
                high=np.array([[0.6, 0.6, 0.3]] * self.n_keypoints),
                dtype=np.float32,
            )
        if self.image_obs:
            n_cameras = len(self.camera.cameras)
            images_space = gym.spaces.Box(
                low=0,
                high=255,
                shape=(
                    n_cameras,
                    3,
                    self.camera.cameras[0]._output_height,
                    self.camera.cameras[0]._output_width,
                ),
                dtype=np.uint8,
            )
            camera_obs_space_dict["images"] = images_space
        camera_obs_space = gym.spaces.Dict(camera_obs_space_dict)

        # goal space
        if self.difficulty == 4:
            if self.keypoint_obs:
                goal_space = gym.spaces.Dict(
                    {"object_keypoints": camera_obs_space["object_keypoints"]}
                )
            else:
                goal_space = gym.spaces.Dict(
                    {
                        k: camera_obs_space[k]
                        for k in ["object_position", "object_orientation"]
                    }
                )
        else:
            goal_space = gym.spaces.Dict(
                {"object_position": camera_obs_space["object_position"]}
            )

        # action space
        self.action_space = robot_torque_space
        self._initial_action = trifingerpro_limits.robot_torque.default

        # NOTE: The order of dictionary items matters as it determines how
        # the observations are flattened/unflattened. The observation space
        # is therefore sorted by key.

        def sort_by_key(d):
            return {
                k: (
                    gym.spaces.Dict(sort_by_key(v.spaces))
                    if isinstance(v, gym.spaces.Dict)
                    else v
                )
                for k, v in sorted(d.items(), key=lambda item: item[0])
            }

        # complete observation space
        self.observation_space = gym.spaces.Dict(
            sort_by_key(
                {
                    "robot_observation": gym.spaces.Dict(
                        {
                            "position": robot_position_space,
                            "velocity": robot_velocity_space,
                            "torque": robot_torque_space,
                            "fingertip_force": robot_fingertip_force_space,
                            "fingertip_position": robot_fingertip_pos_space,
                            "fingertip_velocity": robot_fingertip_vel_space,
                            "robot_id": robot_id_space,
                        }
                    ),
                    "camera_observation": camera_obs_space,
                    "action": self.action_space,
                    "desired_goal": goal_space,
                    "achieved_goal": goal_space,
                }
            )
        )

        self._old_camera_obs: Optional[Dict[str, Any]] = None
        self.t_obs: int = 0

        # Count consecutive steps where timing is violated (to decide when the show a
        # warning)
        self._timing_violation_counter = 0

    def _kernel_reward(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray
    ) -> float:
        """Compute reward by evaluating a logistic kernel on the pairwise distance of
        points.

        Parameters can be either a 1 dim. array of size 3 (positions) or a two dim.
        array with last dim. of size 3 (keypoints)

        Args:
            achieved_goal: Position or keypoints of current pose of the object.
            desired_goal: Position or keypoints of goal pose of the object.
        """

        diff = achieved_goal - desired_goal
        dist = np.linalg.norm(diff, axis=-1)
        scaled = self._logkern_scale * dist
        # Use logistic kernel
        rew = self._kernel_reward_weight * np.mean(
            1.0 / (np.exp(scaled) + self._logkern_offset + np.exp(-scaled))
        )
        return rew

    def _append_desired_action(self, robot_action):
        """Append desired action to queue and wait if real time is enabled."""

        t = self.platform.append_desired_action(robot_action)
        if self.visualization and self.real_time:
            sleep(max(0.001 - (time() - self.time_of_last_step), 0.0))
            self.time_of_last_step = time()
        return t

    def compute_reward(
        self, achieved_goal: dict, desired_goal: dict, info: dict
    ) -> float:
        """Compute the reward for the given achieved and desired goal.

        Args:
            achieved_goal: Current pose of the object.
            desired_goal: Goal pose of the object.
            info: An info dictionary containing a field "time_index" which
                contains the time index of the achieved_goal.

        Returns:
            The reward that corresponds to the provided achieved goal w.r.t. to
            the desired goal.
        """

        if self.reward_type == "dense":
            if self.difficulty == 4:
                # Use full keypoints if available as only difficulty 4 considers
                # orientation
                return self._kernel_reward(
                    achieved_goal["object_keypoints"], desired_goal["object_keypoints"]
                )
            else:
                # use position for all other difficulties
                return self._kernel_reward(
                    achieved_goal["object_position"], desired_goal["object_position"]
                )
        elif self.reward_type == "sparse":
            return self.has_achieved(achieved_goal, desired_goal)
        else:
            raise NotImplementedError(
                f"Reward type {self.reward_type} is not supported"
            )

    def has_achieved(self, achieved_goal: dict, desired_goal: dict) -> bool:
        """Determine whether goal pose is achieved."""
        POSITION_THRESHOLD = 0.02
        ANGLE_THRESHOLD_DEG = 22.0

        desired = desired_goal
        achieved = achieved_goal
        position_diff = np.linalg.norm(
            desired["object_position"] - achieved["object_position"]
        )
        # cast from np.bool_ to bool to make mypy happy
        position_check = bool(position_diff < POSITION_THRESHOLD)

        if self.difficulty < 4:
            return position_check
        else:
            a = to_quat(desired["object_orientation"])
            b = to_quat(achieved["object_orientation"])
            b_conj = b.conjugate()
            quat_prod = a * b_conj
            norm = np.linalg.norm([quat_prod.x, quat_prod.y, quat_prod.z])
            norm = min(norm, 1.0)  # type: ignore
            angle = 2.0 * np.arcsin(norm)
            orientation_check = angle < 2.0 * np.pi * ANGLE_THRESHOLD_DEG / 360.0

            return position_check and orientation_check

    def _check_action(self, action):
        low_check = self.action_space.low <= action
        high_check = self.action_space.high >= action
        return np.all(np.logical_and(low_check, high_check))

    def step(
        self, action: np.ndarray, preappend_actions: bool = True
    ) -> Tuple[dict, float, bool, bool, dict]:
        """Run one timestep of the environment's dynamics.

        When end of episode is reached, you are responsible for calling
        ``reset()`` to reset this environment's state.

        Args:
            action: An action provided by the agent
            preappend_actions (bool): Whether to already append actions that
                will be executed during obs-action delay to action queue.

        Returns:
            tuple:
            - observation (dict): agent's observation of the current environment.
            - reward (float): amount of reward returned after previous action.
            - terminated (bool): whether the MDP has reached a terminal state. If true,
                the user needs to call `reset()`.
            - truncated (bool): Whether the truncation condition outside the scope
                of the MDP is satisfied. For this environment this corresponds to a
                timeout. If true, the user needs to call `reset()`.
            - info (dict): info dictionary containing the current time index.
        """
        if self.platform is None:
            raise RuntimeError("Call `reset()` before starting to step.")

        if not self._check_action(action):
            raise ValueError("Given action is not contained in the action space.")

        self.step_count += 1

        # get robot action
        robot_action = self._gym_action_to_robot_action(action)

        # check timing and show a warning/error if delayed
        # do not check in first iteration as no time index is available yet (would lead
        # to dead-lock)
        if self.t_obs > 0:
            t_now = self.platform.get_current_timeindex()
            t_expected = self.t_obs + self.obs_action_delay
            if t_now > t_expected:
                self._timing_violation_counter += 1
                extreme = t_now > self.t_obs + self._step_size

                if extreme or self._timing_violation_counter >= 3:
                    delay = t_now - t_expected
                    self.logger.warning(
                        f"Control loop got delayed by {delay} ms."
                        " The action will be applied for a shorter time to catch up."
                        " Please check if your policy is fast enough (max. computation"
                        f" time should be <{1 + self.obs_action_delay} ms)."
                    )

                if extreme:
                    self.logger.error(
                        "ERROR: Control loop got delayed by more than a full step."
                        "  Timing of the episode will be significantly affected!"
                    )
            else:
                self._timing_violation_counter = 0

        # send new action to robot until new observation is to be provided
        # Note that by initially setting t the way it is, it is ensured that the loop
        # always runs at least one iteration, even if the actual time step is already
        # ahead by more than one step size.
        t = self.t_obs + self.obs_action_delay
        while t < self.t_obs + self._step_size:
            t = self._append_desired_action(robot_action)
        # time of the new observation
        self.t_obs = t

        observation, info = self._create_observation(self.t_obs, action)
        reward = self.compute_reward(
            observation["achieved_goal"], observation["desired_goal"], info
        )
        truncated = self.step_count >= self.episode_length

        if not truncated and preappend_actions:
            # Append action to action queue of robot for as many time
            # steps as the obs_action_delay dictates. This gives the
            # user time to evaluate the policy.
            for _ in range(self.obs_action_delay):
                self._append_desired_action(robot_action)

        return observation, reward, False, truncated, info

    def reset(  # type: ignore
        self, preappend_actions: bool = True
    ):
        """Reset the environment."""

        super().reset()

        # hard-reset simulation
        del self.platform

        # initialize simulation
        initial_robot_position = trifingerpro_limits.robot_position.default
        initial_object_pose = sample_initial_cube_pose()
        initial_object_pose.position[2] += 0.0005  # avoid negative z of keypoint
        self.platform = trifinger_simulation.TriFingerPlatform(
            visualization=self.visualization,
            initial_robot_position=initial_robot_position,
            initial_object_pose=initial_object_pose,
            enable_cameras=self.image_obs,
        )
        if self.image_obs:
            # overwrite camera with wrapped version which uses software rendering
            self.platform.tricamera = CameraWrapper(self.camera)
            first_camera_obs = self.platform._get_current_camera_observation(0)
            self.platform._delayed_camera_observation = first_camera_obs
            self.platform._camera_observation_t = first_camera_obs
        # sample goal
        self.active_goal = task.sample_goal(difficulty=self.difficulty)
        # visualize the goal (but not if image observations are used)
        if self.visualization and not self.image_obs:
            self.goal_marker = trifinger_simulation.visual_objects.CubeMarker(
                width=task._CUBE_WIDTH,
                position=self.active_goal.position,
                orientation=self.active_goal.orientation,
                pybullet_client_id=self.platform.simfinger._pybullet_client_id,
            )
        self.step_count = 0
        self.time_of_last_step = time()
        # need to already do one step to get initial observation
        self.t_obs = 0
        obs, _, _, _, info = self.step(
            self._initial_action, preappend_actions=preappend_actions
        )
        info = {"time_index": -1}

        return obs, info

    def reset_fingers(self, reset_wait_time: int = 3000):
        """Reset fingers to initial position.

        This resets neither the frontend nor the cube. This method is
        supposed to be used for 'soft resets' between episodes in one
        job.
        """
        assert self.platform is not None, "Environment is not initialised."

        action = self.platform.Action(position=self._initial_finger_position)
        for _ in range(reset_wait_time):
            t = self._append_desired_action(action)
        self.t_obs = t
        # reset step_count even though this is not a full reset
        self.step_count = 0
        # block until reset wait time has passed and return observation
        obs, info = self._create_observation(t, self._initial_action)
        return obs, info

    def sample_new_goal(self, goal=None):
        """Sample a new desired goal."""
        if goal is None:
            self.active_goal = task.sample_goal(difficulty=self.difficulty)
        else:
            self.active_goal.position = np.array(goal["position"], dtype=np.float32)
            self.active_goal.orientation = np.array(
                goal["orientation"], dtype=np.float32
            )

        # update goal visualisation
        if self.visualization and not self.image_obs:
            self.goal_marker.set_state(
                self.active_goal.position, self.active_goal.orientation
            )

    def _get_pose_delay(self, camera_observation, t):
        """Get delay between when the object pose was captured and now."""

        return t / 1000.0 - camera_observation.cameras[0].timestamp

    def _clip_observation(self, obs):
        """Clip observation."""

        def clip_recursively(o, space):
            for k, v in space.spaces.items():
                if isinstance(v, gym.spaces.Box):
                    np.clip(o[k], v.low, v.high, dtype=v.dtype, out=o[k])
                else:
                    clip_recursively(o[k], v)

        clip_recursively(obs, self.observation_space)

    def _create_observation(self, t: int, action: np.ndarray) -> Tuple[dict, dict]:
        assert self.platform is not None, "Environment is not initialised."

        robot_observation = self.platform.get_robot_observation(t)
        camera_observation = self.platform.get_camera_observation(t)
        object_observation = camera_observation.object_pose

        info: Dict[str, Any] = {"time_index": t}

        # camera observation
        camera_obs_processed = {
            "object_position": object_observation.position.astype(np.float32),
            "object_orientation": object_observation.orientation.astype(np.float32),
            # time elapsed since capturing of pose in seconds
            "delay": np.array(
                [self._get_pose_delay(camera_observation, t)], dtype=np.float32
            ),
            "confidence": np.array([object_observation.confidence], dtype=np.float32),
        }
        if self.image_obs:
            if len(camera_observation.cameras[0].image.shape) == 2:
                # images from real platform have to be debayered
                images = np.array([cv2.cvtColor(cam.image, cv2.COLOR_BAYER_BG2RGB) for cam in camera_observation.cameras])
            else:
                # RGB camera images created with software renderer
                # (using openGL requires GUI to run)
                images = np.array([cam.image for cam in camera_observation.cameras])
            # convert to channel first
            images = np.transpose(images, (0, 3, 1, 2))
            camera_obs_processed["images"] = images
        if self.keypoint_obs:
            camera_obs_processed["object_keypoints"] = get_keypoints_from_pose(
                object_observation
            )
        if self._old_camera_obs is not None:
            # handle quaternion flipping
            q_sum = (
                self._old_camera_obs["object_orientation"]
                + camera_obs_processed["object_orientation"]
            )
            if np.linalg.norm(q_sum) < 0.2:
                camera_obs_processed["object_orientation"] = -camera_obs_processed[
                    "object_orientation"
                ]
        self._old_camera_obs = camera_obs_processed

        # goal represented as position and orientation
        desired_goal_pos_ori = {
            "object_position": self.active_goal.position.astype(np.float32),
            "object_orientation": self.active_goal.orientation.astype(np.float32),
        }
        achieved_goal_pos_ori = {
            "object_position": camera_obs_processed["object_position"],
            "object_orientation": camera_obs_processed["object_orientation"],
        }
        # goal as shown to agent
        if self.difficulty == 4:
            if self.keypoint_obs:
                desired_goal = {
                    "object_keypoints": get_keypoints_from_pose(self.active_goal)
                }
                achieved_goal = {
                    "object_keypoints": camera_obs_processed["object_keypoints"]
                }
            else:
                desired_goal = desired_goal_pos_ori
                achieved_goal = achieved_goal_pos_ori
        else:
            desired_goal = {
                "object_position": self.active_goal.position.astype(np.float32)
            }
            achieved_goal = {"object_position": camera_obs_processed["object_position"]}

        # fingertip positions and velocities
        fingertip_position, fingertip_velocity = self.platform.forward_kinematics(
            robot_observation.position, robot_observation.velocity
        )
        fingertip_position = np.array(fingertip_position, dtype=np.float32)
        fingertip_velocity = np.array(fingertip_velocity, dtype=np.float32)

        observation = {
            "robot_observation": {
                "position": robot_observation.position.astype(np.float32),
                "velocity": robot_observation.velocity.astype(np.float32),
                "torque": robot_observation.torque.astype(np.float32),
                "fingertip_force": robot_observation.tip_force.astype(np.float32),
                "fingertip_position": fingertip_position,
                "fingertip_velocity": fingertip_velocity,
                "robot_id": np.array([self.robot_id], dtype=np.int_),
            },
            "camera_observation": camera_obs_processed,
            "action": action.astype(np.float32),
            "desired_goal": desired_goal,
            "achieved_goal": achieved_goal,
        }
        # clip observation
        self._clip_observation(observation)

        has_achieved = self.has_achieved(achieved_goal_pos_ori, desired_goal_pos_ori)
        info["has_achieved"] = has_achieved
        info["desired_goal"] = desired_goal_pos_ori

        return observation, info

    def _gym_action_to_robot_action(self, gym_action: np.ndarray):
        assert self.platform is not None, "Environment is not initialised."

        # robot action is torque
        robot_action = self.platform.Action(torque=gym_action)
        return robot_action

    def render(self, mode: str = "human"):
        pass

    def reset_cube(self):
        """Replay a recorded trajectory to move cube to center of arena."""

        for position in self._cube_reset_traj[: self._reset_trajectory_length : 2]:
            robot_action = self.platform.Action(position=position)
            self._append_desired_action(robot_action)
