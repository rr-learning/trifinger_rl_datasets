import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass

import gymnasium as gym
import numpy as np

ObservationType = typing.Union[np.ndarray, typing.Dict[str, typing.Any]]


@dataclass
class PolicyConfig:
    """Policy configuration specifying what kind of observations the policy expects.
    
    Args:
        flatten_obs:  If True, the policy expects observations as flattened arrays.
            Otherwise, it expects them as dictionaries.
        image_obs: If True, the policy expects the observations to contain camera
            images. Otherwise, images are not included. If images_obs is True and
            flatten_obs is True, the observation is a tuple containing the flattened
            observation excluding the images and the images in a numpy array. If 
            flatten_obs is False, the images are included in the observation
            dictionary.
            """

    flatten_obs: bool = True
    image_obs: bool = False


class PolicyBase(ABC):
    """Base class defining interface for policies."""

    def __init__(
        self, action_space: gym.Space, observation_space: gym.Space, episode_length: int
    ):
        """
        Args:
            action_space:  Action space of the environment.
            observation_space:  Observation space of the environment.
            episode_length:  Number of steps in one episode.
        """
        pass

    @staticmethod
    def get_policy_config() -> PolicyConfig:
        """Returns the policy configuration.
        
        This specifies what kind of observations the policy expects.
        """
        return PolicyConfig()

    def reset(self) -> None:
        """Will be called at the beginning of each episode."""
        pass

    @abstractmethod
    def get_action(self, observation: ObservationType) -> np.ndarray:
        """Returns action that is executed on the robot.

        Args:
            observation: Observation of the current time step.

        Returns:
            Action that is sent to the robot.
        """
        pass
