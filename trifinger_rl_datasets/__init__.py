__version__ = "1.0.0"

from gymnasium.envs.registration import register

from .dataset_env import TriFingerDatasetEnv
from .evaluation import Evaluation
from .policy_base import PolicyBase, PolicyConfig


base_url = "https://robots.real-robot-challenge.com/public/trifinger_rl_datasets/"

dataset_names = [
    "trifinger-cube-push-real-expert-v0",
    "trifinger-cube-push-real-expert-image-v0",
    "trifinger-cube-push-real-weak-n-expert-v0",
    "trifinger-cube-push-real-weak-n-expert-image-v0",
    "trifinger-cube-push-real-half-expert-v0",
    "trifinger-cube-push-real-half-expert-image-v0",
    "trifinger-cube-push-real-mixed-v0",
    "trifinger-cube-push-real-mixed-image-v0",
    "trifinger-cube-push-sim-expert-v0",
    "trifinger-cube-push-sim-expert-image-v0",
    "trifinger-cube-push-sim-weak-n-expert-v0",
    "trifinger-cube-push-sim-weak-n-expert-image-v0",
    "trifinger-cube-push-sim-half-expert-v0",
    "trifinger-cube-push-sim-half-expert-image-v0",
    "trifinger-cube-push-sim-mixed-v0",
    "trifinger-cube-push-sim-mixed-image-v0",
    "trifinger-cube-lift-real-smooth-expert-v0",
    "trifinger-cube-lift-real-smooth-expert-image-v0",
    "trifinger-cube-lift-real-expert-v0",
    "trifinger-cube-lift-real-expert-image-v0",
    "trifinger-cube-lift-real-weak-n-expert-v0",
    "trifinger-cube-lift-real-weak-n-expert-image-v0",
    "trifinger-cube-lift-real-half-expert-v0",
    "trifinger-cube-lift-real-half-expert-image-v0",
    "trifinger-cube-lift-real-mixed-v0",
    "trifinger-cube-lift-real-mixed-image-v0",
    "trifinger-cube-lift-sim-expert-v0",
    "trifinger-cube-lift-sim-expert-image-v0",
    "trifinger-cube-lift-sim-weak-n-expert-v0",
    "trifinger-cube-lift-sim-weak-n-expert-image-v0",
    "trifinger-cube-lift-sim-half-expert-v0",
    "trifinger-cube-lift-sim-half-expert-image-v0",
    "trifinger-cube-lift-sim-mixed-v0",
    "trifinger-cube-lift-sim-mixed-image-v0",
]

task_params = {
    "push": {
        "ref_min_score": 0.0,
        "ref_max_score": 1.0 * 15000 / 20,
        "trifinger_kwargs": {
            "episode_length": 750,
            "difficulty": 1,
            "keypoint_obs": True,
            "obs_action_delay": 10,
        },
    },
    "lift": {
        "ref_min_score": 0.0,
        "ref_max_score": 1.0 * 30000 / 20,
        "trifinger_kwargs": {
            "episode_length": 1500,
            "difficulty": 4,
            "keypoint_obs": True,
            "obs_action_delay": 2,
        },
    }
}

# add the missing parameters for all environments
dataset_params = []
for dataset_name in dataset_names:
    dataset_url = base_url + f"{dataset_name}.zarr/dataset.yaml"
    params  = {
        "name": dataset_name,
        "dataset_url": dataset_url,
        "real_robot": "real" in dataset_name,
        "image_obs": "image" in dataset_name,
    }
    task = dataset_name.split("-")[2]
    params.update(task_params[task])
    dataset_params.append(params)


def get_env(**kwargs):
    return TriFingerDatasetEnv(**kwargs)


for params in dataset_params:
    register(
        id=params["name"], entry_point="trifinger_rl_datasets:get_env", kwargs=params
    )


__all__ = ("TriFingerDatasetEnv", "Evaluation", "PolicyBase", "PolicyConfig", "get_env")
