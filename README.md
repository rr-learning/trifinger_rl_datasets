# TriFinger RL Datasets

This repository provides offline reinforcement learning datasets collected on the real TriFinger platform and in a simulated version of the environment. The paper ["Benchmarking Offline Reinforcement Learning on Real-Robot Hardware"](https://openreview.net/pdf?id=3k5CUGDLNdd) provides more details on the datasets and benchmarks offline RL algorithms on them. All datasets are available with camera images as well.

More detailed information about the simulated environment, the datasets and on how to run experiments on a cluster of real TriFinger robots can be found in the [documentation](https://webdav.tuebingen.mpg.de/trifinger-rl/docs/).

Some of the datasets were used during the [Real Robot Challenge 2022](https://real-robot-challenge.com).

## Installation

To install the package run with python 3.8 in the root directory of the repository (we recommend doing this in a virtual environment):

```bash
pip install --upgrade pip  # make sure the most recent version of pip is installed
pip install .
```

## Usage

This section provides short examples of how to load datasets and evaluate a policy in simulation. More details on how to work with the datasets can be found in the [documentation](https://webdav.tuebingen.mpg.de/trifinger-rl/docs/).


### Loading a dataset

The datasets are accessible via gym environments which are automatically registered when importing the package. They are automatically downloaded when requested and stored in `~/.trifinger_rl_datasets` as Zarr files by default (see the [documentation](https://webdav.tuebingen.mpg.de/trifinger-rl/docs/) for custom paths to the datasets). The code for loading the datasets follows the interface suggested by [D4RL](https://github.com/rail-berkeley/d4rl) and extends it where needed. 

The datasets are named following the pattern `trifinger-cube-task-source-type-v0` where `task` is either `push` or `lift`, `source` is either `sim` or `real` and `type` can be either `mixed`, `weak-n-expert` or `expert`.

By default the observations are loaded as flat arrays. For the simulated datasets the environment can be stepped and visualized. Example usage (also see `demo/load_dataset.py`):

```python
import gymnasium as gym

import trifinger_rl_datasets

env = gym.make(
    "trifinger-cube-push-sim-expert-v0",
    visualization=True,  # enable visualization
)

dataset = env.get_dataset()

print("First observation: ", dataset["observations"][0])
print("First action: ", dataset["actions"][0])
print("First reward: ", dataset["rewards"][0])

obs, info = env.reset()
truncated = False

while not truncated:
    obs, rew, terminated, truncated, info = env.step(env.action_space.sample())
```

Alternatively, the observations can be obtained as nested dictionaries. This simplifies working with the data. As some parts of the observations might be more useful than others, it is also possible to filter the observations when requesting dictionaries (see `demo/load_filtered_dicts.py`):

```python
    # Nested dictionary defines which observations to keep.
    # Everything that is not included or has value False
    # will be dropped.
    obs_to_keep = {
        "robot_observation": {
            "position": True,
            "velocity": True,
            "fingertip_force": False,
        },
        "object_observation": {"keypoints": True},
    }
    env = gym.make(
        args.env_name,
        # filter observations,
        obs_to_keep=obs_to_keep,
    )
```

All datasets come in two versions: with and without camera observations. The versions with camera observations contain `-image` in their name. Despite PNG image compression they are more than one order of magnitude bigger than the imageless versions. To avoid running out of memory, a part of a dataset can be loaded by specifying a range of timesteps:

```python
env = gym.make(
    "trifinger-cube-push-real-expert-image-v0",
    disable_env_checker=True
)

# load only a subset of obervations, actions and rewards
dataset = env.get_dataset(rng=(1000, 2000))
```

The camera observations corresponding to this range are then returned in `dataset["images"]` with the following dimensions:

```python
n_timesteps, n_cameras, n_channels, height, width = dataset["images"].shape
```

### Evaluating a policy in simulation

This package contains an executable module `trifinger_rl_datasets.evaluate_sim`, which
can be used to evaluate a policy in simulation.  As arguments it expects the task
("push" or "lift") and a Python class that implements the policy, following the
`PolicyBase` interface:

    python3 -m trifinger_rl_datasets.evaluate_sim push my_package.MyPolicy

For more options see `--help`.

## How to cite

The paper ["Benchmarking Offline Reinforcement Learning on Real-Robot Hardware"](https://openreview.net/pdf?id=3k5CUGDLNdd) introducing the datasets was published at ICLR 2023:

```
@inproceedings{
guertler2023benchmarking,
title={Benchmarking Offline Reinforcement Learning on Real-Robot Hardware},
author={Nico G{\"u}rtler and Sebastian Blaes and Pavel Kolev and Felix Widmaier and Manuel Wuthrich and Stefan Bauer and Bernhard Sch{\"o}lkopf and Georg Martius},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=3k5CUGDLNdd}
}
```