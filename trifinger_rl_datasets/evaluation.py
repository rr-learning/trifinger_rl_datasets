from time import time
import typing

import numpy as np

from .policy_base import PolicyBase


class Evaluation:

    _reset_time = 500

    def __init__(self, env, time_policy=False):
        self.env = env
        self.time_policy = time_policy

    def run_episode(
        self, initial_obs: dict, initial_info: dict, policy: PolicyBase
    ) -> typing.Dict[str, typing.Union[int, float]]:
        """Run one episode/do one rollout."""

        obs = initial_obs
        info = initial_info
        n_steps = 0
        momentary_successes = 0
        ep_return = 0.0
        max_reward = 0.0
        transient_success = False

        policy.reset()

        while True:
            if self.time_policy:
                time1 = time()
            action = policy.get_action(obs)
            if self.time_policy:
                print("policy execution time: ", time() - time1)
            obs, rew, _, truncated, info = self.env.step(action)
            ep_return += rew
            max_reward = max(max_reward, rew)
            if info["has_achieved"]:
                transient_success = True
                momentary_successes += 1
            self.env.render()
            n_steps += 1
            if truncated:
                if info["has_achieved"]:
                    print("Success: Goal achieved at end of episode.")
                else:
                    print("Goal not reached at the end of the episode.")
                break

        ep_stats = {
            "success_rate": int(info["has_achieved"]),
            "mean_momentary_success": momentary_successes / n_steps,
            "transient_success_rate": int(transient_success),
            "return": ep_return,
            "max_reward": max_reward,
        }
        return ep_stats

    def evaluate(self, policy, n_episodes):
        """Evaluate policy in given environment."""

        difficulty = self.env.sim_env.difficulty
        episode_batch_size = 8 if difficulty == 1 else 6
        ep_stats_list = []
        for i in range(n_episodes):
            print("Start episode {}".format(i))
            # reset episode periodically to simulate start of a new robot job
            if i % episode_batch_size == 0:
                initial_obs, initial_info = self.env.reset()
            # run episode
            ep_stats = self.run_episode(initial_obs, initial_info, policy)
            ep_stats_list.append(ep_stats)
            # move fingers to initial position and wait until cube has settled down
            self.env.reset_fingers(self._reset_time)
            if i < n_episodes - 1:
                # retrieve cube from barrier and center it approximately
                self.env.sim_env.reset_cube()
            # Sample new goal
            self.env.sim_env.sample_new_goal()
            # move fingers to initial position and wait until cube has settled down
            initial_obs, initial_info = self.env.reset_fingers(self._reset_time)

        overall_stats = {"n_episodes": n_episodes}
        for k in ep_stats_list[0]:
            overall_stats[k] = np.mean([ep_stats[k] for ep_stats in ep_stats_list])

        return overall_stats
