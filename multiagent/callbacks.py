import os
from typing import Optional

import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.type_aliases import GymEnv

from utils import ListAsValue


class MACallback:

    def __init__(self):
        self.model = None

    def init_callback(self, model):
        self.model = model
        self.logger = ListAsValue(self.model.models).logger

    def on_training_start(self):
        pass
    
    def on_rollout_start(self):
        pass
    
    def on_step(self):
        pass
    
    def on_rollout_end(self):
        pass
    
    def on_training_end(self):
        pass

class MAEvalCallback(MACallback):

    def __init__(self, eval_env: GymEnv, n_eval_episodes: int = 5, eval_freq: int = 10000,
                 log_path: Optional[str] = None, model_save_path: Optional[str] = None, deterministic: bool = False):

        super().__init__()
        self.max_reward = float("-inf")

        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.model_save_path = model_save_path
        self.deterministic = deterministic

        self.evaluations_timesteps = []
        self.evaluations_results = []
        self.evaluations_length = []

        self.eval_path = log_path
        if log_path:
            os.makedirs(log_path, exist_ok=True)
            self.eval_path = os.path.join(log_path, "evaluations")

        self.last_eval_time = 0
        self.n_calls = 0

    def on_step(self):

        self.n_calls +=1

        if self.n_calls - self.last_eval_time > self.eval_freq:
            self.last_eval_time = self.n_calls
            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                return_episode_rewards=True,
                deterministic=self.deterministic
            )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length = np.mean(episode_lengths)

            if mean_reward > self.max_reward:
                self.max_reward = mean_reward
                self.model.save(f"{self.model_save_path}/best")

            if self.eval_path:
                self.evaluations_timesteps.append(self.n_calls)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                np.savez(
                    self.eval_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                )

            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.model.time, exclude="tensorboard")
            self.logger.dump(self.model.time)

    def on_training_end(self):
        if self.model_save_path:
            self.model.save(f"{self.model_save_path}/last")