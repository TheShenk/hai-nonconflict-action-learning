from typing import Optional, Union

import gym
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback

import torch as th
import numpy as np
from stable_baselines3.common.utils import obs_as_tensor


def multiagent_learn(models, timesteps, env, n_records_count, model_save_path):
    for model in models:
        model.start_learning(timesteps)

    observation = env.reset()
    total_reward = 0
    max_step_reward = 0
    time = 0
    while time < timesteps:
        current_step_reward = 0

        for model in models:
            model.start_record()

        for step in range(n_records_count):
            actions = [model.predict(observation)[0] for model in models]
            total_action = np.concatenate(np.concatenate(actions))
            time += env.num_envs

            next_observation, reward, done, info = env.step(np.array([total_action]))
            total_reward += reward
            current_step_reward += reward

            for model, action in zip(models, actions):
                model.record(observation, action, next_observation, reward, done, info)

            observation = next_observation

        for model in models:
            model.end_record()
            model.train()

        print(time, current_step_reward)

        if current_step_reward > max_step_reward:
            max_step_reward = current_step_reward
            for index, model in enumerate(models):
                model.model.save(f"{model_save_path}-best-{index}")

    for index, model in enumerate(models):
        model.model.save(f"{model_save_path}-last-{index}")


class MultiAgentOnPolicyProxy:
    def __init__(self, model: Union[OnPolicyAlgorithm, OffPolicyAlgorithm]):
        self.model = model

    def start_learning(
            self,
            total_timesteps: int,
            eval_env: Optional[GymEnv] = None,
            callback: MaybeCallback = None,
            eval_freq: int = 10000,
            n_eval_episodes: int = 5,
            eval_log_path: Optional[str] = None,
            reset_num_timesteps: bool = True,
            tb_log_name: str = "run",
            progress_bar: bool = False,
    ):
        self.model._setup_learn(
            total_timesteps,
            eval_env,
            callback,
            eval_freq,
            n_eval_episodes,
            eval_log_path,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

    def record(self, observation, actions, next_observation, rewards, dones, infos):
        if self.model.use_sde and self.model.sde_sample_freq > 0:
            # Sample a new noise matrix
            self.model.policy.reset_noise(self.model.env.num_envs)

        with th.no_grad():
            # Convert to pytorch tensor or to TensorDict
            obs_tensor = obs_as_tensor(self.model._last_obs, self.model.device)
            actions, values, log_probs = self.model.policy(obs_tensor)
        actions = actions.cpu().numpy()

        self.model.num_timesteps += self.model.env.num_envs

        self.model._update_info_buffer(infos)

        if isinstance(self.model.action_space, gym.spaces.Discrete):
            # Reshape in case of discrete action
            actions = actions.reshape(-1, 1)

        # Handle timeout by bootstraping with value function
        # see GitHub issue #633
        for idx, done in enumerate(dones):
            if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
            ):
                terminal_obs = self.model.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                with th.no_grad():
                    terminal_value = self.model.policy.predict_values(terminal_obs)[0]
                rewards[idx] += self.model.gamma * terminal_value

        self.model.rollout_buffer.add(self.model._last_obs, actions, rewards, self.model._last_episode_starts, values,
                                      log_probs)
        self.model._last_obs = next_observation
        self.model._last_episode_starts = dones

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)

    def train(self, *args, **kwargs):
        return self.model.train(*args, **kwargs)

    def start_record(self):
        assert self.model._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.model.policy.set_training_mode(False)

        n_steps = 0
        self.model.rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.model.use_sde:
            self.model.policy.reset_noise(self.model.env.num_envs)

    def end_record(self):
        with th.no_grad():
            # Compute value for the last timestep
            values = self.model.policy.predict_values(obs_as_tensor(self.model._last_obs, self.model.device))

        self.model.rollout_buffer.compute_returns_and_advantage(last_values=values, dones=self.model._last_episode_starts)