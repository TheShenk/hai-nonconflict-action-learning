from typing import Optional, Union, List

import gym
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.noise import VectorizedActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, TrainFrequencyUnit, RolloutReturn

import torch as th
import numpy as np
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv


class MultiAgentOnPolicyProxy:
    def __init__(self, model: OnPolicyAlgorithm):
        self.model = model

    def sample_action(self):
        with th.no_grad():
            # Convert to pytorch tensor or to TensorDict
            obs_tensor = obs_as_tensor(self.model._last_obs, self.model.device)
            actions, values, log_probs = self.model.policy(obs_tensor)
        actions = actions.cpu().numpy()

        # Rescale and perform action
        clipped_actions = actions
        # Clip the actions to avoid out of bound error
        if isinstance(self.model.action_space, gym.spaces.Box):
            clipped_actions = np.clip(actions, self.model.action_space.low, self.model.action_space.high)

        return clipped_actions, actions, values, log_probs

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

    def record(self, observation, actions, next_observation, rewards, dones, infos, sample_actions_result):
        clipped_actions, actions, values, log_probs = sample_actions_result

        if self.model.use_sde and self.model.sde_sample_freq > 0:
            # Sample a new noise matrix
            self.model.policy.reset_noise(self.model.env.num_envs)

        self.model.num_timesteps += self.model.env.num_envs
        self.model._update_info_buffer(infos)

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

        self.model.rollout_buffer.compute_returns_and_advantage(last_values=values,
                                                                dones=self.model._last_episode_starts)


class MultiAgentOffPolicyProxy:

    def __init__(self, model: OffPolicyAlgorithm, log_interval: Optional[int] = None):
        self.rollout = None
        self.model = model
        self.log_interval = log_interval

    def sample_action(self):
        return self.model._sample_action(self.model.learning_starts, self.model.action_noise,
                                         self.model.env.num_envs)

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

    def record(self, observation, actions, next_observation, rewards, dones, infos, sample_actions_result):
        if self.model.use_sde and self.model.sde_sample_freq > 0 and self.num_collected_steps % self.model.sde_sample_freq == 0:
            # Sample a new noise matrix
            self.model.actor.reset_noise(self.model.env.num_envs)

        # Select action randomly or according to policy
        actions, buffer_actions = sample_actions_result

        self.model.num_timesteps += self.model.env.num_envs
        self.num_collected_steps += 1

        # Retrieve reward and episode length if using Monitor wrapper
        self.model._update_info_buffer(infos, dones)

        # Store data in replay buffer (normalized action and unnormalized observation)
        self.model._store_transition(self.model.replay_buffer, buffer_actions, next_observation, rewards, dones, infos)

        self.model._update_current_progress_remaining(self.model.num_timesteps, self.model._total_timesteps)

        # For DQN, check if the target network should be updated
        # and update the exploration schedule
        # For SAC/TD3, the update is dones as the same time as the gradient update
        # see https://github.com/hill-a/stable-baselines/issues/900
        self.model._on_step()

        for idx, done in enumerate(dones):
            if done:
                # Update stats
                self.num_collected_episodes += 1
                self.model._episode_num += 1

                if self.model.action_noise is not None:
                    kwargs = dict(indices=[idx]) if self.model.env.num_envs > 1 else {}
                    self.model.action_noise.reset(**kwargs)

                # Log training infos
                if self.log_interval is not None and self.model._episode_num % self.log_interval == 0:
                    self.model._dump_logs()

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)

    def train(self, *args, **kwargs):
        if self.model.num_timesteps > 0 and self.model.num_timesteps > self.model.learning_starts:
            # If no `gradient_steps` is specified,
            # do as many gradients steps as steps performed during the rollout
            gradient_steps = self.model.gradient_steps if self.model.gradient_steps >= 0 else self.rollout.episode_timesteps
            # Special case when the user passes `gradient_steps=0`
            if gradient_steps > 0:
                return self.train(batch_size=self.batch_size, gradient_steps=gradient_steps)

    def start_record(self):
        self.model.policy.set_training_mode(False)

        self.num_collected_steps, self.num_collected_episodes = 0, 0

        assert isinstance(self.model.env, VecEnv), "You must pass a VecEnv"
        assert self.model.train_freq.frequency > 0, "Should at least collect one step or episode."

        if self.model.env.num_envs > 1:
            assert self.model.train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

        # Vectorize action noise if needed
        if self.model.action_noise is not None and self.model.env.num_envs > 1 and not isinstance(
                self.model.action_noise, VectorizedActionNoise):
            action_noise = VectorizedActionNoise(self.model.action_noise, self.model.env.num_envs)

        if self.model.use_sde:
            self.model.actor.reset_noise(self.model.env.num_envs)

    def end_record(self):
        self.rollout = RolloutReturn(self.num_collected_steps * self.model.env.num_envs, self.num_collected_episodes,
                                     True)


def multiagent_learn(models: List[Union[MultiAgentOffPolicyProxy, MultiAgentOffPolicyProxy]], timesteps, env,
                     n_records_count, model_save_path):
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
            sample_actions_results = [model.sample_action() for model in models]
            actions = list(map(lambda x: x[0], sample_actions_results))
            total_action = np.reshape(actions, (1, -1))
            time += env.num_envs

            next_observation, reward, done, info = env.step(total_action)
            total_reward += reward
            current_step_reward += reward

            for model, action, sample_actions_result in zip(models, actions, sample_actions_results):
                model.record(observation, action, next_observation, reward, done, info, sample_actions_result)

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
