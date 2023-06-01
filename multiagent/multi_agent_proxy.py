import io
import pathlib
import sys
import time
from abc import abstractmethod
from typing import Optional, Union, Iterable, Type

import gym
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.noise import VectorizedActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, TrainFrequencyUnit, RolloutReturn

import torch as th
import numpy as np
from stable_baselines3.common.utils import obs_as_tensor, should_collect_more_steps, safe_mean
from stable_baselines3.common.vec_env import VecEnv


class MultiAgentProxy:

    def __init__(self, model: BaseAlgorithm):
        self.model = model

    def __getattr__(self, item):
        return getattr(self.model, item)

    def save(self, path: Union[str, pathlib.Path, io.BufferedIOBase],
             exclude: Optional[Iterable[str]] = None,
             include: Optional[Iterable[str]] = None):
        self.model.save(path, exclude, include)

    @classmethod
    def load(cls, model_cls, *args, **kwargs):
        model = model_cls.load(*args, **kwargs)
        return cls(model)

    @abstractmethod
    def sample_action(self): pass

    @abstractmethod
    def start_learning(
            self,
            total_timesteps: int,
            eval_env: Optional[GymEnv] = None,
            callback: MaybeCallback = None,
            eval_freq: int = 10000,
            n_eval_episodes: int = 5,
            eval_log_path: Optional[str] = None,
            reset_num_timesteps: bool = True,
            progress_bar: bool = False
    ): pass

    @abstractmethod
    def record(self, observation, next_observation, rewards, dones, infos, sample_actions_result): pass

    @abstractmethod
    def predict(self, *args, **kwargs): pass

    @abstractmethod
    def train(self, *args, **kwargs): pass

    @abstractmethod
    def start_record(self): pass

    @abstractmethod
    def end_record(self): pass

    @abstractmethod
    def continue_record(self): pass

class MultiAgentOnPolicyProxy(MultiAgentProxy):
    def __init__(self,
                 model: OnPolicyAlgorithm,
                 tb_log_name: str = "OnPolicy",
                 two_side_reward_log: bool = False):
        self.n_steps = 0
        self.iteration = 0
        self.model = model
        self.tb_log_name = tb_log_name
        self.two_side_reward_log = two_side_reward_log

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
            progress_bar: bool = False
    ):
        self.model._setup_learn(
            total_timesteps,
            eval_env,
            callback,
            eval_freq,
            n_eval_episodes,
            eval_log_path,
            reset_num_timesteps,
            self.tb_log_name,
            progress_bar,
        )

    def record(self, observation, next_observation, rewards, dones, infos, sample_actions_result):
        clipped_actions, actions, values, log_probs = sample_actions_result

        if self.model.use_sde and self.model.sde_sample_freq > 0:
            # Sample a new noise matrix
            self.model.policy.reset_noise(self.model.env.num_envs)

        self.model.num_timesteps += self.model.env.num_envs
        self.model._update_info_buffer(infos)
        self.n_steps += 1

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
        self.iteration += 1
        log_interval = 1
        # Display training infos
        if log_interval is not None and self.iteration % log_interval == 0:
            time_elapsed = max((time.time_ns() - self.model.start_time) / 1e9, sys.float_info.epsilon)
            fps = int((self.model.num_timesteps - self.model._num_timesteps_at_start) / time_elapsed)
            self.model.logger.record("time/iterations", self.iteration, exclude="tensorboard")
            if len(self.model.ep_info_buffer) > 0 and len(self.model.ep_info_buffer[0]) > 0:
                self.model.logger.record("rollout/ep_rew_mean",
                                         safe_mean([ep_info["r"] for ep_info in self.model.ep_info_buffer]))
                self.model.logger.record("rollout/ep_len_mean",
                                         safe_mean([ep_info["l"] for ep_info in self.model.ep_info_buffer]))
                if self.two_side_reward_log:
                    self.model.logger.record("rollout/ep_left_rew",
                                             safe_mean([ep_info["total_left_reward"] for ep_info in self.model.ep_info_buffer]))
                    self.model.logger.record("rollout/ep_right_rew",
                                             safe_mean([ep_info["total_right_reward"] for ep_info in self.model.ep_info_buffer]))
            self.model.logger.record("time/fps", fps)
            self.model.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
            self.model.logger.record("time/total_timesteps", self.model.num_timesteps, exclude="tensorboard")
            self.model.logger.dump(step=self.model.num_timesteps)

        return self.model.train(*args, **kwargs)

    def start_record(self):
        assert self.model._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.model.policy.set_training_mode(False)

        self.n_steps = 0
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

    def continue_record(self):
        return self.n_steps < self.model.n_steps


class MultiAgentOffPolicyProxy(MultiAgentProxy):

    def __init__(self,
                 model: OffPolicyAlgorithm,
                 log_interval: Optional[int] = 1,
                 tb_log_name: str = "OffPolicy",
                 two_side_reward_log: bool = False):
        self.num_collected_steps = 0
        self.rollout = None
        self.model = model
        self.log_interval = log_interval
        self.tb_log_name = tb_log_name
        self.two_side_reward_log = two_side_reward_log

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
            progress_bar: bool = False
    ):
        self.model._setup_learn(
            total_timesteps,
            eval_env,
            callback,
            eval_freq,
            n_eval_episodes,
            eval_log_path,
            reset_num_timesteps,
            self.tb_log_name,
            progress_bar,
        )

    def record(self, observation, next_observation, rewards, dones, infos, sample_actions_result):
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
                    episode_end_available = len(self.model.ep_info_buffer) > 0 and len(self.model.ep_info_buffer[0]) > 0
                    if episode_end_available and self.two_side_reward_log:
                        self.model.logger.record("rollout/ep_left_rew",
                                                 safe_mean([ep_info["total_left_reward"] for ep_info in
                                                            self.model.ep_info_buffer]))
                        self.model.logger.record("rollout/ep_right_rew",
                                                 safe_mean([ep_info["total_right_reward"] for ep_info in
                                                            self.model.ep_info_buffer]))


    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)

    def train(self, *args, **kwargs):
        if self.model.num_timesteps > 0 and self.model.num_timesteps > self.model.learning_starts:
            # If no `gradient_steps` is specified,
            # do as many gradients steps as steps performed during the rollout
            gradient_steps = self.model.gradient_steps if self.model.gradient_steps >= 0 else self.rollout.episode_timesteps
            # Special case when the user passes `gradient_steps=0`
            if gradient_steps > 0:
                return self.model.train(batch_size=self.model.batch_size, gradient_steps=gradient_steps)

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

    def continue_record(self):
        return should_collect_more_steps(self.model.train_freq, self.num_collected_steps, self.num_collected_episodes)