from typing import Optional, List
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.vec_env import VecEnv

from agents.base_agent import BaseAgent
from multiagent import action_combiners
from multiagent.callbacks import MAEvalCallback
from multiagent.multi_agent_proxy import MultiAgentProxy
from multiagent.multi_model_agent import MultiModelAgent


def multiagent_learn(models: List[MultiAgentProxy],
                     timesteps: int,
                     env: VecEnv,
                     model_save_path: Optional[str] = None,
                     action_combiner=action_combiners.BOX_COMBINER,
                     eval_env: Optional[GymEnv] = None,
                     eval_log_dir: Optional[str] = None,
                     eval_freq: Optional[int] = None,
                     static_models: Optional[List[BaseAgent]] = None):

    eval_callback = None
    if eval_env:
        eval_callback = MAEvalCallback(eval_env,
                                     model_save_path=model_save_path,
                                     eval_freq=eval_freq,
                                     log_path=eval_log_dir,
                                     n_eval_episodes=10)
    multi_model = MultiModelAgent(env, models, static_models, action_combiner)
    multi_model.learn(timesteps, eval_callback)