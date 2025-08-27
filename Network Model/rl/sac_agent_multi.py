# rl/sac_agent_multi.py
"""
SAC wrapper for multi-EV station control (vector actions).

Train offline on MultiEVGymEnv with recorded sequences of:
  - pmax_seq (headroom),
  - price_seq (tariffs),
  - arrivals_by_t (EV arrivals per step).

Deploy online by calling act(obs) inside your PF loop with MultiEVRuntimeEnv.
"""

from __future__ import annotations
from typing import Optional, Callable, List, Dict
import os
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import DummyVecEnv
    
from rl.multi_env_ev_cs import MultiEVGymEnv

class SACMultiAgent:
    """
    Thin SB3 SAC wrapper for a single station with N plugs (vector action).
    """

    def __init__(self, model):
        """
        Hold a trained (or loaded) SB3 SAC model.
        """
        self.model = model

    # ------------------------- Training -------------------------

    @staticmethod
    def train_offline(n_plugs: int,
                      pmax_seq: np.ndarray,
                      price_seq: np.ndarray,
                      arrivals_by_t: List[List[Dict[str, float]]],
                      total_timesteps: int = 400_000,
                      seed: int = 0,
                      save_dir: Optional[str] = None) -> "SACMultiAgent":
        """
        Build a MultiEVGymEnv and train SAC on it.

        Args:
          n_plugs:       number of simultaneous charging sockets at the station
          pmax_seq:      array[T] of station headroom (kW) per step
          price_seq:     array[T] of prices per step
          arrivals_by_t: list length T; each item is a list of arrival dicts
          total_timesteps: RL training steps
          seed:          RNG seed
          save_dir:      if provided, save model there

        Returns:
          SACMultiAgent ready to act() in deployment.
        """

        def make_env():
            return MultiEVGymEnv(
                n_plugs=n_plugs,
                pmax_seq=pmax_seq,
                price_seq=price_seq,
                arrivals_by_t=arrivals_by_t,
                step_minutes=60,             # keep consistent with your sim
                capacity_kwh=60.0,
                alpha_target=150.0,          # slightly stronger target pressure for multi-EV
                allow_v2g=True,
                rem_norm_den_h=24.0,
                lim_norm_den_kw=22.0,
            )

        vec_env = DummyVecEnv([make_env])

        model = SAC(
            policy="MlpPolicy",
            env=vec_env,
            verbose=0,
            seed=seed,
            learning_rate=3e-4,
            tau=0.005,
            gamma=0.99,
            train_freq=(1, "step"),
            gradient_steps=1,
            buffer_size=300_000,
            batch_size=256,
            ent_coef="auto"
        )
        model.learn(total_timesteps=total_timesteps, log_interval=50)

        agent = SACMultiAgent(model)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            model.save(os.path.join(save_dir, "sac_multi_model.zip"))
        return agent

    # ------------------------- Loading -------------------------

    @staticmethod
    def load(save_dir: str) -> "SACMultiAgent":
        """
        Load a previously saved multi-EV SAC model from save_dir.
        """
        model = SAC.load(os.path.join(save_dir, "sac_multi_model"))
        return SACMultiAgent(model)

    # ------------------------- Acting -------------------------

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Predict an action vector a ∈ [-1,1]^N for a single observation.
        """
        action, _ = self.model.predict(obs[None, :], deterministic=deterministic)
        a = np.asarray(action).reshape(-1).astype(float)
        return np.clip(a, -1.0, 1.0)
