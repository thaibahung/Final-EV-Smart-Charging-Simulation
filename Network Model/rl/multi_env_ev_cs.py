"""
Multi-EV environments for a single charging station.

Two envs, same logic & feature layout:
- MultiEVRuntimeEnv: used ONLINE inside your PF loop (takes live pmax_kw each step).
- MultiEVGymEnv:     used OFFLINE to train SAC with recorded sequences of pmax & price.

EV arrivals are given as "event" lists per step (arrivals only). Each arrival dict:
  {"soc0": 0.35, "stay_h": 2.0, "limit_kw": 11.0, "target": 0.80}

Departures happen automatically when an EV's remaining time <= 0.

Hard safety: we enforce Σ|P_i| ≤ pmax_kw via an L1 projection, after per-EV limits.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import gymnasium as gym
from gymnasium import spaces
import numpy as np


def project_to_station_cap(P: np.ndarray, cap_kw: float) -> np.ndarray:
    """
    L1-ball projection: scales vector P so Σ|P_i| ≤ cap_kw.
    Keeps direction, guarantees the station headroom is never exceeded.
    """
    cap = max(0.0, float(cap_kw))
    total = float(np.sum(np.abs(P)))
    if total <= cap or total == 0.0:
        return P
    return P * (cap / total)

def _pack_per_ev_block(soc, rem_h, lim_kw, target, active,
                       rem_norm_den, lim_norm_den) -> np.ndarray:
    """
    Build per-EV normalized feature block for all slots.
    """
    soc      = np.clip(soc, 0, 1)
    rem_norm = np.clip(rem_h / max(rem_norm_den, 1e-9), 0.0, 1.0)
    lim_norm = np.clip(lim_kw / max(lim_norm_den, 1e-9), 0.0, 1.0)
    target   = np.clip(target, 0, 1)
    active   = np.clip(active, 0, 1)
    return np.stack([soc, rem_norm, lim_norm, target, active], axis=1).reshape(-1)

# --------------------------- RUNTIME ENV (PF-coupled) ---------------------------

class MultiEVRuntimeEnv:
    """
    Minimal runtime env for N-plug station used INSIDE your power-flow loop.

    External world provides at each step:
      - pmax_kw: station headroom (from PF #1)
      - price_t: market price
      - arrivals[t]: list of EV arrival dicts (soc0/stay_h/limit_kw/target)

    Internal state:
      - arrays over slots: soc, rem_h (remaining hours), lim_kw, target, active
      - hard enforcement: per-EV |P_i| ≤ lim_i, and Σ|P_i| ≤ pmax_kw

    Observation (flat vector):
      [ pmax_norm, price_norm ] + N * [ soc, rem_norm, lim_norm, target, active ]
    Action:
      a ∈ [-1, 1]^N  →  P_raw = a * lim_kw  →  P = L1-project(P_raw, pmax_kw)

    Reward:
      step:  - (Σ_i P_i * Δt_h) * price_t
      plus terminal penalties for EVs that depart THIS step:
            - α * Σ_departed |target_i - soc_i|
    """

    def __init__(self,
                 n_plugs: int,
                 step_minutes: int = 60,
                 capacity_kwh: float = 60.0,
                 alpha_target: float = 100.0,
                 allow_v2g: bool = True,
                 # normalization knobs (only affect obs scaling)
                 pmax_clip_kw: float = 100.0,
                 price_max_for_norm: Optional[float] = None,
                 rem_norm_den_h: float = 24.0,
                 lim_norm_den_kw: float = 22.0):
        self.N = int(n_plugs)
        self.dt_h = step_minutes / 60.0
        self.capacity_kwh = float(capacity_kwh)
        self.alpha = float(alpha_target)
        self.allow_v2g = bool(allow_v2g)

        # state arrays per slot
        self.soc     = np.zeros(self.N, dtype=float)
        self.rem_h   = np.zeros(self.N, dtype=float)   # time-to-departure
        self.lim_kw  = np.zeros(self.N, dtype=float)   # per-EV power cap
        self.target  = np.zeros(self.N, dtype=float)   # target SoC at departure
        self.active  = np.zeros(self.N, dtype=float)   # 1 if slot occupied
        # logging (optional)
        self.last_departed_idx: List[int] = []

        # normalization helpers
        self.pmax_clip_kw = float(pmax_clip_kw)
        self.price_max = float(price_max_for_norm) if price_max_for_norm else 1.0
        self.rem_norm_den_h = float(rem_norm_den_h)
        self.lim_norm_den_kw = float(lim_norm_den_kw)

        self.t = 0

    # ---------- schedule management ----------

    def reset(self, t0: int = 0):
        """
        Reset clock and clear all slots. Returns nothing; call obs(...) next step with live pmax/price.
        """
        self.t = int(t0)
        self.soc[:] = 0.0
        self.rem_h[:] = 0.0
        self.lim_kw[:] = 0.0
        self.target[:] = 0.0
        self.active[:] = 0.0
        self.last_departed_idx = []

    def apply_arrivals(self, arrivals: List[Dict[str, float]]):
        """
        Fill free slots with arriving EVs. Extra arrivals beyond available slots are dropped.

        Each arrival dict fields:
          - soc0:     initial SoC in [0,1]
          - stay_h:   planned stay duration (hours)
          - limit_kw: per-EV max power (kW)
          - target:   desired SoC at departure in [0,1]
        """
        # prioritize shorter stays (EDF-style) to reduce missed targets
        arrivals = sorted(arrivals, key=lambda d: d.get("stay_h", 0.0))
        free_idx = list(np.where(self.active == 0.0)[0])
        for a, idx in zip(arrivals, free_idx):
            self.active[idx] = 1.0
            self.soc[idx]    = float(np.clip(a.get("soc0", 0.3), 0.0, 1.0))
            self.rem_h[idx]  = float(max(0.0, a.get("stay_h", 1.0)))
            self.lim_kw[idx] = float(max(0.0, a.get("limit_kw", 11.0)))
            self.target[idx] = float(np.clip(a.get("target", 0.8), 0.0, 1.0))

    # ---------- core step ----------

    def obs(self, pmax_kw: float, price_t: float) -> np.ndarray:
        """
        Build the observation for the current step from internal state + exogenous (pmax, price).
        """
        pmax_norm  = min(max(0.0, float(pmax_kw)), self.pmax_clip_kw) / self.pmax_clip_kw
        price_norm = float(price_t) / max(1e-9, self.price_max)
        per_ev = _pack_per_ev_block(self.soc, self.rem_h, self.lim_kw, self.target, self.active,
                                    self.rem_norm_den_h, self.lim_norm_den_kw)
        return np.concatenate([[pmax_norm, price_norm], per_ev]).astype(np.float32)

    def step(self,
             action: np.ndarray,
             pmax_kw: float,
             price_t: float) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """
        Apply agent command for this step, update SoCs, handle departures, and return:
          (next_obs, reward, info)

        action: shape (N,) in [-1,1]; if allow_v2g=False we clamp to [0,1].
        pmax_kw: current headroom for the station from PF #1.
        price_t: price for this step.
        """
        a = np.asarray(action, dtype=float).reshape(self.N)
        if not self.allow_v2g:
            a = np.clip(a, 0.0, 1.0)

        # per-EV raw command (respect per-EV limits and inactivity)
        lim_i = self.lim_kw * self.active  # 0 for empty slots
        P_raw = a * lim_i

        # project to station cap (Σ|P_i| ≤ pmax_kw)
        P_kw = project_to_station_cap(P_raw, max(0.0, float(pmax_kw)))

        # energy & SoC updates
        e_kwh = P_kw * self.dt_h
        self.soc = np.clip(self.soc + e_kwh / self.capacity_kwh, 0.0, 1.0)

        # decrease remaining times
        self.rem_h = np.maximum(0.0, self.rem_h - self.dt_h)

        # compute cost reward
        energy_cost = float(np.sum(e_kwh) * float(price_t))
        reward = -energy_cost

        # departures now (post-update)
        departed = list(np.where((self.active > 0.5) & (self.rem_h <= 1e-9))[0])
        # Ensure departed list is not empty
        if not departed:
            departed = []

        penalty = 0.0
        for i in departed:
            penalty += abs(float(self.target[i]) - float(self.soc[i]))
            # free slot
            self.active[i] = 0.0
            self.lim_kw[i] = 0.0
            # keep soc (historical), zero rem_h
            self.rem_h[i]  = 0.0

        reward -= self.alpha * penalty
        self.last_departed_idx = departed
        self.t += 1

        # next obs uses same pmax/price caller will provide next step
        obs_next = self.obs(pmax_kw, price_t)
        info = {"P_kw": P_kw, "price": float(price_t),
                "departed_idx": departed, "departed_penalty": penalty,
                "energy_cost": energy_cost}
        return obs_next, float(reward), info

# --------------------------- GYM ENV (offline SAC training) ---------------------------

class MultiEVGymEnv(gym.Env if gym else object):
    """
    Gym-compatible multi-EV env for OFFLINE SAC training.

    Inputs (fixed sequences):
      - pmax_seq[t] : station headroom series (from csXX_caps.csv)
      - price_seq[t]: tariff series
      - arrivals_by_t[t]: list of arrival dicts (soc0, stay_h, limit_kw, target)

    Internal dynamics & reward match MultiEVRuntimeEnv.
    """

    metadata = {"render_modes": []}

    def __init__(self,
                 n_plugs: int,
                 pmax_seq: np.ndarray,
                 price_seq: np.ndarray,
                 arrivals_by_t: List[List[Dict[str, float]]],
                 step_minutes: int = 60,
                 capacity_kwh: float = 60.0,
                 alpha_target: float = 100.0,
                 allow_v2g: bool = True,
                 pmax_clip_kw: Optional[float] = None,
                 rem_norm_den_h: float = 24.0,
                 lim_norm_den_kw: float = 22.0):
        assert gym is not None, "gymnasium is required for MultiEVGymEnv"
        super().__init__()
        self.N = int(n_plugs)
        self.pmax_seq  = np.asarray(pmax_seq,  dtype=float)
        self.price_seq = np.asarray(price_seq, dtype=float)
        self.arrivals  = arrivals_by_t
        assert len(self.pmax_seq) == len(self.price_seq) == len(self.arrivals), \
            "pmax_seq, price_seq, arrivals_by_t must have equal length"
        self.H = len(self.pmax_seq)

        # reuse runtime dynamics
        self.env = MultiEVRuntimeEnv(
            n_plugs=self.N,
            step_minutes=step_minutes,
            capacity_kwh=capacity_kwh,
            alpha_target=alpha_target,
            allow_v2g=allow_v2g,
            pmax_clip_kw=(pmax_clip_kw if pmax_clip_kw is not None else float(np.percentile(self.pmax_seq, 95))),
            price_max_for_norm=float(np.max(self.price_seq)),
            rem_norm_den_h=rem_norm_den_h,
            lim_norm_den_kw=lim_norm_den_kw,
        )

        # Gym spaces: obs = 2 + N*5 features, act = N dims
        obs_dim = 2 + self.N * 5
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.N,), dtype=np.float32)

        self.t = 0

    # Gym API
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.t = 0
        self.env.reset(t0=0)
        # apply arrivals at t=0
        self.env.apply_arrivals(self.arrivals[0])
        obs = self.env.obs(self.pmax_seq[0], self.price_seq[0])
        return obs, {}

    def step(self, action: np.ndarray):
        # use current sequences
        pmax_kw = float(self.pmax_seq[self.t])
        price_t = float(self.price_seq[self.t])

        obs_next, reward, info = self.env.step(action, pmax_kw=pmax_kw, price_t=price_t)

        # move time
        self.t += 1
        terminated = (self.t >= self.H)
        truncated = False

        # arrivals for next step (if any)
        if not terminated:
            self.env.apply_arrivals(self.arrivals[self.t])
            # rebuild obs with next step's pmax & price
            obs_next = self.env.obs(self.pmax_seq[self.t], self.price_seq[self.t])

        return obs_next, float(reward), terminated, truncated, info