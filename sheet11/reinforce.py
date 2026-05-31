"""
REINFORCE and Mini-batch REINFORCE implementations.

Policy Gradient Theorem:

    ∇J(θ) = E^{π_θ}_s [ Σ_{t=0}^{T-1} ∇_θ log π_θ(A_t; S_t) · R_t^T ]

where R_t^T = Σ_{k=t}^{T-1} R_{k+1} is the reward-to-go.

"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
import gymnasium as gym
from typing import List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Policy Network
# ─────────────────────────────────────────────────────────────────────────────

class PolicyNetwork(nn.Module):
    """
    Neural network policy for both discrete and continuous action spaces.

    Discrete  : outputs logits  → Categorical distribution
    Continuous: outputs mean    → Normal(mean, exp(log_std)) distribution
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 64,
                 is_continuous: bool = False, log_std_init: float = -0.5):
        super().__init__()
        self.is_continuous = is_continuous
        self.act_dim = act_dim

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, act_dim),
        )
        # Learnable log standard deviation for continuous policies
        if is_continuous:
            self.log_std = nn.Parameter(torch.ones(act_dim) * log_std_init)

    def _distribution(self, obs: torch.Tensor):
        out = self.net(obs)
        if self.is_continuous:
            std = self.log_std.exp().expand_as(out)
            return Normal(out, std)
        return Categorical(logits=out)

    @torch.no_grad()
    def sample_action(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self._distribution(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        if self.is_continuous:
            log_prob = log_prob.sum(-1)
        return action, log_prob

    def log_prob(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Recompute log-probs with gradient (needed for backprop)."""
        dist = self._distribution(obs)
        lp = dist.log_prob(actions)
        if self.is_continuous:
            lp = lp.sum(-1)
        return lp


def make_policy(env: gym.Env, hidden_dim: int = 64) -> PolicyNetwork:
    obs_dim = env.observation_space.shape[0]
    is_continuous = isinstance(env.action_space, gym.spaces.Box)
    act_dim = (env.action_space.shape[0] if is_continuous
               else env.action_space.n)
    return PolicyNetwork(obs_dim, act_dim, hidden_dim, is_continuous)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_returns(rewards: List[float], gamma: float) -> torch.Tensor:
    """Compute discounted reward-to-go R_t^T = Σ_{k≥t} γ^{k-t} r_{k+1}."""
    G, returns = 0.0, []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return torch.tensor(returns, dtype=torch.float32)


def normalize(x: torch.Tensor) -> torch.Tensor:
    """Standardize tensor; safe if all values are the same."""
    if x.std() > 1e-8:
        return (x - x.mean()) / (x.std() + 1e-8)
    return x - x.mean()


def collect_episode(env: gym.Env,
                    policy: PolicyNetwork) -> Tuple[List, List, List, List, float]:
    """
    Run one full episode with the current policy (no gradients).

    Returns: (obs_tensors, actions, log_probs, rewards, episode_return)
    """
    obs, _ = env.reset()
    obs_list, act_list, lp_list, rew_list = [], [], [], []

    done = False
    while not done:
        obs_t = torch.tensor(obs, dtype=torch.float32)
        action, log_prob = policy.sample_action(obs_t)

        if isinstance(env.action_space, gym.spaces.Discrete):
            env_action = int(action.item())
        else:
            env_action = np.clip(action.numpy(),
                                 env.action_space.low,
                                 env.action_space.high)

        next_obs, reward, terminated, truncated, _ = env.step(env_action)
        done = terminated or truncated

        obs_list.append(obs_t)
        act_list.append(action)
        lp_list.append(log_prob)
        rew_list.append(float(reward))
        obs = next_obs

    return obs_list, act_list, lp_list, rew_list, sum(rew_list)


# ─────────────────────────────────────────────────────────────────────────────
# REINFORCE (Algorithm 32, K=1)
# ─────────────────────────────────────────────────────────────────────────────

class REINFORCE:
    """
    Standard REINFORCE (Williams 1992).

    One gradient update per episode using the reward-to-go as return estimate
    (Theorem 5.1.3 in the lecture notes).
    """

    def __init__(self, env_name: str, hidden_dim: int = 64, lr: float = 3e-3,
                 gamma: float = 0.99, seed: int = 0):
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.gamma = gamma
        self.env = gym.make(env_name)
        self.policy = make_policy(self.env, hidden_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.episode_returns: List[float] = []
        self.timesteps: List[int] = []
        self._steps = 0

    def train(self, total_timesteps: int) -> Tuple[List[float], List[int]]:
        while self._steps < total_timesteps:
            # ── collect episode ──────────────────────────────────────────────
            obs_list, act_list, _, rew_list, ep_ret = collect_episode(
                self.env, self.policy)

            self._steps += len(rew_list)
            self.episode_returns.append(ep_ret)
            self.timesteps.append(self._steps)

            # ── compute & normalize returns ──────────────────────────────────
            returns = normalize(compute_returns(rew_list, self.gamma))

            # ── gradient step ────────────────────────────────────────────────
            self.policy.train()
            obs_t   = torch.stack(obs_list)
            acts_t  = torch.stack(act_list)
            log_probs = self.policy.log_prob(obs_t, acts_t)

            loss = -(log_probs * returns).mean()

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.optimizer.step()

        self.env.close()
        return self.episode_returns, self.timesteps


# ─────────────────────────────────────────────────────────────────────────────
# Mini-batch REINFORCE (Algorithm 32, K > 1)
# ─────────────────────────────────────────────────────────────────────────────

class MiniBatchREINFORCE:
    """
    Mini-batch REINFORCE: accumulates K episodes before each gradient step.

    Reduces gradient variance compared to single-episode REINFORCE by
    averaging over K independent rollouts (Algorithm 32 in lecture notes).
    """

    def __init__(self, env_name: str, hidden_dim: int = 64, lr: float = 3e-3,
                 gamma: float = 0.99, batch_size: int = 8, seed: int = 0):
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.gamma      = gamma
        self.batch_size = batch_size
        self.env        = gym.make(env_name)
        self.policy     = make_policy(self.env, hidden_dim)
        self.optimizer  = optim.Adam(self.policy.parameters(), lr=lr)

        self.episode_returns: List[float] = []
        self.timesteps: List[int] = []
        self._steps = 0

    def train(self, total_timesteps: int) -> Tuple[List[float], List[int]]:
        while self._steps < total_timesteps:
            batch_obs, batch_acts, batch_rets = [], [], []

            # ── collect K episodes ───────────────────────────────────────────
            for _ in range(self.batch_size):
                if self._steps >= total_timesteps:
                    break

                obs_list, act_list, _, rew_list, ep_ret = collect_episode(
                    self.env, self.policy)

                self._steps += len(rew_list)
                self.episode_returns.append(ep_ret)
                self.timesteps.append(self._steps)

                batch_obs.extend(obs_list)
                batch_acts.extend(act_list)
                batch_rets.append(compute_returns(rew_list, self.gamma))

            if not batch_obs:
                break

            # ── normalize returns across the entire batch ────────────────────
            all_rets = normalize(torch.cat(batch_rets))

            # ── gradient step ────────────────────────────────────────────────
            self.policy.train()
            obs_t  = torch.stack(batch_obs)
            acts_t = torch.stack(batch_acts)
            log_probs = self.policy.log_prob(obs_t, acts_t)

            loss = -(log_probs * all_rets).mean()

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.optimizer.step()

        self.env.close()
        return self.episode_returns, self.timesteps
