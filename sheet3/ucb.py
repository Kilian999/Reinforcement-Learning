from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .utilities import require_bandit, safe_argmax


@dataclass
class BaseAlgo:
    bandit: object
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        self.K, self._pull = require_bandit(self.bandit)
        self.rng = np.random.default_rng(self.seed)

        self.t: int = 0
        self.counts = np.zeros(self.K, dtype=int)
        self.sums = np.zeros(self.K, dtype=float)

    def empirical_means(self) -> np.ndarray:
        means = np.zeros(self.K, dtype=float)
        mask = self.counts > 0
        means[mask] = self.sums[mask] / self.counts[mask]
        return means

    def _update(self, a: int, r: float) -> None:
        self.counts[a] += 1
        self.sums[a] += r
        self.t += 1

    def run(self, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
        if horizon <= 0:
            raise ValueError("horizon must be positive")

        arms = np.zeros(horizon, dtype=int)
        rewards = np.zeros(horizon, dtype=float)

        for i in range(horizon):
            a, r = self.step()
            arms[i] = a
            rewards[i] = r

        return arms, rewards


@dataclass
class UCB1(BaseAlgo):
    """
    Standard UCB:
        argmax_a [ Qhat_a + sqrt((c log t) / N_a) ]
    """
    c: float = 2.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.c <= 0:
            raise ValueError("c must be positive")

    def step(self) -> Tuple[int, float]:
        unplayed = np.where(self.counts == 0)[0]
        if unplayed.size > 0:
            arm = int(unplayed[0])
        else:
            t_eff = max(2, self.t)
            q = self.empirical_means()
            bonus = np.sqrt((self.c * np.log(t_eff)) / self.counts)
            arm = safe_argmax(q + bonus)

        reward = float(self._pull(arm))
        self._update(arm, reward)
        return arm, reward


@dataclass
class UCBSubGaussian(BaseAlgo):
    """
    sigma-subgaussian UCB:
        argmax_a [ Qhat_a + sigma * sqrt((2 log t) / N_a) ]
    """
    sigma: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")

    def step(self) -> Tuple[int, float]:
        unplayed = np.where(self.counts == 0)[0]
        if unplayed.size > 0:
            arm = int(unplayed[0])
        else:
            t_eff = max(2, self.t)
            q = self.empirical_means()
            bonus = self.sigma * np.sqrt((2.0 * np.log(t_eff)) / self.counts)
            arm = safe_argmax(q + bonus)

        reward = float(self._pull(arm))
        self._update(arm, reward)
        return arm, reward