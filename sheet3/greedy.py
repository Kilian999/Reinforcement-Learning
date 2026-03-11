from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

from utilities import require_bandit, safe_argmax


EpsSchedule = Callable[[int], float]


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
class Greedy(BaseAlgo):
    """Pure greedy after warm-start."""

    def step(self) -> Tuple[int, float]:
        unplayed = np.where(self.counts == 0)[0]
        if unplayed.size > 0:
            arm = int(unplayed[0])
        else:
            arm = safe_argmax(self.empirical_means())

        reward = float(self._pull(arm))
        self._update(arm, reward)
        return arm, reward


@dataclass
class EpsilonGreedyFixed(BaseAlgo):
    epsilon: float = 0.1

    def __post_init__(self) -> None:
        super().__post_init__()
        if not (0.0 <= self.epsilon <= 1.0):
            raise ValueError("epsilon must be in [0,1]")

    def step(self) -> Tuple[int, float]:
        unplayed = np.where(self.counts == 0)[0]
        if unplayed.size > 0:
            arm = int(unplayed[0])
        else:
            if self.rng.random() < self.epsilon:
                arm = int(self.rng.integers(0, self.K))
            else:
                arm = safe_argmax(self.empirical_means())

        reward = float(self._pull(arm))
        self._update(arm, reward)
        return arm, reward


@dataclass
class EpsilonGreedyDecaying(BaseAlgo):
    """
    Default:
        epsilon_t = min(1, eps0 / sqrt(t+1))
    """
    eps0: float = 1.0
    schedule: Optional[EpsSchedule] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.eps0 < 0:
            raise ValueError("eps0 must be nonnegative")

    def step(self) -> Tuple[int, float]:
        unplayed = np.where(self.counts == 0)[0]
        if unplayed.size > 0:
            arm = int(unplayed[0])
        else:
            if self.schedule is not None:
                eps = float(self.schedule(self.t))
            else:
                eps = min(1.0, self.eps0 / np.sqrt(self.t + 1.0))

            eps = float(np.clip(eps, 0.0, 1.0))

            if self.rng.random() < eps:
                arm = int(self.rng.integers(0, self.K))
            else:
                arm = safe_argmax(self.empirical_means())

        reward = float(self._pull(arm))
        self._update(arm, reward)
        return arm, reward