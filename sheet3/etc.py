from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from utilities import require_bandit, safe_argmax


@dataclass
class ETC:
    """
    Explore-Then-Commit.

    exploration_rounds:
        Number of exploration pulls PER ARM.

    total_exploration_steps:
        Optional total exploration budget. If provided, it overrides
        exploration_rounds and is distributed approximately evenly across arms.
    """

    bandit: object
    exploration_rounds: int = 1
    total_exploration_steps: Optional[int] = None

    def __post_init__(self) -> None:
        self.K, self._pull = require_bandit(self.bandit)

        if self.total_exploration_steps is not None:
            if self.total_exploration_steps <= 0:
                raise ValueError("total_exploration_steps must be positive")
            self.m_per_arm = int(np.ceil(self.total_exploration_steps / self.K))
        else:
            if self.exploration_rounds <= 0:
                raise ValueError("exploration_rounds must be positive")
            self.m_per_arm = int(self.exploration_rounds)

        self.total_exploration = self.m_per_arm * self.K

        self.t: int = 0
        self.counts = np.zeros(self.K, dtype=int)
        self.sums = np.zeros(self.K, dtype=float)
        self.committed_arm: Optional[int] = None

    def empirical_means(self) -> np.ndarray:
        means = np.zeros(self.K, dtype=float)
        mask = self.counts > 0
        means[mask] = self.sums[mask] / self.counts[mask]
        return means

    def _choose_exploration_arm(self) -> int:
        return self.t % self.K

    def step(self) -> Tuple[int, float]:
        if self.committed_arm is None and self.t < self.total_exploration:
            arm = self._choose_exploration_arm()
        else:
            if self.committed_arm is None:
                self.committed_arm = safe_argmax(self.empirical_means())
            arm = int(self.committed_arm)

        reward = float(self._pull(arm))
        self.counts[arm] += 1
        self.sums[arm] += reward
        self.t += 1

        return arm, reward

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