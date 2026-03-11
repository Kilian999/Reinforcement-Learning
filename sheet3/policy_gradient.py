from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from utilities import require_bandit, softmax


@dataclass
class PolicyGradientSoftmax:
    """
    REINFORCE with softmax policy:
        pi_theta(a) = exp(theta_a) / sum_b exp(theta_b)

    Update:
        theta <- theta + alpha * (R - baseline) * grad log pi_theta(A)
    """
    bandit: object
    alpha: float = 0.1
    use_baseline: bool = False
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        self.K, self._pull = require_bandit(self.bandit)
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")

        self.rng = np.random.default_rng(self.seed)

        self.t: int = 0
        self.theta = np.zeros(self.K, dtype=float)
        self.baseline: float = 0.0

        # monitoring
        self.counts = np.zeros(self.K, dtype=int)
        self.sums = np.zeros(self.K, dtype=float)

    def policy(self) -> np.ndarray:
        return softmax(self.theta)

    def empirical_means(self) -> np.ndarray:
        means = np.zeros(self.K, dtype=float)
        mask = self.counts > 0
        means[mask] = self.sums[mask] / self.counts[mask]
        return means

    def step(self) -> Tuple[int, float]:
        pi = self.policy()
        arm = int(self.rng.choice(self.K, p=pi))
        reward = float(self._pull(arm))

        self.counts[arm] += 1
        self.sums[arm] += reward

        baseline = self.baseline if self.use_baseline else 0.0
        advantage = reward - baseline

        grad = -pi
        grad[arm] += 1.0
        self.theta += self.alpha * advantage * grad

        if self.use_baseline:
            self.baseline += (reward - self.baseline) / (self.t + 1.0)

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