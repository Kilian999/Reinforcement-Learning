# etc.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class ETC:
    """
    Explore-Then-Commit (ETC) algorithm.

    Parameters
    ----------
    bandit:
        An object with attributes:
          - n_arms: int
          - pull(arm: int) -> float
    exploration_rounds:
        Number of exploration pulls PER ARM (m). Total exploration steps = m * n_arms.
        Ignored if total_exploration_steps is provided.
    total_exploration_steps:
        Optional total exploration budget (across all arms). If provided, we set
        m_per_arm = ceil(total_exploration_steps / n_arms) to distribute roughly evenly.

    Public state
    ------------
    t:
        Total number of steps executed so far.
    committed_arm:
        None during exploration, then index of the chosen arm after commit.
    counts, sums:
        Per-arm pull counts and reward sums.
    """

    bandit: object
    exploration_rounds: int = 1
    total_exploration_steps: Optional[int] = None

    def __post_init__(self) -> None:
        if not hasattr(self.bandit, "n_arms") or not hasattr(self.bandit, "pull"):
            raise TypeError("bandit must have attributes n_arms and method pull(arm)->reward")

        self.n_arms: int = int(self.bandit.n_arms)
        if self.n_arms <= 0:
            raise ValueError("bandit.n_arms must be positive")

        if self.total_exploration_steps is not None:
            if self.total_exploration_steps <= 0:
                raise ValueError("total_exploration_steps must be positive")
            self.m_per_arm = int(np.ceil(self.total_exploration_steps / self.n_arms))
        else:
            if self.exploration_rounds <= 0:
                raise ValueError("exploration_rounds must be positive")
            self.m_per_arm = int(self.exploration_rounds)

        self.total_exploration = self.m_per_arm * self.n_arms

        self.t: int = 0
        self.counts = np.zeros(self.n_arms, dtype=int)
        self.sums = np.zeros(self.n_arms, dtype=float)
        self.committed_arm: Optional[int] = None

    def empirical_means(self) -> np.ndarray:
        """Empirical mean per arm; 0 for arms not played yet."""
        means = np.zeros(self.n_arms, dtype=float)
        played = self.counts > 0
        means[played] = self.sums[played] / self.counts[played]
        return means

    def _choose_exploration_arm(self) -> int:
        """
        Round-robin exploration: step 0 -> arm 0, step 1 -> arm 1, ...
        Ensures each arm gets exactly m_per_arm pulls during exploration.
        """
        return self.t % self.n_arms

    def step(self) -> Tuple[int, float]:
        """
        Execute ONE step of ETC.
        Returns (chosen_arm, reward).
        """
        # Decide arm
        if self.committed_arm is None and self.t < self.total_exploration:
            arm = self._choose_exploration_arm()
        else:
            # If we haven't committed yet, do it now (first step after exploration ends).
            if self.committed_arm is None:
                means = self.empirical_means()
                self.committed_arm = int(np.argmax(means))
            arm = int(self.committed_arm)

        # Pull arm and update stats
        reward = float(self.bandit.pull(arm))
        self.counts[arm] += 1
        self.sums[arm] += reward
        self.t += 1

        return arm, reward

    def run(self, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convenience: run for 'horizon' steps.
        Returns (arms_played, rewards).
        """
        if horizon <= 0:
            raise ValueError("horizon must be positive")

        arms = np.zeros(horizon, dtype=int)
        rewards = np.zeros(horizon, dtype=float)
        for i in range(horizon):
            a, r = self.step()
            arms[i] = a
            rewards[i] = r
        return arms, rewards