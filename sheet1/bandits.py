"""
Stochastic bandits: Gaussian and Bernoulli arms.

Features
- Distribution: "gaussian" or "bernoulli"
- Means: set manually or generated randomly
  - Gaussian random means: iid N(0, 1)
  - Bernoulli random means: iid Uniform(0, 1)
- Optional "gap mode":
  After sampling initial random means, fix the best mean mu* and replace the
  other means in descending order by mu* - kΔ (k=1,2,...). For Bernoulli,
  clamp negative means to 0.
"""

from __future__ import annotations
# Ohne annotations: Class -> echtes Objekt
# Mit annotations: "Class" -> String

from dataclasses import dataclass
# Benannte Komponenten der Klasse (Tupel mit benannten Variablen)
from typing import Iterable, List, Optional, Sequence, Union

import numpy as np


Number = Union[int, float]
# Number elem {int,float}

"""
Validation of the input distribution
"""
def _validate_dist(dist: str) -> str:
    d = dist.strip().lower()
    if d not in {"gaussian", "bernoulli"}:
        raise ValueError(f"dist must be 'gaussian' or 'bernoulli', got: {dist!r}")
    return d

"""
Check size of means and n_arms, convert means into list of floats
"""
def _as_means_list(means: Optional[Sequence[Number]], n_arms: int) -> Optional[List[float]]:
    if means is None:
        return None
    if len(means) != n_arms:
        raise ValueError(f"means must have length n_arms={n_arms}, got {len(means)}")
    return [float(m) for m in means]

"""
Clamp x onto [0,1] fpr benoulli
"""
def _clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


@dataclass
class StochasticBandit:
    """
    A stochastic multi-armed bandit with independent arms.

    n_arms : int
        Number of arms.
    dist : str
        "gaussian" or "bernoulli".
    means : Optional[Sequence[Number]]
        If provided, used as arm means. If None, means are sampled randomly.
    gap : Optional[float]
        If provided (gap > 0), activate gap mode described in the task statement.
        Gap mode is only applied when means are generated randomly (means=None).
    sigma : float
        Standard deviation for Gaussian rewards (same for all arms).
    seed : Optional[int]
        Random seed for reproducibility.

    Methods
    -------
    pull(arm) -> float
        Draw a reward from the chosen arm.
    reset(seed=None) -> None
        Reset RNG.
    """

    n_arms: int
    dist: str
    means: Optional[Sequence[Number]] = None
    gap: Optional[float] = None
    sigma: float = 1.0
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.n_arms <= 0:
            raise ValueError(f"n_arms must be positive, got {self.n_arms}")
        self.dist = _validate_dist(self.dist)

        if self.sigma <= 0:
            raise ValueError(f"sigma must be positive, got {self.sigma}")

        self._rng = np.random.default_rng(self.seed)

        # Track whether means are random so reset() can resample if desired.
        self._means_were_random = self.means is None

        means_list = _as_means_list(self.means, self.n_arms)

        # Sample if no means are given, apply gap mode
        if means_list is None:
            self._means = self._sample_random_means()
            if self.gap is not None:
                self._apply_gap_mode(float(self.gap))
        # Means given by means_list, clamp fpr Bernoulli
        else:
            self._means = np.array(means_list, dtype=float)
            if self.dist == "bernoulli":
                self._means = _clamp01(self._means)

    @property
    # Function can be reas as an attribute
    def means_array(self) -> np.ndarray:
        """Return a copy of current means as a numpy array."""
        return self._means.copy()

    @property
    def means_list(self) -> List[float]:
        """Return current means as list."""
        return self._means.tolist()

    def _sample_random_means(self) -> np.ndarray:
        if self.dist == "gaussian":
            return self._rng.standard_normal(self.n_arms)
        # bernoulli
        return self._rng.uniform(0.0, 1.0, self.n_arms)

    def _apply_gap_mode(self, delta: float) -> None:
        if delta <= 0:
            raise ValueError(f"gap Δ must be > 0, got {delta}")

        # Sort arms by their initially drawn means (descending).
        order_desc = np.argsort(self._means)[::-1]

        mu_star = float(self._means[order_desc[0]])
        new_means = self._means.copy()

        # k=0 -> best arm unchanged; k>=1 replaced by mu* - kΔ
        for k, arm_idx in enumerate(order_desc):
            if k == 0:
                new_means[arm_idx] = mu_star
            else:
                new_means[arm_idx] = mu_star - k * delta

        if self.dist == "bernoulli":
            # Task explicitly allows clamping negatives to 0.
            # Additionally clamp >1 to 1 to keep p valid if mu_star > 1 ever occurs.
            new_means = _clamp01(new_means)

        self._means = new_means

    def pull(self, arm: int) -> float:
        """
        Draw a reward from the specified arm.

        Gaussian:  reward ~ N(mean[arm], sigma^2)
        Bernoulli: reward ~ Bernoulli(p=mean[arm]) returns 0.0 or 1.0
        """
        if not (0 <= arm < self.n_arms):
            raise IndexError(f"arm index out of range: {arm} (n_arms={self.n_arms})")

        mu = float(self._means[arm])

        if self.dist == "gaussian":
            return float(self._rng.normal(loc=mu, scale=self.sigma))

        # bernoulli
        # mu is already clamped to [0,1] in __post_init__ for manual means and in gap mode.
        return float(self._rng.binomial(n=1, p=mu))

    def reset(self, seed: Optional[int] = None) -> None:
        """
        Reset the RNG. If means were random initially, resample them (and reapply gap mode).
        """
        if seed is not None:
            self.seed = seed
        self._rng = np.random.default_rng(self.seed)

        if self._means_were_random:
            self._means = self._sample_random_means()
            if self.gap is not None:
                self._apply_gap_mode(float(self.gap))

    def best_arm(self) -> int:
        """Return index of an arm with maximal mean (ties broken by smallest index)."""
        return int(np.argmax(self._means))

    def __repr__(self) -> str:
        return (
            f"StochasticBandit(n_arms={self.n_arms}, dist={self.dist!r}, "
            f"means={self.means_list}, gap={self.gap}, sigma={self.sigma}, seed={self.seed})"
        )


if __name__ == "__main__":
    """
    Gaussian with manual means
    b = StochasticBandit(n_arms=3, dist="gaussian", means=[0.0, 0.2, -0.5], seed=0)
    r = b.pull(1)

    Bernoulli with random means
    b = StochasticBandit(n_arms=5, dist="bernoulli", means=None, seed=123)
    print(b.means)

    Gaussian + gap mode gap=0.1
    b = StochasticBandit(n_arms=5, dist="gaussian", means=None, gap=0.1, seed=7)
    print(b.means)  # best stays, others become mu* - k*gap in descending rank order
    """
    b1 = StochasticBandit(n_arms=3, dist="gaussian", means=[0.0, 0.2, -0.5], seed=0)
    print("b1 means:", b1.means_list, "pull arm 1:", b1.pull(1))

    b2 = StochasticBandit(n_arms=5, dist="bernoulli", means=None, seed=123)
    print("b2 means:", b2.means_list, "best_arm:", b2.best_arm(), "pull best:", b2.pull(b2.best_arm()))

    b3 = StochasticBandit(n_arms=5, dist="gaussian", means=None, gap=0.1, seed=7)
    print("b3 means (gap):", b3.means_list, "best_arm:", b3.best_arm())