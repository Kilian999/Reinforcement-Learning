from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import numpy as np


Number = Union[int, float]


def _validate_dist(dist: str) -> str:
    d = dist.strip().lower()
    if d not in {"gaussian", "bernoulli"}:
        raise ValueError(f"dist must be 'gaussian' or 'bernoulli', got: {dist!r}")
    return d


def _as_means_list(means: Optional[Sequence[Number]], n_arms: int) -> Optional[List[float]]:
    if means is None:
        return None
    if len(means) != n_arms:
        raise ValueError(f"means must have length n_arms={n_arms}, got {len(means)}")
    return [float(m) for m in means]


def _clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


@dataclass
class StochasticBandit:
    """
    Stochastic multi-armed bandit with independent arms.

    dist:
        "gaussian" or "bernoulli"

    means:
        If None, means are sampled randomly:
          - gaussian: iid N(0,1)
          - bernoulli: iid Uniform(0,1)

    gap:
        Optional gap mode. If means are random and gap is given, keep best mean mu*
        and set remaining means according to mu* - k*gap in descending order.
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
        self._means_were_random = self.means is None

        means_list = _as_means_list(self.means, self.n_arms)

        if means_list is None:
            self._means = self._sample_random_means()
            if self.gap is not None:
                self._apply_gap_mode(float(self.gap))
        else:
            self._means = np.array(means_list, dtype=float)
            if self.dist == "bernoulli":
                self._means = _clamp01(self._means)

    @property
    def means_array(self) -> np.ndarray:
        return self._means.copy()

    @property
    def means_list(self) -> List[float]:
        return self._means.tolist()

    def _sample_random_means(self) -> np.ndarray:
        if self.dist == "gaussian":
            return self._rng.standard_normal(self.n_arms)
        return self._rng.uniform(0.0, 1.0, self.n_arms)

    def _apply_gap_mode(self, delta: float) -> None:
        if delta <= 0:
            raise ValueError(f"gap must be > 0, got {delta}")

        order_desc = np.argsort(self._means)[::-1]
        mu_star = float(self._means[order_desc[0]])
        new_means = self._means.copy()

        for k, arm_idx in enumerate(order_desc):
            if k == 0:
                new_means[arm_idx] = mu_star
            else:
                new_means[arm_idx] = mu_star - k * delta

        if self.dist == "bernoulli":
            new_means = _clamp01(new_means)

        self._means = new_means

    def pull(self, arm: int) -> float:
        if not (0 <= arm < self.n_arms):
            raise IndexError(f"arm index out of range: {arm} (n_arms={self.n_arms})")

        mu = float(self._means[arm])

        if self.dist == "gaussian":
            return float(self._rng.normal(loc=mu, scale=self.sigma))

        return float(self._rng.binomial(n=1, p=mu))

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.seed = seed
        self._rng = np.random.default_rng(self.seed)

        if self._means_were_random:
            self._means = self._sample_random_means()
            if self.gap is not None:
                self._apply_gap_mode(float(self.gap))

    def best_arm(self) -> int:
        return int(np.argmax(self._means))

    def best_mean(self) -> float:
        return float(np.max(self._means))

    def __repr__(self) -> str:
        return (
            f"StochasticBandit(n_arms={self.n_arms}, dist={self.dist!r}, "
            f"means={self.means_list}, gap={self.gap}, sigma={self.sigma}, seed={self.seed})"
        )