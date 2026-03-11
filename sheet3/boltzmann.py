from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional, Tuple

import numpy as np

from .utilities import require_bandit, safe_argmax, softmax


TauSchedule = Callable[[int], float]
NoiseDist = Literal["gumbel", "cauchy", "beta", "betaprime", "chi"]


def _sample_noise(
    rng: np.random.Generator,
    dist: NoiseDist,
    size: int,
    a: float = 2.0,
    b: float = 2.0,
    df: float = 3.0,
) -> np.ndarray:
    if dist == "gumbel":
        return rng.gumbel(0.0, 1.0, size=size)
    if dist == "cauchy":
        return rng.standard_cauchy(size=size)
    if dist == "beta":
        return rng.beta(a, b, size=size)
    if dist == "betaprime":
        x = rng.gamma(shape=a, scale=1.0, size=size)
        y = rng.gamma(shape=b, scale=1.0, size=size)
        return x / y
    if dist == "chi":
        return np.sqrt(rng.chisquare(df=df, size=size))
    raise ValueError(f"Unknown dist: {dist}")


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
class BoltzmannSoftmax(BaseAlgo):
    """
    P(A=a) proportional to exp(Qhat_a / tau_t)
    """
    tau: float = 0.5
    tau_schedule: Optional[TauSchedule] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.tau <= 0:
            raise ValueError("tau must be positive")

    def step(self) -> Tuple[int, float]:
        unplayed = np.where(self.counts == 0)[0]
        if unplayed.size > 0:
            arm = int(unplayed[0])
        else:
            tau_t = float(self.tau_schedule(self.t)) if self.tau_schedule else float(self.tau)
            tau_t = max(1e-12, tau_t)
            q = self.empirical_means()
            probs = softmax(q / tau_t)
            arm = int(self.rng.choice(self.K, p=probs))

        reward = float(self._pull(arm))
        self._update(arm, reward)
        return arm, reward


@dataclass
class BoltzmannGumbel(BaseAlgo):
    """
    Gumbel trick:
        argmax_a [Qhat_a / tau + G_a]
    """
    tau: float = 0.5

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.tau <= 0:
            raise ValueError("tau must be positive")

    def step(self) -> Tuple[int, float]:
        unplayed = np.where(self.counts == 0)[0]
        if unplayed.size > 0:
            arm = int(unplayed[0])
        else:
            q = self.empirical_means()
            g = self.rng.gumbel(loc=0.0, scale=1.0, size=self.K)
            arm = safe_argmax(q / self.tau + g)

        reward = float(self._pull(arm))
        self._update(arm, reward)
        return arm, reward


@dataclass
class PerturbedGreedy(BaseAlgo):
    """
    argmax_a [Qhat_a + scale * Z_a]
    """
    dist: NoiseDist = "gumbel"
    scale: float = 1.0
    a: float = 2.0
    b: float = 2.0
    df: float = 3.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.scale <= 0:
            raise ValueError("scale must be positive")

    def step(self) -> Tuple[int, float]:
        unplayed = np.where(self.counts == 0)[0]
        if unplayed.size > 0:
            arm = int(unplayed[0])
        else:
            q = self.empirical_means()
            z = _sample_noise(self.rng, self.dist, self.K, a=self.a, b=self.b, df=self.df)
            arm = safe_argmax(q + self.scale * z)

        reward = float(self._pull(arm))
        self._update(arm, reward)
        return arm, reward


@dataclass
class GumbelPerturbedUCB(BaseAlgo):
    """
    argmax_a [Qhat_a + sqrt(C / N_a) * Z_a], Z_a ~ Gumbel(0,1)
    """
    C: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.C <= 0:
            raise ValueError("C must be positive")

    def step(self) -> Tuple[int, float]:
        unplayed = np.where(self.counts == 0)[0]
        if unplayed.size > 0:
            arm = int(unplayed[0])
        else:
            q = self.empirical_means()
            z = self.rng.gumbel(0.0, 1.0, size=self.K)
            bonus = np.sqrt(self.C / self.counts.astype(float)) * z
            arm = safe_argmax(q + bonus)

        reward = float(self._pull(arm))
        self._update(arm, reward)
        return arm, reward