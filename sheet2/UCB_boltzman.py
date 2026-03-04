

# etc.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Literal, Union

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




BanditLike = object
Action = int
Reward = float


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / np.sum(ex)


def _safe_argmax(x: np.ndarray) -> int:
    # deterministic tie-break: smallest index
    return int(np.argmax(x))


def _require_bandit(bandit: BanditLike) -> Tuple[int, Callable[[int], float]]:
    if not hasattr(bandit, "n_arms") or not hasattr(bandit, "pull"):
        raise TypeError("bandit must have attribute n_arms and method pull(arm)->reward")
    K = int(bandit.n_arms)
    if K <= 0:
        raise ValueError("bandit.n_arms must be positive")
    pull = bandit.pull  # type: ignore
    return K, pull


@dataclass
class BaseAlgo:
    bandit: BanditLike
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        self.K, self._pull = _require_bandit(self.bandit)
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

    def step(self) -> Tuple[Action, Reward]:
        raise NotImplementedError


# -------------------------
# (a) Greedy / Epsilon-Greedy
# -------------------------

@dataclass
class Greedy(BaseAlgo):
    """Purely greedy: pick argmax empirical mean (after warm-start)."""

    def step(self) -> Tuple[int, float]:
        # Warm-start: play each arm once to avoid all-zero means bias
        unplayed = np.where(self.counts == 0)[0]
        if unplayed.size > 0:
            a = int(unplayed[0])
        else:
            a = _safe_argmax(self.empirical_means())

        r = float(self._pull(a))
        self._update(a, r)
        return a, r


@dataclass
class EpsilonGreedyFixed(BaseAlgo):
    """ε-greedy with constant ε."""
    epsilon: float = 0.1

    def __post_init__(self) -> None:
        super().__post_init__()
        if not (0.0 <= self.epsilon <= 1.0):
            raise ValueError("epsilon must be in [0,1]")

    def step(self) -> Tuple[int, float]:
        unplayed = np.where(self.counts == 0)[0]
        if unplayed.size > 0:
            a = int(unplayed[0])
        else:
            if self.rng.random() < self.epsilon:
                a = int(self.rng.integers(0, self.K))
            else:
                a = _safe_argmax(self.empirical_means())

        r = float(self._pull(a))
        self._update(a, r)
        return a, r


EpsSchedule = Callable[[int], float]


@dataclass
class EpsilonGreedyDecaying(BaseAlgo):
    """
    ε-greedy with ε_t decreasing over time.
    Default: ε_t = min(1, eps0 / sqrt(t+1))
    """
    eps0: float = 1.0
    schedule: Optional[EpsSchedule] = None

    def step(self) -> Tuple[int, float]:
        unplayed = np.where(self.counts == 0)[0]
        if unplayed.size > 0:
            a = int(unplayed[0])
        else:
            if self.schedule is not None:
                eps = float(self.schedule(self.t))
            else:
                eps = min(1.0, self.eps0 / np.sqrt(self.t + 1.0))

            eps = float(np.clip(eps, 0.0, 1.0))

            if self.rng.random() < eps:
                a = int(self.rng.integers(0, self.K))
            else:
                a = _safe_argmax(self.empirical_means())

        r = float(self._pull(a))
        self._update(a, r)
        return a, r


# -------------------------
# (b) UCB
# -------------------------

@dataclass
class UCB1(BaseAlgo):
    """
    Standard UCB (lecture-style): choose argmax( Qhat_a + bonus_a )
    bonus_a = sqrt( (2 * log(t)) / N_a ) (common default)
    """
    c: float = 2.0  # multiplier inside log-term

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.c <= 0:
            raise ValueError("c must be positive")

    def step(self) -> Tuple[int, float]:
        unplayed = np.where(self.counts == 0)[0]
        if unplayed.size > 0:
            a = int(unplayed[0])
        else:
            t_eff = max(1, self.t)
            q = self.empirical_means()
            bonus = np.sqrt((self.c * np.log(t_eff)) / self.counts)
            a = _safe_argmax(q + bonus)

        r = float(self._pull(a))
        self._update(a, r)
        return a, r


@dataclass
class UCBSubGaussian(BaseAlgo):
    """
    σ-subgaussian adapted UCB:
      bonus_a = sigma * sqrt( (2 * log(t)) / N_a )
    (constant factors depend on lecture; this is the canonical scaling)
    """
    sigma: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")

    def step(self) -> Tuple[int, float]:
        unplayed = np.where(self.counts == 0)[0]
        if unplayed.size > 0:
            a = int(unplayed[0])
        else:
            t_eff = max(1, self.t)
            q = self.empirical_means()
            bonus = self.sigma * np.sqrt((2.0 * np.log(t_eff)) / self.counts)
            a = _safe_argmax(q + bonus)

        r = float(self._pull(a))
        self._update(a, r)
        return a, r


# -------------------------
# (c) Boltzmann Exploration (+ perturbations)
# -------------------------

TauSchedule = Callable[[int], float]


@dataclass
class BoltzmannSoftmax(BaseAlgo):
    """
    Simple Boltzmann exploration:
      P(A=a) ∝ exp(Qhat_a / tau_t)
    """
    tau: float = 0.5
    tau_schedule: Optional[TauSchedule] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.tau <= 0:
            raise ValueError("tau must be positive")

    def step(self) -> Tuple[int, float]:
        # warm-start
        unplayed = np.where(self.counts == 0)[0]
        if unplayed.size > 0:
            a = int(unplayed[0])
        else:
            tau_t = float(self.tau_schedule(self.t)) if self.tau_schedule else float(self.tau)
            tau_t = max(1e-12, tau_t)
            q = self.empirical_means()
            p = _softmax(q / tau_t)
            a = int(self.rng.choice(self.K, p=p))

        r = float(self._pull(a))
        self._update(a, r)
        return a, r


@dataclass
class BoltzmannGumbel(BaseAlgo):
    """
    Boltzmann via Gumbel trick:
      A = argmax_a ( Qhat_a / tau + G_a ),  G_a ~ Gumbel(0,1) i.i.d.
    """
    tau: float = 0.5

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.tau <= 0:
            raise ValueError("tau must be positive")

    def step(self) -> Tuple[int, float]:
        unplayed = np.where(self.counts == 0)[0]
        if unplayed.size > 0:
            a = int(unplayed[0])
        else:
            q = self.empirical_means()
            g = self.rng.gumbel(loc=0.0, scale=1.0, size=self.K)
            a = _safe_argmax(q / self.tau + g)

        r = float(self._pull(a))
        self._update(a, r)
        return a, r


NoiseDist = Literal["gumbel", "cauchy", "beta", "betaprime", "chi"]


def _sample_noise(rng: np.random.Generator, dist: NoiseDist, size: int,
                  a: float = 2.0, b: float = 2.0, df: float = 3.0) -> np.ndarray:
    """
    Standard-ish noise samplers without scipy.
    - gumbel: standard Gumbel(0,1)
    - cauchy: standard Cauchy
    - beta(a,b): Beta in [0,1] (not centered)
    - betaprime(a,b): X/Y with X~Gamma(a,1), Y~Gamma(b,1)
    - chi(df): sqrt(ChiSquare(df))
    """
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
class PerturbedGreedy(BaseAlgo):
    """
    'Arbitrary distribution' perturbation selection:
      A = argmax_a ( Qhat_a + scale * Z_a )
    where Z_a i.i.d. from chosen distribution.
    """
    dist: NoiseDist = "gumbel"
    scale: float = 1.0
    # params for beta/betaprime/chi
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
            a = int(unplayed[0])
        else:
            q = self.empirical_means()
            z = _sample_noise(self.rng, self.dist, self.K, a=self.a, b=self.b, df=self.df)
            a = _safe_argmax(q + self.scale * z)

        r = float(self._pull(a))
        self._update(a, r)
        return a, r


@dataclass
class GumbelPerturbedUCB(BaseAlgo):
    """
    Implements the requested variant:
      A_t ∈ argmax_a ( Qhat_a(t-1) + sqrt(C / T_a(t-1)) * Z_a )
    with Z_a i.i.d. standard Gumbel and C in R (typically C>0).
    """
    C: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.C <= 0:
            raise ValueError("C must be positive")

    def step(self) -> Tuple[int, float]:
        unplayed = np.where(self.counts == 0)[0]
        if unplayed.size > 0:
            a = int(unplayed[0])
        else:
            q = self.empirical_means()
            z = self.rng.gumbel(0.0, 1.0, size=self.K)
            bonus = np.sqrt(self.C / self.counts.astype(float)) * z
            a = _safe_argmax(q + bonus)

        r = float(self._pull(a))
        self._update(a, r)
        return a, r


# -------------------------
# (d) Policy Gradient (REINFORCE) with softmax policy
# -------------------------

@dataclass
class PolicyGradientSoftmax:
    """
    REINFORCE on bandits with softmax policy:
      π_θ(a) = exp(θ_a) / sum_b exp(θ_b)
    Update:
      θ <- θ + α (R - b) * ∇_θ log π_θ(A)
    For softmax: ∇ log π(a) = 1_a - π
    Baseline b can be off (0) or running mean reward.
    """
    bandit: BanditLike
    alpha: float = 0.1
    use_baseline: bool = False
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        self.K, self._pull = _require_bandit(self.bandit)
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        self.rng = np.random.default_rng(self.seed)

        self.t: int = 0
        self.theta = np.zeros(self.K, dtype=float)

        # baseline: running average of rewards
        self.baseline: float = 0.0

        # for monitoring only
        self.counts = np.zeros(self.K, dtype=int)
        self.sums = np.zeros(self.K, dtype=float)

    def policy(self) -> np.ndarray:
        return _softmax(self.theta)

    def step(self) -> Tuple[int, float]:
        pi = self.policy()
        a = int(self.rng.choice(self.K, p=pi))
        r = float(self._pull(a))

        # update monitoring stats
        self.counts[a] += 1
        self.sums[a] += r

        b = self.baseline if self.use_baseline else 0.0
        adv = r - b

        # gradient of log pi(a): one_hot(a) - pi
        grad = -pi
        grad[a] += 1.0
        self.theta += self.alpha * adv * grad

        # update baseline AFTER using it
        if self.use_baseline:
            # running mean baseline
            self.baseline += (r - self.baseline) / (self.t + 1.0)

        self.t += 1
        return a, r

