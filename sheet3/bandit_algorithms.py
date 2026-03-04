# simulate_bandits.py
# Usage example:
#   python simulate_bandits.py
#
# Assumes you have these classes available somewhere (adjust imports):
# - StochasticBandit (your environment)
# - ETC (from etc.py)
# - EpsilonGreedyFixed, EpsilonGreedyDecaying, UCB1, BoltzmannSoftmax, PolicyGradientSoftmax (from your algos file)

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

import time

from sheet1.bandits import StochasticBandit
from sheet1.etc import ETC
from sheet2.UCB_boltzman import (
     EpsilonGreedyFixed, EpsilonGreedyDecaying,
     UCB1, BoltzmannSoftmax, PolicyGradientSoftmax
)

# If your classes are in the same file/notebook, you can remove these imports.

# -----------------------------
# Helpers
# -----------------------------

def cumulative_pseudo_regret(arms: np.ndarray, means: np.ndarray) -> np.ndarray:
    """
    Pseudo-regret: sum_t (mu* - mu_{A_t})
    """
    mu_star = float(np.max(means))
    inst_regret = mu_star - means[arms]
    return np.cumsum(inst_regret)


def run_algo(algo: object, means: np.ndarray, T: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generic runner: calls algo.step() T times, returns (arms, cum_pseudo_regret).
    """
    arms = np.zeros(T, dtype=int)
    for t in range(T):
        a, _r = algo.step()
        arms[t] = int(a)
    cum_reg = cumulative_pseudo_regret(arms, means)
    return arms, cum_reg


def mean_and_stderr(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    x shape: (n_runs, T)
    returns (mean, stderr) each shape (T,)
    """
    m = x.mean(axis=0)
    s = x.std(axis=0, ddof=1) / np.sqrt(x.shape[0])
    return m, s


# -----------------------------
# Experiment configuration
# -----------------------------

@dataclass
class BanditSpec:
    dist: str = "bernoulli"   # "bernoulli" or "gaussian"
    K: int = 10
    sigma: float = 1.0        # only used for gaussian rewards
    gap: Optional[float] = 0.1
    seed: int = 0             # for generating means per run


def make_fixed_means_bandit(spec: BanditSpec, means: np.ndarray, seed: int):
    """
    Build a bandit with fixed means and its own reward RNG seed.
    """
    return StochasticBandit(
        n_arms=spec.K,
        dist=spec.dist,
        means=means.tolist(),
        gap=None,              # IMPORTANT: means are fixed; gap mode should not reapply
        sigma=spec.sigma,
        seed=seed,
    )


def sample_means(spec: BanditSpec, rng: np.random.Generator) -> np.ndarray:
    """
    Sample means according to your environment rules,
    then optionally apply 'gap mode' exactly like your StochasticBandit does.
    """
    if spec.dist == "gaussian":
        means = rng.standard_normal(spec.K)
    elif spec.dist == "bernoulli":
        means = rng.uniform(0.0, 1.0, spec.K)
    else:
        raise ValueError("dist must be 'gaussian' or 'bernoulli'")

    if spec.gap is not None:
        delta = float(spec.gap)
        if delta <= 0:
            raise ValueError("gap must be > 0")

        order_desc = np.argsort(means)[::-1]
        mu_star = float(means[order_desc[0]])
        new_means = means.copy()
        for k, arm_idx in enumerate(order_desc):
            if k == 0:
                new_means[arm_idx] = mu_star
            else:
                new_means[arm_idx] = mu_star - k * delta

        if spec.dist == "bernoulli":
            new_means = np.clip(new_means, 0.0, 1.0)

        means = new_means

    return means


# -----------------------------
# Algorithms factory
# -----------------------------

@dataclass
class AlgoSpec:
    name: str
    ctor: Callable[..., object]
    kwargs: Dict


def build_algos() -> List[AlgoSpec]:
    """
    Pick reasonable default hyperparameters.
    You can/should tune these a bit and report what you used.
    """
 


    def tau_schedule(t: int) -> float:
        # mild decay: starts exploratory, becomes greedier
        return 0.5 / np.sqrt(t + 1.0)

    algos: List[AlgoSpec] = [
        AlgoSpec(
            name="ETC (m_per_arm=10)",
            ctor=ETC,
            kwargs={"exploration_rounds": 10},
        ),
        AlgoSpec(
            name="ε-greedy (ε=0.1)",
            ctor=EpsilonGreedyFixed,
            kwargs={"epsilon": 0.1},
        ),
        AlgoSpec(
            name="ε-greedy (ε_t=1/sqrt(t))",
            ctor=EpsilonGreedyDecaying,
            kwargs={"eps0": 1.0, "schedule": None},
        ),
        AlgoSpec(
            name="UCB1 (c=2)",
            ctor=UCB1,
            kwargs={"c": 2.0},
        ),
        AlgoSpec(
            name="Boltzmann (τ_t=0.5/sqrt(t))",
            ctor=BoltzmannSoftmax,
            kwargs={"tau": 0.5, "tau_schedule": tau_schedule},
        ),
        AlgoSpec(
            name="PolicyGrad (α=0.05, baseline)",
            ctor=PolicyGradientSoftmax,
            kwargs={"alpha": 0.05, "use_baseline": True},
        ),
    ]
    return algos


# -----------------------------
# Main evaluation loop
# -----------------------------


def evaluate(
    spec: BanditSpec,
    T: int = 20_000,
    n_runs: int = 100,
    base_seed: int = 123,
    show_step_progress: bool = False,   # <- optional
    step_every: int = 2000,             # <- optional
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Returns:
      t_grid: (T,)
      algo_names: list of names length A
      regrets: (A, n_runs, T) cumulative pseudo-regret
    """
    algos = build_algos()
    A = len(algos)
    regrets = np.zeros((A, n_runs, T), dtype=float)

    means_rng = np.random.default_rng(spec.seed)

    total_jobs = n_runs * A
    job_done = 0
    t0 = time.time()

    def fmt_time(sec: float) -> str:
        sec = max(0.0, sec)
        m, s = divmod(int(sec), 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h:d}h{m:02d}m{s:02d}s"
        return f"{m:d}m{s:02d}s"

    for run in range(n_runs):
        # Fix a bandit instance (means) for this run; reuse for all algos.
        means = sample_means(spec, means_rng)

        for j, algo_spec in enumerate(algos):
            # Each algo gets its own reward RNG seed (but same means).
            bandit_seed = base_seed + 10_000 * run + 100 * j
            bandit = make_fixed_means_bandit(spec, means, seed=bandit_seed)

            # Algo seed separate from bandit seed
            algo_seed = base_seed + 20_000 * run + 200 * j + 7

            # Instantiate algo; note ETC signature differs (no seed arg)
            if algo_spec.ctor.__name__ == "ETC":
                algo = algo_spec.ctor(bandit=bandit, **algo_spec.kwargs)
            else:
                algo = algo_spec.ctor(bandit=bandit, seed=algo_seed, **algo_spec.kwargs)

            # ---- progress header for this job
            job_done += 1
            elapsed = time.time() - t0
            avg_per_job = elapsed / job_done
            eta = avg_per_job * (total_jobs - job_done)
            pct = 100.0 * job_done / total_jobs

            print(
                f"[{job_done:4d}/{total_jobs}] {pct:6.2f}%  "
                f"run {run+1:3d}/{n_runs}, algo {j+1:2d}/{A}: {algo_spec.name}  "
                f"elapsed {fmt_time(elapsed)}  ETA {fmt_time(eta)}",
                flush=True
            )

            # ---- run algo (optional step progress)
            if not show_step_progress:
                _arms, cum_reg = run_algo(algo, means=means, T=T)
                regrets[j, run, :] = cum_reg
            else:
                arms = np.zeros(T, dtype=int)
                for t in range(T):
                    a, _r = algo.step()
                    arms[t] = int(a)
                    if (t + 1) % step_every == 0 or (t + 1) == T:
                        print(f"    steps {t+1:6d}/{T}", end="\r", flush=True)
                if T >= step_every:
                    print(" " * 40, end="\r")  # clear line
                regrets[j, run, :] = cumulative_pseudo_regret(arms, means)

    t_grid = np.arange(1, T + 1)
    names = [a.name for a in algos]
    return t_grid, names, regrets


def plot_regrets(t: np.ndarray, names: List[str], regrets: np.ndarray, title: str) -> None:
    """
    regrets shape: (A, n_runs, T)
    """
    plt.figure()
    for j, name in enumerate(names):
        m, se = mean_and_stderr(regrets[j])
        plt.plot(t, m, label=name)
        plt.fill_between(t, m - 2 * se, m + 2 * se, alpha=0.2)
    plt.xlabel("t")
    plt.ylabel("cumulative pseudo-regret")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    # log-x version (often helps see O(log t) vs O(sqrt t) vs O(t))
    plt.figure()
    for j, name in enumerate(names):
        m, se = mean_and_stderr(regrets[j])
        plt.plot(t, m, label=name)
        plt.fill_between(t, m - 2 * se, m + 2 * se, alpha=0.2)
    plt.xscale("log")
    plt.xlabel("t (log scale)")
    plt.ylabel("cumulative pseudo-regret")
    plt.title(title + " (log x-axis)")
    plt.legend()
    plt.tight_layout()

    # log-log version (rough “slope” impression)
    plt.figure()
    for j, name in enumerate(names):
        m, se = mean_and_stderr(regrets[j])
        plt.plot(t, m, label=name)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("t (log)")
    plt.ylabel("regret (log)")
    plt.title(title + " (log-log)")
    plt.legend()
    plt.tight_layout()


if __name__ == "__main__":
    # Example: Bernoulli gap bandit (clean gap-dependent behavior)
    spec = BanditSpec(dist="bernoulli", K=10, gap=0.1, seed=0)

    T = 20_000
    n_runs = 100

    t, names, regrets = evaluate(spec, T=T, n_runs=n_runs, base_seed=123)
    plot_regrets(t, names, regrets, title=f"Bernoulli bandit (K={spec.K}, gap={spec.gap})")

    plt.show()