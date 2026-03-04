# sheet1/exp_bernoulli_compare.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

from .bandits import StochasticBandit
from .etc import ETC
from .algorithms import (
    Greedy,
    EpsilonGreedyFixed,
    EpsilonGreedyDecaying,
    UCB1,
    UCBSubGaussian,
    BoltzmannSoftmax,
    BoltzmannGumbel,
    PerturbedGreedy,
    GumbelPerturbedUCB,
    PolicyGradientSoftmax,
)


# -------------------------
# Streaming mean/variance over time (Welford)
# -------------------------

@dataclass
class RunningStatsTime:
    n: int
    mean: np.ndarray
    M2: np.ndarray
    k: int = 0

    @classmethod
    def zeros(cls, n: int) -> "RunningStatsTime":
        return cls(n=n, mean=np.zeros(n, dtype=float), M2=np.zeros(n, dtype=float), k=0)

    def update(self, x: np.ndarray) -> None:
        # x shape (n,)
        self.k += 1
        delta = x - self.mean
        self.mean += delta / self.k
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def var(self) -> np.ndarray:
        if self.k <= 1:
            return np.zeros(self.n, dtype=float)
        return self.M2 / (self.k - 1)


# -------------------------
# One run for one algorithm on fixed means
# -------------------------

AlgoFactory = Callable[[StochasticBandit, int], object]  # (bandit, seed) -> algo with step()

def run_one(
    means: np.ndarray,
    n: int,
    algo_factory: AlgoFactory,
    seed: int,
) -> Dict[str, np.ndarray]:
    """
    Runs one algorithm for n steps on a Bernoulli bandit with fixed means.
    Returns:
      - cum_regret (n,)
      - counts_end (K,)
      - est_means_end (K,)  (empirical from observed rewards)
      - true_means (K,)
    """
    K = len(means)

    bandit = StochasticBandit(n_arms=K, dist="bernoulli", means=means.tolist(), seed=seed)
    algo = algo_factory(bandit, seed)

    true_means = bandit.means_array
    best = int(np.argmax(true_means))
    mu_star = float(true_means[best])

    # External tracking (works for all algos, incl. policy gradient)
    counts = np.zeros(K, dtype=int)
    sums = np.zeros(K, dtype=float)
    cum_regret = np.zeros(n, dtype=float)

    running = 0.0
    for t in range(n):
        a, r = algo.step()
        a = int(a)
        r = float(r)

        counts[a] += 1
        sums[a] += r

        inst_reg = mu_star - float(true_means[a])  # expected regret
        running += inst_reg
        cum_regret[t] = running

    est_means = np.zeros(K, dtype=float)
    mask = counts > 0
    est_means[mask] = sums[mask] / counts[mask]

    return {
        "cum_regret": cum_regret,
        "counts_end": counts.astype(float),
        "est_means_end": est_means,
        "true_means": true_means,
    }


# -------------------------
# Aggregate N runs (means fixed per run shared across algos)
# -------------------------

@dataclass
class Summary:
    # time series
    regret_mean: np.ndarray
    regret_var: np.ndarray
    # end-of-horizon distributions for boxplots
    true_means_runs: np.ndarray       # (N,K)
    est_means_runs: np.ndarray        # (N,K)
    probs_runs: np.ndarray            # (N,K) = counts/n
    final_regret_runs: np.ndarray     # (N,)

def run_many(
    K: int,
    n: int,
    N: int,
    algo_factory: AlgoFactory,
    base_seed: int = 0,
) -> Summary:
    stats = RunningStatsTime.zeros(n)

    true_means_runs = np.zeros((N, K), dtype=float)
    est_means_runs = np.zeros((N, K), dtype=float)
    probs_runs = np.zeros((N, K), dtype=float)
    final_regret_runs = np.zeros(N, dtype=float)

    for i in range(N):
        # sample random Bernoulli means ~ Unif(0,1)
        rng = np.random.default_rng(base_seed + i)
        means = rng.uniform(0.0, 1.0, size=K)

        out = run_one(means=means, n=n, algo_factory=algo_factory, seed=10_000 * (base_seed + i) + 17)

        stats.update(out["cum_regret"])

        true_means_runs[i, :] = out["true_means"]
        est_means_runs[i, :] = out["est_means_end"]
        probs_runs[i, :] = out["counts_end"] / n
        final_regret_runs[i] = out["cum_regret"][-1]

    return Summary(
        regret_mean=stats.mean,
        regret_var=stats.var(),
        true_means_runs=true_means_runs,
        est_means_runs=est_means_runs,
        probs_runs=probs_runs,
        final_regret_runs=final_regret_runs,
    )


# -------------------------
# Pilot parameter search (cheap)
# -------------------------

def pick_best_param(
    K: int,
    n_pilot: int,
    N_pilot: int,
    factories: Dict[str, AlgoFactory],
    base_seed: int = 0,
) -> str:
    """
    Choose best key by minimal mean final regret on a pilot experiment.
    """
    best_key = None
    best_val = float("inf")

    for key, fac in factories.items():
        summ = run_many(K=K, n=n_pilot, N=N_pilot, algo_factory=fac, base_seed=base_seed)
        val = float(np.mean(summ.final_regret_runs))
        if val < best_val:
            best_val = val
            best_key = key

    assert best_key is not None
    return best_key


# -------------------------
# Plotting
# -------------------------

def plot_regret_with_ci(
    summaries: Dict[str, Summary],
    n: int,
    title: str,
) -> None:
    t = np.arange(n)
    plt.figure(figsize=(12, 6))
    plt.title(title)
    plt.xlabel("t")
    plt.ylabel("Cumulative regret")

    for name, summ in summaries.items():
        mean = summ.regret_mean
        se = np.sqrt(summ.regret_var / max(1, summ.true_means_runs.shape[0]))
        ci = 1.96 * se
        plt.plot(t, mean, label=name)
        plt.fill_between(t, mean - ci, mean + ci, alpha=0.15)

    plt.legend()
    plt.tight_layout()
    plt.show()


def boxplot_true_vs_estimates(
    summaries: Dict[str, Summary],
    arm: int,
    title: str,
) -> None:
    """
    For a fixed arm index, show boxplots of:
      - true mean (same distribution across runs)
      - each algorithm's estimated mean
    """
    labels = ["true"] + list(summaries.keys())
    data = [next(iter(summaries.values())).true_means_runs[:, arm]]  # true distribution (identical per algo if same seeds)
    for name in summaries.keys():
        data.append(summaries[name].est_means_runs[:, arm])

    plt.figure(figsize=(12, 5))
    plt.title(f"{title} (arm {arm})")
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.ylabel("mean / estimate")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.show()


def boxplot_probs_per_arm(
    summaries: Dict[str, Summary],
    title: str,
) -> None:
    """
    For each algorithm, boxplot of probabilities for each arm.
    Produces one figure per algorithm (cleaner than huge combined plot).
    """
    for name, summ in summaries.items():
        K = summ.probs_runs.shape[1]
        data = [summ.probs_runs[:, a] for a in range(K)]
        plt.figure(figsize=(10, 5))
        plt.title(f"{title} — {name}")
        plt.boxplot(data, labels=[f"a{a}" for a in range(K)], showfliers=False)
        plt.ylabel("P(play arm)")
        plt.tight_layout()
        plt.show()


def boxplot_final_regret(
    summaries: Dict[str, Summary],
    title: str,
) -> None:
    labels = list(summaries.keys())
    data = [summaries[name].final_regret_runs for name in labels]
    plt.figure(figsize=(12, 5))
    plt.title(title)
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.ylabel("Final cumulative regret")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.show()


# -------------------------
# Algorithm factories
# -------------------------

def make_factories() -> Dict[str, AlgoFactory]:
    """
    Factories return an algorithm instance with .step().
    Seeds are forwarded where meaningful.
    """
    return {
        # (a) Greedy family
        "Greedy": lambda bandit, seed: Greedy(bandit, seed=seed),
        "eps-greedy(0.1)": lambda bandit, seed: EpsilonGreedyFixed(bandit, epsilon=0.1, seed=seed),
        "eps-decay(eps0=1.0/sqrt)": lambda bandit, seed: EpsilonGreedyDecaying(bandit, eps0=1.0, seed=seed),

        # ETC (from previous sheet) – include if exercise 4 included it
        "ETC(m=20)": lambda bandit, seed: ETC(bandit, exploration_rounds=20),

        # (b) UCB
        "UCB1(c=2)": lambda bandit, seed: UCB1(bandit, c=2.0, seed=seed),
        "UCB-subG(sigma=1)": lambda bandit, seed: UCBSubGaussian(bandit, sigma=1.0, seed=seed),

        # (c) Boltzmann / perturbations
        "Boltzmann(tau=0.5)": lambda bandit, seed: BoltzmannSoftmax(bandit, tau=0.5, seed=seed),
        "Boltzmann-Gumbel(tau=0.5)": lambda bandit, seed: BoltzmannGumbel(bandit, tau=0.5, seed=seed),
        "Perturbed(Cauchy,scale=0.1)": lambda bandit, seed: PerturbedGreedy(bandit, dist="cauchy", scale=0.1, seed=seed),
        "GumbelPertUCB(C=1.0)": lambda bandit, seed: GumbelPerturbedUCB(bandit, C=1.0, seed=seed),

        # (d) Policy gradient
        "PG(alpha=0.1)": lambda bandit, seed: PolicyGradientSoftmax(bandit, alpha=0.1, use_baseline=False, seed=seed),
        "PG+baseline(alpha=0.1)": lambda bandit, seed: PolicyGradientSoftmax(bandit, alpha=0.1, use_baseline=True, seed=seed),
    }


def tune_some_params(K: int) -> Dict[str, AlgoFactory]:
    """
    Example: numerically pick good parameters cheaply for a subset (tau, epsilon, alpha, etc.).
    Uses pilot runs.
    """
    tuned: Dict[str, AlgoFactory] = {}

    # ε grid
    eps_grid = [0.01, 0.05, 0.1, 0.2]
    eps_factories = {f"eps={e}": (lambda e=e: (lambda b, s: EpsilonGreedyFixed(b, epsilon=e, seed=s)))() for e in eps_grid}
    best_eps = pick_best_param(K, n_pilot=2000, N_pilot=150, factories=eps_factories, base_seed=123)
    best_eps_val = float(best_eps.split("=")[1])
    tuned[f"eps-greedy({best_eps_val})"] = lambda bandit, seed, e=best_eps_val: EpsilonGreedyFixed(bandit, epsilon=e, seed=seed)

    # tau grid (Boltzmann)
    tau_grid = [0.1, 0.2, 0.5, 1.0]
    tau_factories = {f"tau={t}": (lambda t=t: (lambda b, s: BoltzmannSoftmax(b, tau=t, seed=s)))() for t in tau_grid}
    best_tau = pick_best_param(K, n_pilot=2000, N_pilot=150, factories=tau_factories, base_seed=456)
    best_tau_val = float(best_tau.split("=")[1])
    tuned[f"Boltzmann(tau={best_tau_val})"] = lambda bandit, seed, t=best_tau_val: BoltzmannSoftmax(bandit, tau=t, seed=seed)

    # alpha grid (Policy Gradient)
    alpha_grid = [0.01, 0.05, 0.1, 0.2]
    alpha_factories = {f"alpha={a}": (lambda a=a: (lambda b, s: PolicyGradientSoftmax(b, alpha=a, use_baseline=True, seed=s)))() for a in alpha_grid}
    best_alpha = pick_best_param(K, n_pilot=2000, N_pilot=150, factories=alpha_factories, base_seed=789)
    best_alpha_val = float(best_alpha.split("=")[1])
    tuned[f"PG+baseline(alpha={best_alpha_val})"] = lambda bandit, seed, a=best_alpha_val: PolicyGradientSoftmax(bandit, alpha=a, use_baseline=True, seed=seed)

    return tuned


# -------------------------
# Main
# -------------------------

def main() -> None:
    K = 5
    n = 10_000
    N = 1_000

    # Start with default set
    factories = make_factories()

    # Optional: replace some with tuned versions (cheap pilot search)
    tuned = tune_some_params(K)
    # overwrite defaults with tuned (and keep others)
    factories.update(tuned)

    # Run all algorithms
    summaries: Dict[str, Summary] = {}
    for name, fac in factories.items():
        print(f"Running: {name}")
        summaries[name] = run_many(K=K, n=n, N=N, algo_factory=fac, base_seed=0)

    # (a) regret curves with 95% CI
    plot_regret_with_ci(summaries, n=n, title=f"5-armed Bernoulli, N={N}, n={n}: Cumulative Regret (95% CI)")

    # (b.i) true means vs estimates (boxplot per arm)
    for arm in range(K):
        boxplot_true_vs_estimates(summaries, arm=arm, title="True means vs algorithm estimates at horizon")

    # (b.ii) probabilities of choosing each arm (per algorithm)
    boxplot_probs_per_arm(summaries, title="Choice probabilities at horizon (counts/n)")

    # (b.iii) final regrets
    boxplot_final_regret(summaries, title="Final cumulative regret at horizon")


if __name__ == "__main__":
    main()