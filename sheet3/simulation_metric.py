from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import time
import numpy as np
import matplotlib.pyplot as plt

from .bandit import StochasticBandit
from .etc import ETC
from .greedy import EpsilonGreedyFixed, EpsilonGreedyDecaying
from .ucb import UCB1
from .boltzmann import BoltzmannSoftmax
from .policy_gradient import PolicyGradientSoftmax


# =========================================================
# Helpers
# =========================================================

def cumulative_pseudo_regret(arms: np.ndarray, means: np.ndarray) -> np.ndarray:
    """
    Pseudo-regret:
        R_T = sum_{t=1}^T (mu* - mu_{A_t})
    """
    mu_star = float(np.max(means))
    inst_regret = mu_star - means[arms]
    return np.cumsum(inst_regret)


def run_algo(algo: object, means: np.ndarray, T: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generic runner: calls algo.step() T times.
    Returns:
        arms: shape (T,)
        cum_reg: cumulative pseudo-regret, shape (T,)
    """
    arms = np.zeros(T, dtype=int)
    for t in range(T):
        a, _r = algo.step()
        arms[t] = int(a)

    cum_reg = cumulative_pseudo_regret(arms, means)
    return arms, cum_reg


def mean_and_stderr(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Input:
        x shape = (n_runs, T)
    Output:
        mean, stderr each shape = (T,)
    """
    m = x.mean(axis=0)
    s = x.std(axis=0, ddof=1) / np.sqrt(x.shape[0])
    return m, s


# =========================================================
# Experiment configuration
# =========================================================

@dataclass
class BanditSpec:
    dist: str = "bernoulli"   # "bernoulli" or "gaussian"
    K: int = 10
    sigma: float = 1.0        # only used for gaussian rewards
    gap: Optional[float] = 0.1
    seed: int = 0             # for generating means across runs


def make_fixed_means_bandit(spec: BanditSpec, means: np.ndarray, seed: int) -> StochasticBandit:
    """
    Build a bandit with fixed means and its own reward RNG seed.
    """
    return StochasticBandit(
        n_arms=spec.K,
        dist=spec.dist,
        means=means.tolist(),
        gap=None,      # means are already fixed; do not reapply gap mode
        sigma=spec.sigma,
        seed=seed,
    )


def sample_means(spec: BanditSpec, rng: np.random.Generator) -> np.ndarray:
    """
    Sample means exactly in the same spirit as the environment:
      - gaussian: iid N(0,1)
      - bernoulli: iid Uniform(0,1)
    Then optionally apply gap mode.
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


# =========================================================
# Algorithms factory
# =========================================================

@dataclass
class AlgoSpec:
    name: str
    ctor: Callable[..., object]
    kwargs: Dict


def build_algos() -> List[AlgoSpec]:
    """
    Choose a baseline set of algorithms and hyperparameters.
    """

    def tau_schedule(t: int) -> float:
        return 0.5 / np.sqrt(t + 1.0)

    algos: List[AlgoSpec] = [
        AlgoSpec(
            name="ETC (m_per_arm=10)",
            ctor=ETC,
            kwargs={"exploration_rounds": 10},
        ),
        AlgoSpec(
            name="epsilon-greedy (epsilon=0.1)",
            ctor=EpsilonGreedyFixed,
            kwargs={"epsilon": 0.1},
        ),
        AlgoSpec(
            name="epsilon-greedy (epsilon_t=1/sqrt(t))",
            ctor=EpsilonGreedyDecaying,
            kwargs={"eps0": 1.0, "schedule": None},
        ),
        AlgoSpec(
            name="UCB1 (c=2)",
            ctor=UCB1,
            kwargs={"c": 2.0},
        ),
        AlgoSpec(
            name="Boltzmann (tau_t=0.5/sqrt(t))",
            ctor=BoltzmannSoftmax,
            kwargs={"tau": 0.5, "tau_schedule": tau_schedule},
        ),
        AlgoSpec(
            name="PolicyGrad (alpha=0.05, baseline)",
            ctor=PolicyGradientSoftmax,
            kwargs={"alpha": 0.05, "use_baseline": True},
        ),
    ]
    return algos


# =========================================================
# Main evaluation loop: regret
# =========================================================

def evaluate(
    spec: BanditSpec,
    T: int = 20_000,
    n_runs: int = 100,
    base_seed: int = 123,
    show_step_progress: bool = False,
    step_every: int = 2000,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Returns:
        t_grid: shape (T,)
        algo_names: list of length A
        regrets: shape (A, n_runs, T)
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
        means = sample_means(spec, means_rng)

        for j, algo_spec in enumerate(algos):
            bandit_seed = base_seed + 10_000 * run + 100 * j
            algo_seed = base_seed + 20_000 * run + 200 * j + 7

            bandit = make_fixed_means_bandit(spec, means, seed=bandit_seed)

            if algo_spec.ctor.__name__ == "ETC":
                algo = algo_spec.ctor(bandit=bandit, **algo_spec.kwargs)
            else:
                algo = algo_spec.ctor(bandit=bandit, seed=algo_seed, **algo_spec.kwargs)

            job_done += 1
            elapsed = time.time() - t0
            avg_per_job = elapsed / job_done
            eta = avg_per_job * (total_jobs - job_done)
            pct = 100.0 * job_done / total_jobs

            print(
                f"[REG {job_done:4d}/{total_jobs}] {pct:6.2f}%  "
                f"run {run + 1:3d}/{n_runs}, algo {j + 1:2d}/{A}: {algo_spec.name}  "
                f"elapsed {fmt_time(elapsed)}  ETA {fmt_time(eta)}",
                flush=True
            )

            if not show_step_progress:
                _arms, cum_reg = run_algo(algo, means=means, T=T)
                regrets[j, run, :] = cum_reg
            else:
                arms = np.zeros(T, dtype=int)
                for t in range(T):
                    a, _r = algo.step()
                    arms[t] = int(a)

                    if (t + 1) % step_every == 0 or (t + 1) == T:
                        print(f"    steps {t + 1:6d}/{T}", end="\r", flush=True)

                if T >= step_every:
                    print(" " * 50, end="\r")

                regrets[j, run, :] = cumulative_pseudo_regret(arms, means)

    t_grid = np.arange(1, T + 1)
    names = [algo.name for algo in algos]
    return t_grid, names, regrets


# =========================================================
# Plotting: regret
# =========================================================

def plot_regrets(t: np.ndarray, names: List[str], regrets: np.ndarray, title: str) -> None:
    """
    regrets shape: (A, n_runs, T)
    """

    plt.figure(figsize=(10, 6))
    for j, name in enumerate(names):
        m, se = mean_and_stderr(regrets[j])
        plt.plot(t, m, label=name)
        plt.fill_between(t, m - 2 * se, m + 2 * se, alpha=0.2)

    plt.xlabel("t")
    plt.ylabel("cumulative pseudo-regret")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(10, 6))
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

    plt.figure(figsize=(10, 6))
    for j, name in enumerate(names):
        m, _se = mean_and_stderr(regrets[j])
        plt.plot(t, m, label=name)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("t (log)")
    plt.ylabel("regret (log)")
    plt.title(title + " (log-log)")
    plt.legend()
    plt.tight_layout()


# =========================================================
# Alternative metrics
# =========================================================

def _current_mean_estimates(algo: object, K: int) -> np.ndarray:
    """
    Return current estimates of arm means.
    Assumes the algorithm has empirical_means().
    """
    if hasattr(algo, "empirical_means"):
        est = algo.empirical_means()
        return np.asarray(est, dtype=float)

    raise AttributeError(
        f"Algorithm {type(algo).__name__} has no empirical_means() method."
    )


def _estimated_best_arm(algo: object, K: int) -> int:
    """
    Final estimated best arm based on empirical mean estimates.
    """
    est = _current_mean_estimates(algo, K)
    return int(np.argmax(est))


def run_algo_with_alt_metrics(
    algo: object,
    means: np.ndarray,
    T: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Run one algorithm for T steps and collect:

    Returns
    -------
    arms : shape (T,)
        Played arms.
    p_opt_indicator : shape (T,)
        Indicator 1{A_t = a*}.
    mse_over_time : shape (T,)
        MSE of empirical mean estimates after each step.
    final_best_correct : int in {0,1}
        Indicator that estimated best arm at time T equals the true best arm.
    """
    K = len(means)
    true_best = int(np.argmax(means))

    arms = np.zeros(T, dtype=int)
    p_opt_indicator = np.zeros(T, dtype=float)
    mse_over_time = np.zeros(T, dtype=float)

    for t in range(T):
        a, _r = algo.step()
        a = int(a)
        arms[t] = a
        p_opt_indicator[t] = 1.0 if a == true_best else 0.0

        est = _current_mean_estimates(algo, K)
        mse_over_time[t] = np.mean((est - means) ** 2)

    final_best = _estimated_best_arm(algo, K)
    final_best_correct = 1 if final_best == true_best else 0

    return arms, p_opt_indicator, mse_over_time, final_best_correct


def evaluate_alternative_metrics(
    spec: BanditSpec,
    T: int = 20_000,
    n_runs: int = 100,
    base_seed: int = 123,
    show_step_progress: bool = False,
    step_every: int = 2000,
):
    """
    Evaluate the 3 requested alternative metrics.

    Returns
    -------
    t_grid : shape (T,)
    names : list of algorithm names
    p_opt : shape (A, n_runs, T)
        Indicators for choosing the optimal arm over time.
    mse : shape (A, n_runs, T)
        Mean squared error of mean estimates over time.
    final_best_found : shape (A, n_runs)
        Indicators whether the true best arm was identified at the end.
    """
    algos = build_algos()
    A = len(algos)

    p_opt = np.zeros((A, n_runs, T), dtype=float)
    mse = np.zeros((A, n_runs, T), dtype=float)
    final_best_found = np.zeros((A, n_runs), dtype=float)

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
        means = sample_means(spec, means_rng)

        for j, algo_spec in enumerate(algos):
            bandit_seed = base_seed + 10_000 * run + 100 * j
            algo_seed = base_seed + 20_000 * run + 200 * j + 7

            bandit = make_fixed_means_bandit(spec, means, seed=bandit_seed)

            if algo_spec.ctor.__name__ == "ETC":
                algo = algo_spec.ctor(bandit=bandit, **algo_spec.kwargs)
            else:
                algo = algo_spec.ctor(bandit=bandit, seed=algo_seed, **algo_spec.kwargs)

            job_done += 1
            elapsed = time.time() - t0
            avg_per_job = elapsed / job_done
            eta = avg_per_job * (total_jobs - job_done)
            pct = 100.0 * job_done / total_jobs

            print(
                f"[ALT {job_done:4d}/{total_jobs}] {pct:6.2f}%  "
                f"run {run + 1:3d}/{n_runs}, algo {j + 1:2d}/{A}: {algo_spec.name}  "
                f"elapsed {fmt_time(elapsed)}  ETA {fmt_time(eta)}",
                flush=True
            )

            if not show_step_progress:
                _arms, p_opt_ind, mse_t, found_best = run_algo_with_alt_metrics(
                    algo=algo,
                    means=means,
                    T=T,
                )
            else:
                K = len(means)
                true_best = int(np.argmax(means))

                arms = np.zeros(T, dtype=int)
                p_opt_ind = np.zeros(T, dtype=float)
                mse_t = np.zeros(T, dtype=float)

                for t in range(T):
                    a, _r = algo.step()
                    a = int(a)
                    arms[t] = a
                    p_opt_ind[t] = 1.0 if a == true_best else 0.0

                    est = _current_mean_estimates(algo, K)
                    mse_t[t] = np.mean((est - means) ** 2)

                    if (t + 1) % step_every == 0 or (t + 1) == T:
                        print(f"    steps {t + 1:6d}/{T}", end="\r", flush=True)

                if T >= step_every:
                    print(" " * 50, end="\r")

                found_best = 1 if _estimated_best_arm(algo, K) == true_best else 0

            p_opt[j, run, :] = p_opt_ind
            mse[j, run, :] = mse_t
            final_best_found[j, run] = found_best

    t_grid = np.arange(1, T + 1)
    names = [algo.name for algo in algos]
    return t_grid, names, p_opt, mse, final_best_found


# =========================================================
# Plotting: alternative metrics
# =========================================================

def plot_alternative_metrics(
    t: np.ndarray,
    names: List[str],
    p_opt: np.ndarray,
    mse: np.ndarray,
    final_best_found: np.ndarray,
    title_prefix: str,
) -> None:
    """
    Parameters
    ----------
    p_opt : shape (A, n_runs, T)
        Indicator for playing the optimal arm.
    mse : shape (A, n_runs, T)
        MSE of empirical mean estimates.
    final_best_found : shape (A, n_runs)
        Indicator for correct final best-arm identification.
    """

    # 1) Probability of choosing the optimal arm over time
    plt.figure(figsize=(10, 6))
    for j, name in enumerate(names):
        m, se = mean_and_stderr(p_opt[j])
        plt.plot(t, m, label=name)
        plt.fill_between(t, np.clip(m - 2 * se, 0.0, 1.0), np.clip(m + 2 * se, 0.0, 1.0), alpha=0.2)

    plt.xlabel("t")
    plt.ylabel("P(A_t = a*)")
    plt.title(title_prefix + " - Probability of choosing the optimal arm")
    plt.legend()
    plt.tight_layout()

    # 2) MSE of mean estimates over time
    plt.figure(figsize=(10, 6))
    for j, name in enumerate(names):
        m, se = mean_and_stderr(mse[j])
        plt.plot(t, m, label=name)
        plt.fill_between(t, np.maximum(m - 2 * se, 0.0), m + 2 * se, alpha=0.2)

    plt.xlabel("t")
    plt.ylabel("MSE of mean estimates")
    plt.title(title_prefix + " - Accuracy of arm-mean estimates")
    plt.legend()
    plt.tight_layout()

    # 3) Final probability of correctly identifying the best arm
    final_probs = final_best_found.mean(axis=1)
    final_se = final_best_found.std(axis=1, ddof=1) / np.sqrt(final_best_found.shape[1])

    plt.figure(figsize=(10, 6))
    x = np.arange(len(names))
    plt.bar(x, final_probs, yerr=2 * final_se, capsize=5)
    plt.xticks(x, names, rotation=20, ha="right")
    plt.ylim(0.0, 1.05)
    plt.ylabel("P(best arm correctly identified at T)")
    plt.title(title_prefix + " - Final best-arm identification probability")
    plt.tight_layout()


def print_alternative_metrics_table(
    names: List[str],
    p_opt: np.ndarray,
    mse: np.ndarray,
    final_best_found: np.ndarray,
) -> None:
    """
    Prints final summary table for the three alternative metrics.
    """
    print("\n" + "=" * 95)
    print("FINAL SUMMARY OF ALTERNATIVE METRICS")
    print("=" * 95)
    print(
        f"{'Algorithm':40s} | {'P(A_T=a*)':>12s} | {'Final MSE':>12s} | {'P(best found)':>14s}"
    )
    print("-" * 95)

    for j, name in enumerate(names):
        pT = p_opt[j, :, -1].mean()
        mseT = mse[j, :, -1].mean()
        pbest = final_best_found[j].mean()

        print(f"{name:40s} | {pT:12.4f} | {mseT:12.6f} | {pbest:14.4f}")

    print("=" * 95 + "\n")


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":
    spec = BanditSpec(
        dist="bernoulli",
        K=10,
        gap=0.1,
        seed=0,
    )

    T = 20_000
    n_runs = 100
    base_seed = 123

    # -------------------------
    # Regret evaluation
    # -------------------------
    t, names, regrets = evaluate(
        spec=spec,
        T=T,
        n_runs=n_runs,
        base_seed=base_seed,
        show_step_progress=False,
    )

    plot_regrets(
        t=t,
        names=names,
        regrets=regrets,
        title=f"Bernoulli bandit (K={spec.K}, gap={spec.gap})",
    )

    # -------------------------
    # Alternative metrics
    # -------------------------
    t_alt, names_alt, p_opt, mse, final_best_found = evaluate_alternative_metrics(
        spec=spec,
        T=T,
        n_runs=n_runs,
        base_seed=base_seed,
        show_step_progress=False,
    )

    plot_alternative_metrics(
        t=t_alt,
        names=names_alt,
        p_opt=p_opt,
        mse=mse,
        final_best_found=final_best_found,
        title_prefix=f"Bernoulli bandit (K={spec.K}, gap={spec.gap})",
    )

    print_alternative_metrics_table(
        names=names_alt,
        p_opt=p_opt,
        mse=mse,
        final_best_found=final_best_found,
    )

    plt.show()

    # python -m sheet3.simulation_metric