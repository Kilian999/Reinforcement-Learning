# sheet1/testing.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from .bandits import StochasticBandit
from .etc import ETC


# ---------- Metrics container ----------

@dataclass
class Aggregates:
    # over full horizon n
    regret_sum: np.ndarray          # shape (n,)
    regret_sumsq: np.ndarray        # shape (n,)
    correct_sum: np.ndarray         # shape (n,)
    correct_sumsq: np.ndarray       # shape (n,)
    action_counts: np.ndarray       # shape (n, K)  counts of choosing arm k at time t
    action_counts_sumsq: np.ndarray # shape (n, K)  for variance logging (Bernoulli indicators)

    # over subsampled times for mean-estimates plot
    t_grid: np.ndarray              # shape (G,)
    est_sum: np.ndarray             # shape (G, K)
    est_sumsq: np.ndarray           # shape (G, K)

    # true means (for reference)
    true_means_sum: np.ndarray      # shape (K,)
    true_means_sumsq: np.ndarray    # shape (K,)


def make_time_grid(n: int, G: int = 200) -> np.ndarray:
    """
    Subsample grid for logging estimated means over time.
    Uses more density early, but still covers the full horizon.
    """
    # log-spaced + ensure endpoints
    grid = np.unique(np.clip(np.round(np.geomspace(1, n, G)).astype(int), 1, n)) - 1
    if grid[0] != 0:
        grid = np.insert(grid, 0, 0)
    if grid[-1] != n - 1:
        grid = np.append(grid, n - 1)
    return grid


def init_aggregates(n: int, K: int, t_grid: np.ndarray) -> Aggregates:
    G = len(t_grid)
    return Aggregates(
        regret_sum=np.zeros(n, dtype=float),
        regret_sumsq=np.zeros(n, dtype=float),
        correct_sum=np.zeros(n, dtype=float),
        correct_sumsq=np.zeros(n, dtype=float),
        action_counts=np.zeros((n, K), dtype=float),
        action_counts_sumsq=np.zeros((n, K), dtype=float),
        t_grid=t_grid,
        est_sum=np.zeros((G, K), dtype=float),
        est_sumsq=np.zeros((G, K), dtype=float),
        true_means_sum=np.zeros(K, dtype=float),
        true_means_sumsq=np.zeros(K, dtype=float),
    )


# ---------- One run ----------

def run_one_etc_gaussian(
    K: int,
    n: int,
    m: int,
    seed: int,
    t_grid: np.ndarray,
    sigma: float = 1.0,
) -> Dict[str, np.ndarray]:
    """
    One run:
    - Gaussian bandit with random means ~ N(0,1)
    - ETC with m exploration pulls per arm
    Returns arrays needed for aggregation.
    """
    # separate RNG streams for reproducibility
    bandit = StochasticBandit(n_arms=K, dist="gaussian", means=None, seed=seed, sigma=sigma)
    algo = ETC(bandit, exploration_rounds=m)

    true_means = bandit.means_array  # shape (K,)
    best_arm = int(np.argmax(true_means))
    mu_star = float(true_means[best_arm])

    regret = np.zeros(n, dtype=float)
    correct = np.zeros(n, dtype=float)
    actions = np.zeros((n, K), dtype=float)  # one-hot per time
    est_on_grid = np.zeros((len(t_grid), K), dtype=float)

    # to quickly check if current t is in grid
    grid_pos = {int(t): i for i, t in enumerate(t_grid)}

    for t in range(n):
        arm, reward = algo.step()
        arm = int(arm)

        # instantaneous (expected) regret uses true means:
        regret[t] = mu_star - float(true_means[arm])
        correct[t] = 1.0 if arm == best_arm else 0.0
        actions[t, arm] = 1.0

        if t in grid_pos:
            est_on_grid[grid_pos[t], :] = algo.empirical_means()

    return {
        "true_means": true_means,
        "regret": regret,
        "correct": correct,
        "actions": actions,
        "est_grid": est_on_grid,
    }


# ---------- Many runs (aggregate mean + variance) ----------

def run_many(
    K: int,
    n: int,
    m: int,
    N: int,
    base_seed: int = 0,
    grid_points: int = 200,
    sigma: float = 1.0,
) -> Dict[str, np.ndarray]:
    t_grid = make_time_grid(n, G=grid_points)
    agg = init_aggregates(n=n, K=K, t_grid=t_grid)

    for i in range(N):
        seed = base_seed + i
        out = run_one_etc_gaussian(K=K, n=n, m=m, seed=seed, t_grid=t_grid, sigma=sigma)

        # true means (for reference; average over experiments)
        tm = out["true_means"]
        agg.true_means_sum += tm
        agg.true_means_sumsq += tm * tm

        r = out["regret"]
        agg.regret_sum += r
        agg.regret_sumsq += r * r

        c = out["correct"]
        agg.correct_sum += c
        agg.correct_sumsq += c * c

        A = out["actions"]  # one-hot; variance also possible via Bernoulli var
        agg.action_counts += A
        agg.action_counts_sumsq += A * A

        E = out["est_grid"]
        agg.est_sum += E
        agg.est_sumsq += E * E

    # convert to means/vars
    mean_regret = agg.regret_sum / N
    var_regret = agg.regret_sumsq / N - mean_regret**2

    mean_correct = agg.correct_sum / N
    var_correct = agg.correct_sumsq / N - mean_correct**2

    prob_choose = agg.action_counts / N              # P(choose arm k at time t)
    var_choose = agg.action_counts_sumsq / N - prob_choose**2

    mean_true_means = agg.true_means_sum / N
    var_true_means = agg.true_means_sumsq / N - mean_true_means**2

    mean_est_grid = agg.est_sum / N
    var_est_grid = agg.est_sumsq / N - mean_est_grid**2

    # cumulative regret over time
    cum_regret = np.cumsum(mean_regret)

    return {
        "m": np.array([m]),
        "t": np.arange(n),
        "t_grid": agg.t_grid,
        "mean_regret_inst": mean_regret,
        "var_regret_inst": var_regret,
        "mean_regret_cum": cum_regret,
        "mean_correct": mean_correct,
        "var_correct": var_correct,
        "prob_choose": prob_choose,
        "var_choose": var_choose,
        "mean_true_means": mean_true_means,
        "var_true_means": var_true_means,
        "mean_est_grid": mean_est_grid,
        "var_est_grid": var_est_grid,
    }


# ---------- Plotting ----------

def plot_results(results: Dict[str, np.ndarray], title_prefix: str = "") -> None:
    t = results["t"]
    t_grid = results["t_grid"]
    K = results["prob_choose"].shape[1]
    m = int(results["m"][0])

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"{title_prefix} ETC (m={m})", fontsize=14)

    # (a) regret over time (cumulative)
    ax = axes[0, 0]
    ax.plot(t, results["mean_regret_cum"])
    ax.set_title("Cumulative regret over time")
    ax.set_xlabel("t")
    ax.set_ylabel("E[Regret_1:t]")

    # (b) correct action rate over time
    ax = axes[0, 1]
    ax.plot(t, results["mean_correct"])
    ax.set_title("Correct action rate over time")
    ax.set_xlabel("t")
    ax.set_ylabel("P(play best arm)")

    # (c) estimated means vs true means over time (on grid)
    ax = axes[1, 0]
    true_means = results["mean_true_means"]
    est = results["mean_est_grid"]  # shape (G,K)
    for k in range(K):
        ax.plot(t_grid, est[:, k], linewidth=1)
    # show true means as horizontal lines
    for k in range(K):
        ax.hlines(true_means[k], xmin=t_grid[0], xmax=t_grid[-1], linestyles="dashed", linewidth=1)
    ax.set_title("Estimated means (solid) vs true means (dashed)")
    ax.set_xlabel("t (subsampled)")
    ax.set_ylabel("mean")

    # (d) average probabilities of choosing each arm over time
    ax = axes[1, 1]
    P = results["prob_choose"]  # shape (n,K)
    for k in range(K):
        ax.plot(t, P[:, k], linewidth=1)
    ax.set_title("P(choose arm k) over time")
    ax.set_xlabel("t")
    ax.set_ylabel("probability")

    plt.tight_layout()
    plt.show()


# ---------- Main experiment ----------

def main() -> None:
    K = 10
    n = 10_000
    N = 1_000

    # different m choices (exploration pulls per arm)
    ms = [5, 10, 20, 30, 50, 80, 120]

    for m in ms:
        res = run_many(K=K, n=n, m=m, N=N, base_seed=0, grid_points=250, sigma=1.0)
        plot_results(res, title_prefix="Gaussian 10-armed,")


if __name__ == "__main__":
    main()