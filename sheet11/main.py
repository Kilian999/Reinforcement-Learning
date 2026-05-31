# -*- coding: utf-8 -*-
"""
Exercise Sheet 11 - Task 4: Actor-Critic Algorithms Evaluation Study
======================================================================

Evaluates the following algorithms across all Gymnasium classic-control
environments using multiple random seeds:

  Own implementations : REINFORCE, Mini-batch REINFORCE
  SB3 / sb3-contrib   : ARS, A2C, DDPG, PPO, SAC, TD3, TQC, TRPO
  (ARS uses sb3_contrib.ARS with LinearPolicy, matching Mania et al. 2018)

Usage
-----
  python main.py               # full run
  python main.py --plot-only   # re-plot from saved results
  python main.py --envs CartPole-v1 Pendulum-v1   # subset of envs
  python main.py --algos PPO SAC REINFORCE         # subset of algos
  python main.py --seeds 5     # change seed count

Output
------
  sheet11/results/results.json        serialised results
  sheet11/results/<env>_results.png   per-environment figure
  sheet11/results/overview.png        5-env overview
  sheet11/results/metrics.png         cross-env metric comparison
"""

import argparse
import json
import os
import sys
import time
import warnings

# Ensure UTF-8 output on Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics

warnings.filterwarnings("ignore")

# ── SB3 imports ───────────────────────────────────────────────────────────────
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import ARS, TRPO, TQC   # ARS: pre-implemented in sb3_contrib

# ── Own implementations ───────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from reinforce import REINFORCE, MiniBatchREINFORCE

# ═════════════════════════════════════════════════════════════════════════════
# Configuration
# ═════════════════════════════════════════════════════════════════════════════

ENVS_CONFIG: Dict[str, Dict] = {
    "CartPole-v1":             {"discrete": True,  "timesteps": 50_000},
    "MountainCar-v0":          {"discrete": True,  "timesteps": 80_000},
    "MountainCarContinuous-v0":{"discrete": False, "timesteps": 60_000},
    "Pendulum-v1":             {"discrete": False, "timesteps": 30_000},
    "Acrobot-v1":              {"discrete": True,  "timesteps": 80_000},
}

# Algorithms that require continuous action spaces
CONTINUOUS_ONLY = {"DDPG", "SAC", "TD3", "TQC"}

# Off-policy algorithms train online (1 gradient step / env step) and are
# significantly slower on CPU.  We cap their timesteps at 15k so the
# experiment completes in reasonable time while still showing their
# learning behaviour.
OFFPOLICY_TS_CAP = 60_000   # same budget as MountainCarContinuous / Pendulum

ALL_ALGORITHMS = [
    "REINFORCE", "MiniBatch-REINFORCE",     # own implementations
    "ARS", "A2C", "PPO", "TRPO",            # sb3_contrib / SB3 (all action types)
    "DDPG", "SAC", "TD3", "TQC",            # SB3 / sb3_contrib (continuous only)
]

SB3_CLASSES = {
    "ARS":  ARS,   # sb3_contrib – linear policy, random search (Mania et al. 2018)
    "A2C":  A2C,
    "DDPG": DDPG,
    "PPO":  PPO,
    "SAC":  SAC,
    "TD3":  TD3,
    "TRPO": TRPO,
    "TQC":  TQC,
}

ALGO_COLORS = {
    "REINFORCE":          "#1f77b4",
    "MiniBatch-REINFORCE":"#ff7f0e",
    "ARS":                "#2ca02c",
    "A2C":                "#d62728",
    "PPO":                "#9467bd",
    "TRPO":               "#8c564b",
    "DDPG":               "#e377c2",
    "SAC":                "#7f7f7f",
    "TD3":                "#bcbd22",
    "TQC":                "#17becf",
}

# Different line styles so overlapping curves (e.g. all stuck at −200)
# remain distinguishable.
ALGO_LINESTYLES = {
    "REINFORCE":           "-",
    "MiniBatch-REINFORCE": "--",
    "ARS":                 "-",
    "A2C":                 "-.",
    "PPO":                 ":",
    "TRPO":                (0, (3, 1, 1, 1)),   # dash-dot-dot
    "DDPG":                "--",
    "SAC":                 "-",
    "TD3":                 "-.",
    "TQC":                 ":",
}

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# ═════════════════════════════════════════════════════════════════════════════
# SB3 callback
# ═════════════════════════════════════════════════════════════════════════════

class EpisodeCallback(BaseCallback):
    """
    Logs episode returns and the timestep at which they finished.

    Standard SB3 algorithms report episodes via infos['episode'] in _on_step.
    ARS is special: it runs complete episodes internally and stores results
    in model.ep_info_buffer instead.  We drain that buffer in _on_rollout_end.
    """

    def __init__(self):
        super().__init__(verbose=0)
        self.episode_returns: List[float] = []
        self.episode_timesteps: List[int] = []

    def _on_step(self) -> bool:
        # A2C, PPO, TRPO, SAC, TD3, DDPG, TQC all report via infos['episode']
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_returns.append(float(info["episode"]["r"]))
                self.episode_timesteps.append(int(self.num_timesteps))
        return True

    def _on_rollout_end(self) -> None:
        # ARS resets ep_info_buffer at the start of each update step and fills
        # it with all 2*n_delta candidate returns.  _on_step never sees these
        # because ARS bypasses the standard infos mechanism.
        # Solution: drain the buffer completely on every rollout_end.
        # (For all other algorithms this method is a no-op.)
        if type(self.model).__name__ != "ARS":
            return
        buf = getattr(self.model, "ep_info_buffer", None) or []
        for entry in list(buf):
            self.episode_returns.append(float(entry["r"]))
            self.episode_timesteps.append(int(self.num_timesteps))

# ═════════════════════════════════════════════════════════════════════════════
# Run a single experiment
# ═════════════════════════════════════════════════════════════════════════════

def _run_custom(algo_name: str, env_name: str,
                total_timesteps: int, seed: int
                ) -> Tuple[List[float], List[int]]:
    """Run REINFORCE or MiniBatch-REINFORCE (own PyTorch implementations)."""
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)

    if algo_name == "REINFORCE":
        algo = REINFORCE(env_name, lr=3e-3, gamma=0.99, seed=seed)
    elif algo_name == "MiniBatch-REINFORCE":
        algo = MiniBatchREINFORCE(env_name, lr=3e-3, gamma=0.99,
                                  batch_size=8, seed=seed)
    else:
        raise ValueError(f"Unknown custom algorithm: {algo_name}")

    return algo.train(total_timesteps)


def _run_sb3(algo_name: str, env_name: str,
             total_timesteps: int, seed: int
             ) -> Tuple[List[float], List[int]]:
    env = Monitor(gym.make(env_name))

    # ── policy type ──────────────────────────────────────────────────────────
    # ARS uses a linear policy (original paper: M * phi(s)).
    # All other algorithms use a 2-layer MLP policy.
    policy = "LinearPolicy" if algo_name == "ARS" else "MlpPolicy"

    # ── per-algorithm kwargs to keep training fast on CPU ────────────────────
    extra: Dict = {}
    if algo_name == "ARS":
        # Match our custom ARS: 16 directions, top-8, same lr and delta_std.
        extra = {"n_delta": 16, "n_top": 8, "learning_rate": 0.02,
                 "delta_std": 0.03}
    elif algo_name == "A2C":
        # Default n_steps=5 → ~10k gradient updates per 50k steps (very slow).
        # n_steps=64 → ~780 updates (same compute per update, much fewer total).
        extra = {"n_steps": 64, "ent_coef": 0.01}
    elif algo_name == "PPO":
        # Default n_epochs=10, n_steps=2048. Reduce for faster wall-clock time.
        extra = {"n_steps": 512, "n_epochs": 4, "batch_size": 128}
    elif algo_name == "TRPO":
        extra = {"n_steps": 512, "batch_size": 128}
    elif algo_name in ("DDPG", "SAC", "TD3", "TQC"):
        # Off-policy: 1 gradient step per env step by default (very slow on CPU).
        # train_freq=8 reduces gradient steps 8x with minimal quality loss.
        extra = {
            "learning_starts": 200,
            "batch_size": 64,
            "train_freq": 8,
            "gradient_steps": 1,
        }

    model = SB3_CLASSES[algo_name](
        policy, env, seed=seed, verbose=0, device="cpu", **extra
    )
    cb = EpisodeCallback()
    model.learn(total_timesteps=total_timesteps,
                callback=cb, reset_num_timesteps=True, progress_bar=False)
    env.close()
    return cb.episode_returns, cb.episode_timesteps


def run_experiment(algo_name: str, env_name: str,
                   total_timesteps: int, seed: int
                   ) -> Tuple[List[float], List[int]]:
    # Cap off-policy algorithms to avoid multi-hour CPU runs
    if algo_name in CONTINUOUS_ONLY:
        total_timesteps = min(total_timesteps, OFFPOLICY_TS_CAP)
    # REINFORCE and MiniBatch-REINFORCE are our own implementations.
    # Everything else (including ARS) uses sb3_contrib / SB3.
    if algo_name in ("REINFORCE", "MiniBatch-REINFORCE"):
        return _run_custom(algo_name, env_name, total_timesteps, seed)
    return _run_sb3(algo_name, env_name, total_timesteps, seed)


def compatible_algorithms(env_name: str,
                           requested: Optional[List[str]] = None) -> List[str]:
    is_discrete = ENVS_CONFIG[env_name]["discrete"]
    pool = requested or ALL_ALGORITHMS
    return [a for a in pool
            if not (a in CONTINUOUS_ONLY and is_discrete)]

# ═════════════════════════════════════════════════════════════════════════════
# Serialisation
# ═════════════════════════════════════════════════════════════════════════════

def save_results(results: dict, path: str):
    serial = {}
    for env, env_r in results.items():
        serial[env] = {}
        for algo, (ret_l, ts_l) in env_r.items():
            serial[env][algo] = {
                "returns":   [list(r) for r in ret_l],
                "timesteps": [list(t) for t in ts_l],
            }
    with open(path, "w") as f:
        json.dump(serial, f)
    print(f"\nResults saved → {path}")


def load_results(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    return {
        env: {
            algo: ([list(r) for r in v["returns"]],
                   [list(t) for t in v["timesteps"]])
            for algo, v in env_r.items()
        }
        for env, env_r in data.items()
    }

# ═════════════════════════════════════════════════════════════════════════════
# Main training loop
# ═════════════════════════════════════════════════════════════════════════════

def run_all(envs: List[str], algos: Optional[List[str]],
            n_seeds: int) -> dict:
    # ── pre-compute total number of (algo, env, seed) runs for progress bar ──
    total_runs = sum(
        len(compatible_algorithms(e, algos)) * n_seeds for e in envs
    )
    done_runs  = 0

    results = {}
    for env_name in envs:
        print(f"\n{'═'*65}")
        print(f"  ENV : {env_name}")
        print(f"{'═'*65}")

        total_ts = ENVS_CONFIG[env_name]["timesteps"]
        compat   = compatible_algorithms(env_name, algos)
        env_r    = {}

        for algo in compat:
            print(f"\n  ▶ {algo}")
            seed_ret, seed_ts = [], []

            for s in range(n_seeds):
                pct = 100 * done_runs / total_runs
                print(f"      seed {s}  [{done_runs}/{total_runs}  {pct:.0f}%]… ",
                      end="", flush=True)
                t0 = time.time()
                try:
                    import concurrent.futures as _cf
                    # 10-minute per-seed timeout to guard against CPU spikes
                    with _cf.ThreadPoolExecutor(max_workers=1) as ex:
                        fut = ex.submit(run_experiment, algo, env_name, total_ts, s)
                        try:
                            ret, ts = fut.result(timeout=600)
                        except _cf.TimeoutError:
                            raise RuntimeError("timeout (>600s)")
                    elapsed = time.time() - t0
                    final = (np.mean(ret[-max(1, len(ret)//5):])
                             if ret else float("nan"))
                    print(f"✓ {elapsed:.0f}s | {len(ret)} ep | "
                          f"final≈{final:.1f}")
                    seed_ret.append(ret)
                    seed_ts.append(ts)
                except Exception as exc:
                    elapsed = time.time() - t0
                    print(f"✗ {elapsed:.0f}s — {exc}")
                    seed_ret.append([])
                    seed_ts.append([])

                done_runs += 1

            env_r[algo] = (seed_ret, seed_ts)

        results[env_name] = env_r
    return results

# ═════════════════════════════════════════════════════════════════════════════
# Plotting utilities
# ═════════════════════════════════════════════════════════════════════════════

def _interp_to_grid(ret_list: List[List[float]],
                    ts_list: List[List[int]],
                    n: int = 200):
    valid = [(r, t) for r, t in zip(ret_list, ts_list)
             if len(r) >= 2 and len(t) >= 2]
    if not valid:
        return None, None
    # Use the MINIMUM final timestep across all valid seeds.
    # This guarantees every seed has real data at every grid point — no
    # extrapolation/clamping at the right edge, which would look like a
    # spurious drop if the last recorded episode happened to be noisy.
    max_ts = min(t[-1] for _, t in valid)
    if max_ts <= 0:
        return None, None
    grid = np.linspace(0, max_ts, n)
    mat  = np.array([np.interp(grid, t, r) for r, t in valid])
    return grid, mat


def _smooth(x: np.ndarray, w: int = 15) -> np.ndarray:
    """Rolling mean with edge-padding to avoid boundary artefacts.

    np.convolve(..., mode='same') zero-pads at the borders, which creates
    spurious spikes when the signal is far from zero (e.g. −200 near 0).
    Instead we pad with the edge value so the window average is unbiased.
    """
    if len(x) <= w:
        return x
    padded   = np.pad(x, (w // 2, w // 2), mode="edge")
    smoothed = np.convolve(padded, np.ones(w) / w, mode="valid")
    return smoothed[: len(x)]


def _final_perf(ret_list: List[List[float]]) -> List[float]:
    out = []
    for r in ret_list:
        if r:
            n = max(1, len(r) // 5)
            out.append(float(np.mean(r[-n:])))
    return out


# ── per-environment figure ────────────────────────────────────────────────────

def plot_env(results: dict, env_name: str, ax_lc: plt.Axes, ax_fp: plt.Axes,
             show_legend: bool = True):
    env_r = results[env_name]

    # Learning curves
    for algo, (ret_list, ts_list) in env_r.items():
        grid, mat = _interp_to_grid(ret_list, ts_list)
        if grid is None:
            continue
        mu  = _smooth(mat.mean(0))
        sig = _smooth(mat.std(0))
        c   = ALGO_COLORS.get(algo, "gray")
        ls  = ALGO_LINESTYLES.get(algo, "-")
        ax_lc.plot(grid, mu, label=algo, color=c, linewidth=1.8, linestyle=ls)
        ax_lc.fill_between(grid, mu - sig, mu + sig, alpha=0.15, color=c)

    ax_lc.set_title(env_name, fontsize=11, fontweight="bold")
    ax_lc.set_xlabel("Environment Steps", fontsize=9)
    ax_lc.set_ylabel("Episode Return",    fontsize=9)
    ax_lc.grid(True, alpha=0.3)
    if show_legend:
        ax_lc.legend(loc="upper left", fontsize=7, ncol=2)

    # Final performance bar chart
    names, means, stds = [], [], []
    for algo, (ret_list, _) in env_r.items():
        fp = _final_perf(ret_list)
        if fp:
            names.append(algo)
            means.append(np.mean(fp))
            stds.append(np.std(fp))

    if names:
        x = np.arange(len(names))
        c = [ALGO_COLORS.get(a, "gray") for a in names]
        ax_fp.bar(x, means, yerr=stds, capsize=4, color=c,
                  alpha=0.85, edgecolor="black", linewidth=0.5)
        ax_fp.set_xticks(x)
        ax_fp.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax_fp.set_ylabel("Mean Final Return", fontsize=9)
        ax_fp.set_title(f"{env_name} – Final Perf.", fontsize=10,
                        fontweight="bold")
        ax_fp.grid(True, alpha=0.3, axis="y")


# ── per-environment standalone figure ────────────────────────────────────────

def save_env_figure(results: dict, env_name: str):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    plot_env(results, env_name, axes[0], axes[1])
    fig.suptitle(f"Algorithm Comparison: {env_name}", fontsize=13,
                 fontweight="bold")
    plt.tight_layout()
    fname = env_name.replace("-", "_") + "_results.png"
    fig.savefig(os.path.join(SAVE_DIR, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fname}")


# ── 5-env overview ────────────────────────────────────────────────────────────

def save_overview(results: dict):
    envs = list(results)
    n    = len(envs)
    fig  = plt.figure(figsize=(6 * n, 10))
    gs   = gridspec.GridSpec(2, n, hspace=0.55, wspace=0.38)

    for col, env_name in enumerate(envs):
        ax_lc = fig.add_subplot(gs[0, col])
        ax_fp = fig.add_subplot(gs[1, col])
        plot_env(results, env_name, ax_lc, ax_fp, show_legend=(col == 0))

    # Remove individual legends; add one shared legend
    all_handles, all_labels = [], []
    for ax in fig.axes:
        lg = ax.get_legend()
        if lg:
            all_handles += lg.legend_handles
            all_labels  += [t.get_text() for t in lg.get_texts()]
            lg.remove()

    # deduplicate
    seen, h2, l2 = set(), [], []
    for h, l in zip(all_handles, all_labels):
        if l not in seen:
            seen.add(l); h2.append(h); l2.append(l)

    fig.legend(h2, l2, loc="lower center", ncol=min(5, len(l2)),
               fontsize=9, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("RL Algorithm Comparison – Classic Control Environments",
                 fontsize=15, fontweight="bold", y=1.01)

    fig.savefig(os.path.join(SAVE_DIR, "overview.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved overview.png")


# ── cross-environment metric comparison ───────────────────────────────────────

def save_metrics_figure(results: dict):
    """
    Three-panel comparison.

    Panel 1 – Performance score [0,1]:
      Uses fixed theoretical bounds per environment so that scores are
      comparable across environments with different reward scales.
        score = (return − worst) / (best − worst)
      e.g. Pendulum: worst=−1600, best=0  →  −800 maps to 0.5

    Panel 2 – Raw std over seeds (lower = more stable, no normalisation needed).

    Panel 3 – Raw timesteps to reach 80 % of final return (lower = more efficient).
    """

    # Theoretical (worst, best) return bounds per environment
    ENV_BOUNDS: Dict[str, Tuple[float, float]] = {
        "CartPole-v1":              (0.,    500.),
        "MountainCar-v0":           (-200., -100.),
        "MountainCarContinuous-v0": (-100.,  90.),
        "Pendulum-v1":              (-3254.,  0.),  # true min: -(pi^2+6.4+0.004)*200
        "Acrobot-v1":               (-500., -72.),
    }

    # Separate discrete and continuous environments for a fair comparison:
    # off-policy algorithms (DDPG, SAC, TD3, TQC) only run on continuous envs,
    # so mixing them with discrete-only runs would bias the aggregate scores.
    DISCRETE_ENVS    = {"CartPole-v1", "MountainCar-v0", "Acrobot-v1"}
    CONTINUOUS_ENVS  = {"MountainCarContinuous-v0", "Pendulum-v1"}

    def _collect(env_subset):
        perf: Dict[str, List[float]] = {}   # final performance score [0,1]
        stab: Dict[str, List[float]] = {}   # raw std over seeds
        auc:  Dict[str, List[float]] = {}   # area under normalised learning curve

        for env_name, env_r in results.items():
            if env_name not in env_subset:
                continue
            worst, best = ENV_BOUNDS.get(env_name, (None, None))
            if worst is None:
                continue
            rng = best - worst + 1e-8

            for algo, (ret_list, ts_list) in env_r.items():
                fp = _final_perf(ret_list)
                if not fp:
                    continue

                # ── final performance score ──────────────────────────────────
                mean_fp = float(np.mean(fp))
                score   = float(np.clip((mean_fp - worst) / rng, 0., 1.))
                perf.setdefault(algo, []).append(score)
                stab.setdefault(algo, []).append(float(np.std(fp)))

                # ── AUC: area under the normalised learning curve ────────────
                # Interpolate each seed to a common grid, normalise to [0,1]
                # using ENV_BOUNDS, then take the mean over time.
                # AUC = 1  →  perfect score throughout all timesteps
                # AUC = 0  →  worst possible score throughout all timesteps
                grid, mat = _interp_to_grid(ret_list, ts_list, n=200)
                if grid is not None:
                    norm_mat = np.clip((mat - worst) / rng, 0., 1.)
                    # mean over time per seed, then mean over seeds
                    seed_aucs = norm_mat.mean(axis=1)   # shape (n_seeds,)
                    auc.setdefault(algo, []).append(float(seed_aucs.mean()))

        return perf, stab, auc

    disc_perf, disc_stab, disc_auc = _collect(DISCRETE_ENVS)
    cont_perf, cont_stab, cont_auc = _collect(CONTINUOUS_ENVS)

    # ── plot: 2 rows (discrete / continuous) × 3 panels ─────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    def _bar(ax, data, title, ylabel, ascending=False, ylim=None):
        items = [(a, float(np.mean(v)), float(np.std(v)))
                 for a, v in data.items() if v]
        if not items:
            ax.set_visible(False)
            return
        items.sort(key=lambda x: x[1], reverse=not ascending)
        names  = [i[0] for i in items]
        vals   = [i[1] for i in items]
        errs   = [i[2] for i in items]
        colors = [ALGO_COLORS.get(a, "gray") for a in names]
        ls     = [ALGO_LINESTYLES.get(a, "-") for a in names]
        bars   = ax.bar(names, vals, yerr=errs, capsize=4, color=colors,
                        alpha=0.85, edgecolor="black", linewidth=0.5)
        # hatch pattern mirrors line style so colours + patterns are distinct
        hatches = {"--": "//", ":": "..", "-.": "xx", "-": ""}
        for bar, a in zip(bars, names):
            bar.set_hatch(hatches.get(ALGO_LINESTYLES.get(a, "-"), ""))
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        if ylim:
            ax.set_ylim(*ylim)

    # Row 0: discrete environments (CartPole, MountainCar, Acrobot)
    _bar(axes[0, 0], disc_perf,
         "Performance Score  [discrete envs]\n(1=perfect, 0=worst  |  ↑ better)",
         "Score [0–1]", ylim=(0, 1.15))
    axes[0, 0].axhline(1.0, color="black", linewidth=0.8,
                       linestyle="--", alpha=0.35)
    _bar(axes[0, 1], disc_stab,
         "Stability  [discrete envs]\n(raw std over seeds, ↓ better)",
         "Std of Final Return", ascending=True)
    _bar(axes[0, 2], disc_auc,
         "AUC  [discrete envs]\n(area under normalised curve, ↑ better)",
         "AUC [0–1]", ylim=(0, 1.15))
    axes[0, 2].axhline(1.0, color="black", linewidth=0.8,
                       linestyle="--", alpha=0.35)

    # Row 1: continuous environments (MountainCarContinuous, Pendulum)
    _bar(axes[1, 0], cont_perf,
         "Performance Score  [continuous envs]\n(1=perfect, 0=worst  |  ↑ better)",
         "Score [0–1]", ylim=(0, 1.15))
    axes[1, 0].axhline(1.0, color="black", linewidth=0.8,
                       linestyle="--", alpha=0.35)
    _bar(axes[1, 1], cont_stab,
         "Stability  [continuous envs]\n(raw std over seeds, ↓ better)",
         "Std of Final Return", ascending=True)
    _bar(axes[1, 2], cont_auc,
         "AUC  [continuous envs]\n(area under normalised curve, ↑ better)",
         "AUC [0–1]", ylim=(0, 1.15))
    axes[1, 2].axhline(1.0, color="black", linewidth=0.8,
                       linestyle="--", alpha=0.35)

    fig.suptitle(
        "Algorithm Comparison: Key Metrics\n"
        "Top: discrete action spaces (all algorithms)  |  "
        "Bottom: continuous action spaces (all algorithms)",
        fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "metrics.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved metrics.png")

# ═════════════════════════════════════════════════════════════════════════════
# Summary table (console)
# ═════════════════════════════════════════════════════════════════════════════

def print_summary(results: dict):
    line = "─" * 72
    print(f"\n{'═'*72}")
    print("  FINAL PERFORMANCE SUMMARY  (mean ± std, last 20 % of episodes)")
    print(f"{'═'*72}")

    for env_name, env_r in results.items():
        ts = ENVS_CONFIG.get(env_name, {}).get("timesteps", "?")
        print(f"\n  {env_name}  [{ts:,} timesteps]")
        print(f"  {line}")
        print(f"  {'Algorithm':<28} {'Final Return':>14}  {'±Std':>8}  "
              f"{'N_episodes':>11}")
        print(f"  {line}")
        for algo, (ret_list, _) in env_r.items():
            fp = _final_perf(ret_list)
            if fp:
                n_ep = np.mean([len(r) for r in ret_list if r])
                print(f"  {algo:<28} {np.mean(fp):>14.2f}  "
                      f"{np.std(fp):>8.2f}  {n_ep:>11.0f}")
            else:
                print(f"  {algo:<28} {'N/A':>14}")

# ═════════════════════════════════════════════════════════════════════════════
# Part (b): Discussion of evaluation metrics
# ═════════════════════════════════════════════════════════════════════════════

METRICS_DISCUSSION = """
╔══════════════════════════════════════════════════════════════════════════════╗
║  PART (b) – EVALUATION METRICS: TABULAR vs. DEEP / FUNCTION-APPROXIMATION ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  TABULAR RL  (Q-learning / SARSA on GridWorld)                              ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  • Exact optimal known  → compare Q_n directly to Q* (or V*)               ║
║  • Convergence provable under Robbins-Monro step-size conditions            ║
║  • Primary metric: ‖Q_n − Q*‖_∞ or fraction of states with optimal action  ║
║  • Every state-action pair visited infinitely often → full coverage          ║
║  • Low variance:  update targets are "exact" samples of Bellman operator    ║
║                                                                              ║
║  DEEP / CONTINUOUS-STATE RL  (classic control)                              ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  • Optimal value function unknown → only empirical episode returns           ║
║  • No convergence guarantees in general (deadly triad:                       ║
║    bootstrapping + off-policy + function approximation)                     ║
║  • Moving targets: critic loss landscape changes as policy changes           ║
║  • Large/continuous state space → generalisation via neural networks         ║
║                                                                              ║
║  KEY DIFFERENCES IN EVALUATION                                              ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  Criterion              │ Tabular RL          │ Deep RL                      ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  Optimality reference   │ Exact (Q*)          │ None (empirical max)         ║
║  Convergence proof      │ Yes (Q-learning)    │ Rarely / partially           ║
║  Stability              │ High                │ Low (hyper-param sensitive)  ║
║  Sample efficiency      │ Low (tabular)       │ Varies widely (PPO < SAC)    ║
║  Generalisation         │ None                │ Via function approx.         ║
║  Environment complexity │ Small / discrete    │ Continuous / large           ║
║                                                                              ║
║  IMPROVED METRICS FOR DEEP RL (see Agarwal et al. 2021)                     ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  1. IQM (Interquartile Mean)  – robust to outlier runs                      ║
║  2. Performance profiles (reliability diagrams)  – full distribution        ║
║  3. Optimality gap  – distance to human / expert level                      ║
║  4. Sample efficiency curves  – return vs. environment steps                ║
║  5. Wallclock efficiency  – return vs. training time                        ║
║  In this study we use: mean±std final return, timesteps to 80 %-final,     ║
║  and seed-stability (std over seeds).                                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ═════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Sheet 11 – Actor-Critic evaluation study")
    p.add_argument("--envs",  nargs="*", default=None,
                   help="subset of environments (default: all 5)")
    p.add_argument("--algos", nargs="*", default=None,
                   help="subset of algorithms (default: all)")
    p.add_argument("--seeds", type=int, default=3,
                   help="number of random seeds (default: 3)")
    p.add_argument("--plot-only", action="store_true",
                   help="skip training; re-plot from saved results.json")
    return p.parse_args()


def main():
    args = parse_args()

    envs  = args.envs  or list(ENVS_CONFIG)
    algos = args.algos  # None = all compatible
    n_seeds = args.seeds

    os.makedirs(SAVE_DIR, exist_ok=True)
    json_path = os.path.join(SAVE_DIR, "results.json")

    # ── load or run experiments ───────────────────────────────────────────────
    if args.plot_only:
        if not os.path.exists(json_path):
            sys.exit(f"No results file found at {json_path}. "
                     "Run without --plot-only first.")
        print(f"Loading results from {json_path}")
        results = load_results(json_path)
    else:
        print("═" * 65)
        print("  SHEET 11 – ACTOR-CRITIC ALGORITHMS EVALUATION STUDY")
        print("═" * 65)
        print(f"  Environments : {envs}")
        print(f"  Seeds        : {n_seeds}")
        print(f"  Results dir  : {SAVE_DIR}")

        t0 = time.time()
        results = run_all(envs, algos, n_seeds)
        elapsed = time.time() - t0
        print(f"\nTotal training time: {elapsed/60:.1f} min")
        save_results(results, json_path)

    # ── console summary ───────────────────────────────────────────────────────
    print_summary(results)

    # ── plots ─────────────────────────────────────────────────────────────────
    print(f"\nCreating plots in {SAVE_DIR}/")

    for env_name in results:
        save_env_figure(results, env_name)

    if len(results) > 1:
        save_overview(results)

    save_metrics_figure(results)

    # ── part (b) discussion ───────────────────────────────────────────────────
    print(METRICS_DISCUSSION)
    print("Done.")


if __name__ == "__main__":
    main()
