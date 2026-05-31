# Sheet 11 – Actor-Critic Algorithms: Evaluation Study

Exercise 4 from Sheet 11 of the Reinforcement Learning course (FSS 2026, Uni Mannheim).

---

## Overview

Evaluation study comparing 10 RL algorithms across all 5 Gymnasium classic-control
environments using 3 random seeds each.  Results and plots are saved to `results/`.

---

## Files

| File | Description |
|------|-------------|
| `reinforce.py` | Own implementations: REINFORCE and Mini-batch REINFORCE |
| `main.py` | Full evaluation pipeline: training, plotting, metrics |
| `results/` | Saved plots (PNG) and serialised results (`results.json`) |

---

## Algorithms

### Own implementations (`reinforce.py`)

#### REINFORCE
Standard policy gradient algorithm (Williams 1992), implementing the
**Policy Gradient Theorem** (Theorem 5.1.3 in the lecture notes):

```
∇J(θ) = E [ Σ_t  ∇_θ log π_θ(A_t | S_t) · R_t^T ]
```

where `R_t^T = Σ_{k≥t} γ^{k-t} R_{k+1}` is the **reward-to-go**.

- One gradient update per episode (K=1 in Algorithm 32)
- Policy network: 2-layer MLP (64 units, Tanh), Adam optimiser
- Discrete actions: Categorical distribution over logits
- Continuous actions: Normal distribution with learnable log-std
- Returns are normalised (zero mean, unit variance) before each update

#### Mini-batch REINFORCE
Same as REINFORCE but collects **K=8 episodes** before each gradient step
(Algorithm 32 with K>1).  Gradient is averaged over the batch, reducing
variance compared to single-episode updates.

---

### Pre-implemented algorithms (`sb3_contrib` / `stable-baselines3`)

| Algorithm | Library | Action space | Type |
|-----------|---------|--------------|------|
| ARS | `sb3_contrib` | both | Derivative-free, **linear policy** |
| A2C | `stable_baselines3` | both | On-policy, advantage actor-critic |
| PPO | `stable_baselines3` | both | On-policy, clipped surrogate |
| TRPO | `sb3_contrib` | both | On-policy, trust-region |
| DDPG | `stable_baselines3` | continuous | Off-policy, deterministic |
| SAC | `stable_baselines3` | continuous | Off-policy, max-entropy |
| TD3 | `stable_baselines3` | continuous | Off-policy, twin-critic |
| TQC | `sb3_contrib` | continuous | Off-policy, quantile critic |

**ARS (Augmented Random Search, Mania et al. 2018)** uses a linear policy
`π(s) = M · φ(s)` and updates `M` by evaluating random perturbations — no
gradient computation.  It uses `LinearPolicy` in sb3_contrib.

Off-policy algorithms (DDPG, SAC, TD3, TQC) are capped at `OFFPOLICY_TS_CAP`
timesteps because they perform one gradient step per environment step, making
them significantly slower on CPU than on-policy methods.

---

## Environments

| Environment | Actions | Timesteps | Reward range | Notes |
|-------------|---------|-----------|-------------|-------|
| CartPole-v1 | discrete | 50 000 | [0, 500] | Classic balancing task |
| MountainCar-v0 | discrete | 80 000 | [−200, −100] | Sparse reward — car must reach top |
| MountainCarContinuous-v0 | continuous | 60 000 | [−100, 90] | Energy penalty creates lazy-policy trap |
| Pendulum-v1 | continuous | 30 000 | [−3254, 0] | True min = −(π²+6.4+0.004)×200 |
| Acrobot-v1 | discrete | 80 000 | [−500, −72] | Double pendulum swing-up |

---

## Usage

```powershell
# Full study (all 5 envs, all algorithms, 3 seeds, ~2 hours on CPU)
$env:PYTHONIOENCODING="utf-8"
.\venv\Scripts\python.exe sheet11\main.py --seeds 3

# Re-generate all plots from existing results (seconds)
.\venv\Scripts\python.exe sheet11\main.py --plot-only

# Subset run
.\venv\Scripts\python.exe sheet11\main.py --envs CartPole-v1 Pendulum-v1 --algos PPO SAC --seeds 2
```

If `results/results.json` already exists, `--plot-only` is the fastest option.

---

## Output plots

| File | Description |
|------|-------------|
| `CartPole_v1_results.png` | Learning curves + final performance bar chart |
| `MountainCar_v0_results.png` | same |
| `MountainCarContinuous_v0_results.png` | same |
| `Pendulum_v1_results.png` | same |
| `Acrobot_v1_results.png` | same |
| `overview.png` | All 5 environments side by side |
| `metrics.png` | Aggregated metrics (see below) |

### metrics.png — 2×3 layout

Split into **discrete** (top row) and **continuous** (bottom row) environments
to avoid the bias introduced by off-policy algorithms only running on 2 of 5
environments.

| Panel | Metric | Description |
|-------|--------|-------------|
| Left | Performance Score | `(final_return − worst) / (best − worst)` using fixed theoretical bounds per env.  1 = perfect, 0 = worst. |
| Middle | Stability | Raw std of final return over seeds.  Lower = more stable. |
| Right | AUC | Area under the normalised learning curve.  Combines speed and quality: fast learning at a good level → high AUC. |

**Why AUC instead of "steps to 80 %":** The naive efficiency metric rewards
algorithms that quickly reach 80 % of *their own* (possibly bad) final
performance.  AUC integrates the normalised return over the whole training
budget, penalising both slow learning and low final performance.

---

## Key findings

### CartPole-v1
TRPO (426) and ARS (379) lead.  ARS converges fast but plateaus — the linear
policy has a natural ceiling.  REINFORCE shows high seed variance (±150) vs.
MiniBatch-REINFORCE (±33), demonstrating the variance-reduction effect of
batch averaging.

### MountainCar-v0
**Sparse reward problem**: all neural-network methods score exactly −200
(episode always times out, goal never reached).  Only ARS (≈−200, barely)
finds any signal through random linear-policy search.
*Known fix*: reward shaping, e.g. `r_extra = 300·position + 40·|velocity|`.

### MountainCarContinuous-v0
**Lazy policy trap**: ARS, TRPO, DDPG, TD3 converge to `action ≈ 0`, which
avoids the energy penalty `−0.1·u²` and yields return ≈ 0 — better numerically
than actually trying (−27) but the task is never solved.
REINFORCE and PPO at −27 are more useful in practice despite a lower return.
*Known fix*: reward shaping or more training steps for SAC/TQC (entropy
regularisation prevents the lazy-policy collapse).

### Pendulum-v1
Off-policy methods (DDPG −932, SAC −985, TQC −987) outperform on-policy
(PPO −1134, TRPO −1161) despite only 20 000 training steps vs. 30 000.
Replay buffers enable more efficient use of experience on this continuous
control task.

### Acrobot-v1
PPO (−92) and TRPO (−88) nearly optimal.  ARS (−103) competitive but limited
by the linear policy.  REINFORCE (−403) and A2C (−368) fail due to high
gradient variance over long episodes.

---

## Part (b) — Evaluation Metrics: Tabular vs. Deep RL

| Criterion | Tabular RL (Q-learning / GridWorld) | Deep RL (this study) |
|-----------|--------------------------------------|----------------------|
| Optimal reference | Known exactly (Q*) | Unknown — empirical max only |
| Convergence proof | Yes (Robbins-Monro conditions) | Rarely (deadly triad) |
| Primary metric | `‖Q_n − Q*‖_∞` | Mean episode return |
| Stability | High (exact Bellman updates) | Low (hyperparameter-sensitive) |
| Generalisation | None (separate entry per state) | Via neural function approximation |

**Improved metrics used in this study** (following Agarwal et al. 2021):
- Performance score with absolute environment bounds (not relative to empirical best)
- Stability measured as std over seeds
- AUC of the normalised learning curve (combines speed and quality)
- Separate aggregation for discrete and continuous action spaces to avoid
  coverage bias (off-policy algorithms run on fewer environments)
