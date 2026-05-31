# Sheet 11 – Actor-Critic Algorithms: Evaluation Study

Exercise 4 from Sheet 11 of the Reinforcement Learning course (FSS 2026, Uni Mannheim).

---

## Overview

Evaluation study comparing 10 RL algorithms across all 5 Gymnasium classic-control
environments using 3 random seeds each.  Results and plots are saved to `results/`.

---

## Algorithms

### Own implementations

**REINFORCE** — standard policy gradient (Williams 1992), Policy Gradient Theorem:

```
∇J(θ) = E [ Σ_t  ∇_θ log π_θ(A_t | S_t) · R_t^T ]
```

One gradient update per episode.  Returns normalised before each update.

**Mini-batch REINFORCE** — same as REINFORCE but averages the gradient over K=8 episodes per update, reducing variance.

Both use a 2-layer MLP (64 units, Tanh) with a Categorical distribution (discrete) or Normal distribution with learnable log-std (continuous).

---

### Pre-implemented algorithms (`sb3_contrib` / `stable-baselines3`)

| Algorithm | Action space | Type |
|-----------|--------------|------|
| ARS | both | Derivative-free, linear policy `π(s) = M·φ(s)` |
| A2C | both | On-policy, advantage actor-critic |
| PPO | both | On-policy, clipped surrogate |
| TRPO | both | On-policy, trust-region |
| DDPG | continuous | Off-policy, deterministic |
| SAC | continuous | Off-policy, max-entropy |
| TD3 | continuous | Off-policy, twin-critic |
| TQC | continuous | Off-policy, quantile critic |

Off-policy algorithms are trained for fewer timesteps due to CPU runtime constraints (one gradient step per environment step).

---

## Environments

| Environment | Actions | Timesteps | Reward range |
|-------------|---------|-----------|-------------|
| CartPole-v1 | discrete | 50 000 | [0, 500] |
| MountainCar-v0 | discrete | 80 000 | [−200, −100] |
| MountainCarContinuous-v0 | continuous | 60 000 | [−100, 90] |
| Pendulum-v1 | continuous | 30 000 | [−3254, 0] |
| Acrobot-v1 | discrete | 80 000 | [−500, −72] |

---

## Usage

```powershell
# Full study (~2 hours on CPU)
$env:PYTHONIOENCODING="utf-8"
.\venv\Scripts\python.exe sheet11\main.py --seeds 3

# Re-generate plots from existing results (seconds)
.\venv\Scripts\python.exe sheet11\main.py --plot-only

# Subset run
.\venv\Scripts\python.exe sheet11\main.py --envs CartPole-v1 Pendulum-v1 --algos PPO SAC --seeds 2
```

---

## Output plots

Each environment gets a figure with learning curves (mean ± std over seeds) and a
final performance bar chart.  `overview.png` shows all 5 environments side by side.

`metrics.png` is a **2×3 grid** split by action space type (discrete top, continuous
bottom) to avoid coverage bias — off-policy algorithms only run on continuous environments.

| Panel | Metric |
|-------|--------|
| Left | Performance score: `(return − worst) / (best − worst)`, theoretical bounds per env |
| Middle | Stability: raw std of final return over seeds (lower = better) |
| Right | AUC: area under the normalised learning curve — combines speed and quality |

AUC is used instead of "steps to 80 % of final return" because the naive metric
rewards algorithms that quickly converge to a *bad* final performance.

---

## Key findings

**CartPole-v1**: TRPO (426) and ARS (379) lead.  MiniBatch-REINFORCE (±33) is much
more stable than single-episode REINFORCE (±150).

**MountainCar-v0**: All neural-network methods fail (return = −200, goal never
reached) due to the sparse reward.  ARS finds marginal improvement through
random linear-policy search.

**MountainCarContinuous-v0**: ARS, TRPO, DDPG, TD3 converge to `action ≈ 0`
(lazy policy), avoiding the energy penalty but never solving the task.
REINFORCE and PPO (return ≈ −27) actually attempt the task despite a lower score.

**Pendulum-v1**: Off-policy methods (DDPG −932, SAC −985) outperform on-policy
(PPO −1134) despite fewer training steps, demonstrating the sample efficiency
of replay-buffer-based learning.

**Acrobot-v1**: PPO (−92) and TRPO (−88) nearly optimal.  REINFORCE (−403) and
A2C (−368) fail due to high gradient variance over long episodes.

---

## Part (b) — Evaluation Metrics: Tabular vs. Deep RL

### What changes compared to the tabular setting

In the tabular exercises (Q-learning, SARSA on GridWorld), the optimal
Q-function Q* could be computed exactly via Value Iteration.  This allowed
measuring convergence directly as `‖Q_n − Q*‖_∞` — the distance between the
current estimate and the true optimum.  The algorithm was considered solved
once this error fell below a threshold.

In the deep / continuous-state setting of this exercise, no such reference
exists.  CartPole has a theoretical maximum of 500, but whether any algorithm
actually *reaches* it depends on training time and random seeds.  For
MountainCar the reward is so sparse that most algorithms score −200 without
ever seeing a positive signal — comparing to an optimum makes no sense here.
The metrics therefore shift from absolute error to empirical comparisons.

### Metrics used in this study and why

**Mean final return (last 20 % of episodes)**  
The most direct metric: what does the algorithm achieve after training?
Used for the bar charts in the per-environment figures.  Problem: incomparable
across environments because reward scales differ (CartPole 0–500 vs.
Pendulum −3254–0).

**Performance score `(return − worst) / (best − worst)`**  
Normalises to [0, 1] using fixed theoretical bounds per environment (e.g.
−3254 and 0 for Pendulum), making results comparable across environments.
A score of 1 means the algorithm reached the theoretically best possible
return; 0 means it performed as badly as the worst case.  Used in `metrics.png`.

**Std over seeds**  
In the tabular setting, the same algorithm always converges to the same Q*
(given Robbins-Monro step sizes), so stability is not a meaningful concern.
With neural networks the result depends heavily on random initialisation,
and a single run is unreliable.  The std over 3 seeds therefore replaces the
convergence guarantee: low std means the algorithm is robust, high std means
success is largely a matter of luck.  REINFORCE shows std ±150 on CartPole
while PPO shows ±22 — the same difference that convergence proofs capture
analytically in the tabular case.

**AUC (area under the normalised learning curve)**  
The learning curve plots (episode return over timesteps) replace the
convergence-rate plots from the tabular setting.  The AUC summarises the
entire curve in one number: an algorithm that learns quickly *and* reaches a
high final return gets a high AUC, whereas an algorithm that plateaus early
at a poor level gets a low AUC even if it converged fast.  This is important
because a naive efficiency metric ("steps to 80 % of own final return")
incorrectly rewards algorithms like REINFORCE that settle quickly on a bad
solution.

**Separate aggregation for discrete and continuous environments**  
In the tabular setting all algorithms ran on the same GridWorld, so
aggregating results was trivial.  Here, off-policy algorithms (DDPG, SAC,
TD3, TQC) require continuous action spaces and therefore only run on 2 of
the 5 environments.  Averaging their scores together with on-policy
algorithms that ran on all 5 would bias the comparison.  Splitting the
metrics plot into a discrete row and a continuous row ensures that each
aggregation only covers algorithms with the same environment coverage.
