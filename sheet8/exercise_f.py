"""
Exercise 4f: Optimale Parameter fuer Q-learning und Double Q-learning
======================================================================
Szenario (wie in der Aufgabenstellung):
  4x4 Grid World, gamma=0.9
  Start:            oben rechts  (0,3) = State  3
  Goal  (r=+1.0):   unten, 2. Sp. (3,1) = State 13   (terminal)
  Fake Goal (r=0.65): oben links  (0,0) = State  0   (terminal)
  Stochastic Region: 2x2 unten rechts = States {10,11,14,15}
    Reward: {-2.1, +2.0} mit gleicher Wahrscheinlichkeit (E=-0.05)
  Default-Reward:   {-0.05, +0.05} mit gleicher Wahrsch. (E=0.0)

Versionen: mit und ohne Rauschen (noise=True/False)
  noise=False: stochastische Rewards durch Erwartungswerte ersetzen

Methoden:
  1. Q-learning       (off-policy, Bellman-Optimalitaetsoperator T*)
  2. Double Q-learning (Algorithmus 23, reduziert Overestimation Bias)

Parameteroptimierung:
  Grid Search ueber alpha in {0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.7}
                und eps   in {0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5}
  Metrik: mittlerer Episode-Reward in den letzten 300 von 1500 Episoden
          gemittelt ueber 8 Seeds

Plots (3x2):
  Zeile 1: Heatmap Q-Lrn (noise)  |  Heatmap DQ-Lrn (noise)
  Zeile 2: Heatmap Q-Lrn (kein Rauschen) | Heatmap DQ-Lrn (kein Rauschen)
  Zeile 3: Lernkurven beste Params (noise) | Lernkurven (kein Rauschen)

Python: 3.x | numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

# =============================================================================
# GridWorld fuer Exercise f
# =============================================================================

class GridWorldF:
    """
    4x4 Grid World gemaess Aufgabe f.
    Gitter-Layout (0-indiziert von oben-links):

       col:  0    1    2    3
      row 0: FG   .    .    S
      row 1: .    .    .    .
      row 2: .    .    SR   SR
      row 3: .    G    SR   SR

    FG = Fake Goal  (terminal, r=0.65)
    S  = Start
    G  = Goal       (terminal, r=1.0)
    SR = Stochastic Region (r ~ {-2.1, +2.0})
    .  = Default           (r ~ {-0.05, +0.05})
    """
    ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # hoch, runter, links, rechts

    def __init__(self, size: int = 4, gamma: float = 0.9, noise: bool = True):
        self.size      = size
        self.gamma     = gamma
        self.noise     = noise
        self.n_s       = size * size
        self.n_a       = 4
        self.start     = size - 1                      # (0,3) = 3
        self.goal      = (size - 1) * size + 1         # (3,1) = 13
        self.fake_goal = 0                             # (0,0) = 0
        self.sr_states = {                             # 2x2 unten rechts
            self._s(2, 2), self._s(2, 3),
            self._s(3, 2), self._s(3, 3),
        }
        self.terminals = {self.goal, self.fake_goal}
        self._build()

    def _s(self, r, c): return r * self.size + c
    def _rc(self, s):   return divmod(s, self.size)

    def _build(self):
        """Deterministischer Uebergang (Wand-Bounce)."""
        self.P = np.zeros((self.n_s, self.n_a, self.n_s))
        for s in range(self.n_s):
            if s in self.terminals:
                for a in range(self.n_a):
                    self.P[s, a, s] = 1.0
                continue
            row, col = self._rc(s)
            for a, (dr, dc) in enumerate(self.ACTIONS):
                r2 = max(0, min(self.size - 1, row + dr))
                c2 = max(0, min(self.size - 1, col + dc))
                self.P[s, a, self._s(r2, c2)] = 1.0

    def _reward(self, s2: int) -> float:
        """Sampelt Reward fuer Ankunft in Zustand s2."""
        if s2 == self.goal:
            return 1.0
        if s2 == self.fake_goal:
            return 0.65
        if s2 in self.sr_states:
            return (np.random.choice([-2.1, 2.0]) if self.noise
                    else (-2.1 + 2.0) / 2.0)   # E = -0.05
        return (np.random.choice([-0.05, 0.05]) if self.noise
                else 0.0)                        # E = 0.0

    def get_arrays(self):
        """Gibt Arrays fuer Numba-Core zurueck."""
        det_next = np.array(
            [[int(np.where(self.P[s, a] > 0)[0][0]) for a in range(self.n_a)]
             for s in range(self.n_s)], dtype=np.int32)
        sr_arr = np.array(sorted(self.sr_states), dtype=np.int32)
        return det_next, sr_arr

    def step(self, s: int, a: int):
        s2   = int(np.where(self.P[s, a] > 0)[0][0])  # deterministisch
        r    = self._reward(s2)
        done = s2 in self.terminals
        return s2, r, done

    def grid_str(self) -> str:
        """ASCII-Darstellung des Grids."""
        icons = {self.start: 'S', self.goal: 'G',
                 self.fake_goal: 'F'}
        rows = []
        for row in range(self.size):
            line = []
            for col in range(self.size):
                s = self._s(row, col)
                if s in icons:
                    line.append(icons[s])
                elif s in self.sr_states:
                    line.append('R')
                else:
                    line.append('.')
            rows.append(' '.join(line))
        return '\n'.join(rows)


# =============================================================================
# Hilfsfunktionen
# =============================================================================

def _greedy(q_s: np.ndarray) -> int:
    best = np.where(q_s == q_s.max())[0]
    return int(np.random.choice(best))


def rolling(x, w: int = 50) -> np.ndarray:
    return np.convolve(x, np.ones(w) / w, mode='valid')


# =============================================================================
# Q-learning
# =============================================================================

def q_learning_run(mdp: GridWorldF, n_episodes: int,
                   alpha: float, eps: float,
                   max_steps: int = 100, seed: int = 0):
    """
    Q-learning mit konstantem alpha und eps.
    Gibt Liste der Episode-Rewards zurueck.
    """
    np.random.seed(seed)
    Q = np.zeros((mdp.n_s, mdp.n_a))
    rewards = []

    for _ in range(n_episodes):
        s = mdp.start
        total_r = 0.0
        for _ in range(max_steps):
            a = (np.random.randint(mdp.n_a) if np.random.rand() < eps
                 else _greedy(Q[s]))
            s2, r, done = mdp.step(s, a)
            Q[s, a] += alpha * (r + mdp.gamma * Q[s2].max() - Q[s, a])
            total_r += r
            s = s2
            if done:
                break
        rewards.append(total_r)
    return rewards


# =============================================================================
# Double Q-learning (Algorithmus 23)
# =============================================================================

def double_q_learning_run(mdp: GridWorldF, n_episodes: int,
                          alpha: float, eps: float,
                          max_steps: int = 100, seed: int = 0):
    """
    Double Q-learning (Algorithmus 23 aus dem Skript).
    Zwei unabhaengige Schaetzer QA, QB.
    Action-Selection: eps-greedy auf QA + QB.
    Update (zufaellig 50/50):
      A-Update: a* = argmax QA(s2); QA(s,a) += alpha*(r + gamma*QB(s2,a*) - QA(s,a))
      B-Update: b* = argmax QB(s2); QB(s,a) += alpha*(r + gamma*QA(s2,b*) - QB(s,a))
    Reduziert Overestimation Bias von Standard Q-learning (besonders im SR).
    """
    np.random.seed(seed)
    QA = np.zeros((mdp.n_s, mdp.n_a))
    QB = np.zeros((mdp.n_s, mdp.n_a))
    rewards = []

    for _ in range(n_episodes):
        s = mdp.start
        total_r = 0.0
        for _ in range(max_steps):
            q_sum = QA[s] + QB[s]
            a = (np.random.randint(mdp.n_a) if np.random.rand() < eps
                 else _greedy(q_sum))
            s2, r, done = mdp.step(s, a)

            if np.random.rand() < 0.5:
                a_star = _greedy(QA[s2])
                QA[s, a] += alpha * (r + mdp.gamma * QB[s2, a_star] - QA[s, a])
            else:
                b_star = _greedy(QB[s2])
                QB[s, a] += alpha * (r + mdp.gamma * QA[s2, b_star] - QB[s, a])

            total_r += r
            s = s2
            if done:
                break
        rewards.append(total_r)
    return rewards


# =============================================================================
# Numba-Cores
# =============================================================================

if HAS_NUMBA:
    @numba.njit(cache=True)
    def _ql_numba(det_next, sr_arr, goal, fake_goal, gamma,
                  alpha, eps, n_episodes, max_steps, noise, seed):
        np.random.seed(seed)
        n_s = det_next.shape[0]; n_a = det_next.shape[1]
        Q = np.zeros((n_s, n_a))
        rewards = np.empty(n_episodes)
        for ep in range(n_episodes):
            s = n_a - 1; total_r = 0.0  # start = size-1 = n_a-1 = 3
            for _ in range(max_steps):
                if np.random.random() < eps:
                    a = np.random.randint(0, n_a)
                else:
                    best_v = Q[s, 0]; best_a = 0
                    for ai in range(1, n_a):
                        v = Q[s, ai] + np.random.random() * 1e-10
                        if v > best_v: best_v = v; best_a = ai
                    a = best_a
                s2 = det_next[s, a]
                if s2 == goal: r = 1.0
                elif s2 == fake_goal: r = 0.65
                else:
                    is_sr = False
                    for k in range(len(sr_arr)):
                        if s2 == sr_arr[k]: is_sr = True; break
                    if is_sr:
                        r = (np.random.choice(np.array([-2.1, 2.0]))
                             if noise else -0.05)
                    else:
                        r = (np.random.choice(np.array([-0.05, 0.05]))
                             if noise else 0.0)
                done = (s2 == goal) or (s2 == fake_goal)
                max_q = Q[s2, 0]
                for ai in range(1, n_a):
                    if Q[s2, ai] > max_q: max_q = Q[s2, ai]
                Q[s, a] += alpha * (r + gamma * max_q - Q[s, a])
                total_r += r; s = s2
                if done: break
            rewards[ep] = total_r
        return rewards

    @numba.njit(cache=True)
    def _dql_numba(det_next, sr_arr, goal, fake_goal, gamma,
                   alpha, eps, n_episodes, max_steps, noise, seed):
        np.random.seed(seed)
        n_s = det_next.shape[0]; n_a = det_next.shape[1]
        QA = np.zeros((n_s, n_a)); QB = np.zeros((n_s, n_a))
        rewards = np.empty(n_episodes)
        for ep in range(n_episodes):
            s = n_a - 1; total_r = 0.0
            for _ in range(max_steps):
                q_sum = QA[s] + QB[s]
                if np.random.random() < eps:
                    a = np.random.randint(0, n_a)
                else:
                    best_v = q_sum[0]; a = 0
                    for ai in range(1, n_a):
                        v = q_sum[ai] + np.random.random() * 1e-10
                        if v > best_v: best_v = v; a = ai
                s2 = det_next[s, a]
                if s2 == goal: r = 1.0
                elif s2 == fake_goal: r = 0.65
                else:
                    is_sr = False
                    for k in range(len(sr_arr)):
                        if s2 == sr_arr[k]: is_sr = True; break
                    if is_sr:
                        r = (np.random.choice(np.array([-2.1, 2.0]))
                             if noise else -0.05)
                    else:
                        r = (np.random.choice(np.array([-0.05, 0.05]))
                             if noise else 0.0)
                done = (s2 == goal) or (s2 == fake_goal)
                if np.random.random() < 0.5:
                    a_star = 0; bv = QA[s2, 0]
                    for ai in range(1, n_a):
                        if QA[s2, ai] > bv: bv = QA[s2, ai]; a_star = ai
                    QA[s, a] += alpha * (r + gamma * QB[s2, a_star] - QA[s, a])
                else:
                    b_star = 0; bv = QB[s2, 0]
                    for ai in range(1, n_a):
                        if QB[s2, ai] > bv: bv = QB[s2, ai]; b_star = ai
                    QB[s, a] += alpha * (r + gamma * QA[s2, b_star] - QB[s, a])
                total_r += r; s = s2
                if done: break
            rewards[ep] = total_r
        return rewards


# =============================================================================
# Top-Level Worker (picklbar)
# =============================================================================

def _worker(args):
    det_next, sr_arr, goal, fake_goal, gamma, alpha, eps, \
        n_ep, max_steps, noise, seed, use_double = args
    if HAS_NUMBA:
        fn = _dql_numba if use_double else _ql_numba
        return fn(det_next, sr_arr, goal, fake_goal, gamma,
                  alpha, eps, n_ep, max_steps, noise, seed)
    # Python fallback
    mdp_stub = type('M', (), {
        'n_s': det_next.shape[0], 'n_a': det_next.shape[1],
        'start': det_next.shape[1] - 1, 'goal': goal,
        'fake_goal': fake_goal, 'gamma': gamma,
        'P': None
    })()
    fn = double_q_learning_run if use_double else q_learning_run
    return np.array(fn(mdp_stub, n_ep, alpha, eps, max_steps, seed))


# =============================================================================
# Grid Search  (parallelisiert)
# =============================================================================

def grid_search(mdp: GridWorldF, use_double: bool, alphas, epsilons,
                n_episodes: int = 1500, n_runs: int = 8,
                eval_last: int = 300):
    det_next, sr_arr = mdp.get_arrays()
    base = (det_next, sr_arr, mdp.goal, mdp.fake_goal, mdp.gamma)
    tasks = [(base + (a, e, n_episodes, 100, mdp.noise, s, use_double))
             for a in alphas for e in epsilons for s in range(n_runs)]

    n_workers = min(len(tasks), mp.cpu_count())
    with mp.Pool(n_workers) as pool:
        raw = pool.map(_worker, tasks)

    results = np.zeros((len(alphas), len(epsilons)))
    idx = 0
    for i in range(len(alphas)):
        for j in range(len(epsilons)):
            vals = [raw[idx + s][-eval_last:].mean() for s in range(n_runs)]
            results[i, j] = np.mean(vals)
            idx += n_runs
    return results


# =============================================================================
# Lernkurven  (parallelisiert)
# =============================================================================

def learning_curves(mdp, use_double, alpha, eps,
                    n_episodes: int = 3000, n_runs: int = 20):
    det_next, sr_arr = mdp.get_arrays()
    base = (det_next, sr_arr, mdp.goal, mdp.fake_goal, mdp.gamma)
    tasks = [(base + (alpha, eps, n_episodes, 100, mdp.noise, s, use_double))
             for s in range(n_runs)]
    n_workers = min(n_runs, mp.cpu_count())
    with mp.Pool(n_workers) as pool:
        raw = pool.map(_worker, tasks)
    arr = np.array(raw)
    return arr.mean(0), arr.std(0)


# =============================================================================
# Heatmap-Plot-Hilfsfunktion
# =============================================================================

def plot_heatmap(ax, data, alphas, epsilons, title, best_idx=None):
    im = ax.imshow(data, aspect='auto', origin='upper',
                   cmap='RdYlGn', vmin=data.min(), vmax=max(data.max(), 0.01))
    plt.colorbar(im, ax=ax, label='Mittl. Reward (letzte 300 Ep.)')
    ax.set_xticks(range(len(epsilons)))
    ax.set_yticks(range(len(alphas)))
    ax.set_xticklabels([f'{e:.2f}' for e in epsilons], fontsize=7, rotation=45)
    ax.set_yticklabels([f'{a:.2f}' for a in alphas], fontsize=7)
    ax.set_xlabel('epsilon', fontsize=9)
    ax.set_ylabel('alpha', fontsize=9)
    ax.set_title(title, fontsize=10)

    # Werte einzeichnen + bestes markieren
    for i in range(len(alphas)):
        for j in range(len(epsilons)):
            color = 'white' if data[i, j] < (data.min() + data.max()) / 2 else 'black'
            ax.text(j, i, f'{data[i,j]:.2f}', ha='center', va='center',
                    fontsize=6, color=color)
    if best_idx is not None:
        ax.add_patch(plt.Rectangle((best_idx[1] - 0.5, best_idx[0] - 0.5), 1, 1,
                                   fill=False, edgecolor='royalblue', lw=3))


# =============================================================================
# Main
# =============================================================================

def main():
    np.random.seed(42)

    ALPHAS   = [0.07, 0.1, 0.13, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.7]
    EPSILONS = [0.05, 0.1, 0.13, 0.15, 0.18, 0.2, 0.23, 0.25, 0.3, 0.4]
    N_SRCH   = 1500   # Episoden fuer Grid Search
    N_RUNS_S = 8      # Seeds fuer Grid Search
    N_RUNS_L = 20     # Seeds fuer Lernkurven
    N_EP_L   = 3000   # Episoden fuer Lernkurven
    EVAL_LAST = 300
    W         = 100

    mdp_n = GridWorldF(noise=True)
    mdp_d = GridWorldF(noise=False)

    print("Grid-Layout:")
    print(mdp_n.grid_str())
    print(f"\nStart={mdp_n.start}, Goal={mdp_n.goal}, "
          f"FakeGoal={mdp_n.fake_goal}, SR={sorted(mdp_n.sr_states)}")

    # ── Grid Search ───────────────────────────────────────────────────────────
    print("\nGrid Search Q-learning (mit Rauschen) ...")
    ql_n = grid_search(mdp_n, False, ALPHAS, EPSILONS, N_SRCH, N_RUNS_S)

    print("Grid Search Double Q-learning (mit Rauschen) ...")
    dq_n = grid_search(mdp_n, True,  ALPHAS, EPSILONS, N_SRCH, N_RUNS_S)

    print("Grid Search Q-learning (kein Rauschen) ...")
    ql_d = grid_search(mdp_d, False, ALPHAS, EPSILONS, N_SRCH, N_RUNS_S)

    print("Grid Search Double Q-learning (kein Rauschen) ...")
    dq_d = grid_search(mdp_d, True,  ALPHAS, EPSILONS, N_SRCH, N_RUNS_S)

    # ── Beste Parameter ───────────────────────────────────────────────────────
    def best(grid):
        idx = np.unravel_index(grid.argmax(), grid.shape)
        return idx, ALPHAS[idx[0]], EPSILONS[idx[1]], grid[idx]

    ql_n_idx,  ql_n_a,  ql_n_e,  ql_n_sc  = best(ql_n)
    dq_n_idx,  dq_n_a,  dq_n_e,  dq_n_sc  = best(dq_n)
    ql_d_idx,  ql_d_a,  ql_d_e,  ql_d_sc  = best(ql_d)
    dq_d_idx,  dq_d_a,  dq_d_e,  dq_d_sc  = best(dq_d)

    print(f"\nOptimale Parameter (Grid Search, {N_RUNS_S} Seeds):")
    print(f"  Q-Lrn  (Rauschen):    alpha={ql_n_a}, eps={ql_n_e}, Score={ql_n_sc:.3f}")
    print(f"  DQ-Lrn (Rauschen):    alpha={dq_n_a}, eps={dq_n_e}, Score={dq_n_sc:.3f}")
    print(f"  Q-Lrn  (kein Rauch.): alpha={ql_d_a}, eps={ql_d_e}, Score={ql_d_sc:.3f}")
    print(f"  DQ-Lrn (kein Rauch.): alpha={dq_d_a}, eps={dq_d_e}, Score={dq_d_sc:.3f}")

    # ── Lernkurven mit besten Parametern ─────────────────────────────────────
    print(f"\nLernkurven mit optimalen Parametern ({N_RUNS_L} Seeds x {N_EP_L} Ep.) ...")

    ql_n_rm,  ql_n_rs  = learning_curves(mdp_n, False, ql_n_a, ql_n_e, N_EP_L, N_RUNS_L)
    dq_n_rm,  dq_n_rs  = learning_curves(mdp_n, True,  dq_n_a, dq_n_e, N_EP_L, N_RUNS_L)
    ql_d_rm,  ql_d_rs  = learning_curves(mdp_d, False, ql_d_a, ql_d_e, N_EP_L, N_RUNS_L)
    dq_d_rm,  dq_d_rs  = learning_curves(mdp_d, True,  dq_d_a, dq_d_e, N_EP_L, N_RUNS_L)

    last = slice(-200, None)
    print(f"\nFinale Performance (letzte 200 Ep., {N_RUNS_L} Seeds):")
    print(f"  Q-Lrn  (Rauschen):    {ql_n_rm[last].mean():.3f}")
    print(f"  DQ-Lrn (Rauschen):    {dq_n_rm[last].mean():.3f}")
    print(f"  Q-Lrn  (kein Rauch.): {ql_d_rm[last].mean():.3f}")
    print(f"  DQ-Lrn (kein Rauch.): {dq_d_rm[last].mean():.3f}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 2, figsize=(12, 15))

    # Zeile 1: Heatmaps mit Rauschen
    plot_heatmap(axes[0, 0], ql_n, ALPHAS, EPSILONS,
                 f'Q-learning  (Rauschen)  — Optimum: α={ql_n_a}, ε={ql_n_e}',
                 best_idx=ql_n_idx)
    plot_heatmap(axes[0, 1], dq_n, ALPHAS, EPSILONS,
                 f'Double Q-learning  (Rauschen)  — Optimum: α={dq_n_a}, ε={dq_n_e}',
                 best_idx=dq_n_idx)

    # Zeile 2: Heatmaps ohne Rauschen
    plot_heatmap(axes[1, 0], ql_d, ALPHAS, EPSILONS,
                 f'Q-learning  (kein Rauschen)  — Optimum: α={ql_d_a}, ε={ql_d_e}',
                 best_idx=ql_d_idx)
    plot_heatmap(axes[1, 1], dq_d, ALPHAS, EPSILONS,
                 f'Double Q-learning  (kein Rauschen)  — Optimum: α={dq_d_a}, ε={dq_d_e}',
                 best_idx=dq_d_idx)

    # Zeile 3: Lernkurven
    ROLL_X = np.arange(W, N_EP_L + 1)

    ax = axes[2, 0]
    for name, rm, rs, c in [('Q-learning',        ql_n_rm, ql_n_rs, 'steelblue'),
                             ('Double Q-learning', dq_n_rm, dq_n_rs, 'darkorange')]:
        sm = rolling(rm, W)
        ax.plot(ROLL_X, sm, color=c, lw=2.5,
                label=f'{name}  (α={ql_n_a if name[0]=="Q" else dq_n_a},'
                      f' ε={ql_n_e if name[0]=="Q" else dq_n_e})')
        ax.fill_between(ROLL_X,
                        rolling(np.maximum(rm - rs, -5), W),
                        rolling(rm + rs, W),
                        alpha=0.15, color=c)
    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel(f'Gegl. Reward  (Fenster={W})', fontsize=11)
    ax.set_title('Lernkurven  (mit Rauschen)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', lw=0.7, ls='--')

    ax = axes[2, 1]
    for name, rm, rs, c in [('Q-learning',        ql_d_rm, ql_d_rs, 'steelblue'),
                             ('Double Q-learning', dq_d_rm, dq_d_rs, 'darkorange')]:
        sm = rolling(rm, W)
        ax.plot(ROLL_X, sm, color=c, lw=2.5,
                label=f'{name}  (α={ql_d_a if name[0]=="Q" else dq_d_a},'
                      f' ε={ql_d_e if name[0]=="Q" else dq_d_e})')
        ax.fill_between(ROLL_X,
                        rolling(np.maximum(rm - rs, -5), W),
                        rolling(rm + rs, W),
                        alpha=0.15, color=c)
    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel(f'Gegl. Reward  (Fenster={W})', fontsize=11)
    ax.set_title('Lernkurven  (kein Rauschen)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', lw=0.7, ls='--')

    plt.suptitle(
        f'Q-learning & Double Q-learning  |  4x4 GridWorld  |  gamma=0.9',
        fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig('sheet8/exercise_f.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Gespeichert: sheet8/exercise_f.png")


if __name__ == "__main__":
    main()
