"""
Exercise 4d: Stepsizes and Exploration Parameters for Q-learning
=================================================================
Aufgabe:
  - Bedeutung der Schrittweite alpha und des Explorationsparameters epsilon
    fuer Q-learning herausarbeiten.
  - Faustregeln fuer die Wahl von Schedules entwickeln.
  - Exploration vs. Exploitation und Committal Behavior demonstrieren.

MDP: 4x4 GridWorld mit Fake Goal (angelehnt an Exercise f)
  - Start:      oben rechts  (State 3)
  - Fake Goal:  oben links   (State 0,  Reward 0.65) -- nah am Start
  - Real Goal:  unten links  (State 12, Reward 1.00) -- weiter entfernt
  - gamma = 0.9

Plots (2x2):
  (0,0) Konstante alpha-Werte       -> ||Q_n - Q*||_inf  (Robbins-Monro-Verletzung)
  (0,1) Alpha-Schedules             -> ||Q_n - Q*||_inf  (RM-Bedingung vs. Verletzung)
  (1,0) Konstante epsilon-Werte     -> episodischer Reward (Exploration/Exploitation)
  (1,1) Committal Behavior          -> Anteil Episoden die Real Goal erreichen

Python: 3.x | numpy, matplotlib
Hyperparameter: gamma=0.9, N_RUNS=20, N_EP=2000, Rollenfenster=100
"""

import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("Hinweis: numba nicht installiert, Python-Fallback aktiv.")


# =============================================================================
# Picklable Schedule-Callables (benoetigt fuer multiprocessing)
# =============================================================================

class ConstAlpha:
    def __init__(self, val): self.val = float(val)
    def __call__(self, n):   return self.val

class PowerAlpha:
    def __init__(self, p): self.p = float(p)
    def __call__(self, n): return float(n) ** (-self.p)

class ConstEps:
    def __init__(self, val): self.val = float(val)
    def __call__(self, ep):  return self.val

class DecayEps:
    def __init__(self, scale, shift, min_val):
        self.scale = float(scale); self.shift = float(shift)
        self.min_val = float(min_val)
    def __call__(self, ep): return max(self.min_val, self.scale / (ep + self.shift))

# =============================================================================
# GridWorld mit Fake Goal
# =============================================================================

class GridWorld:
    """
    4x4 GridWorld.
    Start:     oben rechts  = state size-1
    Fake Goal: oben links   = state 0,          Reward fake_r  (terminal)
    Real Goal: unten links  = state (size-1)*size, Reward 1.0  (terminal)
    Alle anderen Transitions: Reward 0.

    noise_prob: Wahrscheinlichkeit, statt der gewaehlten Aktion eine zufaellige
                zu nehmen (uniform ueber alle 4 Aktionen). Macht den Unterschied
                zwischen alpha-Schedules sichtbar, da grosse alpha bei Rauschen
                staerker oszillieren.
    """
    ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]   # oben, unten, links, rechts

    def __init__(self, size: int = 4, gamma: float = 0.95,
                 fake_goal_reward: float = 0.65, real_goal_reward: float = 1.0,
                 noise_prob: float = 0.0, step_cost: float = 0.0):
        self.size       = size
        self.gamma      = gamma
        self.n_s        = size * size
        self.n_a        = 4
        self.start      = size - 1              # (row=0, col=size-1)
        self.fake_goal  = 0                     # (row=0, col=0)
        self.real_goal  = (size - 1) * size     # (row=size-1, col=0)
        self.fake_r     = fake_goal_reward
        self.real_r     = real_goal_reward
        self.noise_prob = noise_prob
        self.step_cost  = step_cost
        self._build()

    def _s(self, r, c): return r * self.size + c
    def _rc(self, s):   return divmod(s, self.size)

    def _next_state(self, row, col, a):
        dr, dc = self.ACTIONS[a]
        r2 = max(0, min(self.size - 1, row + dr))
        c2 = max(0, min(self.size - 1, col + dc))
        return self._s(r2, c2)

    def _build(self):
        self.P = np.zeros((self.n_s, self.n_a, self.n_s))
        self.R = np.zeros((self.n_s, self.n_a))
        self._terminals = {self.fake_goal, self.real_goal}

        for s in range(self.n_s):
            if s in self._terminals:
                for a in range(self.n_a):
                    self.P[s, a, s] = 1.0
                continue
            row, col = self._rc(s)
            for a in range(self.n_a):
                for a2 in range(self.n_a):
                    s2 = self._next_state(row, col, a2)
                    p  = (1.0 - self.noise_prob) * (1 if a2 == a else 0)
                    p += self.noise_prob / self.n_a
                    self.P[s, a, s2] += p
                for a2 in range(self.n_a):
                    s2  = self._next_state(row, col, a2)
                    p   = (1.0 - self.noise_prob) * (1 if a2 == a else 0)
                    p  += self.noise_prob / self.n_a
                    r_s2 = (self.real_r if s2 == self.real_goal else
                            self.fake_r if s2 == self.fake_goal else 0.0)
                    self.R[s, a] += p * r_s2
                self.R[s, a] += self.step_cost

        if self.noise_prob == 0.0:
            # Deterministischer Fast-Path: direktes (n_s x n_a) Integer-Array.
            # Kein searchsorted, kein numpy-Overhead pro step().
            self._det_next = np.array(
                [[int(np.where(self.P[s, a] > 0)[0][0]) for a in range(self.n_a)]
                 for s in range(self.n_s)], dtype=np.int32)
            self._T_states = self._T_cumprob = None
        else:
            self._det_next = None
            self._T_states, self._T_cumprob = [], []
            for s in range(self.n_s):
                row_s, row_cp = [], []
                for a in range(self.n_a):
                    nz = np.where(self.P[s, a] > 0)[0]
                    cp = np.cumsum(self.P[s, a, nz])
                    cp[-1] = 1.0
                    row_s.append(nz)
                    row_cp.append(cp)
                self._T_states.append(row_s)
                self._T_cumprob.append(row_cp)

    def get_trans_arrays(self):
        """Gepaddte Transition-Arrays fuer den Numba-Core (max 4 Nachfolger)."""
        T_s  = np.zeros((self.n_s, self.n_a, 4), dtype=np.int32)
        T_cp = np.ones ((self.n_s, self.n_a, 4))
        T_n  = np.ones ((self.n_s, self.n_a), dtype=np.int32)
        if self._det_next is not None:
            T_s[:, :, 0] = self._det_next
        else:
            for s in range(self.n_s):
                for a in range(self.n_a):
                    nb = len(self._T_states[s][a])
                    T_n[s, a] = nb
                    T_s [s, a, :nb] = self._T_states[s][a]
                    T_cp[s, a, :nb] = self._T_cumprob[s][a]
        return T_s, T_cp, T_n

    def step(self, s, a, rand_val=None):
        if self._det_next is not None:
            s2 = int(self._det_next[s, a])
        else:
            if rand_val is None:
                rand_val = np.random.random()
            states = self._T_states[s][a]
            s2 = int(states[np.searchsorted(self._T_cumprob[s][a], rand_val)])
        return s2, self.R[s, a], s2 in self._terminals


# =============================================================================
# Q* via Value Iteration (modellbasiert)
# =============================================================================

def compute_Q_star(mdp, eps=1e-13) -> np.ndarray:
    V = np.zeros(mdp.n_s)
    for _ in range(20_000):
        Q     = mdp.R + mdp.gamma * np.einsum('ijk,k->ij', mdp.P, V)
        V_new = np.max(Q, axis=1)
        if np.max(np.abs(V_new - V)) < eps:
            return Q
        V = V_new
    return mdp.R + mdp.gamma * np.einsum('ijk,k->ij', mdp.P, V)


# =============================================================================
# Q-learning (ein Seed)
# =============================================================================

_MAX_VISITS = 50_000   # obere Schranke fuer Alpha-Lookup-Table

if HAS_NUMBA:
    @numba.njit(cache=True)
    def _ql_core(Q, N, gamma, start, exploring_starts, non_terminals,
                 fake_goal, real_goal, n_episodes, max_steps,
                 alpha_table, eps_arr, R, T_s, T_cp, T_n,
                 Q_star, compute_errors, seed):
        np.random.seed(seed)
        n_a = Q.shape[1]
        rewards      = np.empty(n_episodes)
        reached_real = np.zeros(n_episodes, dtype=np.int8)
        q_errors     = np.empty(n_episodes)

        for ep in range(n_episodes):
            if exploring_starts:
                s = non_terminals[np.random.randint(0, len(non_terminals))]
            else:
                s = start
            eps     = eps_arr[ep]
            total_r = 0.0
            end_s   = -1

            for _ in range(max_steps):
                if np.random.random() < eps:
                    a = np.random.randint(0, n_a)
                else:
                    best_v, best_a = -1e18, 0
                    for ai in range(n_a):
                        v = Q[s, ai] + np.random.random() * 1e-10
                        if v > best_v:
                            best_v, best_a = v, ai
                    a = best_a

                # Transition
                n_nb = T_n[s, a]
                rv   = np.random.random()
                s2   = T_s[s, a, n_nb - 1]
                for k in range(n_nb):
                    if rv <= T_cp[s, a, k]:
                        s2 = T_s[s, a, k]
                        break

                r    = R[s, a]
                done = (s2 == fake_goal) or (s2 == real_goal)
                N[s, a] += 1
                alpha = alpha_table[min(N[s, a] - 1, len(alpha_table) - 1)]

                max_q = Q[s2, 0]
                for ai in range(1, n_a):
                    if Q[s2, ai] > max_q:
                        max_q = Q[s2, ai]

                Q[s, a] += alpha * (r + gamma * max_q - Q[s, a])
                total_r += r
                s = s2
                if done:
                    end_s = s2
                    break

            rewards[ep]      = total_r
            reached_real[ep] = 1 if end_s == real_goal else 0
            if compute_errors:
                err = 0.0
                for si in range(Q.shape[0]):
                    for ai in range(n_a):
                        d = abs(Q[si, ai] - Q_star[si, ai])
                        if d > err:
                            err = d
                q_errors[ep] = err

        return rewards, reached_real, q_errors


def q_learning(mdp, n_episodes, eps_fn, alpha_fn,
               Q_star=None, max_steps=100, seed=0, exploring_starts=False):
    alpha_table   = np.array([alpha_fn(n) for n in range(1, _MAX_VISITS + 1)])
    eps_arr       = np.array([eps_fn(ep)  for ep in range(n_episodes)])
    Q             = np.zeros((mdp.n_s, mdp.n_a))
    N             = np.zeros((mdp.n_s, mdp.n_a), dtype=np.int32)
    non_terminals = np.array([s for s in range(mdp.n_s) if s not in mdp._terminals])

    if HAS_NUMBA:
        T_s, T_cp, T_n = mdp.get_trans_arrays()
        q_star_arr     = Q_star if Q_star is not None else np.zeros_like(Q)
        compute_errors = Q_star is not None
        rewards, reached_real, q_errors_arr = _ql_core(
            Q, N, mdp.gamma, mdp.start, exploring_starts, non_terminals,
            mdp.fake_goal, mdp.real_goal, n_episodes, max_steps,
            alpha_table, eps_arr, mdp.R, T_s, T_cp, T_n,
            q_star_arr, compute_errors, seed)
        q_errors = q_errors_arr if compute_errors else None
    else:
        rng          = np.random.default_rng(seed)
        rewards      = np.empty(n_episodes)
        reached_real = np.zeros(n_episodes, dtype=np.int8)
        q_errors     = np.empty(n_episodes) if Q_star is not None else None
        gamma        = mdp.gamma
        for ep in range(n_episodes):
            s       = (int(rng.choice(non_terminals)) if exploring_starts
                       else mdp.start)
            eps     = eps_arr[ep]
            total_r = 0.0
            end_s   = None
            r_eps   = rng.random(max_steps)
            r_act   = rng.integers(0, mdp.n_a, max_steps)
            r_tie   = rng.random((max_steps, mdp.n_a))
            r_trans = rng.random(max_steps)
            for i in range(max_steps):
                if r_eps[i] < eps:
                    a = int(r_act[i])
                else:
                    a = int(np.argmax(Q[s] + r_tie[i] * 1e-10))
                s2, r, done = mdp.step(s, a, r_trans[i])
                N[s, a] += 1
                alpha    = alpha_table[min(int(N[s, a]) - 1, _MAX_VISITS - 1)]
                Q[s, a] += alpha * (r + gamma * Q[s2].max() - Q[s, a])
                total_r += r
                s = s2
                if done:
                    end_s = s2
                    break
            rewards[ep]      = total_r
            reached_real[ep] = 1 if end_s == mdp.real_goal else 0
            if Q_star is not None:
                q_errors[ep] = np.max(np.abs(Q - Q_star))

    return rewards, q_errors, reached_real


# =============================================================================
# Averaging ueber Runs  (parallelisiert ueber Seeds)
# =============================================================================

def _run_one(args):
    """Top-Level Worker fuer multiprocessing (muss picklbar sein)."""
    seed, mdp, n_ep, eps_fn, alpha_fn, Q_star, max_steps, expl = args
    return q_learning(mdp, n_ep, eps_fn, alpha_fn, Q_star, max_steps, seed, expl)


def avg_runs(mdp, n_episodes, eps_fn, alpha_fn, Q_star=None,
             n_runs=20, max_steps=100, desc="", exploring_starts=False):
    print(f"  {desc:<40s} ...", end='', flush=True)
    args_list = [(seed, mdp, n_episodes, eps_fn, alpha_fn,
                  Q_star, max_steps, exploring_starts)
                 for seed in range(n_runs)]
    n_workers = min(n_runs, mp.cpu_count())
    with mp.Pool(processes=n_workers) as pool:
        results = pool.map(_run_one, args_list)
    print(" fertig")

    r_arr = np.array([res[0] for res in results])
    g_arr = np.array([res[2] for res in results])
    out = {'r_mean': r_arr.mean(0), 'r_std': r_arr.std(0),
           'g_mean': g_arr.mean(0), 'g_std': g_arr.std(0)}
    errors = [res[1] for res in results if res[1] is not None]
    if errors:
        e_arr = np.array(errors)
        out['e_mean'] = e_arr.mean(0)
        out['e_std']  = e_arr.std(0)
    return out


def rolling(x, w):
    return np.convolve(x, np.ones(w) / w, mode='valid')


# =============================================================================
# Main
# =============================================================================

def run_gamma(gamma: float):
    np.random.seed(42)

    mdp        = GridWorld(size=7, gamma=gamma, fake_goal_reward=0.65,
                           real_goal_reward=1.5, noise_prob=0.0, step_cost=-0.001)
    mdp_noisy  = GridWorld(size=7, gamma=gamma, fake_goal_reward=0.65,
                           real_goal_reward=1.5, noise_prob=0.2)
    Q_star        = compute_Q_star(mdp)
    Q_star_noisy  = compute_Q_star(mdp_noisy)

    print(f"\n=== gamma={gamma} ===")
    print(f"GridWorld 7x7 (deterministisch): start={mdp.start}, "
          f"fake_goal={mdp.fake_goal}, real_goal={mdp.real_goal}")
    print(f"GridWorld 7x7 (noise=0.2):       start={mdp_noisy.start}, "
          f"fake_goal={mdp_noisy.fake_goal}, real_goal={mdp_noisy.real_goal}")

    N_RUNS   = 20
    N_EP      = 3000
    N_EP_P1   = 8000      # fuer Plot 1: konstante alpha brauchen mehr Episoden
    N_EP_LONG = 15000     # fuer Plot 2: Plateau sichtbar machen
    W         = 150       # Rollenfenster
    EPS_X_P1 = np.arange(1, N_EP_P1 + 1)
    ROLL_X   = np.arange(W, N_EP + 1)

    fixed_eps = ConstEps(0.15)

    # ─────────────────────────────────────────────────────────────────────────
    # Alpha-Sweep (laeuft immer, auch bei ONLY_SWEEP=True)
    # ─────────────────────────────────────────────────────────────────────────
    print("  Alpha-Sweep ...")
    sweep_alphas = np.concatenate([
        np.linspace(0.01, 0.09, 5),
        np.linspace(0.1,  0.5,  9),
        np.linspace(0.55, 1.0,  6),
    ])
    sweep_means, sweep_stds = [], []
    for a_val in sweep_alphas:
        res = avg_runs(mdp_noisy, N_EP, fixed_eps, ConstAlpha(a_val),
                       Q_star_noisy, N_RUNS,
                       desc=f"  sweep alpha={a_val:.2f}",
                       exploring_starts=True)
        sweep_means.append(res['e_mean'][-1])
        sweep_stds.append(res['e_std'][-1])
    sweep_means = np.array(sweep_means)
    sweep_stds  = np.array(sweep_stds)

    fig_s, ax_s = plt.subplots(figsize=(8, 5))
    ax_s.plot(sweep_alphas, sweep_means, lw=2, color='steelblue', marker='o', ms=5)
    ax_s.fill_between(sweep_alphas,
                      np.maximum(sweep_means - sweep_stds, 1e-6),
                      sweep_means + sweep_stds, alpha=0.2, color='steelblue')
    ax_s.set_xlabel('Konstantes alpha', fontsize=11)
    ax_s.set_ylabel(f'||Q_{N_EP} - Q*||_inf', fontsize=11)
    ax_s.set_title(f'Fehler nach {N_EP} Episoden vs. alpha  |  gamma={gamma}', fontsize=11)
    ax_s.grid(True, alpha=0.3)
    fig_s.tight_layout()
    fname_s = f'sheet8/exercise_d_alpha_sweep_gamma{str(gamma).replace(".", "")}.png'
    fig_s.savefig(fname_s, dpi=150, bbox_inches='tight')
    plt.close(fig_s)
    print(f"Gespeichert: {fname_s}")

    if ONLY_SWEEP:
        return

    # ─────────────────────────────────────────────────────────────────────────
    # 2x2 Plots
    # ─────────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # Plot 1 (oben links): Konstante alpha -> ||Q_n - Q*||_inf
    ax = axes[0, 0]
    alphas    = [0.01, 0.1, 0.3, 0.7, 1.0]
    colors_a  = plt.cm.Blues(np.linspace(0.35, 0.95, len(alphas)))

    for color, a_val in zip(colors_a, alphas):
        res = avg_runs(mdp_noisy, N_EP_P1, fixed_eps, ConstAlpha(a_val),
                       Q_star_noisy, N_RUNS, desc=f"[1/4] alpha={a_val}",
                       exploring_starts=True)
        ax.semilogy(EPS_X_P1, res['e_mean'], lw=2, color=color,
                    label=f'alpha = {a_val}')
        ax.fill_between(EPS_X_P1,
                        np.maximum(res['e_mean'] - res['e_std'], 1e-6),
                        res['e_mean'] + res['e_std'], alpha=0.1, color=color)

    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel('||Q_n - Q*||_inf  (log)', fontsize=11)
    ax.set_title('Konstante alpha  (RM verletzt)', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.text(0.97, 0.97, 'RM verletzt:\nsum(alpha^2) = inf',
            transform=ax.transAxes, ha='right', va='top', fontsize=8.5,
            bbox=dict(boxstyle='round', fc='lightyellow', ec='orange', alpha=0.8))

    # ─────────────────────────────────────────────────────────────────────────
    # Plot 2 (oben rechts): Alpha-Schedules -> ||Q_n - Q*||_inf
    # Zeigt: nur abklingende alpha erfuellen RM -> echte Konvergenz
    # ─────────────────────────────────────────────────────────────────────────
    ax = axes[0, 1]
    schedules = [
        ('konstant 0.3  (RM verletzt)',              'firebrick',  ConstAlpha(0.3)),
        ('1/n^0.5  (RM Grenzfall)',                  'darkorange', PowerAlpha(0.5)),
        ('1/n^0.7  (RM erfuellt)',                   'steelblue',  PowerAlpha(0.7)),
        ('1/n      (RM erfuellt, zu schnell decay)', 'green',      PowerAlpha(1.0)),
    ]

    for name, color, alpha_fn in schedules:
        # exploring_starts=True: alle States werden besucht -> RM-Bedingung greift.
        # noise_prob=0.2: stochastische Uebergaenge -> konstantes alpha
        # oszilliert um Q* statt darunter zu fallen (Plateau sichtbar bei viel Ep.)
        res = avg_runs(mdp_noisy, N_EP_LONG, fixed_eps, alpha_fn, Q_star_noisy,
                       N_RUNS, desc=f"[2/4] {name[:22]}",
                       exploring_starts=True)
        ax.semilogy(np.arange(1, N_EP_LONG + 1), res['e_mean'],
                    lw=2, color=color, label=name)
        ax.fill_between(np.arange(1, N_EP_LONG + 1),
                        np.maximum(res['e_mean'] - res['e_std'], 1e-6),
                        res['e_mean'] + res['e_std'], alpha=0.12, color=color)

    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel('||Q_n - Q*||_inf  (log)', fontsize=11)
    ax.set_title('Alpha-Schedules  (Robbins-Monro)', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    # Faustregeln als Text
    ax.text(0.97, 0.97,
            'Faustregel:\n  1/n^p mit p in (0.5, 1]\n'
            '  p=0.7 guter Kompromiss;\n  1/n klingt zu schnell ab\n'
            '  -> Plateau in endlicher Zeit',
            transform=ax.transAxes, ha='right', va='top', fontsize=8.5,
            bbox=dict(boxstyle='round', fc='lightcyan', ec='steelblue', alpha=0.8))

    # ─────────────────────────────────────────────────────────────────────────
    # Plot 3 (unten links): Konstante epsilon -> Episode-Reward
    # Zeigt: eps=0 -> Committal; eps zu gross -> zu viel Exploration
    # ─────────────────────────────────────────────────────────────────────────
    ax = axes[1, 0]
    best_alpha = PowerAlpha(0.7)
    eps_vals   = [0.0, 0.05, 0.15, 0.3, 0.5]
    colors_e   = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(eps_vals)))

    for color, e_val in zip(colors_e, eps_vals):
        eps_fn = ConstEps(e_val)
        res    = avg_runs(mdp, N_EP, eps_fn, best_alpha, None, N_RUNS,
                          desc=f"[3/4] eps={e_val}")
        sm     = rolling(res['r_mean'], W)
        ax.plot(ROLL_X, sm, lw=2, color=color,
                label=f'eps = {e_val}')
        ax.fill_between(ROLL_X,
                        rolling(np.maximum(res['r_mean'] - res['r_std'], -2), W),
                        rolling(res['r_mean'] + res['r_std'], W),
                        alpha=0.12, color=color)

    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel(f'Gegl. Episode-Reward  (Fenster={W})', fontsize=11)
    ax.set_title('Exploration vs. Exploitation', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', lw=0.8, ls='--')
    ax.text(0.97, 0.03,
            'Faustregel: eps in [0.05, 0.2],\ndann abklingen',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8.5,
            bbox=dict(boxstyle='round', fc='lightyellow', ec='orange', alpha=0.8))

    # ─────────────────────────────────────────────────────────────────────────
    # Plot 4 (unten rechts): Committal Behavior
    # Zeigt: kleines konstantes eps -> committed fruehzeitig an Fake Goal
    #        grosses eps oder Decay -> lernt Real Goal
    # Y-Achse: rollender Anteil der Episoden die den Real Goal erreichen
    # ─────────────────────────────────────────────────────────────────────────
    ax = axes[1, 1]
    commit_cfgs = [
        ('eps=0.01  (stark committal)',   'crimson',    ConstEps(0.01)),
        ('eps=0.05  (schwach committal)', 'darkorange', ConstEps(0.05)),
        ('eps=0.2   (gut)',               'steelblue',  ConstEps(0.2)),
        ('eps decay 5/(ep+6) -> 0.05',   'green',      DecayEps(5.0, 6, 0.05)),
    ]

    for name, color, eps_fn in commit_cfgs:
        res = avg_runs(mdp, N_EP, eps_fn, best_alpha, None, N_RUNS,
                       desc=f"[4/4] {name[:22]}")
        sm  = rolling(res['g_mean'], W)
        sm_std = rolling(res['g_std'], W)
        ax.plot(ROLL_X, sm, lw=2, color=color, label=name)
        ax.fill_between(ROLL_X,
                        np.maximum(sm - sm_std, 0),
                        np.minimum(sm + sm_std, 1),
                        alpha=0.12, color=color)

    ax.axhline(1.0, color='k', ls=':', lw=1.2, alpha=0.5,
               label='Optimal: immer Real Goal')
    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel(f'Anteil Episoden -> Real Goal  (Fenster={W})', fontsize=11)
    ax.set_title('Committal Behavior', fontsize=11)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.05, 1.08])
    ax.text(0.03, 0.03,
            f'Fake Goal r={mdp.fake_r}  |  Real Goal r={mdp.real_r}',
            transform=ax.transAxes, ha='left', va='bottom', fontsize=8.5,
            bbox=dict(boxstyle='round', fc='lightyellow', ec='gray', alpha=0.8))

    plt.suptitle(
        f'Q-learning: Schrittweiten & Exploration  |  7x7 GridWorld  |  gamma={gamma}',
        fontsize=13, y=1.01)
    plt.tight_layout()
    fname = f'sheet8/exercise_d_gamma{str(gamma).replace(".", "")}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Gespeichert: {fname}")


ALL_GAMMAS   = True   # True -> [0.8, 0.9, 0.95];  False -> nur 0.9
ONLY_SWEEP   = True  # True -> nur Alpha-Sweep, kein 2x2-Plot

def main():
    gammas = [0.8, 0.9, 0.95] if ALL_GAMMAS else [0.9]
    for gamma in gammas:
        run_gamma(gamma)


if __name__ == "__main__":
    main()
