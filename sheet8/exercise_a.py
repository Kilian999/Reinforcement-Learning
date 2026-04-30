"""
Exercise 4a: Convergence rates of iterative vs. sample-based methods
=====================================================================
Model-based (bekannte Uebergangswahrscheinlichkeiten):
  * Value Iteration     (Alg. 7,  Theorem 3.2.5: lineare Konvergenz mit Rate gamma)
  * Iterative Policy Eval (Alg. 8, Theorem 3.3.2: lineare Konvergenz mit Rate gamma)

Sample-based (model-free):
  * First-Visit Monte Carlo  (Alg. 15, ~ O(1/sqrt(n)))
  * TD(0) / SARS             (Alg. 18/19, Bootstrapping -> niedrigere Varianz)

Linker Plot:  Mehrere MDPs (GridWorld, ChainMDP) mit verschiedenen gamma-Werten,
              theoretische Rate gamma^n durch gesamten Plot.
Rechter Plot: Gerichtete Policy (bewegt sich zum Ziel) damit MC tatsaechlich konvergiert,
              gemittelt ueber N_RUNS Seeds mit Standardabweichungsband.

Python: 3.x | numpy, matplotlib
Hyperparameter: alpha_exp=0.7 (TD), N_RUNS=30 Seeds, n_episodes=4000
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
# MDP 1: Grid World
# =============================================================================

class GridWorld:
    """
    Deterministisches NxN Grid World MDP.
    Ziel (terminal, Reward +1): unten links.
    """
    ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def __init__(self, size: int = 5, gamma: float = 0.95):
        self.size  = size
        self.gamma = gamma
        self.n_s   = size * size
        self.n_a   = 4
        self.goal  = (size - 1) * size
        self._build_mdp()

    def _s(self, r, c): return r * self.size + c
    def _rc(self, s):   return divmod(s, self.size)

    def _build_mdp(self):
        n_s, n_a = self.n_s, self.n_a
        self.P = np.zeros((n_s, n_a, n_s))
        self.R = np.zeros((n_s, n_a))
        for s in range(n_s):
            if s == self.goal:
                for a in range(n_a):
                    self.P[s, a, s] = 1.0
                continue
            row, col = self._rc(s)
            for a, (dr, dc) in enumerate(self.ACTIONS):
                r2 = max(0, min(self.size - 1, row + dr))
                c2 = max(0, min(self.size - 1, col + dc))
                s2 = self._s(r2, c2)
                self.P[s, a, s2] = 1.0
                self.R[s, a]     = 1.0 if s2 == self.goal else 0.0

    def get_trans_arrays(self):
        T_s  = np.zeros((self.n_s, self.n_a, 4), dtype=np.int32)
        T_cp = np.ones ((self.n_s, self.n_a, 4))
        T_n  = np.ones ((self.n_s, self.n_a), dtype=np.int32)
        for s in range(self.n_s):
            for a in range(self.n_a):
                nz = np.where(self.P[s, a] > 0)[0]
                T_n[s, a] = len(nz)
                T_s[s, a, :len(nz)] = nz
                cp = np.cumsum(self.P[s, a, nz]); cp[-1] = 1.0
                T_cp[s, a, :len(nz)] = cp
        return T_s, T_cp, T_n

    def step(self, s, a):
        s2 = int(np.random.choice(self.n_s, p=self.P[s, a]))
        return s2, self.R[s, a], (s2 == self.goal)

    def directed_policy(self, goal_bias: float = 0.7) -> np.ndarray:
        """Policy die mit Wahrscheinlichkeit goal_bias in Richtung Ziel geht."""
        pi = np.zeros((self.n_s, self.n_a))
        g_row, g_col = self._rc(self.goal)
        for s in range(self.n_s):
            row, col = self._rc(s)
            if s == self.goal:
                pi[s] = 0.25
                continue
            best = []
            for a, (dr, dc) in enumerate(self.ACTIONS):
                r2 = max(0, min(self.size-1, row+dr))
                c2 = max(0, min(self.size-1, col+dc))
                d_old = abs(row-g_row) + abs(col-g_col)
                d_new = abs(r2-g_row)  + abs(c2-g_col)
                if d_new < d_old:
                    best.append(a)
            if not best:
                pi[s] = 0.25
            else:
                n_other = self.n_a - len(best)
                for a in range(self.n_a):
                    if a in best:
                        pi[s, a] = goal_bias / len(best)
                    else:
                        pi[s, a] = (1 - goal_bias) / n_other if n_other > 0 else 0.0
        return pi

    def uniform_policy(self): return np.full((self.n_s, self.n_a), 0.25)


# =============================================================================
# MDP 2: Chain MDP (stochastisch, klarer gamma^n-Effekt sichtbar)
# =============================================================================

class ChainMDP:
    """
    Chain MDP: Zustaende 0..N. Aktionen: 0=links, 1=rechts.
    Stochastisch: mit Prob slip geht man in falsche Richtung.
    Terminal: Zustand N (rechtes Ende), Reward +1.
    """
    def __init__(self, n_states: int = 15, gamma: float = 0.9, slip: float = 0.1):
        self.n_s   = n_states
        self.n_a   = 2
        self.gamma = gamma
        self.goal  = n_states - 1
        self.slip  = slip
        self._build_mdp()

    def _build_mdp(self):
        n_s, n_a = self.n_s, self.n_a
        self.P = np.zeros((n_s, n_a, n_s))
        self.R = np.zeros((n_s, n_a))
        for s in range(n_s):
            if s == self.goal:
                for a in range(n_a):
                    self.P[s, a, s] = 1.0
                continue
            for a in range(n_a):
                # a=1 -> rechts (Richtung Ziel), a=0 -> links
                intended = +1 if a == 1 else -1
                s_int = max(0, min(n_s-1, s + intended))
                s_slip = max(0, min(n_s-1, s - intended))
                self.P[s, a, s_int]  += (1 - self.slip)
                self.P[s, a, s_slip] += self.slip
                self.R[s, a] = (1 - self.slip) if s_int == self.goal else 0.0
                if s_slip == self.goal:
                    self.R[s, a] += self.slip

    def get_trans_arrays(self):
        T_s  = np.zeros((self.n_s, self.n_a, 4), dtype=np.int32)
        T_cp = np.ones ((self.n_s, self.n_a, 4))
        T_n  = np.ones ((self.n_s, self.n_a), dtype=np.int32)
        for s in range(self.n_s):
            for a in range(self.n_a):
                nz = np.where(self.P[s, a] > 0)[0]
                T_n[s, a] = len(nz)
                T_s[s, a, :len(nz)] = nz
                cp = np.cumsum(self.P[s, a, nz]); cp[-1] = 1.0
                T_cp[s, a, :len(nz)] = cp
        return T_s, T_cp, T_n

    def step(self, s, a):
        s2 = int(np.random.choice(self.n_s, p=self.P[s, a]))
        return s2, self.R[s, a], (s2 == self.goal)

    def directed_policy(self, goal_bias=0.8):
        pi = np.zeros((self.n_s, self.n_a))
        for s in range(self.n_s):
            pi[s, 1] = goal_bias       # rechts (Richtung Ziel)
            pi[s, 0] = 1 - goal_bias
        return pi


# =============================================================================
# Model-based Methoden
# =============================================================================

def compute_V_star(mdp, eps=1e-14) -> np.ndarray:
    V = np.zeros(mdp.n_s)
    for _ in range(20_000):
        Q     = mdp.R + mdp.gamma * np.einsum('ijk,k->ij', mdp.P, V)
        V_new = np.max(Q, axis=1)
        if np.max(np.abs(V_new - V)) < eps:
            return V_new
        V = V_new
    return V


def compute_V_pi(mdp, pi) -> np.ndarray:
    """Exakte Loesung ueber Matrixinversion: V^pi = (I - gamma P^pi)^{-1} r^pi."""
    P_pi = np.einsum('ijk,ij->ik', mdp.P, pi)
    r_pi = np.einsum('ij,ij->i',   mdp.R, pi)
    return np.linalg.solve(np.eye(mdp.n_s) - mdp.gamma * P_pi, r_pi)


def value_iteration_errors(mdp, V_star, n_iters=400):
    """
    Iteriert T* von V=0.
    Gibt (errors, cum_interactions) zurueck.
    Eine Iteration = n_s * n_a Interaktionen (voller Sweep ueber alle (s,a)).
    """
    interactions_per_iter = mdp.n_s * mdp.n_a
    V = np.zeros(mdp.n_s)
    errors, cum_inter = [], []
    total = 0
    for _ in range(n_iters):
        Q   = mdp.R + mdp.gamma * np.einsum('ijk,k->ij', mdp.P, V)
        V   = np.max(Q, axis=1)
        total += interactions_per_iter
        err   = np.max(np.abs(V - V_star))
        errors.append(err)
        cum_inter.append(total)
        if err < 1e-15:
            break
    return errors, cum_inter


def policy_eval_errors(mdp, pi, V_pi, n_iters=400):
    """
    Iteriert T^pi von V=0.
    Gibt (errors, cum_interactions) zurueck.
    Eine Iteration = n_s * n_a Interaktionen.
    """
    interactions_per_iter = mdp.n_s * mdp.n_a
    P_pi = np.einsum('ijk,ij->ik', mdp.P, pi)
    r_pi = np.einsum('ij,ij->i',   mdp.R, pi)
    V = np.zeros(mdp.n_s)
    errors, cum_inter = [], []
    total = 0
    for _ in range(n_iters):
        V   = r_pi + mdp.gamma * (P_pi @ V)
        total += interactions_per_iter
        err   = np.max(np.abs(V - V_pi))
        errors.append(err)
        cum_inter.append(total)
        if err < 1e-15:
            break
    return errors, cum_inter


# =============================================================================
# Sample-based Methoden (ein Seed)
# =============================================================================

def monte_carlo_eval_run(mdp, pi, V_ref, n_episodes=4000, max_steps=2000):
    """
    First-Visit MC, ein Seed.
    Gibt (errors, cum_interactions) zurueck.
    Jeder Environment-Step zaehlt als 1 Interaktion.
    """
    V, N   = np.zeros(mdp.n_s), np.zeros(mdp.n_s)
    errors, cum_inter = [], []
    total = 0
    for _ in range(n_episodes):
        s    = np.random.randint(mdp.n_s)
        traj = []
        for _ in range(max_steps):
            a       = np.random.choice(mdp.n_a, p=pi[s])
            s2, r, done = mdp.step(s, a)
            traj.append((s, r))
            total += 1
            s = s2
            if done:
                break
        G, visited = 0.0, set()
        for s_t, r_t in reversed(traj):
            G = r_t + mdp.gamma * G
            if s_t not in visited:
                visited.add(s_t)
                N[s_t] += 1
                V[s_t] += (G - V[s_t]) / N[s_t]
        errors.append(np.max(np.abs(V - V_ref)))
        cum_inter.append(total)
    return errors, cum_inter


def td0_eval_run(mdp, pi, V_ref, n_episodes=4000, max_steps=2000, alpha_exp=0.7):
    """
    TD(0), ein Seed. alpha(s) = 1/N(s)^alpha_exp.
    Gibt (errors, cum_interactions) zurueck.
    Jeder Environment-Step zaehlt als 1 Interaktion.
    """
    V, N   = np.zeros(mdp.n_s), np.zeros(mdp.n_s)
    errors, cum_inter = [], []
    total = 0
    for _ in range(n_episodes):
        s = np.random.randint(mdp.n_s)
        for _ in range(max_steps):
            a       = np.random.choice(mdp.n_a, p=pi[s])
            s2, r, done = mdp.step(s, a)
            N[s]  += 1
            alpha  = 1.0 / (N[s] ** alpha_exp)
            V[s]  += alpha * (r + mdp.gamma * V[s2] - V[s])
            total += 1
            s = s2
            if done:
                break
        errors.append(np.max(np.abs(V - V_ref)))
        cum_inter.append(total)
    return errors, cum_inter


# =============================================================================
# Numba-Cores fuer MC und TD (falls verfuegbar)
# =============================================================================

if HAS_NUMBA:
    @numba.njit(cache=True)
    def _mc_core(T_s, T_cp, T_n, R, pi_cdf, V_ref, V, N,
                 goal, gamma, n_episodes, max_steps, seed):
        np.random.seed(seed)
        n_s, n_a = R.shape
        errors    = np.empty(n_episodes)
        cum_inter = np.empty(n_episodes)
        traj_s    = np.empty(max_steps, dtype=np.int32)
        traj_r    = np.empty(max_steps)
        total     = 0
        for ep in range(n_episodes):
            s = np.random.randint(0, n_s)
            t = 0
            while t < max_steps:
                rv = np.random.random()
                a  = n_a - 1
                for ai in range(n_a):
                    if rv <= pi_cdf[s, ai]:
                        a = ai; break
                nb  = T_n[s, a]
                rv2 = np.random.random()
                s2  = T_s[s, a, nb - 1]
                for k in range(nb):
                    if rv2 <= T_cp[s, a, k]:
                        s2 = T_s[s, a, k]; break
                traj_s[t] = s; traj_r[t] = R[s, a]
                t += 1; total += 1; s = s2
                if s == goal: break
            G       = 0.0
            visited = np.zeros(n_s, dtype=numba.boolean)
            for k in range(t - 1, -1, -1):
                G = traj_r[k] + gamma * G
                st = traj_s[k]
                if not visited[st]:
                    visited[st] = True
                    N[st]  += 1
                    V[st]  += (G - V[st]) / N[st]
            err = 0.0
            for si in range(n_s):
                d = abs(V[si] - V_ref[si])
                if d > err: err = d
            errors[ep]    = err
            cum_inter[ep] = total
        return errors, cum_inter

    @numba.njit(cache=True)
    def _td_core(T_s, T_cp, T_n, R, pi_cdf, V_ref, V, N,
                 goal, gamma, alpha_exp, n_episodes, max_steps, seed):
        np.random.seed(seed)
        n_s, n_a = R.shape
        errors    = np.empty(n_episodes)
        cum_inter = np.empty(n_episodes)
        total     = 0
        for ep in range(n_episodes):
            s = np.random.randint(0, n_s)
            for _ in range(max_steps):
                rv = np.random.random()
                a  = n_a - 1
                for ai in range(n_a):
                    if rv <= pi_cdf[s, ai]:
                        a = ai; break
                nb  = T_n[s, a]
                rv2 = np.random.random()
                s2  = T_s[s, a, nb - 1]
                for k in range(nb):
                    if rv2 <= T_cp[s, a, k]:
                        s2 = T_s[s, a, k]; break
                N[s]  += 1
                alpha  = 1.0 / (N[s] ** alpha_exp)
                V[s]  += alpha * (R[s, a] + gamma * V[s2] - V[s])
                total += 1; s = s2
                if s == goal: break
            err = 0.0
            for si in range(n_s):
                d = abs(V[si] - V_ref[si])
                if d > err: err = d
            errors[ep]    = err
            cum_inter[ep] = total
        return errors, cum_inter


# =============================================================================
# Worker-Funktionen (Top-Level fuer multiprocessing)
# =============================================================================

def _mc_worker(args):
    seed, T_s, T_cp, T_n, R, pi_cdf, V_ref, goal, gamma, n_ep, max_steps = args
    V, N = np.zeros(R.shape[0]), np.zeros(R.shape[0])
    if HAS_NUMBA:
        return _mc_core(T_s, T_cp, T_n, R, pi_cdf, V_ref, V, N,
                        goal, gamma, n_ep, max_steps, seed)
    np.random.seed(seed)
    pi = np.diff(np.hstack([np.zeros((R.shape[0],1)), pi_cdf]), axis=1)
    class _S:
        def __init__(self): self.n_s=R.shape[0]; self.n_a=R.shape[1]; self.R=R; self.goal=goal; self.gamma=gamma; self._T_s=T_s; self._T_cp=T_cp; self._T_n=T_n
        def step(self, s, a):
            nb=self._T_n[s,a]; rv=np.random.random(); s2=int(self._T_s[s,a,nb-1])
            for k in range(nb):
                if rv<=self._T_cp[s,a,k]: s2=int(self._T_s[s,a,k]); break
            return s2, self.R[s,a], (s2==self.goal)
    mdp = _S()
    e, c = monte_carlo_eval_run(mdp, pi, V_ref, n_ep, max_steps)
    return np.array(e), np.array(c)

def _td_worker(args):
    seed, T_s, T_cp, T_n, R, pi_cdf, V_ref, goal, gamma, alpha_exp, n_ep, max_steps = args
    V, N = np.zeros(R.shape[0]), np.zeros(R.shape[0])
    if HAS_NUMBA:
        return _td_core(T_s, T_cp, T_n, R, pi_cdf, V_ref, V, N,
                        goal, gamma, alpha_exp, n_ep, max_steps, seed)
    np.random.seed(seed + 1000)
    pi = np.diff(np.hstack([np.zeros((R.shape[0],1)), pi_cdf]), axis=1)
    class _S:
        def __init__(self): self.n_s=R.shape[0]; self.n_a=R.shape[1]; self.R=R; self.goal=goal; self.gamma=gamma; self._T_s=T_s; self._T_cp=T_cp; self._T_n=T_n
        def step(self, s, a):
            nb=self._T_n[s,a]; rv=np.random.random(); s2=int(self._T_s[s,a,nb-1])
            for k in range(nb):
                if rv<=self._T_cp[s,a,k]: s2=int(self._T_s[s,a,k]); break
            return s2, self.R[s,a], (s2==self.goal)
    mdp = _S()
    e, c = td0_eval_run(mdp, pi, V_ref, n_ep, max_steps)
    return np.array(e), np.array(c)


# =============================================================================
# Averaging ueber Runs  (parallelisiert)
# =============================================================================

def run_sample_methods(mdp, pi, V_ref, n_runs=30, n_episodes=4000, max_steps=2000):
    T_s, T_cp, T_n = mdp.get_trans_arrays()
    pi_cdf = np.cumsum(pi, axis=1); pi_cdf[:, -1] = 1.0
    n_workers = min(n_runs, mp.cpu_count())

    mc_args = [(s, T_s, T_cp, T_n, mdp.R, pi_cdf, V_ref,
                mdp.goal, mdp.gamma, n_episodes, max_steps) for s in range(n_runs)]
    td_args = [(s, T_s, T_cp, T_n, mdp.R, pi_cdf, V_ref,
                mdp.goal, mdp.gamma, 0.7, n_episodes, max_steps) for s in range(n_runs)]

    with mp.Pool(n_workers) as pool:
        mc_res = pool.map(_mc_worker, mc_args)
        td_res = pool.map(_td_worker, td_args)

    mc_e = np.array([r[0] for r in mc_res])
    mc_i = np.array([r[1] for r in mc_res])
    td_e = np.array([r[0] for r in td_res])
    td_i = np.array([r[1] for r in td_res])
    return (mc_e.mean(0), mc_e.std(0), mc_i.mean(0),
            td_e.mean(0), td_e.std(0), td_i.mean(0))


# =============================================================================
# Hilfsfunktion
# =============================================================================

def smooth(x, w=30):
    return np.convolve(x, np.ones(w) / w, mode='valid')


# =============================================================================
# Main
# =============================================================================

def main():
    np.random.seed(42)
    N_RUNS    = 30
    N_EPISODES = 4000

    # ── MDPs fuer linken Plot ─────────────────────────────────────────────────
    mdps_left = [
        ("GridWorld 4x4, g=0.99",  GridWorld(size=4, gamma=0.99)),
        ("GridWorld 5x5, g=0.99",  GridWorld(size=5, gamma=0.99)),
        ("ChainMDP N=15, g=0.99",  ChainMDP(n_states=15, gamma=0.99)),
    ]

    # ── Rechter Plot: GridWorld 5x5 mit gerichteter Policy ───────────────────
    mdp_right = GridWorld(size=5, gamma=0.99)
    pi_right  = mdp_right.directed_policy(goal_bias=0.7)
    V_pi_right = compute_V_pi(mdp_right, pi_right)
    print(f"Rechter Plot: V^pi max={V_pi_right.max():.3f}, min={V_pi_right.min():.3f}")

    # ── Linker Plot vorbereiten ───────────────────────────────────────────────
    colors_vi = ['steelblue',  'royalblue',   'teal']
    colors_pe = ['darkorange', 'saddlebrown', 'firebrick']
    linestyles = ['-', '--', '-.']

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # ─────────────────────────────────────────────────────────────────────────
    # Plot 1 (links): Model-based, normierte Fehler vs. Iteration
    # ─────────────────────────────────────────────────────────────────────────
    ax = axes[0]
    max_iters = 0
    vi_inter_ref = None   # Interaktionen fuer GridWorld 5x5 (fuer Plot 3)
    pe_inter_ref = None
    vi_err_ref   = None
    pe_err_ref   = None

    for i, (label, mdp) in enumerate(mdps_left):
        print(f"Model-based: {label} ...")
        pi    = mdp.directed_policy() if hasattr(mdp, 'directed_policy') else mdp.uniform_policy()
        V_star = compute_V_star(mdp)
        V_pi   = compute_V_pi(mdp, pi)

        vi_errs, vi_inter = value_iteration_errors(mdp, V_star)
        pe_errs, pe_inter = policy_eval_errors(mdp, pi, V_pi)

        vi_norm = np.array(vi_errs) / vi_errs[0]
        pe_norm = np.array(pe_errs) / pe_errs[0]
        max_iters = max(max_iters, len(pe_errs))

        ax.semilogy(vi_norm, color=colors_vi[i], lw=1.5, ls=linestyles[i],
                    label=f'VI  {label}')
        ax.semilogy(pe_norm, color=colors_pe[i], lw=1.5, ls=linestyles[i],
                    label=f'PE  {label}')

        # Speichere GridWorld 5x5 fuer Plot 3
        if '5x5' in label:
            vi_inter_ref = vi_inter; vi_err_ref = vi_errs
            pe_inter_ref = pe_inter; pe_err_ref = pe_errs

    x_full = np.arange(max_iters + 1)
    ax.semilogy(x_full, 0.99 ** x_full, 'k', lw=2.0, ls=(0, (3, 2)),
                alpha=0.7, label='Theor. gamma=0.99: gamma^n')

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('||V_n - V*||  (normiert, log)', fontsize=11)
    ax.set_title('Modellbasierte Methoden', fontsize=12)
    ax.legend(fontsize=7, ncol=2, loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([1e-15, 2])

    # ─────────────────────────────────────────────────────────────────────────
    # Plot 2 (mitte): Sample-based, gemittelt ueber Seeds, vs. Episode
    # ─────────────────────────────────────────────────────────────────────────
    print(f"Sample-based ({N_RUNS} Seeds x {N_EPISODES} Episoden) ...")
    (mc_mean, mc_std, mc_inter_mean,
     td_mean, td_std, td_inter_mean) = run_sample_methods(
        mdp_right, pi_right, V_pi_right, n_runs=N_RUNS, n_episodes=N_EPISODES)

    ax  = axes[1]
    eps = np.arange(1, N_EPISODES + 1)

    ax.fill_between(eps, np.maximum(mc_mean - mc_std, 1e-6), mc_mean + mc_std,
                    alpha=0.2, color='darkorange')
    ax.semilogy(eps, mc_mean, color='darkorange', lw=2.5,
                label='Monte Carlo')

    ax.fill_between(eps, np.maximum(td_mean - td_std, 1e-6), td_mean + td_std,
                    alpha=0.2, color='crimson')
    ax.semilogy(eps, td_mean, color='crimson', lw=2.5,
                label='TD(0)')

    ref_at_50 = mc_mean[49] * np.sqrt(50)
    ax.semilogy(eps, ref_at_50 / np.sqrt(eps), 'k--', lw=1.5, alpha=0.7,
                label='1/sqrt(n)')

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('||V_n - V^pi||  (log)', fontsize=12)
    ax.set_title('Samplebasierte Methoden', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # ─────────────────────────────────────────────────────────────────────────
    # Plot 3 (rechts): Fairer Vergleich – alle Methoden vs. Interaktionen
    # Alle 4 Methoden auf GridWorld 5x5 (n_s*n_a = 100 pro Model-based-Iter.)
    # Fehler normiert auf Startwert -> gleiche Skala trotz verschiedener Ziele
    # ─────────────────────────────────────────────────────────────────────────
    ax = axes[2]

    # Model-based: normiert
    vi_n = np.array(vi_err_ref) / vi_err_ref[0]
    pe_n = np.array(pe_err_ref) / pe_err_ref[0]
    ax.semilogy(vi_inter_ref, vi_n, color='steelblue', lw=2.5,
                label='Value Iteration')
    ax.semilogy(pe_inter_ref, pe_n, color='darkorange', lw=2.5,
                label='Iterative Policy Eval')

    # Sample-based: normiert (MC und TD starten aehnlich)
    mc_n = mc_mean / mc_mean[0]
    td_n = td_mean / td_mean[0]
    ax.fill_between(mc_inter_mean,
                    np.maximum(mc_n - mc_std / mc_mean[0], 1e-6),
                    mc_n + mc_std / mc_mean[0], alpha=0.15, color='green')
    ax.semilogy(mc_inter_mean, mc_n, color='green', lw=2.5,
                label='Monte Carlo')

    ax.fill_between(td_inter_mean,
                    np.maximum(td_n - td_std / td_mean[0], 1e-6),
                    td_n + td_std / td_mean[0], alpha=0.15, color='crimson')
    ax.semilogy(td_inter_mean, td_n, color='crimson', lw=2.5,
                label='TD(0)')

    ax.set_xlabel('Kumulative Interaktionen', fontsize=11)
    ax.set_ylabel('||V_n - V*||  (normiert, log)', fontsize=11)
    ax.set_title('Fairer Vergleich vs. Interaktionen', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([1e-15, 2])

    # Ausgabe
    n_inter_vi = vi_inter_ref[-1]
    n_inter_mc = int(mc_inter_mean[-1])
    print(f"\nInteraktionen bis Konvergenz:")
    print(f"  VI:  {n_inter_vi:6d}  ({len(vi_err_ref)} Iterationen x {mdp_right.n_s * mdp_right.n_a})")
    print(f"  PE:  {pe_inter_ref[-1]:6d}  ({len(pe_err_ref)} Iterationen x {mdp_right.n_s * mdp_right.n_a})")
    print(f"  MC:  ~{n_inter_mc:6d}  (nach {N_EPISODES} Episoden, Faktor {n_inter_mc // max(n_inter_vi,1)}x mehr)")
    print(f"  TD:  ~{int(td_inter_mean[-1]):6d}  (nach {N_EPISODES} Episoden)")
    print(f"\nFehler Sample-based (letzte 200 Ep):")
    print(f"  MC:  {mc_mean[-200:].mean():.5f} +- {mc_std[-200:].mean():.5f}")
    print(f"  TD:  {td_mean[-200:].mean():.5f} +- {td_std[-200:].mean():.5f}")

    plt.suptitle('Konvergenzraten  |  GridWorld 5x5, gamma=0.99',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig('sheet8/exercise_a_convergence.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Gespeichert: sheet8/exercise_a_convergence.png")


if __name__ == "__main__":
    main()
