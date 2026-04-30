"""
Exercise 4e: General Actor-Critic vs. direkte stochastische Kontrollalgorithmen
=================================================================================
Vergleich von:
  1. Q-learning  (off-policy,  Bellman-Optimalitaetsoperator T*)
  2. SARSA        (on-policy,   Bellman-Erwartungsoperator T^pi)
  3. Actor-Critic (AC, Abschnitt 5.4.1): tabular Softmax-Aktor + TD(0)-Kritiker

Kernunterschiede (aus dem Skript):
  - Q-learning / SARSA: lernen implizit eine Policy ueber Q-Werte
    (value-based, nutzen Bellman-Operator)
  - AC: optimiert Policy DIREKT via Policy-Gradient (Theorem 5.2.8),
    Kritiker schaetzt V^pi als Baseline -> Varianzreduktion

MDP: 4x4 GridWorld mit Fake Goal (wie in Exercise d)
  Start:     oben rechts (State 3)
  Fake Goal: oben links  (State 0,  Reward 0.65, terminal)
  Real Goal: unten links (State 12, Reward 1.00, terminal)
  gamma = 0.9

Plots (2x2):
  (0,0) Episode-Reward (rollender MW)              -- alle 3 Methoden
  (0,1) Anteil Episoden -> Real Goal (rollender MW) -- alle 3 Methoden
  (1,0) Varianz ueber Seeds (Stabilitaet)           -- alle 3 Methoden
  (1,1) Finale mittlere Performance (Balkendiagramm)

Python: 3.x | numpy, matplotlib
Hyperparameter: gamma=0.9, N_RUNS=20, N_EP=3000, W=100
  AC:    alpha_actor=0.05, alpha_critic=0.1 (konstant)
  Q/SARSA: eps=0.15, alpha=1/n^0.7 (Robbins-Monro)
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
# GridWorld mit Fake Goal (identisch zu exercise_d.py)
# =============================================================================

class GridWorld:
    """
    4x4 GridWorld.
    Start:     oben rechts  = state size-1
    Fake Goal: oben links   = state 0,          Reward fake_r  (terminal)
    Real Goal: unten links  = state (size-1)*size, Reward 1.0  (terminal)
    """
    ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def __init__(self, size=4, gamma=0.95, fake_goal_reward=0.65, real_goal_reward=1.0):
        self.size      = size
        self.gamma     = gamma
        self.n_s       = size * size
        self.n_a       = 4
        self.start     = size - 1
        self.fake_goal = 0
        self.real_goal = (size - 1) * size
        self.fake_r    = fake_goal_reward
        self.real_r    = real_goal_reward
        self._build()

    def _s(self, r, c): return r * self.size + c
    def _rc(self, s):   return divmod(s, self.size)

    def _build(self):
        self.P = np.zeros((self.n_s, self.n_a, self.n_s))
        self.R = np.zeros((self.n_s, self.n_a))
        terminals = {self.fake_goal, self.real_goal}
        for s in range(self.n_s):
            if s in terminals:
                for a in range(self.n_a):
                    self.P[s, a, s] = 1.0
                continue
            row, col = self._rc(s)
            for a, (dr, dc) in enumerate(self.ACTIONS):
                r2 = max(0, min(self.size - 1, row + dr))
                c2 = max(0, min(self.size - 1, col + dc))
                s2 = self._s(r2, c2)
                self.P[s, a, s2] = 1.0
                self.R[s, a] = (self.real_r if s2 == self.real_goal else
                                self.fake_r if s2 == self.fake_goal else 0.0)

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
        s2   = int(np.random.choice(self.n_s, p=self.P[s, a]))
        done = s2 in {self.fake_goal, self.real_goal}
        return s2, self.R[s, a], done


# =============================================================================
# Hilfsfunktionen
# =============================================================================

def _softmax(theta_s: np.ndarray) -> np.ndarray:
    """Numerisch stabile Softmax."""
    e = np.exp(theta_s - theta_s.max())
    return e / e.sum()


def _greedy(q_s: np.ndarray) -> int:
    """Zufaelliges Tie-Breaking beim Greedy-Schritt."""
    best = np.where(q_s == q_s.max())[0]
    return int(np.random.choice(best))


def rolling(x, w):
    return np.convolve(x, np.ones(w) / w, mode='valid')


# =============================================================================
# Algorithmus 1: Actor-Critic (ein-Schritt, tabular Softmax + TD(0))
# =============================================================================

def actor_critic(mdp, n_episodes, alpha_actor, alpha_critic,
                 max_steps=100, seed=0):
    """
    One-step Actor-Critic (Algorithmus 34 aus dem Skript, tabular).

    Aktor:   Softmax-Policy pi_theta(a|s) = exp(theta[s,a]) / Z_s
    Kritiker: V(s), geschaetzt mit TD(0).

    Update pro Schritt (s, a, r, s'):
      delta  = r + gamma * V(s') - V(s)          (TD-Fehler = Advantage-Schaetzer)
      V(s)   <- V(s)   + alpha_critic * delta     (Kritiker-Update)
      theta[s,:] <- theta[s,:] + alpha_actor * delta * (e_a - pi_s)   (Aktor-Update)

    Die score-Funktion ist nabla log pi_theta(a|s) = e_a - pi_theta(·|s)
    (Proposition 5.4.1, Tabular Softmax).
    """
    np.random.seed(seed)
    theta = np.zeros((mdp.n_s, mdp.n_a))   # Policy-Parameter
    V     = np.zeros(mdp.n_s)              # Wertefunktions-Schaetzer

    rewards, reached_real = [], []

    for ep in range(n_episodes):
        s       = mdp.start
        total_r = 0.0
        end_s   = None

        for _ in range(max_steps):
            pi_s = _softmax(theta[s])
            a    = np.random.choice(mdp.n_a, p=pi_s)
            s2, r, done = mdp.step(s, a)

            # TD-Fehler (Advantage-Schaetzer)
            delta = r + mdp.gamma * V[s2] - V[s]

            # Kritiker-Update
            V[s] += alpha_critic * delta

            # Aktor-Update: theta[s,:] += alpha_a * delta * (e_a - pi_s)
            e_a          = np.zeros(mdp.n_a)
            e_a[a]       = 1.0
            theta[s, :] += alpha_actor * delta * (e_a - pi_s)

            total_r += r
            s = s2
            if done:
                end_s = s2
                break

        rewards.append(total_r)
        reached_real.append(1 if end_s == mdp.real_goal else 0)

    return rewards, reached_real


# =============================================================================
# Algorithmus 2: Q-learning (off-policy, T*)
# =============================================================================

def q_learning(mdp, n_episodes, eps, alpha_fn, max_steps=100, seed=0):
    """
    Q-learning mit epsilon-greedy-Exploration und zufaelligem Tie-Breaking.
    Update: Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a)).
    Off-policy: Verhaltenspolicy (eps-greedy) != Zielpolicy (greedy).
    """
    np.random.seed(seed)
    Q = np.zeros((mdp.n_s, mdp.n_a))
    N = np.zeros((mdp.n_s, mdp.n_a))
    rewards, reached_real = [], []

    for ep in range(n_episodes):
        s       = mdp.start
        total_r = 0.0
        end_s   = None

        for _ in range(max_steps):
            a = (np.random.randint(mdp.n_a) if np.random.rand() < eps
                 else _greedy(Q[s]))
            s2, r, done = mdp.step(s, a)
            N[s, a] += 1
            alpha    = alpha_fn(N[s, a])
            Q[s, a] += alpha * (r + mdp.gamma * Q[s2].max() - Q[s, a])
            total_r += r
            s = s2
            if done:
                end_s = s2
                break

        rewards.append(total_r)
        reached_real.append(1 if end_s == mdp.real_goal else 0)

    return rewards, reached_real


# =============================================================================
# Algorithmus 3: SARSA (on-policy, T^pi)
# =============================================================================

def sarsa(mdp, n_episodes, eps, alpha_fn, max_steps=100, seed=0):
    """
    SARSA mit epsilon-greedy-Exploration.
    Update: Q(s,a) <- Q(s,a) + alpha * (r + gamma * Q(s',a') - Q(s,a)).
    On-policy: naechste Aktion a' wird ebenfalls eps-greedy gewaehlt.
    """
    np.random.seed(seed)
    Q = np.zeros((mdp.n_s, mdp.n_a))
    N = np.zeros((mdp.n_s, mdp.n_a))
    rewards, reached_real = [], []

    for ep in range(n_episodes):
        s       = mdp.start
        a       = (np.random.randint(mdp.n_a) if np.random.rand() < eps
                   else _greedy(Q[s]))
        total_r = 0.0
        end_s   = None

        for _ in range(max_steps):
            s2, r, done = mdp.step(s, a)
            N[s, a] += 1
            alpha = alpha_fn(N[s, a])

            # Naechste Aktion on-policy
            a2 = (np.random.randint(mdp.n_a) if np.random.rand() < eps
                  else _greedy(Q[s2]))

            Q[s, a] += alpha * (r + mdp.gamma * Q[s2, a2] - Q[s, a])
            total_r += r
            s, a = s2, a2
            if done:
                end_s = s2
                break

        rewards.append(total_r)
        reached_real.append(1 if end_s == mdp.real_goal else 0)

    return rewards, reached_real


# =============================================================================
# Picklbare Alpha-Callable
# =============================================================================

class PowerAlpha:
    def __init__(self, p): self.p = float(p)
    def __call__(self, n): return float(n) ** (-self.p)


# =============================================================================
# Numba-Cores
# =============================================================================

_MAX_N = 200_000

if HAS_NUMBA:
    @numba.njit(cache=True)
    def _ac_core(T_s, T_cp, T_n, R, theta, V, N_s, start, fake_goal, real_goal,
                 gamma, alpha_table_a, alpha_table_c, n_episodes, max_steps, seed,
                 non_terminals, exploring_starts):
        np.random.seed(seed)
        n_s, n_a = R.shape
        rewards      = np.empty(n_episodes)
        reached_real = np.zeros(n_episodes, dtype=np.int8)
        max_idx = len(alpha_table_a) - 1
        for ep in range(n_episodes):
            if exploring_starts:
                s = non_terminals[np.random.randint(0, len(non_terminals))]
            else:
                s = start
            total_r = 0.0; end_s = -1
            for _ in range(max_steps):
                # softmax
                row = theta[s]; mx = row[0]
                for ai in range(1, n_a):
                    if row[ai] > mx: mx = row[ai]
                sm = 0.0
                pi_s = np.empty(n_a)
                for ai in range(n_a):
                    pi_s[ai] = np.exp(row[ai] - mx); sm += pi_s[ai]
                for ai in range(n_a): pi_s[ai] /= sm
                # sample action
                rv = np.random.random(); a = n_a - 1; cum = 0.0
                for ai in range(n_a):
                    cum += pi_s[ai]
                    if rv <= cum: a = ai; break
                # transition
                nb = T_n[s, a]; rv2 = np.random.random(); s2 = T_s[s, a, nb-1]
                for k in range(nb):
                    if rv2 <= T_cp[s, a, k]: s2 = T_s[s, a, k]; break
                r = R[s, a]
                done = (s2 == fake_goal) or (s2 == real_goal)
                N_s[s] += 1
                alpha_a = alpha_table_a[min(N_s[s] - 1, max_idx)]
                alpha_c = alpha_table_c[min(N_s[s] - 1, max_idx)]
                delta = r + gamma * V[s2] - V[s]
                V[s] += alpha_c * delta
                for ai in range(n_a):
                    theta[s, ai] += alpha_a * delta * ((1.0 if ai == a else 0.0) - pi_s[ai])
                total_r += r; s = s2
                if done: end_s = s2; break
            rewards[ep] = total_r
            if end_s == real_goal: reached_real[ep] = 1
        return rewards, reached_real

    @numba.njit(cache=True)
    def _ql_core(T_s, T_cp, T_n, R, Q, N, alpha_table, start, fake_goal, real_goal,
                 gamma, eps, n_episodes, max_steps, seed):
        np.random.seed(seed)
        n_s, n_a = R.shape
        rewards      = np.empty(n_episodes)
        reached_real = np.zeros(n_episodes, dtype=np.int8)
        for ep in range(n_episodes):
            s = start; total_r = 0.0; end_s = -1
            for _ in range(max_steps):
                if np.random.random() < eps:
                    a = np.random.randint(0, n_a)
                else:
                    best_v = Q[s, 0]; best_a = 0
                    for ai in range(1, n_a):
                        v = Q[s, ai] + np.random.random() * 1e-10
                        if v > best_v: best_v = v; best_a = ai
                    a = best_a
                nb = T_n[s, a]; rv = np.random.random(); s2 = T_s[s, a, nb-1]
                for k in range(nb):
                    if rv <= T_cp[s, a, k]: s2 = T_s[s, a, k]; break
                r = R[s, a]; done = (s2 == fake_goal) or (s2 == real_goal)
                N[s, a] += 1
                alpha = alpha_table[min(N[s, a] - 1, len(alpha_table) - 1)]
                max_q = Q[s2, 0]
                for ai in range(1, n_a):
                    if Q[s2, ai] > max_q: max_q = Q[s2, ai]
                Q[s, a] += alpha * (r + gamma * max_q - Q[s, a])
                total_r += r; s = s2
                if done: end_s = s2; break
            rewards[ep] = total_r
            if end_s == real_goal: reached_real[ep] = 1
        return rewards, reached_real

    @numba.njit(cache=True)
    def _sarsa_core(T_s, T_cp, T_n, R, Q, N, alpha_table, start, fake_goal, real_goal,
                    gamma, eps, n_episodes, max_steps, seed):
        np.random.seed(seed)
        n_s, n_a = R.shape
        rewards      = np.empty(n_episodes)
        reached_real = np.zeros(n_episodes, dtype=np.int8)
        for ep in range(n_episodes):
            s = start; total_r = 0.0; end_s = -1
            if np.random.random() < eps:
                a = np.random.randint(0, n_a)
            else:
                best_v = Q[s, 0]; a = 0
                for ai in range(1, n_a):
                    v = Q[s, ai] + np.random.random() * 1e-10
                    if v > best_v: best_v = v; a = ai
            for _ in range(max_steps):
                nb = T_n[s, a]; rv = np.random.random(); s2 = T_s[s, a, nb-1]
                for k in range(nb):
                    if rv <= T_cp[s, a, k]: s2 = T_s[s, a, k]; break
                r = R[s, a]; done = (s2 == fake_goal) or (s2 == real_goal)
                N[s, a] += 1
                alpha = alpha_table[min(N[s, a] - 1, len(alpha_table) - 1)]
                if np.random.random() < eps:
                    a2 = np.random.randint(0, n_a)
                else:
                    best_v = Q[s2, 0]; a2 = 0
                    for ai in range(1, n_a):
                        v = Q[s2, ai] + np.random.random() * 1e-10
                        if v > best_v: best_v = v; a2 = ai
                Q[s, a] += alpha * (r + gamma * Q[s2, a2] - Q[s, a])
                total_r += r; s = s2; a = a2
                if done: end_s = s2; break
            rewards[ep] = total_r
            if end_s == real_goal: reached_real[ep] = 1
        return rewards, reached_real


# =============================================================================
# Top-Level Worker (picklbar fuer multiprocessing)
# =============================================================================

def _worker_ac(args):
    seed, T_s, T_cp, T_n, R, start, fake_goal, real_goal, gamma, \
        alpha_a, alpha_c, n_ep, max_steps, non_terminals = args
    n_s = R.shape[0]
    theta = np.zeros((n_s, R.shape[1])); V = np.zeros(n_s)
    N_s   = np.zeros(n_s, dtype=np.int32)
    alpha_table_a = np.full(_MAX_N, alpha_a)
    alpha_table_c = np.full(_MAX_N, alpha_c)
    if HAS_NUMBA:
        r, g = _ac_core(T_s, T_cp, T_n, R, theta, V, N_s, start, fake_goal, real_goal,
                        gamma, alpha_table_a, alpha_table_c, n_ep, max_steps, seed,
                        non_terminals, True)
    else:
        mdp = _make_stub(T_s, T_cp, T_n, R, start, fake_goal, real_goal, gamma)
        r, g = actor_critic(mdp, n_ep, alpha_a, alpha_c, max_steps, seed)
        r, g = np.array(r), np.array(g)
    return r, g

def _worker_ql(args):
    seed, T_s, T_cp, T_n, R, start, fake_goal, real_goal, gamma, \
        eps, alpha_p, n_ep, max_steps = args
    Q = np.zeros(R.shape); N = np.zeros(R.shape, dtype=np.int32)
    alpha_table = np.array([float(i)**(-alpha_p) for i in range(1, _MAX_N+1)])
    if HAS_NUMBA:
        r, g = _ql_core(T_s, T_cp, T_n, R, Q, N, alpha_table, start, fake_goal,
                        real_goal, gamma, eps, n_ep, max_steps, seed)
    else:
        mdp = _make_stub(T_s, T_cp, T_n, R, start, fake_goal, real_goal, gamma)
        r, g = q_learning(mdp, n_ep, eps, PowerAlpha(alpha_p), max_steps, seed)
        r, g = np.array(r), np.array(g)
    return r, g

def _worker_sarsa(args):
    seed, T_s, T_cp, T_n, R, start, fake_goal, real_goal, gamma, \
        eps, alpha_p, n_ep, max_steps = args
    Q = np.zeros(R.shape); N = np.zeros(R.shape, dtype=np.int32)
    alpha_table = np.array([float(i)**(-alpha_p) for i in range(1, _MAX_N+1)])
    if HAS_NUMBA:
        r, g = _sarsa_core(T_s, T_cp, T_n, R, Q, N, alpha_table, start, fake_goal,
                           real_goal, gamma, eps, n_ep, max_steps, seed)
    else:
        mdp = _make_stub(T_s, T_cp, T_n, R, start, fake_goal, real_goal, gamma)
        r, g = sarsa(mdp, n_ep, eps, PowerAlpha(alpha_p), max_steps, seed)
        r, g = np.array(r), np.array(g)
    return r, g

def _make_stub(T_s, T_cp, T_n, R, start, fake_goal, real_goal, gamma):
    """Minimaler MDP-Stub fuer Python-Fallback."""
    class _S:
        pass
    m = _S()
    m.n_s = R.shape[0]; m.n_a = R.shape[1]; m.R = R
    m.start = start; m.fake_goal = fake_goal; m.real_goal = real_goal; m.gamma = gamma
    m._T_s = T_s; m._T_cp = T_cp; m._T_n = T_n
    def step(s, a):
        nb = m._T_n[s,a]; rv = np.random.random(); s2 = int(m._T_s[s,a,nb-1])
        for k in range(nb):
            if rv <= m._T_cp[s,a,k]: s2 = int(m._T_s[s,a,k]); break
        return s2, m.R[s,a], (s2 == fake_goal or s2 == real_goal)
    m.step = step
    return m


# =============================================================================
# Averaging ueber Seeds  (parallelisiert)
# =============================================================================

def avg_runs(worker_fn, args_list, n_runs):
    n_workers = min(n_runs, mp.cpu_count())
    with mp.Pool(n_workers) as pool:
        results = pool.map(worker_fn, args_list)
    r_arr = np.array([res[0] for res in results])
    g_arr = np.array([res[1] for res in results])
    return r_arr.mean(0), r_arr.std(0), g_arr.mean(0), g_arr.std(0)


# =============================================================================
# Main
# =============================================================================

def main():
    np.random.seed(42)

    mdp    = GridWorld(size=4, gamma=0.9, fake_goal_reward=0.65, real_goal_reward=1.5)
    N_RUNS = 20
    N_EP   = 5000
    W      = 100
    ROLL_X = np.arange(W, N_EP + 1)

    print(f"GridWorld 4x4: start={mdp.start}, "
          f"fake_goal={mdp.fake_goal} (r={mdp.fake_r}), "
          f"real_goal={mdp.real_goal} (r={mdp.real_r})")

    ALPHA_A = 0.05    # AC Aktor:   konstant
    ALPHA_C = 0.10    # AC Kritiker: konstant
    EPS       = 0.15
    ALPHA_P   = 0.7   # Exponent fuer Q/SARSA

    T_s, T_cp, T_n = mdp.get_trans_arrays()
    terminals     = {mdp.fake_goal, mdp.real_goal}
    non_terminals = np.array([s for s in range(mdp.n_s) if s not in terminals],
                              dtype=np.int32)
    base = (T_s, T_cp, T_n, mdp.R, mdp.start, mdp.fake_goal, mdp.real_goal, mdp.gamma)

    print(f"Actor-Critic ({N_RUNS} Seeds, Exploring Starts) ...")
    ac_args = [(s,) + base + (ALPHA_A, ALPHA_C, N_EP, 100, non_terminals)
               for s in range(N_RUNS)]
    ac_r_m, ac_r_s, ac_g_m, ac_g_s = avg_runs(_worker_ac, ac_args, N_RUNS)

    print(f"Q-learning ({N_RUNS} Seeds) ...")
    ql_args = [(s,) + base + (EPS, ALPHA_P, N_EP, 100) for s in range(N_RUNS)]
    ql_r_m, ql_r_s, ql_g_m, ql_g_s = avg_runs(_worker_ql, ql_args, N_RUNS)

    print(f"SARSA ({N_RUNS} Seeds) ...")
    sa_args = [(s,) + base + (EPS, ALPHA_P, N_EP, 100) for s in range(N_RUNS)]
    sa_r_m, sa_r_s, sa_g_m, sa_g_s = avg_runs(_worker_sarsa, sa_args, N_RUNS)

    # ── Ergebnisse drucken ────────────────────────────────────────────────────
    last = slice(-200, None)
    print(f"\nFinale Performance (letzte 200 Episoden):")
    print(f"  AC:     Reward={ac_r_m[last].mean():.3f}  "
          f"Real Goal={ac_g_m[last].mean():.3f}")
    print(f"  Q-Lrn:  Reward={ql_r_m[last].mean():.3f}  "
          f"Real Goal={ql_g_m[last].mean():.3f}")
    print(f"  SARSA:  Reward={sa_r_m[last].mean():.3f}  "
          f"Real Goal={sa_g_m[last].mean():.3f}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    COLORS = {'AC': 'steelblue', 'Q-learning': 'darkorange', 'SARSA': 'green'}
    methods = {
        'AC':        (ac_r_m, ac_r_s, ac_g_m, ac_g_s),
        'Q-learning':(ql_r_m, ql_r_s, ql_g_m, ql_g_s),
        'SARSA':     (sa_r_m, sa_r_s, sa_g_m, sa_g_s),
    }

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # ── (0,0) Episode-Reward ──────────────────────────────────────────────────
    ax = axes[0, 0]
    for name, (r_m, r_s, g_m, g_s) in methods.items():
        c  = COLORS[name]
        sm = rolling(r_m, W)
        ax.plot(ROLL_X, sm, color=c, lw=2.5, label=name)
        ax.fill_between(ROLL_X,
                        rolling(np.maximum(r_m - r_s, -2), W),
                        rolling(r_m + r_s, W),
                        alpha=0.15, color=c)
    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel(f'Gegl. Episode-Reward  (Fenster={W})', fontsize=11)
    ax.set_title('Episode-Reward', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', lw=0.7, ls='--')

    # ── (0,1) Anteil Episoden -> Real Goal ───────────────────────────────────
    ax = axes[0, 1]
    for name, (r_m, r_s, g_m, g_s) in methods.items():
        c  = COLORS[name]
        sm = rolling(g_m, W)
        ax.plot(ROLL_X, sm, color=c, lw=2.5, label=name)
        ax.fill_between(ROLL_X,
                        np.maximum(rolling(g_m - g_s, W), 0),
                        np.minimum(rolling(g_m + g_s, W), 1),
                        alpha=0.15, color=c)
    ax.axhline(1.0, color='k', ls=':', lw=1.2, alpha=0.4,
               label='Optimal (immer Real Goal)')
    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel(f'Anteil Real Goal-Episoden  (Fenster={W})', fontsize=11)
    ax.set_title('Real Goal Rate', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.05, 1.08])

    # ── (1,0) Stabilitaet: Std des Rewards ueber Seeds ───────────────────────
    ax = axes[1, 0]
    for name, (r_m, r_s, g_m, g_s) in methods.items():
        c  = COLORS[name]
        sm = rolling(r_s, W)
        ax.plot(ROLL_X, sm, color=c, lw=2.5, label=name)
    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel(f'Std. Episode-Reward  (Fenster={W})', fontsize=11)
    ax.set_title('Stabilitaet (Std ueber Seeds)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # ── (1,1) Finale Performance (Balken + Tabelle) ───────────────────────────
    ax = axes[1, 1]
    names  = list(methods.keys())
    r_final = [methods[n][0][last].mean() for n in names]
    g_final = [methods[n][2][last].mean() for n in names]
    r_std   = [methods[n][1][last].mean() for n in names]
    g_std   = [methods[n][3][last].mean() for n in names]

    x = np.arange(len(names))
    w = 0.35
    bars1 = ax.bar(x - w/2, r_final, w, yerr=r_std, capsize=5,
                   color=[COLORS[n] for n in names], alpha=0.85, label='Reward')
    bars2 = ax.bar(x + w/2, g_final, w, yerr=g_std, capsize=5,
                   color=[COLORS[n] for n in names], alpha=0.45, label='Real Goal Rate',
                   hatch='//')

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel('Mittlerer Wert (letzte 200 Ep)', fontsize=11)
    ax.set_title('Finale Performance', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.15])

    plt.suptitle(
        f'Actor-Critic vs. Q-learning vs. SARSA  |  4x4 GridWorld  |  gamma=0.9  |  {N_RUNS} Seeds',
        fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig('sheet8/exercise_e.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Gespeichert: sheet8/exercise_e.png")


if __name__ == "__main__":
    main()
