# Sheet 8 — Tabular Reinforcement Learning

**Aufgabe 4 (Programmieraufgabe):** Vergleich verschiedener tabellarischer RL-Algorithmen aus der Vorlesung. Gewählte Teilaufgaben: **a, d, e, f**.

---

## Umgebungen

Alle Experimente verwenden Varianten der **GridWorld**:

### Standard GridWorld (Aufgabe a)
- N×N deterministisches Gitter, terminaler Zustand (Ziel) unten links, Reward +1
- Zusätzlich: **ChainMDP** (lineares MDP für isolierte Konvergenzmessungen)

### 4×4 GridWorld mit Fake Goal (Aufgaben d, e)
```
col:  0     1     2     3
row 0: FG    .     .     S
row 1: .     .     .     .
row 2: .     .     .     .
row 3: RG    .     .     .
```
- **S** = Start (oben rechts, State 3), **FG** = Fake Goal (oben links, State 0, Reward 0.65, terminal), **RG** = Real Goal (unten links, State 12, Reward 1.0, terminal), γ = 0.9

### 4×4 GridWorld für Aufgabe f (Aufgabenstellung)
```
col:  0     1     2     3
row 0: FG    .     .     S
row 1: .     .     .     .
row 2: .     .     SR    SR
row 3: .     G     SR    SR
```
- **G** = Ziel (State 13, Reward +1.0, terminal), **FG** = Fake Goal (State 0, Reward 0.65, terminal), **SR** = Stochastic Region (States 10, 11, 14, 15; Reward ∈ {−2.1, +2.0} mit gleicher Wahrscheinlichkeit), Default-Reward: {−0.05, +0.05} mit gleicher Wahrscheinlichkeit, γ = 0.9

---

## Aufgabe a — Konvergenzraten: modellbasiert vs. samplebasiert

**Datei:** `exercise_a.py` | **Plot:** `exercise_a_convergence.png`

### Algorithmen
| Algorithmus | Typ | Konvergenzrate |
|---|---|---|
| Value Iteration | modellbasiert | linear, Rate γ |
| Iterative Policy Eval. | modellbasiert | linear, Rate γ |
| First-Visit Monte Carlo | modelfrei | O(1/√n) |
| TD(0) / SARS | modelfrei | stochastisch, Bootstrapping |

**Linker Plot** (modellbasierte Methoden): Der Fehler ‖Vₙ − V*‖∞ fällt für Value Iteration und iterative Policy Evaluation sogar unter der theoretischen Schranke γⁿ. Kleineres γ bedeutet schnellere Konvergenz, da der Kontraktionsfaktor des Bellman-Operators direkt γ ist.

**Rechter Plot** (samplebasierte Methoden): Monte Carlo konvergiert nach schnellem Start deutlich langsamer (kein Bootstrapping) und zeigt größere Varianz über Seeds. TD(0) nutzt den Bellman-Erwartungsoperator als stochastische Fixpunktiteration und konvergiert durch Bootstrapping mit niedrigerer Varianz, obwohl keine Modellkenntnis vorausgesetzt wird.

---

## Aufgabe d — Schrittweiten und Explorationsparameter für Q-Learning

**Datei:** `exercise_d.py` | **Plots:** `exercise_d_gamma{0.8,0.9,0.95}.png`, `exercise_d_alpha_sweep_gamma{0.8,0.9,0.95}.png`

### Was untersucht wird

**Plot 1 (2×2, pro γ):**
- **(0,0) Konstante α-Werte** → ‖Qₙ − Q*‖∞: Verletzung der Robbins-Monro-Bedingung (ΣαN < ∞ nicht erfüllt) verhindert echte Konvergenz — der Fehler fluktuiert dauerhaft um einen von α abhängigen Wert.
- **(0,1) α-Schedules** → ‖Qₙ − Q*‖∞: αₙ = n⁻ᵖ mit p ∈ (0.5, 1] erfüllt die Robbins-Monro-Bedingungen und konvergiert.
- **(1,0) Konstante ε-Werte** → episodischer Reward: Zu großes ε erzwingt zu viel Exploration (viele suboptimale Aktionen), zu kleines ε führt zu Committal Behavior.
- **(1,1) Committal Behavior** → Anteil Episoden die Real Goal erreichen: Mit kleinem ε konvergiert der Agent früh auf das Fake Goal (nahe am Start) und verlässt es nie wieder — klassisches Committal Behavior.

**Plot 2 (Alpha-Sweep):** Heatmap des finalen Fehlers über α-Schedules und Episodenzahl, um robuste Parameterregionen zu identifizieren.

---

## Aufgabe e — Actor-Critic vs. direkte Kontrollalgorithmen

**Datei:** `exercise_e.py` | **Plot:** `exercise_e.png`

### Algorithmen
| Algorithmus | Typ | Bellman-Operator |
|---|---|---|
| Q-learning | off-policy, value-based | T* (Optimalitätsoperator) |
| SARSA | on-policy, value-based | Tᵖⁱ (Erwartungsoperator) |
| Actor-Critic (AC) | policy-based | Policy Gradient |

### Was zu sehen ist

**Plot (2×2):**
- **(0,0) Episode-Reward (rollender MW):** Q-learning und SARSA konvergieren in dieser kleinen Umgebung schnell und stabil. AC braucht mehr Episoden.
- **(0,1) Anteil Episoden → Real Goal:** Q-learning vermeidet das Fake Goal am zuverlässigsten. SARSA ist etwas konservativer. AC wieder langsamer..
- **(1,0) Varianz über Seeds:** AC zeigt höhere Varianz aufgrund der stochastischen Policy und des Monte-Carlo-Gradienten. Q-learning ist am stabilsten.
- **(1,1) Finale Performance (Balken):** Alle drei Methoden erreichen ähnliche finale Performance — der Unterschied liegt in Konvergenzgeschwindigkeit und Varianz.

**Kernaussage:** Value-basierte Methoden (Q-learning, SARSA) konvergieren in tabellarischen Umgebungen schneller und stabiler als AC, weil sie direkt den Bellman-Operator nutzen. AC ist universeller dafür mit höherer Varianz.

---

## Aufgabe f — Parameteroptimierung für Q-Learning und Double Q-Learning

**Datei:** `exercise_f.py` | **Plot:** `exercise_f.png`

### Warum Double Q-Learning?

Q-learning überschätzt systematisch Q-Werte, weil maxa Q(s', a) als Schätzer für maxa Q*(s', a) positiv verzerrt ist (Overestimation Bias). Double Q-learning trennt Aktionswahl und Bewertung auf zwei unabhängige Schätzer QA, QB, um diesen Bias zu reduzieren.

### Parameteroptimierung

Grid Search über:
- α ∈ {0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7}
- ε ∈ {0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5}

Metrik: mittlerer Episode-Reward der letzten 300 von 1500 Episoden, gemittelt über 8 Seeds.

**Getestet:** mit Rauschen (stochastische Rewards) und ohne Rauschen (Erwartungswerte).

### Was zu sehen ist

**Plot (3×2):**
- **Zeilen 1–2 (Heatmaps):** Optimale Parameter für Q-learning vs. Double Q-learning, mit/ohne Rauschen. Double Q-learning bevorzugt in der stochastischen Region etwas höhere α-Werte.
- **Zeile 3 (Lernkurven):** Mit den besten gefundenen Parametern zeigt Double Q-learning in der stochastischen Umgebung weniger Überoptimismus und stabilere Konvergenz, da die Stochastic Region hohe Varianz einbringt, die Q-learning durch Overestimation noch verstärkt.

---

## Hyperparameter & Reproduzierbarkeit

| Aufgabe | N_RUNS | N_EP | α | ε | γ |
|---|---|---|---|---|---|
| a | 30 | 4000 | n⁻⁰·⁷ (TD) | 0.1 | 0.8 / 0.9 / 0.95 |
| d | 20 | 2000 | variiert | variiert | 0.8 / 0.9 / 0.95 |
| e | 20 | 3000 | 0.05 (AC), n⁻⁰·⁷ (Q/SARSA) | 0.15 | 0.9 |
| f | 8 | 1500 | Grid Search | Grid Search | 0.9 |

**Python:** 3.x, numpy, matplotlib. Optional: numba (automatische Erkennung, Python-Fallback aktiv).

Reproduzierbar via festem `np.random.seed` pro Run-Index.
