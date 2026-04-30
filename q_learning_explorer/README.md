# Q-Learning Interactive Explorer — Eis-Gitter-Welt

Interaktive Browser-App zum Erkunden von **Q-Learning** und **Value Iteration** in einer konfigurierbaren GridWorld. Keine Installation, keine Abhängigkeiten — einfach `q_learning_ice_grid.html` im Browser öffnen.

---

## Umgebung (MDP)

8×8-Gitter mit vier Aktionen (↑↓←→). Läuft der Agent gegen eine Wand, bleibt er stehen.

| Zelltyp | Effekt |
|---|---|
| Normal | Standard, kein besonderer Effekt |
| Eis | Agent gleitet in Bewegungsrichtung weiter bis zum nächsten Nicht-Eis-Feld |
| Wand | Nicht betretbar |
| Bombe | Negativer Reward, Episode endet sofort |
| Ziel | Positiver Reward, Episode endet sofort |
| Fake-Ziel | Konfigurierbarer Reward, Episode endet sofort |

**Standard-Rewards:** Ziel +10, Fake-Ziel +3, Bombe −10, Schritt −0.1, Eis-Schritt −0.05 — alle zur Laufzeit editierbar.

---

## Algorithmen

### Q-Learning (modell-frei, off-policy)

```
Q(s, a) ← Q(s, a) + α · [ R + γ · max_a' Q(s', a') − Q(s, a) ]
```

Bei Terminalzuständen entfällt der Bootstrapping-Term.

### Value Iteration (modell-basiert, Referenz)

```
V(s) ← max_a [ R(s,a) + γ · V(s') ]   bis  max_s |V_neu − V_alt| < 0.001
```

Liefert die exakt optimale Policy als Vergleich — kein Sampling nötig, aber setzt Modellkenntnis voraus.

---

## Explorations-Strategien

| Strategie | Beschreibung |
|---|---|
| **Random** | Gleichmäßig zufällig — reine Exploration, konvergiert nicht |
| **Greedy** | Immer `argmax Q(s,a)` — keine Exploration, bleibt bei erster Lösung |
| **ε-Greedy** | Mit Prob. ε zufällig, sonst greedy — Standardstrategie |
| **ε-Greedy Decay** | ε sinkt nach jeder Episode: `ε ← max(ε_min, ε · decay)` |
| **Softmax** | `P(a|s) ∝ exp(Q(s,a)/τ)` — Temperatur τ steuert Schärfe: τ→0 wie Greedy, τ→∞ wie Random; Exploration skaliert relativ zu Q-Wert-Unterschieden statt absolut |
| **UCB** | `argmax [Q(s,a) + c·√(ln t / n(s,a))]` — systematische Exploration |
| **Behaviour Policy** | Off-Policy: Daten mit gemischter Policy, lernt greedy Target |

---

## Parameter

| Parameter | Symbol | Bereich |
|---|---|---|
| Lernrate | α | 0.01 – 1.0 |
| Diskontfaktor | γ | 0.1 – 0.99 |
| Epsilon | ε | 0 – 1.0 |
| Epsilon Decay | — | 0.990 – 1.0 |
| Temperatur (Softmax) | τ | 0.1 – 5.0 |
| UCB-Gewicht | c | 0.1 – 5.0 |
| Schritte/Frame | — | 1 – 300 |

---

## Visualisierungen

- **Q-Max-Heatmap:** Farbkodierter Maximalwert `max_a Q(s,a)` pro Zelle
- **Q-Pfeile:** Alle vier Q-Werte als Pfeile, bester hervorgehoben
- **Besuchshäufigkeit:** Wie oft jeder Zustand besucht wurde
- **Optimale Policy:** Nur der beste Pfeil pro Zelle

