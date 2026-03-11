from __future__ import annotations

from typing import Callable, Tuple

import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - np.max(x)  # numerical stability
    ex = np.exp(x)
    return ex / np.sum(ex)


def safe_argmax(x: np.ndarray) -> int:
    """Deterministic tie-break: smallest index."""
    return int(np.argmax(x))


def require_bandit(bandit: object) -> Tuple[int, Callable[[int], float]]:
    """
    Validate that the object behaves like a bandit:
      - has attribute n_arms
      - has method pull(arm) -> reward
    Returns (K, pull_function).
    """
    if not hasattr(bandit, "n_arms") or not hasattr(bandit, "pull"):
        raise TypeError("bandit must have attribute n_arms and method pull(arm)->reward")

    K = int(bandit.n_arms)
    if K <= 0:
        raise ValueError("bandit.n_arms must be positive")

    pull = bandit.pull  # type: ignore[attr-defined]
    return K, pull