from __future__ import annotations

import math
from typing import Iterable, Tuple


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def prob_3pl(theta: float, a: float, b: float, c: float = 0.0) -> float:
    """
    3PL: p = c + (1-c)*sigmoid(a*(theta-b))
    (If c=0 -> 2PL.)
    """
    pstar = _sigmoid(a * (theta - b))
    p = c + (1.0 - c) * pstar
    return min(1.0 - 1e-9, max(1e-9, p))


def info_3pl(theta: float, a: float, b: float, c: float = 0.0) -> float:
    """
    Fisher information for 3PL:
      I = (p'²) / (p(1-p))
    where p' = d/dtheta p = (1-c)*a*pstar*(1-pstar)
    """
    pstar = _sigmoid(a * (theta - b))
    p = c + (1.0 - c) * pstar
    p = min(1.0 - 1e-9, max(1e-9, p))
    dp = (1.0 - c) * a * pstar * (1.0 - pstar)
    denom = p * (1.0 - p)
    return float((dp * dp) / max(1e-12, denom))


def online_theta_update(theta: float, u: int, a: float, b: float, c: float = 0.0, lr: float = 0.25) -> float:
    """
    One-step gradient ascent on log-likelihood with step size lr.
    """
    pstar = _sigmoid(a * (theta - b))
    p = c + (1.0 - c) * pstar
    p = min(1.0 - 1e-9, max(1e-9, p))
    dp = (1.0 - c) * a * pstar * (1.0 - pstar)
    grad = (u - p) * (dp / max(1e-12, p * (1.0 - p)))
    return theta + lr * grad


def theta_sem(theta: float, items: Iterable[Tuple[float, float, float]]) -> float:
    """
    SEM(theta) = 1 / sqrt(I(theta)), with I as sum of item information.
    items: iterable of (a,b,c)
    """
    info = 0.0
    for a, b, c in items:
        info += info_3pl(theta, a, b, c)
    if info <= 1e-12:
        return 3.0
    return float(1.0 / math.sqrt(info))


def reliability_from_sem(sem: float, prior_var: float = 1.0) -> float:
    """
    A common heuristic: rel ≈ 1 - SEM^2 / Var(theta)
    (Clamp to [0,0.99]).
    """
    rel = 1.0 - (sem * sem) / max(1e-9, prior_var)
    return float(min(0.99, max(0.0, rel)))
