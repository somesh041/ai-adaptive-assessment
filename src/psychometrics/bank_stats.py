from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from src.psychometrics.irt import info_3pl
from src.schemas import ItemBank
from src.utils import difficulty_to_b


def _default_irt(item) -> Tuple[float, float, float]:
    if item.irt is not None:
        return float(item.irt.a), float(item.irt.b), float(item.irt.c)
    # sensible defaults if no priors
    b = difficulty_to_b(item.difficulty_label)
    a = 1.0
    c = 0.0
    return a, b, c


def bank_information_curve(bank: ItemBank, thetas: List[float] | None = None) -> List[Tuple[float, float]]:
    if thetas is None:
        thetas = [x / 2 for x in range(-6, 7)]  # -3..3 step 0.5
    curve = []
    for th in thetas:
        info = 0.0
        for it in bank.items:
            a, b, c = _default_irt(it)
            info += info_3pl(th, a, b, c)
        curve.append((float(th), float(info)))
    return curve


def marginal_reliability_heuristic(bank: ItemBank) -> float:
    """
    Approximates E[SEM(theta)^2] under N(0,1) using Gauss-Hermite,
    then rel ≈ 1 - E[SEM^2] / Var(theta) (Var(theta)=1).
    """
    # Hermite nodes integrate f(x)*exp(-x^2) dx; adjust for N(0,1)
    nodes, weights = np.polynomial.hermite.hermgauss(21)
    # transform: theta = sqrt(2)*x for N(0,1)
    infos = []
    for x, w in zip(nodes, weights):
        theta = (2.0**0.5) * float(x)
        info = 0.0
        for it in bank.items:
            a, b, c = _default_irt(it)
            info += info_3pl(theta, a, b, c)
        infos.append((theta, float(info), float(w)))

    # E[SEM^2] = E[1/I(theta)]
    num = 0.0
    den = 0.0
    for _theta, info, w in infos:
        sem2 = 1.0 / max(1e-9, info)
        num += w * sem2
        den += w
    e_sem2 = num / max(1e-12, den)

    rel = 1.0 - e_sem2 / 1.0
    return float(min(0.99, max(0.0, rel)))


def bank_summary_metrics(bank: ItemBank) -> Dict[str, object]:
    skills = sorted({it.skill for it in bank.items})
    bloom = sorted({it.bloom_level for it in bank.items})
    diff_counts = {"easy": 0, "med": 0, "hard": 0}
    has_irt = 0
    flagged = 0

    for it in bank.items:
        diff_counts[it.difficulty_label] += 1
        if it.irt is not None:
            has_irt += 1
        if it.review and (it.review.flags or it.review.revision_needed):
            flagged += 1

    reliab = marginal_reliability_heuristic(bank)

    return {
        "n_items": len(bank.items),
        "n_skills": len(skills),
        "skills": skills[:20] + (["..."] if len(skills) > 20 else []),
        "bloom_levels": bloom,
        "difficulty_counts": diff_counts,
        "items_with_irt": has_irt,
        "items_flagged": flagged,
        "marginal_reliability_heuristic": round(reliab, 3),
    }
