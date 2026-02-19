from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from src.psychometrics.irt import prob_3pl
from src.schemas import ItemBank, IRTParams
from src.storage import load_json


def _extract_response_matrix(bank: ItemBank, session_paths: List[str]) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    Build a sparse-ish matrix with missing as nan:
      persons x items -> u in {0,1} or nan
    Returns (item_ids, U, mask)
    """
    item_ids = [it.id for it in bank.items]
    item_index = {iid: j for j, iid in enumerate(item_ids)}

    rows = []
    for p in session_paths:
        d = load_json(p)
        ans = d.get("answers", [])
        row = np.full((len(item_ids),), np.nan, dtype=float)
        for ev in ans:
            iid = ev.get("item_id")
            if iid in item_index:
                row[item_index[iid]] = 1.0 if ev.get("was_correct") else 0.0
        if np.isfinite(row).sum() >= 4:
            rows.append(row)

    if not rows:
        U = np.empty((0, len(item_ids)), dtype=float)
    else:
        U = np.vstack(rows)

    mask = np.isfinite(U)
    return item_ids, U, mask


def _init_item_params(bank: ItemBank) -> Tuple[np.ndarray, np.ndarray]:
    a = np.ones((len(bank.items),), dtype=float)
    b = np.zeros((len(bank.items),), dtype=float)
    for j, it in enumerate(bank.items):
        if it.irt:
            a[j] = float(max(0.2, min(3.0, it.irt.a)))
            b[j] = float(max(-3.0, min(3.0, it.irt.b)))
        else:
            # defaults by difficulty label
            b[j] = {"easy": -1.0, "med": 0.0, "hard": 1.0}.get(it.difficulty_label, 0.0)
            a[j] = 1.0
    return a, b


def _fit_theta_for_person(u_row: np.ndarray, mask_row: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    theta = 0.0
    for _ in range(15):
        idx = np.where(mask_row)[0]
        if idx.size == 0:
            return 0.0
        p = 1.0 / (1.0 + np.exp(-(a[idx] * (theta - b[idx]))))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        # gradient & hessian for 2PL
        grad = np.sum(a[idx] * (u_row[idx] - p))
        hess = -np.sum((a[idx] ** 2) * p * (1 - p))
        if abs(hess) < 1e-8:
            break
        step = grad / hess
        theta = float(np.clip(theta - step, -4.0, 4.0))
        if abs(step) < 1e-3:
            break
    return float(theta)


def _fit_item_ab(theta: np.ndarray, u_col: np.ndarray, mask_col: np.ndarray, a0: float, b0: float) -> Tuple[float, float]:
    """
    Newton fit for logistic regression: logit(p)=a*(theta-b)=a*theta + intercept where intercept=-a*b
    We optimize (slope=a, intercept=d), then b=-d/a.
    """
    idx = np.where(mask_col)[0]
    if idx.size < 8:
        return a0, b0

    x = theta[idx]
    y = u_col[idx]

    a = float(max(0.2, min(3.0, a0)))
    d = float(-a * b0)

    for _ in range(20):
        z = a * x + d
        p = 1.0 / (1.0 + np.exp(-z))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        w = p * (1 - p)

        grad_a = np.sum((y - p) * x)
        grad_d = np.sum(y - p)

        h_aa = -np.sum(w * x * x)
        h_ad = -np.sum(w * x)
        h_dd = -np.sum(w)

        H = np.array([[h_aa, h_ad], [h_ad, h_dd]], dtype=float)
        g = np.array([grad_a, grad_d], dtype=float)

        if np.linalg.cond(H) > 1e10:
            break
        step = np.linalg.solve(H, g)
        a_new = a - step[0]
        d_new = d - step[1]

        a_new = float(np.clip(a_new, 0.2, 3.5))
        if abs(step[0]) + abs(step[1]) < 1e-3:
            a, d = a_new, d_new
            break
        a, d = a_new, d_new

    b = -d / max(1e-6, a)
    b = float(np.clip(b, -4.0, 4.0))
    return float(a), float(b)


def calibrate_2pl_jml_from_sessions(
    bank: ItemBank,
    session_paths: List[str],
    min_sessions: int = 20,
    max_iter: int = 8,
) -> Tuple[ItemBank, Dict[str, object]]:
    """
    Joint Maximum Likelihood (JML) calibration:
    - alternating updates of theta(person) and a/b(item) using observed responses
    Handles missing responses (adaptive tests).

    Requires enough sessions; otherwise returns bank unchanged.
    """
    item_ids, U, mask = _extract_response_matrix(bank, session_paths)
    n_persons = U.shape[0]
    if n_persons < min_sessions:
        return bank, {"status": "skipped", "reason": f"need >= {min_sessions} usable sessions", "usable_sessions": n_persons}

    a, b = _init_item_params(bank)
    theta = np.zeros((n_persons,), dtype=float)

    for _ in range(max_iter):
        # update theta
        for i in range(n_persons):
            theta[i] = _fit_theta_for_person(U[i], mask[i], a, b)

        # update items
        for j in range(len(item_ids)):
            a[j], b[j] = _fit_item_ab(theta, U[:, j], mask[:, j], a0=a[j], b0=b[j])

    # write back
    id_to_params = {item_ids[j]: (float(a[j]), float(b[j])) for j in range(len(item_ids))}
    updated = 0
    for it in bank.items:
        if it.id in id_to_params:
            aj, bj = id_to_params[it.id]
            it.irt = it.irt or IRTParams(a=aj, b=bj, c=0.0)
            it.irt.a = aj
            it.irt.b = bj
            updated += 1

    return bank, {"status": "ok", "usable_sessions": n_persons, "items_updated": updated}
