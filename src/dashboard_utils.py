from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple, Optional

import matplotlib.pyplot as plt

from src.openai_client import get_client
from src.adaptive_engine import AdaptiveSession
from src.schemas import ItemBank


# ----------------------------
# Robust helpers (no hard deps on sess.theta_path)
# ----------------------------
def _safe_get(obj: Any, name: str, default: Any) -> Any:
    return getattr(obj, name, default)


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _theta_series_from_answers(sess: AdaptiveSession) -> List[float]:
    """
    Try best-effort extraction from sess.answers, supporting multiple key names.
    Returns a series aligned to question index (1..n). If nothing found, returns [sess.theta].
    """
    answers = _safe_get(sess, "answers", []) or []
    series: List[float] = []

    # Prefer explicit stored paths
    tp = getattr(sess, "theta_path", None)
    if isinstance(tp, list) and tp:
        return [_as_float(v, 0.0) for v in tp]

    # Otherwise derive from answers
    for a in answers:
        if isinstance(a, dict):
            for k in ("theta_after", "theta", "theta_est", "theta_hat"):
                if k in a:
                    series.append(_as_float(a.get(k), 0.0))
                    break

    if series:
        return series

    # Fallback: flat line at final theta
    return [_as_float(_safe_get(sess, "theta", 0.0), 0.0)]


def _difficulty_series_from_answers(sess: AdaptiveSession) -> List[float]:
    """
    Returns numeric difficulty path.
    Uses sess.difficulty_path if present, else maps labels from answers.
    """
    dp = getattr(sess, "difficulty_path", None)
    if isinstance(dp, list) and dp:
        return [_as_float(v, 0.0) for v in dp]

    answers = _safe_get(sess, "answers", []) or []
    labels: List[str] = []
    for a in answers:
        if isinstance(a, dict):
            lab = a.get("difficulty_label") or a.get("difficulty") or a.get("diff")
            if lab is not None:
                labels.append(str(lab).strip().lower())

    mapping = {"easy": -1.0, "med": 0.0, "medium": 0.0, "hard": 1.0}
    vals: List[float] = []
    for lab in labels:
        vals.append(mapping.get(lab, 0.0))
    return vals


def _mastery_dict(sess: AdaptiveSession) -> Dict[str, float]:
    m = _safe_get(sess, "mastery", None)
    if isinstance(m, dict):
        out = {}
        for k, v in m.items():
            out[str(k)] = max(0.0, min(1.0, _as_float(v, 0.5)))
        return out
    # fallback: try other field names
    m2 = _safe_get(sess, "skill_mastery", None)
    if isinstance(m2, dict):
        out = {}
        for k, v in m2.items():
            out[str(k)] = max(0.0, min(1.0, _as_float(v, 0.5)))
        return out
    return {}


# ----------------------------
# Dashboard computations
# ----------------------------
def compute_confidence_heuristic(sess: AdaptiveSession) -> float:
    n = max(1, len(_safe_get(sess, "answers", []) or []))
    sem = _as_float(_safe_get(sess, "theta_sem", 1.0), 1.0)
    sem_term = max(0.0, min(1.0, 1.0 - (sem / 1.5)))
    n_term = max(0.0, min(1.0, n / max(10, _as_float(_safe_get(sess, "total_questions", 10), 10.0))))
    return max(0.0, min(1.0, 0.65 * sem_term + 0.35 * n_term))


def summarize_misconceptions(sess: AdaptiveSession, top_k: int = 8) -> List[Tuple[str, int]]:
    counters = _safe_get(sess, "misconception_counts", {}) or {}
    if not isinstance(counters, dict):
        return []
    items = sorted(((str(k), int(v)) for k, v in counters.items() if k), key=lambda kv: kv[1], reverse=True)
    return [(k, v) for k, v in items[:top_k] if v > 0]


def generate_ai_insights_one_call(
    sess: AdaptiveSession,
    model: str,
    top_misconceptions: List[Tuple[str, int]],
    confidence: float,
) -> Dict[str, Any]:
    client = get_client()

    answers = _safe_get(sess, "answers", []) or []
    correct_count = int(_safe_get(sess, "correct_count", 0) or 0)
    accuracy = correct_count / max(1, len(answers))

    payload = {
        "theta": round(_as_float(_safe_get(sess, "theta", 0.0), 0.0), 3),
        "sem": round(_as_float(_safe_get(sess, "theta_sem", 1.0), 1.0), 3),
        "confidence": round(float(confidence), 3),
        "answered": len(answers),
        "accuracy": round(float(accuracy), 3),
        "skill_mastery": {k: round(v, 3) for k, v in _mastery_dict(sess).items()},
        "top_misconceptions": [{"label": k, "count": v} for k, v in top_misconceptions],
        "difficulty_path": [a.get("difficulty_label") for a in answers if isinstance(a, dict)],
    }

    system = "You are an assessment diagnostician. Return ONLY JSON. No markdown."
    user = f"""
Create a short diagnostic summary for a learner.

Constraints:
- narrative: <=160 words
- recommendations: exactly 5 bullet strings
- return STRICT JSON:
{{"narrative":"...","recommendations":["...","...","...","...","..."]}}

Data:
{json.dumps(payload, ensure_ascii=False)}
""".strip()

    resp = client.responses.create(
        model=model,
        input=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_output_tokens=500,
    )
    raw = (resp.output_text or "").strip()

    try:
        obj = json.loads(raw)
        nar = str(obj.get("narrative", "")).strip()
        rec = obj.get("recommendations", [])
        if not isinstance(rec, list):
            rec = []
        rec = [str(x).strip() for x in rec if str(x).strip()]
        if len(rec) < 5:
            rec = rec + ["Practice 10 questions in the weakest skill."] * (5 - len(rec))
        rec = rec[:5]
        return {"narrative": nar, "recommendations": rec}
    except Exception:
        # fallback if JSON fails
        return {
            "narrative": raw[:900] if raw else "Performance summary unavailable.",
            "recommendations": [
                "Revise the weakest skill with worked examples.",
                "Do 10 practice questions at easy difficulty, then 10 at medium.",
                "Review mistakes and note the misconception behind each wrong choice.",
                "Use place-value/number-line visuals to build intuition.",
                "Re-attempt a short quiz after 24 hours to check retention.",
            ],
        }


# ----------------------------
# Matplotlib charts (NO seaborn)
# ----------------------------
def make_theta_fig(sess: AdaptiveSession):
    ys = _theta_series_from_answers(sess)
    xs = list(range(1, len(ys) + 1))
    fig = plt.figure()
    plt.plot(xs, ys)
    plt.xlabel("Question #")
    plt.ylabel("Theta")
    plt.title("Theta over time")
    return fig


def make_mastery_heatmap_fig(sess: AdaptiveSession):
    mastery = _mastery_dict(sess)
    skills = sorted(mastery.keys())
    vals = [mastery[s] for s in skills] if skills else []
    fig = plt.figure()
    if skills:
        plt.imshow([vals], aspect="auto")
        plt.yticks([])
        plt.xticks(range(len(skills)), skills, rotation=45, ha="right")
        plt.colorbar()
    plt.title("Skill mastery heatmap")
    return fig


def make_difficulty_path_fig(sess: AdaptiveSession):
    ys = _difficulty_series_from_answers(sess)
    xs = list(range(1, len(ys) + 1))
    fig = plt.figure()
    if ys:
        plt.plot(xs, ys, marker="o")
    plt.xlabel("Question #")
    plt.ylabel("Difficulty (mapped)")
    plt.title("Difficulty path")
    return fig


def make_bank_info_curve_fig(bank: ItemBank):
    # Approx info curve using stored item IRT params if present; otherwise neutral defaults.
    fig = plt.figure()
    thetas = [x / 10 for x in range(-30, 31)]
    infos = []
    import math

    for th in thetas:
        total = 0.0
        for it in bank.items:
            a = getattr(it, "irt_a", None) or 1.0
            b = getattr(it, "irt_b", None) or 0.0
            p = 1.0 / (1.0 + math.exp(-float(a) * (th - float(b))))
            total += (float(a) ** 2) * p * (1 - p)
        infos.append(total)

    plt.plot(thetas, infos)
    plt.xlabel("Theta")
    plt.ylabel("Information")
    plt.title("Test information curve (approx.)")
    return fig
