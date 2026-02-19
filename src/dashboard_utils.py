from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt

from src.openai_client import get_client
from src.adaptive_engine import AdaptiveSession
from src.schemas import ItemBank


# ----------------------------
# Small robust utils
# ----------------------------
def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _ans_get(a: Any, key: str, default: Any = None) -> Any:
    """
    Read field from either dict answers OR Pydantic / object answers.
    """
    if a is None:
        return default
    if isinstance(a, dict):
        return a.get(key, default)
    # Pydantic BaseModel or plain object
    if hasattr(a, key):
        return getattr(a, key)
    # last resort: mapping-like
    try:
        return a[key]  # type: ignore[index]
    except Exception:
        return default


def _answers_list(sess: AdaptiveSession) -> List[Any]:
    ans = getattr(sess, "answers", None)
    if isinstance(ans, list):
        return ans
    return []


# ----------------------------
# Series extraction for charts
# ----------------------------
def _theta_series(sess: AdaptiveSession) -> List[float]:
    """
    Prefer per-item theta updates.
    Supports:
      - sess.theta_path (list)
      - answers[*].theta_after
      - answers[*].theta
    """
    # If session has explicit theta_path, use it
    tp = getattr(sess, "theta_path", None)
    if isinstance(tp, list) and tp:
        return [_as_float(v, 0.0) for v in tp]

    ans = _answers_list(sess)
    series: List[float] = []
    for a in ans:
        v = _ans_get(a, "theta_after", None)
        if v is None:
            v = _ans_get(a, "theta", None)
        if v is None:
            v = _ans_get(a, "theta_est", None)
        if v is None:
            v = _ans_get(a, "theta_hat", None)
        if v is not None:
            series.append(_as_float(v, 0.0))

    # If we have answers but no theta fields, at least show final theta
    if not series:
        final_theta = _as_float(getattr(sess, "theta", 0.0), 0.0)
        if ans:
            return [final_theta] * max(1, len(ans))
        return [final_theta]

    return series


def _difficulty_series(sess: AdaptiveSession) -> List[float]:
    """
    Difficulty path:
      - if answers contain numeric item difficulty b -> use that
      - else map difficulty_label: easy=-1, med=0, hard=+1
    """
    dp = getattr(sess, "difficulty_path", None)
    if isinstance(dp, list) and dp:
        return [_as_float(v, 0.0) for v in dp]

    ans = _answers_list(sess)
    out: List[float] = []

    # Prefer numeric b if present (1PL/2PL style)
    any_b = any(_ans_get(a, "b", None) is not None for a in ans)
    if any_b:
        for a in ans:
            out.append(_as_float(_ans_get(a, "b", 0.0), 0.0))
        return out

    mapping = {"easy": -1.0, "med": 0.0, "medium": 0.0, "hard": 1.0}
    for a in ans:
        lab = _ans_get(a, "difficulty_label", None)
        if lab is None:
            lab = _ans_get(a, "difficulty", None)
        if lab is None:
            lab = _ans_get(a, "diff", None)
        lab_s = str(lab).strip().lower() if lab is not None else "med"
        out.append(mapping.get(lab_s, 0.0))

    return out


def _mastery_dict(sess: AdaptiveSession) -> Dict[str, float]:
    """
    Supports sess.mastery or sess.skill_mastery.
    """
    m = getattr(sess, "mastery", None)
    if isinstance(m, dict):
        return {str(k): _clamp01(_as_float(v, 0.5)) for k, v in m.items()}

    m2 = getattr(sess, "skill_mastery", None)
    if isinstance(m2, dict):
        return {str(k): _clamp01(_as_float(v, 0.5)) for k, v in m2.items()}

    return {}


# ----------------------------
# Metrics / summaries
# ----------------------------
def compute_confidence_heuristic(sess: AdaptiveSession) -> float:
    n = max(1, len(_answers_list(sess)))
    sem = _as_float(getattr(sess, "theta_sem", 1.0), 1.0)
    sem_term = max(0.0, min(1.0, 1.0 - (sem / 1.5)))
    n_term = max(0.0, min(1.0, n / max(10, _as_float(getattr(sess, "total_questions", 10), 10.0))))
    return max(0.0, min(1.0, 0.65 * sem_term + 0.35 * n_term))


def summarize_misconceptions(sess: AdaptiveSession, top_k: int = 8) -> List[Tuple[str, int]]:
    counters = getattr(sess, "misconception_counts", {}) or {}
    if not isinstance(counters, dict):
        return []
    items = sorted(((str(k), int(v)) for k, v in counters.items() if k), key=lambda kv: kv[1], reverse=True)
    return [(k, v) for k, v in items[:top_k] if v > 0]


# ----------------------------
# ONE LLM call insights
# ----------------------------
def generate_ai_insights_one_call(
    sess: AdaptiveSession,
    model: str,
    top_misconceptions: List[Tuple[str, int]],
    confidence: float,
) -> Dict[str, Any]:
    client = get_client()

    answers = _answers_list(sess)
    correct_count = int(getattr(sess, "correct_count", 0) or 0)
    accuracy = correct_count / max(1, len(answers))

    payload = {
        "theta": round(_as_float(getattr(sess, "theta", 0.0), 0.0), 3),
        "sem": round(_as_float(getattr(sess, "theta_sem", 1.0), 1.0), 3),
        "confidence": round(float(confidence), 3),
        "answered": len(answers),
        "accuracy": round(float(accuracy), 3),
        "skill_mastery": {k: round(v, 3) for k, v in _mastery_dict(sess).items()},
        "top_misconceptions": [{"label": k, "count": v} for k, v in top_misconceptions],
        "difficulty_path": [str(_ans_get(a, "difficulty_label", "")) for a in answers],
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
        return {
            "narrative": raw[:900] if raw else "Performance summary unavailable.",
            "recommendations": [
                "Revise the weakest skill with worked examples.",
                "Do 10 practice questions at easy difficulty, then 10 at medium.",
                "Review mistakes and note the misconception behind each wrong choice.",
                "Use number-line/visual fraction models to build intuition.",
                "Re-attempt a short quiz after 24 hours to check retention.",
            ],
        }


# ----------------------------
# Matplotlib charts (NO seaborn)
# ----------------------------
def make_theta_fig(sess: AdaptiveSession):
    ys = _theta_series(sess)
    xs = list(range(1, len(ys) + 1))
    fig = plt.figure()
    # marker ensures even 1-point series is visible
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Question #")
    plt.ylabel("Theta")
    plt.title("Theta over time")
    plt.grid(True, alpha=0.2)
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
    ys = _difficulty_series(sess)
    xs = list(range(1, len(ys) + 1))
    fig = plt.figure()
    if ys:
        plt.plot(xs, ys, marker="o")
    plt.xlabel("Question #")
    plt.ylabel("Item difficulty (b) or mapped label")
    plt.title("Difficulty path")
    plt.grid(True, alpha=0.2)
    return fig


def make_bank_info_curve_fig(bank: ItemBank):
    """
    Approx info curve using stored IRT params if present; otherwise neutral defaults.
    """
    fig = plt.figure()
    thetas = [x / 10 for x in range(-30, 31)]
    infos = []

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
    plt.grid(True, alpha=0.2)
    return fig
