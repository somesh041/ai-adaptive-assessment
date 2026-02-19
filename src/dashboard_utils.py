from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt

from src.openai_client import get_client
from src.schemas import ItemBank
from src.adaptive_engine import AdaptiveSession


def compute_confidence_heuristic(sess: AdaptiveSession) -> float:
    # simple bounded heuristic: lower SEM + more answers => higher confidence
    n = max(1, len(sess.answers))
    sem_term = max(0.0, min(1.0, 1.0 - (sess.theta_sem / 1.5)))
    n_term = max(0.0, min(1.0, n / max(10, sess.total_questions)))
    return max(0.0, min(1.0, 0.65 * sem_term + 0.35 * n_term))


def summarize_misconceptions(sess: AdaptiveSession, top_k: int = 8) -> List[Tuple[str, int]]:
    counters = sess.misconception_counts or {}
    items = sorted(counters.items(), key=lambda kv: kv[1], reverse=True)
    return [(k, v) for k, v in items[:top_k] if k and v > 0]


def generate_ai_insights_one_call(
    sess: AdaptiveSession,
    model: str,
    top_misconceptions: List[Tuple[str, int]],
    confidence: float,
) -> Dict[str, Any]:
    """
    ONE LLM call. Robust parsing: if JSON fails, fall back to plain text.
    Output:
      { "narrative": "...<=160 words...", "recommendations": ["...", ...] }
    """
    client = get_client()

    prompt = {
        "theta": round(sess.theta, 3),
        "sem": round(sess.theta_sem, 3),
        "confidence": round(confidence, 3),
        "answered": len(sess.answers),
        "accuracy": round(sess.correct_count / max(1, len(sess.answers)), 3),
        "skill_mastery": {k: round(v, 3) for k, v in (sess.mastery or {}).items()},
        "top_misconceptions": [{"label": k, "count": v} for k, v in top_misconceptions],
        "difficulty_path": [a.get("difficulty_label") for a in sess.answers],
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
{json.dumps(prompt, ensure_ascii=False)}
""".strip()

    resp = client.responses.create(
        model=model,
        input=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_output_tokens=500,
    )
    raw = (resp.output_text or "").strip()

    # Robust parse
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
        return {"narrative": nar[:1200], "recommendations": rec}
    except Exception:
        # fallback: treat raw as narrative, synthesize recommendations
        lines = [ln.strip("-• ").strip() for ln in raw.splitlines() if ln.strip()]
        narrative = " ".join(lines)[:900] if lines else "Performance summary unavailable."
        rec = [
            "Revise the weakest skill with worked examples.",
            "Do 10 practice questions at easy difficulty, then 10 at medium.",
            "Review mistakes and note the misconception behind each wrong choice.",
            "Use number line / place value charts to build intuition (if applicable).",
            "Re-attempt a short quiz after 24 hours to check retention.",
        ]
        return {"narrative": narrative, "recommendations": rec}


# ---- chart helpers (matplotlib only) ----

def make_theta_fig(sess: AdaptiveSession):
    xs = list(range(1, len(sess.theta_path) + 1))
    ys = sess.theta_path or [sess.theta]
    fig = plt.figure()
    plt.plot(xs, ys)
    plt.xlabel("Question #")
    plt.ylabel("Theta")
    plt.title("Theta over time")
    return fig


def make_mastery_heatmap_fig(sess: AdaptiveSession):
    skills = sorted((sess.mastery or {}).keys())
    vals = [sess.mastery[s] for s in skills] if skills else []
    fig = plt.figure()
    if skills:
        plt.imshow([vals], aspect="auto")
        plt.yticks([])
        plt.xticks(range(len(skills)), skills, rotation=45, ha="right")
        plt.colorbar()
    plt.title("Skill mastery heatmap")
    return fig


def make_difficulty_path_fig(sess: AdaptiveSession):
    xs = list(range(1, len(sess.difficulty_path) + 1))
    ys = sess.difficulty_path or []
    fig = plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Question #")
    plt.ylabel("Difficulty (mapped)")
    plt.title("Difficulty path")
    return fig


def make_bank_info_curve_fig(bank: ItemBank):
    # simple placeholder curve using stored a/b if available
    fig = plt.figure()
    thetas = [x / 10 for x in range(-30, 31)]
    infos = []
    for th in thetas:
        total = 0.0
        for it in bank.items:
            a = getattr(it, "irt_a", None) or 1.0
            b = getattr(it, "irt_b", None) or 0.0
            import math
            p = 1.0 / (1.0 + math.exp(-a * (th - b)))
            total += (a * a) * p * (1 - p)
        infos.append(total)
    plt.plot(thetas, infos)
    plt.xlabel("Theta")
    plt.ylabel("Information")
    plt.title("Test information curve (approx.)")
    return fig
