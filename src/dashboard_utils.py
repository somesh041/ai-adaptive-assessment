from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

from src.openai_client import get_client
from src.adaptive_engine import AdaptiveSession
from src.utils import sigmoid


def compute_confidence_heuristic(sess: AdaptiveSession) -> float:
    """
    Confidence heuristic:
      - Approximate IRT information: sum p(1-p) across administered items
      - Standard error ~ 1/sqrt(info); confidence ~ 1/(1+se), clamped
    """
    if not sess.answers:
        return 0.0

    info = 0.0
    theta = sess.theta
    for ev in sess.answers:
        p = sigmoid(theta - ev.b)
        info += p * (1.0 - p)

    if info <= 1e-9:
        return 0.1

    se = (1.0 / (info**0.5))
    conf = 1.0 / (1.0 + se)
    # also scale slightly by test length (short tests should be less confident)
    n = len(sess.answers)
    length_factor = 1.0 - pow(2.718281828, -n / 10.0)
    conf = conf * length_factor
    return float(max(0.0, min(0.98, conf)))


def make_theta_fig(sess: AdaptiveSession):
    xs = list(range(1, len(sess.answers) + 1))
    ys = [ev.theta_after for ev in sess.answers]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xs, ys, marker="o")
    ax.set_title("Theta over time")
    ax.set_xlabel("Question #")
    ax.set_ylabel("Theta")
    ax.grid(True, alpha=0.3)
    return fig


def make_mastery_heatmap_fig(sess: AdaptiveSession):
    skills = sorted(sess.mastery.keys())
    values = [sess.mastery[s] for s in skills]

    # Heatmap as 1xN image for compactness
    fig = plt.figure(figsize=(max(6, len(skills) * 0.6), 2.5))
    ax = fig.add_subplot(111)

    import numpy as np

    data = np.array(values).reshape(1, -1)
    im = ax.imshow(data, aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_yticks([])
    ax.set_xticks(range(len(skills)))
    ax.set_xticklabels(skills, rotation=35, ha="right")
    ax.set_title("Per-skill mastery (0 to 1)")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    return fig


def make_difficulty_path_fig(sess: AdaptiveSession):
    xs = list(range(1, len(sess.answers) + 1))
    ys = [ev.b for ev in sess.answers]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xs, ys, marker="o")
    ax.set_title("Difficulty path over time (b)")
    ax.set_xlabel("Question #")
    ax.set_ylabel("Difficulty (easy=-1, med=0, hard=1)")
    ax.set_yticks([-1, 0, 1])
    ax.grid(True, alpha=0.3)
    return fig


def summarize_misconceptions(sess: AdaptiveSession, top_k: int = 5) -> List[Tuple[str, int]]:
    items = sorted(sess.misconception_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return items[:top_k]


def _ai_prompt(sess: AdaptiveSession, top_misconceptions: List[Tuple[str, int]], confidence: float) -> str:
    weak_skills = sorted(sess.mastery.items(), key=lambda kv: (kv[1], kv[0]))[:5]
    strong_skills = sorted(sess.mastery.items(), key=lambda kv: (-kv[1], kv[0]))[:5]

    payload = {
        "theta": round(sess.theta, 3),
        "confidence_heuristic": round(confidence, 3),
        "answered": len(sess.answers),
        "accuracy": round(sess.correct_count / max(1, len(sess.answers)), 3),
        "weak_skills": [{"skill": s, "mastery": round(m, 3)} for s, m in weak_skills],
        "strong_skills": [{"skill": s, "mastery": round(m, 3)} for s, m in strong_skills],
        "top_misconceptions": [{"label": k, "count": v} for k, v in top_misconceptions],
        "difficulty_path": [ev.b for ev in sess.answers[-12:]],
    }

    return f"""
You are an educational diagnostician.

Using ONLY the data below, produce:
1) A narrative summary of performance, <=160 words.
2) Exactly 5 bullet recommendations (each <=18 words), focused on targeted practice and remediation.

Return STRICT JSON only in this format:
{{
  "narrative": "...",
  "recommendations": ["...", "...", "...", "...", "..."]
}}

Data:
{json.dumps(payload, ensure_ascii=False)}
""".strip()


def generate_ai_insights_one_call(
    sess: AdaptiveSession,
    model: str,
    top_misconceptions: List[Tuple[str, int]],
    confidence: float,
) -> Dict[str, Any]:
    """
    ONE LLM call for:
      - <=160 word narrative
      - 5 bullet recommendations
    """
    client = get_client()

    prompt = _ai_prompt(sess, top_misconceptions, confidence)

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": "Return ONLY JSON. No markdown. No extra keys."},
            {"role": "user", "content": prompt},
        ],
        max_output_tokens=350,
    )

    raw = (resp.output_text or "").strip()
    data = json.loads(raw)

    # Minimal validation & guardrails
    narrative = str(data.get("narrative", "")).strip()
    recs = data.get("recommendations", [])

    if not narrative:
        raise ValueError("AI insights missing narrative.")
    if len(narrative.split()) > 160:
        # Soft clamp by trimming words (no additional LLM call)
        narrative = " ".join(narrative.split()[:160])

    if not isinstance(recs, list) or len(recs) != 5:
        raise ValueError("AI insights must include exactly 5 recommendations.")

    recs2 = []
    for r in recs:
        s = str(r).strip()
        # <= 18 words soft clamp
        if len(s.split()) > 18:
            s = " ".join(s.split()[:18])
        recs2.append(s)

    return {"narrative": narrative, "recommendations": recs2}
