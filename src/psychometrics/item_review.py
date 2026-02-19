from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from src.openai_client import get_client
from src.schemas import IRTParams, Item, ItemReview
from src.utils import now_timestamp_id


REVIEW_SYSTEM = """You are a psychometrics-aware assessment quality reviewer.
Return ONLY valid JSON (no markdown).
You will:
- detect item-writing flaws (ambiguity, multiple correct answers, implausible distractors, cueing, off-skill)
- check construct alignment: skill + Bloom match
- estimate initial IRT-like priors (a,b,c) as rough starting values (NOT final calibration)
Be conservative: if unsure, flag the issue.
"""


class ReviewItemIn(BaseModel):
    id: str
    skill: str
    difficulty_label: str
    bloom_level: str
    stem: str
    options: List[str]
    correct_index: int
    explanation_short: str


class ReviewItemOut(BaseModel):
    id: str

    skill_pred: str
    bloom_pred: str
    skill_match: bool
    bloom_match: bool
    difficulty_label_ok: bool

    # IRT priors
    a: float = Field(..., ge=0.05, le=5.0)
    b: float = Field(..., ge=-6.0, le=6.0)
    c: float = Field(..., ge=0.0, le=0.40)

    flags: List[str] = Field(default_factory=list)
    revision_needed: bool = False
    revision_instructions: Optional[str] = None

    @field_validator("flags")
    @classmethod
    def _flags_clean(cls, v: List[str]) -> List[str]:
        return [str(x).strip()[:80] for x in (v or []) if str(x).strip()]


class ReviewPayloadOut(BaseModel):
    items: List[ReviewItemOut] = Field(..., min_length=1)

    @model_validator(mode="after")
    def _unique_ids(self) -> "ReviewPayloadOut":
        ids = [x.id for x in self.items]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate ids in review output")
        return self


def _build_prompt(topic: str, grade: str, bank_skills: List[str], items: List[ReviewItemIn]) -> str:
    payload = [it.model_dump() for it in items]
    return f"""
Context:
- Topic: {topic}
- Grade: {grade}
- Allowed skills list: {bank_skills}

Review each item.

Output STRICT JSON:
{{
  "items": [
    {{
      "id": "...",
      "skill_pred": "...",
      "bloom_pred": "...",
      "skill_match": true/false,
      "bloom_match": true/false,
      "difficulty_label_ok": true/false,
      "a": 0.8,
      "b": -0.5,
      "c": 0.20,
      "flags": ["..."],
      "revision_needed": true/false,
      "revision_instructions": "..."
    }}
  ]
}}

Guidance:
- skill_pred should be one of the allowed skills if possible; otherwise closest description.
- difficulty_label_ok: does the presented difficulty label seem consistent with the item?
- a: 0.3..2.5 typical; higher if item sharply differentiates.
- b: -2..2 typical; negative easier, positive harder.
- c: guessing lower bound; for 4-option MCQ typical ~0.25, but reduce if distractors strong.
- If revision_needed=true, provide short, actionable revision_instructions (<=240 chars).

Items:
{json.dumps(payload, ensure_ascii=False)}
""".strip()


def review_items_llm(
    items: List[Item],
    topic: str,
    grade: str,
    skills: List[str],
    model: str,
    chunk_size: int = 20,
    max_attempts: int = 3,
) -> Dict[str, ReviewItemOut]:
    """
    Returns mapping item_id -> ReviewItemOut
    """
    out: Dict[str, ReviewItemOut] = {}
    client = get_client()

    for i in range(0, len(items), chunk_size):
        chunk = items[i : i + chunk_size]
        in_items = [
            ReviewItemIn(
                id=it.id,
                skill=it.skill,
                difficulty_label=it.difficulty_label,
                bloom_level=it.bloom_level,
                stem=it.stem,
                options=it.options,
                correct_index=it.correct_index,
                explanation_short=it.explanation_short,
            )
            for it in chunk
        ]

        prompt = _build_prompt(topic, grade, skills, in_items)

        last_raw = ""
        for attempt in range(1, max_attempts + 1):
            kwargs: Dict[str, Any] = dict(
                model=model,
                input=[{"role": "system", "content": REVIEW_SYSTEM}, {"role": "user", "content": prompt}],
                max_output_tokens=2200,
            )
            try:
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "ItemReviewPayload", "schema": ReviewPayloadOut.model_json_schema(), "strict": True},
                }
            except Exception:
                pass

            if attempt == 1:
                resp = client.responses.create(**kwargs)
                raw = resp.output_text
            else:
                repair_prompt = f"""
The review JSON failed validation. Errors:
{errors_str}

Return ONLY corrected JSON in the exact same schema.
Invalid JSON:
{last_raw}
""".strip()
                resp = client.responses.create(
                    model=model,
                    input=[{"role": "system", "content": REVIEW_SYSTEM}, {"role": "user", "content": repair_prompt}],
                    max_output_tokens=2200,
                )
                raw = resp.output_text

            last_raw = raw
            try:
                data = json.loads((raw or "").strip())
                payload = ReviewPayloadOut.model_validate(data)
                for ri in payload.items:
                    out[ri.id] = ri
                break
            except (json.JSONDecodeError, ValidationError, ValueError) as e:
                errors_str = str(e)
                if attempt == max_attempts:
                    raise RuntimeError(
                        "Failed to produce valid review payload.\n"
                        f"Last error: {errors_str}\n"
                        f"Last raw (truncated): {last_raw[:1200]}"
                    ) from e

    return out


def apply_review_to_item(item: Item, r: ReviewItemOut, source: str = "llm") -> Item:
    item.irt = IRTParams(a=float(r.a), b=float(r.b), c=float(r.c))
    item.review = ItemReview(
        reviewer=source,  # type: ignore[arg-type]
        skill_pred=r.skill_pred,
        bloom_pred=r.bloom_pred,
        skill_match=bool(r.skill_match),
        bloom_match=bool(r.bloom_match),
        difficulty_label_ok=bool(r.difficulty_label_ok),
        flags=list(r.flags),
        revision_needed=bool(r.revision_needed),
        revision_instructions=(r.revision_instructions or None),
        updated_at=now_timestamp_id(),
    )
    return item
