from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError

from src.openai_client import get_client


# -----------------------------
# Pydantic output schema (minimal + robust)
# -----------------------------
class ItemReviewOut(BaseModel):
    id: str
    flags: List[str] = Field(default_factory=list)
    revision_needed: bool = False
    revision_instructions: str = ""


class ReviewPayloadOut(BaseModel):
    items: List[ItemReviewOut] = Field(default_factory=list)


SYSTEM = "You are a psychometric and content QA reviewer for MCQ items. Return ONLY JSON."
REVIEW_PROMPT = """Review the following multiple-choice items for:
- clarity/grammar,
- correct answer consistency,
- distractor quality,
- skill alignment,
- age/grade appropriateness,
- ambiguity / trickiness,
- numerical/symbol formatting issues.

Return STRICT JSON only in this format:
{
  "items": [
    {
      "id": "...",
      "flags": ["...", "..."],                 // empty list if no issues
      "revision_needed": true/false,
      "revision_instructions": "..."           // empty string if no revision needed
    }
  ]
}

Rules:
- Keep flags short (2–8 words).
- revision_instructions should be concise.
- If stem is garbled or explanation mismatches correct answer, set revision_needed=true.
"""

REPAIR_PROMPT = """The JSON output was invalid or didn't match the schema.

Errors:
{errors}

Return ONLY corrected JSON using the exact required format.
Here is the invalid output:
{raw}
"""


def _extract_json(text: str) -> str:
    """
    Robustly extract the first JSON object from a string.
    Prevents failures when the model accidentally adds commentary.
    """
    s = (text or "").strip()
    if not s:
        return s

    # Fast path: already valid JSON object
    if s.startswith("{") and s.endswith("}"):
        return s

    # Try to find a JSON object block
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    return m.group(0).strip() if m else s


def _responses_create_safe(client, **kwargs):
    """
    Some OpenAI SDK versions do not support `response_format` on responses.create().
    This wrapper retries without unsupported kwargs.
    """
    try:
        return client.responses.create(**kwargs)
    except TypeError as e:
        msg = str(e)
        if "response_format" in msg and "unexpected keyword argument" in msg:
            kwargs.pop("response_format", None)
            return client.responses.create(**kwargs)
        raise


def _call_review_llm(items_payload: List[Dict[str, Any]], topic: str, grade: str, skills: List[str], model: str) -> str:
    client = get_client()
    user = (
        f"{REVIEW_PROMPT}\n\n"
        f"Context:\nTopic: {topic}\nGrade: {grade}\nSkills: {', '.join(skills)}\n\n"
        f"Items:\n{json.dumps(items_payload, ensure_ascii=False)}"
    )

    kwargs: Dict[str, Any] = {
        "model": model,
        "input": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        "max_output_tokens": 1800,
    }

    # Try to enforce schema if supported; wrapper will remove if SDK rejects.
    try:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "ReviewPayloadOut",
                "schema": ReviewPayloadOut.model_json_schema(),
                "strict": True,
            },
        }
    except Exception:
        pass

    resp = _responses_create_safe(client, **kwargs)
    return resp.output_text or ""


def _call_repair_llm(raw: str, errors: str, model: str) -> str:
    client = get_client()
    user = REPAIR_PROMPT.format(errors=errors[:2000], raw=(raw or "")[:6000])

    kwargs: Dict[str, Any] = {
        "model": model,
        "input": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        "max_output_tokens": 1800,
    }

    # Again: try schema, but fallback if unsupported
    try:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "ReviewPayloadOut",
                "schema": ReviewPayloadOut.model_json_schema(),
                "strict": True,
            },
        }
    except Exception:
        pass

    resp = _responses_create_safe(client, **kwargs)
    return resp.output_text or ""


def review_items_llm(
    items: List[Any],
    topic: str,
    grade: str,
    skills: List[str],
    model: str,
    chunk_size: int = 20,
    max_attempts: int = 2,
) -> Dict[str, ItemReviewOut]:
    """
    Returns: { item_id -> ItemReviewOut }
    Works even if SDK doesn't support response_format.
    """
    out: Dict[str, ItemReviewOut] = {}

    # Convert items to minimal payload to reduce tokens
    def to_payload(it: Any) -> Dict[str, Any]:
        return {
            "id": getattr(it, "id", ""),
            "skill": getattr(it, "skill", ""),
            "difficulty_label": getattr(it, "difficulty_label", ""),
            "bloom_level": getattr(it, "bloom_level", ""),
            "stem": getattr(it, "stem", ""),
            "options": getattr(it, "options", []),
            "correct_index": getattr(it, "correct_index", 0),
            "explanation_short": getattr(it, "explanation_short", ""),
            "distractor_misconceptions": getattr(it, "distractor_misconceptions", []),
        }

    for i in range(0, len(items), chunk_size):
        chunk = items[i : i + chunk_size]
        payload = [to_payload(it) for it in chunk]

        last_raw = ""
        err = ""
        for attempt in range(1, max_attempts + 1):
            raw = _call_review_llm(payload, topic=topic, grade=grade, skills=skills, model=model) if attempt == 1 else _call_repair_llm(last_raw, err, model=model)
            last_raw = raw

            try:
                jtxt = _extract_json(raw)
                parsed = json.loads(jtxt)
                validated = ReviewPayloadOut.model_validate(parsed)
                for r in validated.items:
                    out[r.id] = r
                break
            except (json.JSONDecodeError, ValidationError, ValueError) as e:
                err = str(e)
                if attempt == max_attempts:
                    # If review fails, default to "no flags" so QC doesn't crash.
                    for it in chunk:
                        iid = getattr(it, "id", "")
                        if iid and iid not in out:
                            out[iid] = ItemReviewOut(id=iid, flags=["review_failed"], revision_needed=False, revision_instructions="")
    return out


def apply_review_to_item(item: Any, review: ItemReviewOut, source: str = "llm") -> None:
    """
    Attaches review to item in a way that's compatible with most schema designs.
    """
    try:
        # If your schemas define ItemReview model, prefer it.
        from src.schemas import ItemReview  # type: ignore

        item.review = ItemReview(
            source=source,
            flags=review.flags,
            revision_needed=bool(review.revision_needed),
            revision_instructions=review.revision_instructions or "",
        )
    except Exception:
        # Fallback: store as dict
        item.review = {
            "source": source,
            "flags": review.flags,
            "revision_needed": bool(review.revision_needed),
            "revision_instructions": review.revision_instructions or "",
        }
