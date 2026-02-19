from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

from pydantic import ValidationError

from src.openai_client import get_client
from src.schemas import ItemBank, ItemBankRoot
from src.storage import ensure_data_dirs, load_json, save_json_atomic
from src.utils import stable_hash


SYSTEM_PROMPT = """You are an expert assessment designer.
Return ONLY valid JSON. No markdown, no commentary, no code fences.
All strings must be plain text.
"""

GEN_PROMPT_TEMPLATE = """Create a multiple-choice item bank for:

Topic: {topic}
Grade: {grade}
Skills: {skills}
Number of items: {n_items}

Rules:
- Output STRICT JSON only.
- Output must be either:
  (A) an object: {{ "items": [ ... ] }}
  OR
  (B) a raw list: [ ... ]  (we will wrap it)

Each item MUST have fields:
id, skill, difficulty_label ("easy"|"med"|"hard"), bloom_level,
stem, options (array of 4 strings), correct_index (0-3), explanation_short (<=240 chars),
distractor_misconceptions (array of 4 strings) where:
  - distractor_misconceptions[correct_index] MUST be ""
  - every wrong option MUST have a non-empty misconception label (short phrase)

Additional constraints:
- IDs must be unique and stable-looking, like "{topic_slug}-{i:03d}".
- Options should be plausible and aligned to the skill.
- Exactly 4 options.
- Avoid trick questions, avoid ambiguous wording.
- Keep stems concise.
- explanation_short must be <= 240 characters (hard limit).

Return JSON only.
"""

REPAIR_PROMPT_TEMPLATE = """The JSON you produced failed schema validation.

Validation errors:
{errors}

Please RETURN ONLY corrected JSON, following the same rules exactly.
Do NOT add any new keys beyond the specified schema.
Ensure:
- unique ids
- exactly 4 options
- correct_index 0-3
- explanation_short <= 240 chars
- distractor_misconceptions is length 4 and has "" for the correct option index
- all wrong options have non-empty misconception labels

Return JSON only.
"""


def _topic_slug(s: str) -> str:
    import re

    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s[:24] or "topic"


def _parse_json_strict(text: str) -> Any:
    # tolerate leading/trailing whitespace only
    return json.loads(text.strip())


def _call_itembank_llm(topic: str, grade: str, skills: List[str], n_items: int, model: str) -> str:
    """
    Calls OpenAI Responses API and returns raw text content (JSON string expected).
    Uses response_format JSON Schema if supported; falls back to plain instruction-only JSON.
    """
    client = get_client()

    prompt = GEN_PROMPT_TEMPLATE.format(
        topic=topic,
        grade=grade,
        skills=", ".join(skills),
        n_items=n_items,
    )

    # Best effort: provide a schema to help the model comply
    schema_obj = ItemBank.model_json_schema()
    response_kwargs: Dict[str, Any] = dict(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_output_tokens=3000,
    )

    # Some SDK versions support response_format for schema-constrained JSON
    try:
        response_kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "ItemBank",
                "schema": schema_obj,
                "strict": True,
            },
        }
    except Exception:
        pass

    try:
        resp = client.responses.create(**response_kwargs)
        return resp.output_text  # SDK convenience: concatenated text outputs
    except TypeError:
        # response_format not supported in this SDK version; retry without it
        response_kwargs.pop("response_format", None)
        resp = client.responses.create(**response_kwargs)
        return resp.output_text


def _call_repair_llm(raw_json: str, errors: str, model: str) -> str:
    client = get_client()

    prompt = REPAIR_PROMPT_TEMPLATE.format(errors=errors)

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "user", "content": "Here is the invalid JSON:"},
            {"role": "user", "content": raw_json.strip()},
        ],
        max_output_tokens=3200,
    )
    return resp.output_text


def _coerce_itembank(parsed: Any) -> ItemBank:
    """
    Accept either {"items": [...]} or raw list [...]
    """
    if isinstance(parsed, dict) and "items" in parsed:
        return ItemBank.model_validate(parsed)
    if isinstance(parsed, list):
        root = ItemBankRoot.model_validate(parsed)
        return ItemBank(items=root.root)
    raise ValueError("JSON must be an object with 'items' or a raw list of items.")


def generate_or_load_item_bank(
    topic: str,
    grade: str,
    skills: List[str],
    n_items: int = 30,
    model: str = "gpt-4.1-mini",
    max_attempts: int = 3,
) -> Tuple[ItemBank, str]:
    """
    File-cache item banks based on input hash:
      data/itembank_<hash>.json
    If exists, loads and validates.
    Otherwise, generates, validates, repairs if needed, and saves.
    """
    ensure_data_dirs()

    cache_key = stable_hash(
        {
            "topic": topic,
            "grade": grade,
            "skills": skills,
            "n_items": n_items,
            "model": model,
            "schema": "Item-v1",
        }
    )
    path = os.path.join("data", f"itembank_{cache_key}.json")

    if os.path.exists(path):
        data = load_json(path)
        bank = ItemBank.model_validate(data)
        return bank, path

    # Generate with ID hints
    topic_slug = _topic_slug(topic)

    last_raw = ""
    for attempt in range(1, max_attempts + 1):
        if attempt == 1:
            raw = _call_itembank_llm(topic, grade, skills, n_items, model)
        else:
            raw = _call_repair_llm(last_raw, errors_str, model)

        last_raw = raw
        try:
            parsed = _parse_json_strict(raw)
            bank = _coerce_itembank(parsed)

            # Optional: ensure IDs look stable-ish; if not, patch (no extra LLM call)
            # This keeps repo usable even if model returns UUIDs.
            ids = [it.id for it in bank.items]
            if not all(ids) or len(set(ids)) != len(ids):
                raise ValueError("IDs missing or not unique (post-parse check).")

            save_json_atomic(path, bank.model_dump(mode="json"))
            return bank, path

        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            errors_str = str(e)

            # If last attempt, raise with context
            if attempt == max_attempts:
                raise RuntimeError(
                    "Failed to generate a valid item bank after multiple attempts.\n"
                    f"Last error: {errors_str}\n"
                    "Last raw output (truncated):\n"
                    f"{last_raw[:2000]}"
                ) from e

    # Unreachable
    raise RuntimeError("Unexpected generation flow.")
