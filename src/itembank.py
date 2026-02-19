from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError

from src.openai_client import get_client
from src.psychometrics.qc_pipeline import QC_VERSION, qc_audit_and_repair_bank
from src.schemas import ItemBank, ItemBankMeta, LangCode
from src.storage import ensure_data_dirs, load_json, save_json_atomic
from src.translate import add_translations_to_bank
from src.utils import now_timestamp_id, stable_hash


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
- Output MUST be an object: {{ "items": [ ... ] }}

Each item MUST have fields:
id, skill, difficulty_label ("easy"|"med"|"hard"), bloom_level,
stem, options (array of 4 strings), correct_index (0-3), explanation_short (<=240 chars),
distractor_misconceptions (array of 4 strings) where:
  - distractor_misconceptions[correct_index] MUST be ""
  - every wrong option MUST have a non-empty misconception label (short phrase)

Constraints:
- IDs must be unique and stable-looking, like "{topic_slug}-{i:03d}" with i starting at 1.
- Exactly 4 options; only one correct.
- Avoid ambiguity, avoid trick questions.
- explanation_short must be <= 240 characters (hard limit).

Return JSON only.
"""

REPAIR_PROMPT_TEMPLATE = """The JSON you produced failed schema validation.

Validation errors:
{errors}

Please RETURN ONLY corrected JSON, following the same rules exactly.
Return JSON only.
"""


def _topic_slug(s: str) -> str:
    import re

    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s[:24] or "topic"


def _parse_json_strict(text: str) -> Any:
    return json.loads(text.strip())


def _call_itembank_llm(topic: str, grade: str, skills: List[str], n_items: int, model: str) -> str:
    client = get_client()
    topic_slug = _topic_slug(topic)

    prompt = GEN_PROMPT_TEMPLATE.format(
        topic=topic,
        grade=grade,
        skills=", ".join(skills),
        n_items=n_items,
        topic_slug=topic_slug,
    )

    schema_obj = ItemBank.model_json_schema()
    kwargs: Dict[str, Any] = dict(
        model=model,
        input=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        max_output_tokens=3200,
    )
    try:
        kwargs["response_format"] = {"type": "json_schema", "json_schema": {"name": "ItemBank", "schema": schema_obj, "strict": True}}
    except Exception:
        pass

    try:
        resp = client.responses.create(**kwargs)
        return resp.output_text
    except TypeError:
        kwargs.pop("response_format", None)
        resp = client.responses.create(**kwargs)
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
        max_output_tokens=3400,
    )
    return resp.output_text


def _ensure_meta(bank: ItemBank, topic: str, grade: str, skills: List[str], model: str) -> None:
    bank.meta = bank.meta or ItemBankMeta()
    bank.meta.topic = bank.meta.topic or topic
    bank.meta.grade = bank.meta.grade or grade
    bank.meta.skills = bank.meta.skills or skills
    bank.meta.created_at = bank.meta.created_at or now_timestamp_id()
    bank.meta.model = bank.meta.model or model


def generate_or_load_item_bank(
    topic: str,
    grade: str,
    skills: List[str],
    n_items: int = 30,
    model: str = "gpt-4.1-mini",
    max_attempts: int = 3,
    translate_langs: Optional[List[LangCode]] = None,
    qc_enabled: bool = True,
) -> Tuple[ItemBank, str]:
    """
    Cache:
      data/itembank_<hash>.json

    File can be progressively enriched:
      - QC can attach irt+review
      - translations can be added later
    """
    ensure_data_dirs()

    cache_key = stable_hash(
        {"topic": topic, "grade": grade, "skills": skills, "n_items": n_items, "model": model, "schema": "Item-v3-psychometrics"}
    )
    path = os.path.join("data", f"itembank_{cache_key}.json")

    if os.path.exists(path):
        data = load_json(path)
        bank = ItemBank.model_validate(data)
        _ensure_meta(bank, topic, grade, skills, model)

        # QC refresh if enabled and not present / version mismatch
        if qc_enabled and (not bank.meta or not bank.meta.qc_enabled or bank.meta.qc_version != QC_VERSION):
            bank, _ = qc_audit_and_repair_bank(bank, topic, grade, skills, model=model)
            save_json_atomic(path, bank.model_dump(mode="json"))

        if translate_langs:
            bank = add_translations_to_bank(bank, translate_langs=translate_langs, model=model)
            save_json_atomic(path, bank.model_dump(mode="json"))

        return bank, path

    last_raw = ""
    for attempt in range(1, max_attempts + 1):
        raw = _call_itembank_llm(topic, grade, skills, n_items, model) if attempt == 1 else _call_repair_llm(last_raw, errors_str, model)
        last_raw = raw
        try:
            parsed = _parse_json_strict(raw)
            bank = ItemBank.model_validate(parsed)
            _ensure_meta(bank, topic, grade, skills, model)
            save_json_atomic(path, bank.model_dump(mode="json"))

            if qc_enabled:
                bank, _ = qc_audit_and_repair_bank(bank, topic, grade, skills, model=model)
                _ensure_meta(bank, topic, grade, skills, model)
                save_json_atomic(path, bank.model_dump(mode="json"))

            if translate_langs:
                bank = add_translations_to_bank(bank, translate_langs=translate_langs, model=model)
                save_json_atomic(path, bank.model_dump(mode="json"))

            return bank, path
        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            errors_str = str(e)
            if attempt == max_attempts:
                raise RuntimeError(
                    "Failed to generate a valid item bank.\n"
                    f"Last error: {errors_str}\n"
                    f"Last raw output (truncated):\n{last_raw[:2000]}"
                ) from e

    raise RuntimeError("Unexpected generation flow.")
