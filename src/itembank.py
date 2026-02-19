from __future__ import annotations

import json
import os
import unicodedata
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
  - EVERY wrong option MUST have a short, non-empty misconception label (2–8 words)

Quality constraints (IMPORTANT):
- Grammar and punctuation must be correct.
- Use standard math formatting:
  - decimals use "."
  - fractions use "1/2" (avoid unicode ½)
  - minus uses "-"
- Do NOT use "None of the above" or "All of the above" or "Cannot determine".
- explanation_short must match the correct answer.

Constraints:
- IDs must be unique and stable-looking, like "{topic_slug}-001", "{topic_slug}-002", ... with numbering starting at 1.
- Exactly 4 options; only one correct.

Return JSON only.
"""

REPAIR_PROMPT_TEMPLATE = """The JSON you produced failed schema validation.

Validation errors:
{errors}

Please RETURN ONLY corrected JSON, following the same rules exactly.

Critical rule reminder:
- distractor_misconceptions must be 4 strings.
- distractor_misconceptions[correct_index] must be ""
- ALL OTHER entries must be NON-EMPTY misconception labels.

Return JSON only.
"""


def _topic_slug(s: str) -> str:
    import re

    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s[:24] or "topic"


def _parse_json_strict(text: str) -> Any:
    return json.loads((text or "").strip())


def _normalize_text(t: str) -> str:
    t = unicodedata.normalize("NFKC", t)
    t = t.replace("½", "1/2").replace("¼", "1/4").replace("¾", "3/4")
    t = t.replace("−", "-").replace("–", "-").replace("—", "-")
    t = t.replace("×", "x")
    t = " ".join(t.split())
    return t.strip()


def _clip(s: Any, n: int) -> str:
    t = "" if s is None else str(s)
    t = _normalize_text(t)
    if len(t) > n:
        t = t[:n].rstrip()
    return t


def _guess_misconception_label(skill: str, option_text: str, idx: int) -> str:
    s = (skill or "").lower()
    o = (option_text or "").strip().lower()

    if "none of the above" in o or "all of the above" in o or "cannot determine" in o:
        return "guesses using meta option"

    if "place value" in s:
        return "confuses place value"
    if "decimal" in s and "compare" in s:
        return "compares by digit count"
    if "equivalent" in s or "equivalence" in s:
        return "does not simplify fractions"
    if "fraction" in s and "decimal" in s:
        return "misapplies division"

    return f"common misconception {idx+1}"


def _sanitize_item_dict(it: Dict[str, Any]) -> Dict[str, Any]:
    it["id"] = _clip(it.get("id", ""), 80)
    it["skill"] = _clip(it.get("skill", ""), 120)

    dl = _clip(it.get("difficulty_label", "med"), 10).lower() or "med"
    it["difficulty_label"] = dl if dl in {"easy", "med", "hard"} else "med"

    bloom = _clip(it.get("bloom_level", "understand"), 40)
    # accept either numeric bloom or label; normalize numbers to labels
    bloom_map = {"1": "remember", "2": "understand", "3": "apply", "4": "analyze", "5": "evaluate", "6": "create"}
    it["bloom_level"] = bloom_map.get(bloom, bloom) or "understand"

    it["stem"] = _clip(it.get("stem", ""), 600) or "Question text unavailable."
    it["explanation_short"] = _clip(it.get("explanation_short", ""), 240) or "Explanation unavailable."

    options = it.get("options")
    if not isinstance(options, list):
        options = []
    options = [_clip(x, 180) for x in options]
    if len(options) > 4:
        options = options[:4]
    while len(options) < 4:
        options.append("N/A")
    it["options"] = options

    try:
        ci = int(it.get("correct_index", 0))
    except Exception:
        ci = 0
    ci = max(0, min(3, ci))
    it["correct_index"] = ci

    dm = it.get("distractor_misconceptions")
    if not isinstance(dm, list):
        dm = ["", "", "", ""]
    dm = ["" if x is None else str(x) for x in dm]
    if len(dm) > 4:
        dm = dm[:4]
    while len(dm) < 4:
        dm.append("")

    for j in range(4):
        if j == ci:
            dm[j] = ""
        else:
            dm[j] = _clip(dm[j], 80) if str(dm[j]).strip() else _guess_misconception_label(it["skill"], options[j], j)

    it["distractor_misconceptions"] = dm
    return it


def _sanitize_itembank_dict(parsed: Any) -> Any:
    if not isinstance(parsed, dict):
        return parsed
    items = parsed.get("items")
    if not isinstance(items, list):
        return parsed
    parsed["items"] = [(_sanitize_item_dict(x) if isinstance(x, dict) else x) for x in items]
    return parsed


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

    # Try strict schema if supported; fallback if provider rejects it.
    try:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "ItemBank", "schema": schema_obj, "strict": True},
        }
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
            {"role": "user", "content": (raw_json or "").strip()},
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
    ensure_data_dirs()

    cache_key = stable_hash(
        {"topic": topic, "grade": grade, "skills": skills, "n_items": n_items, "model": model, "schema": "Item-v3-psychometrics"}
    )
    path = os.path.join("data", f"itembank_{cache_key}.json")

    if os.path.exists(path):
        data = load_json(path)
        bank = ItemBank.model_validate(data)
        _ensure_meta(bank, topic, grade, skills, model)

        if qc_enabled and (not bank.meta or not bank.meta.qc_enabled or bank.meta.qc_version != QC_VERSION):
            bank, _ = qc_audit_and_repair_bank(bank, topic, grade, skills, model=model)
            save_json_atomic(path, bank.model_dump(mode="json"))

        if translate_langs:
            bank = add_translations_to_bank(bank, translate_langs=translate_langs, model=model)
            save_json_atomic(path, bank.model_dump(mode="json"))

        return bank, path

    last_raw = ""
    errors_str = ""
    for attempt in range(1, max_attempts + 1):
        raw = _call_itembank_llm(topic, grade, skills, n_items, model) if attempt == 1 else _call_repair_llm(last_raw, errors_str, model)
        last_raw = raw
        try:
            parsed = _parse_json_strict(raw)
            parsed = _sanitize_itembank_dict(parsed)
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
                    f"Last raw output (truncated):\n{(last_raw or '')[:2000]}"
                ) from e

    raise RuntimeError("Unexpected generation flow.")
