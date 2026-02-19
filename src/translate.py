from __future__ import annotations

import json
from typing import Any, Dict, List

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from src.openai_client import get_client
from src.schemas import ItemBank, LangCode, TranslationText


LANG_LABELS: Dict[LangCode, str] = {
    "hi": "Hindi (हिन्दी)",
    "mr": "Marathi (मराठी)",
    "or": "Odia (ଓଡ଼ିଆ)",
    "te": "Telugu (తెలుగు)",
}

TRANSLATION_SYSTEM = """You are a professional educational translator for Indian school content.
Return ONLY valid JSON. No markdown. No commentary.
Preserve meaning, grade-appropriate vocabulary, and keep numbers/symbols/units unchanged.
Do NOT change option order. Do NOT add or remove options.
"""


class _ItemTranslationIn(BaseModel):
    id: str
    stem: str
    options: List[str] = Field(..., min_length=4, max_length=4)
    explanation_short: str

    @field_validator("options")
    @classmethod
    def _opt4(cls, v: List[str]) -> List[str]:
        if len(v) != 4:
            raise ValueError("options must be length 4")
        return v


class _TranslationPayloadOut(BaseModel):
    class ItemOut(BaseModel):
        id: str
        translations: Dict[LangCode, TranslationText]

        @model_validator(mode="after")
        def _nonempty(self) -> "ItemOut":
            if not self.id.strip():
                raise ValueError("id empty")
            if not self.translations:
                raise ValueError("translations empty")
            return self

    items: List[ItemOut] = Field(..., min_length=1)


def _build_translation_prompt(items: List[_ItemTranslationIn], langs: List[LangCode]) -> str:
    lang_names = ", ".join([f"{c}={LANG_LABELS[c]}" for c in langs])
    payload = [it.model_dump() for it in items]
    return f"""
Translate the following assessment items into these languages: {lang_names}.

Rules:
- Return STRICT JSON only.
- Output format:
{{
  "items": [
    {{
      "id": "<same id>",
      "translations": {{
        "<lang_code>": {{
          "stem": "...",
          "options": ["...","...","...","..."],
          "explanation_short": "..."
        }}
      }}
    }}
  ]
}}

Constraints:
- Preserve numbers, math symbols, units, proper nouns.
- Keep meaning and difficulty stable.
- Keep option order unchanged; do not add/remove options.
- explanation_short <=240 chars per language.
- Output ONLY requested languages.

Items:
{json.dumps(payload, ensure_ascii=False)}
""".strip()


def _call_translate_llm(items: List[_ItemTranslationIn], langs: List[LangCode], model: str) -> str:
    client = get_client()
    prompt = _build_translation_prompt(items, langs)

    kwargs: Dict[str, Any] = dict(
        model=model,
        input=[{"role": "system", "content": TRANSLATION_SYSTEM}, {"role": "user", "content": prompt}],
        max_output_tokens=3500,
    )

    try:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "TranslationPayload", "schema": _TranslationPayloadOut.model_json_schema(), "strict": True},
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
    prompt = f"""
The JSON failed validation.

Errors:
{errors}

Return ONLY corrected JSON in the exact output format.
Here is the invalid JSON:
{raw_json}
""".strip()

    resp = client.responses.create(
        model=model,
        input=[{"role": "system", "content": TRANSLATION_SYSTEM}, {"role": "user", "content": prompt}],
        max_output_tokens=3500,
    )
    return resp.output_text


def _parse_json(text: str) -> Any:
    return json.loads((text or "").strip())


def _chunk(lst: List[Any], n: int) -> List[List[Any]]:
    return [lst[i : i + n] for i in range(0, len(lst), n)]


def _sanitize_translation_payload(raw: Any, originals: Dict[str, _ItemTranslationIn], langs: List[LangCode]) -> Any:
    """
    Prevent translation crashes by filling missing/empty values with original English text.
    This is cheap and avoids wasting credits on retries.
    """
    if not isinstance(raw, dict) or not isinstance(raw.get("items"), list):
        return raw

    for it in raw["items"]:
        if not isinstance(it, dict):
            continue
        iid = str(it.get("id", "")).strip()
        if iid not in originals:
            continue
        orig = originals[iid]

        tr = it.get("translations")
        if not isinstance(tr, dict):
            tr = {}
        # keep only requested languages
        tr2 = {}
        for lc in langs:
            v = tr.get(lc)
            if not isinstance(v, dict):
                v = {}

            stem = str(v.get("stem", "")).strip() or orig.stem
            opts = v.get("options")
            if not isinstance(opts, list):
                opts = []
            opts = [str(x).strip() for x in opts]
            if len(opts) != 4 or any(not x for x in opts):
                opts = orig.options

            expl = str(v.get("explanation_short", "")).strip() or orig.explanation_short
            if len(expl) > 240:
                expl = expl[:240].rstrip()

            tr2[lc] = {"stem": stem, "options": opts, "explanation_short": expl}

        it["translations"] = tr2

    return raw


def add_translations_to_bank(
    bank: ItemBank,
    translate_langs: List[LangCode],
    model: str = "gpt-4.1-mini",
    chunk_size: int = 25,
    max_attempts: int = 3,
) -> ItemBank:
    if not translate_langs:
        return bank

    needed_items = []
    for it in bank.items:
        existing = it.translations or {}
        if any(lc not in existing for lc in translate_langs):
            needed_items.append(it)

    if not needed_items:
        return bank

    for chunk_items in _chunk(needed_items, chunk_size):
        inputs = [
            _ItemTranslationIn(id=it.id, stem=it.stem, options=it.options, explanation_short=it.explanation_short)
            for it in chunk_items
        ]
        originals = {x.id: x for x in inputs}

        last_raw = ""
        errors_str = ""
        for attempt in range(1, max_attempts + 1):
            raw_text = _call_translate_llm(inputs, translate_langs, model=model) if attempt == 1 else _call_repair_llm(last_raw, errors_str, model=model)
            last_raw = raw_text
            try:
                parsed = _parse_json(raw_text)
                parsed = _sanitize_translation_payload(parsed, originals=originals, langs=translate_langs)
                out = _TranslationPayloadOut.model_validate(parsed)

                out_map = {o.id: o.translations for o in out.items}
                for it in chunk_items:
                    got = out_map.get(it.id)
                    if not got:
                        raise ValueError(f"Missing translation for item id={it.id}")
                    it.translations = it.translations or {}
                    for lc in translate_langs:
                        it.translations[lc] = got[lc]
                break
            except (json.JSONDecodeError, ValidationError, ValueError) as e:
                errors_str = str(e)
                if attempt == max_attempts:
                    raise RuntimeError(
                        "Failed to generate valid translations.\n"
                        f"Last error: {errors_str}\n"
                        f"Last raw (truncated): {last_raw[:1500]}"
                    ) from e

    return bank
