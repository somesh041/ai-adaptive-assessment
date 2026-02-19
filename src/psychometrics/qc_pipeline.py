from __future__ import annotations

import json
from typing import Dict, List, Tuple

from pydantic import ValidationError

from src.openai_client import get_client
from src.psychometrics.bank_stats import bank_summary_metrics
from src.psychometrics.item_review import apply_review_to_item, review_items_llm
from src.schemas import Item, ItemBank, ItemBankMeta
from src.utils import now_timestamp_id


QC_VERSION = "qc-v1.0"


SYSTEM_PROMPT = """You are an expert assessment designer.
Return ONLY valid JSON. No markdown, no commentary, no code fences.
All strings must be plain text.
"""

REGEN_ITEM_PROMPT = """Regenerate ONE multiple-choice item.

Context:
Topic: {topic}
Grade: {grade}
Target skill: {skill}
Bloom level: {bloom}
Difficulty label: {difficulty}

Problems found in prior version:
{issues}

Rules:
- Return STRICT JSON only (a single object, not a list).
- Fields required:
  id, skill, difficulty_label (easy|med|hard), bloom_level,
  stem, options[4], correct_index (0-3), explanation_short (<=240 chars),
  distractor_misconceptions[4] with "" for correct option and non-empty labels for wrong options.
- Keep the SAME id exactly: {item_id}
- Exactly 4 options; only one correct.

Return JSON only.
"""


def _clip(s, n: int) -> str:
    t = "" if s is None else str(s)
    t = t.strip()
    if len(t) > n:
        t = t[:n].rstrip()
    return t


def _guess_misconception(skill: str, idx: int) -> str:
    s = (skill or "").lower()
    if "place value" in s:
        return "confuses place value"
    if "decimal" in s:
        return "misreads decimal places"
    if "fraction" in s:
        return "does not simplify fractions"
    return f"common misconception {idx+1}"


def _sanitize_item_dict(it: Dict) -> Dict:
    # normalize difficulty
    dl = _clip(it.get("difficulty_label", "med"), 10).lower()
    if dl not in {"easy", "med", "hard"}:
        dl = "med"
    it["difficulty_label"] = dl

    it["id"] = _clip(it.get("id", ""), 80)
    it["skill"] = _clip(it.get("skill", ""), 120)
    it["bloom_level"] = _clip(it.get("bloom_level", "understand"), 40) or "understand"
    it["stem"] = _clip(it.get("stem", ""), 600) or "Question text unavailable."
    it["explanation_short"] = _clip(it.get("explanation_short", ""), 240) or "Explanation unavailable."

    # options
    opts = it.get("options")
    if not isinstance(opts, list):
        opts = []
    opts = [_clip(x, 180) for x in opts]
    if len(opts) > 4:
        opts = opts[:4]
    while len(opts) < 4:
        opts.append("N/A")
    it["options"] = opts

    # correct index
    try:
        ci = int(it.get("correct_index", 0))
    except Exception:
        ci = 0
    ci = max(0, min(3, ci))
    it["correct_index"] = ci

    # misconceptions
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
            dm[j] = _clip(dm[j], 80) if str(dm[j]).strip() else _guess_misconception(it["skill"], j)
    it["distractor_misconceptions"] = dm

    return it


def _regen_one_item(topic: str, grade: str, item: Item, issues: List[str], model: str) -> Item:
    client = get_client()
    prompt = REGEN_ITEM_PROMPT.format(
        topic=topic,
        grade=grade,
        skill=item.skill,
        bloom=item.bloom_level,
        difficulty=item.difficulty_label,
        issues="; ".join(issues)[:800],
        item_id=item.id,
    )

    resp = client.responses.create(
        model=model,
        input=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        max_output_tokens=800,
    )
    raw = (resp.output_text or "").strip()
    data = json.loads(raw)

    if isinstance(data, dict):
        data = _sanitize_item_dict(data)

    new_item = Item.model_validate(data)

    if new_item.id != item.id:
        new_item.id = item.id

    return new_item


def qc_audit_and_repair_bank(
    bank: ItemBank,
    topic: str,
    grade: str,
    skills: List[str],
    model: str,
    max_repairs: int = 10,
) -> Tuple[ItemBank, Dict[str, object]]:
    review_map = review_items_llm(bank.items, topic=topic, grade=grade, skills=skills, model=model)
    for it in bank.items:
        r = review_map.get(it.id)
        if r:
            apply_review_to_item(it, r, source="llm")

    to_fix = [it for it in bank.items if it.review and (it.review.revision_needed or len(it.review.flags) > 0)]
    to_fix = to_fix[:max_repairs]

    repaired = 0
    for it in to_fix:
        issues = []
        if it.review:
            issues.extend(it.review.flags)
            if it.review.revision_instructions:
                issues.append(it.review.revision_instructions)
        issues = [x for x in issues if x]

        try:
            new_item = _regen_one_item(topic, grade, it, issues, model=model)
        except (json.JSONDecodeError, ValidationError) as e:
            if it.review:
                it.review.flags = list(set(it.review.flags + ["regen_failed"]))
                it.review.revision_needed = True
            continue

        # preserve any translations if existed
        new_item.translations = it.translations

        # re-review just this regenerated item
        r2_map = review_items_llm([new_item], topic=topic, grade=grade, skills=skills, model=model, chunk_size=1)
        r2 = r2_map.get(new_item.id)
        if r2:
            apply_review_to_item(new_item, r2, source="llm")

        idx = next(i for i, x in enumerate(bank.items) if x.id == it.id)
        bank.items[idx] = new_item
        repaired += 1

    bank.meta = bank.meta or ItemBankMeta()
    bank.meta.qc_enabled = True
    bank.meta.qc_version = QC_VERSION
    bank.meta.qc_summary = bank_summary_metrics(bank)
    if not bank.meta.created_at:
        bank.meta.created_at = now_timestamp_id()

    report = {
        "qc_version": QC_VERSION,
        "initial_flagged": len(to_fix),
        "repaired": repaired,
        "summary": bank.meta.qc_summary,
    }
    return bank, report
