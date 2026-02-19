from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


DifficultyLabel = Literal["easy", "med", "hard"]
LangCode = Literal["hi", "mr", "or", "te"]


class TranslationText(BaseModel):
    stem: str = Field(..., description="Translated stem")
    options: List[str] = Field(..., min_length=4, max_length=4, description="Exactly 4 translated options")
    explanation_short: Optional[str] = Field(None, description="Optional translated explanation (<=240 chars)")

    @field_validator("stem")
    @classmethod
    def _strip_stem(cls, v: str) -> str:
        v2 = (v or "").strip()
        if not v2:
            raise ValueError("stem must be non-empty")
        return v2

    @field_validator("options")
    @classmethod
    def _options_valid(cls, v: List[str]) -> List[str]:
        if len(v) != 4:
            raise ValueError("options must be exactly 4")
        cleaned = [s.strip() for s in v]
        if any(not s for s in cleaned):
            raise ValueError("options cannot be empty")
        return cleaned

    @field_validator("explanation_short")
    @classmethod
    def _explanation_len(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v2 = v.strip()
        if len(v2) > 240:
            v2 = v2[:240].rstrip()
        return v2


class IRTParams(BaseModel):
    # 2PL/3PL-style parameters (priors or calibrated)
    a: float = Field(..., ge=0.05, le=5.0, description="Discrimination (>0)")
    b: float = Field(..., ge=-6.0, le=6.0, description="Difficulty")
    c: float = Field(0.0, ge=0.0, le=0.40, description="Guessing (3PL), default 0")


class ItemReview(BaseModel):
    reviewer: Literal["llm", "rules", "calibration"] = "llm"
    skill_pred: Optional[str] = None
    bloom_pred: Optional[str] = None
    skill_match: Optional[bool] = None
    bloom_match: Optional[bool] = None
    difficulty_label_ok: Optional[bool] = None
    flags: List[str] = Field(default_factory=list)
    revision_needed: bool = False
    revision_instructions: Optional[str] = None
    updated_at: Optional[str] = None


class Item(BaseModel):
    id: str = Field(..., description="Unique string ID")
    skill: str = Field(..., description="Primary target skill label")
    difficulty_label: DifficultyLabel = Field(..., description="easy|med|hard")
    bloom_level: str = Field(..., description="Bloom level label")

    stem: str
    options: List[str] = Field(..., min_length=4, max_length=4)
    correct_index: int = Field(..., ge=0, le=3)
    explanation_short: str = Field(..., max_length=240)

    distractor_misconceptions: List[str] = Field(..., min_length=4, max_length=4)

    # Optional enhancements
    translations: Optional[Dict[LangCode, TranslationText]] = None

    # Psychometrics / QC
    irt: Optional[IRTParams] = None
    review: Optional[ItemReview] = None

    @field_validator("skill", "bloom_level", "stem")
    @classmethod
    def _strip_nonempty(cls, v: str) -> str:
        v2 = (v or "").strip()
        if not v2:
            raise ValueError("must be non-empty")
        return v2

    @field_validator("options")
    @classmethod
    def _options_valid(cls, v: List[str]) -> List[str]:
        if len(v) != 4:
            raise ValueError("options must be exactly 4")
        cleaned = [s.strip() for s in v]
        if any(not s for s in cleaned):
            raise ValueError("options cannot be empty")
        return cleaned

    @model_validator(mode="after")
    def _validate_misconceptions(self) -> "Item":
        if len(self.distractor_misconceptions) != 4:
            raise ValueError("distractor_misconceptions must be exactly 4")
        if self.distractor_misconceptions[self.correct_index] != "":
            raise ValueError('distractor_misconceptions[correct_index] must be ""')
        for i in range(4):
            if i == self.correct_index:
                continue
            if not (self.distractor_misconceptions[i] or "").strip():
                raise ValueError("wrong option misconception labels must be non-empty")
        return self

    @model_validator(mode="after")
    def _validate_translations(self) -> "Item":
        if self.translations is not None and len(self.translations) == 0:
            self.translations = None
        return self


class ItemBankMeta(BaseModel):
    topic: Optional[str] = None
    grade: Optional[str] = None
    skills: Optional[List[str]] = None
    created_at: Optional[str] = None
    model: Optional[str] = None

    qc_enabled: bool = False
    qc_version: Optional[str] = None
    qc_summary: Optional[Dict[str, object]] = None


class ItemBank(BaseModel):
    items: List[Item] = Field(..., min_length=1)
    meta: Optional[ItemBankMeta] = None

    @model_validator(mode="after")
    def _ids_unique(self) -> "ItemBank":
        ids = [it.id for it in self.items]
        if len(set(ids)) != len(ids):
            raise ValueError("Item IDs must be unique")
        return self
