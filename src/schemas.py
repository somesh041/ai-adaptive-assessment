from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field, RootModel, field_validator, model_validator


DifficultyLabel = Literal["easy", "med", "hard"]


class Item(BaseModel):
    id: str = Field(..., description="Unique string ID")
    skill: str = Field(..., description="Target skill label")
    difficulty_label: DifficultyLabel = Field(..., description="easy|med|hard")
    bloom_level: str = Field(..., description="Bloom level label (e.g., Remember, Understand, Apply...)")

    stem: str = Field(..., description="Question stem")
    options: List[str] = Field(..., min_length=4, max_length=4, description="Exactly 4 options")
    correct_index: int = Field(..., ge=0, le=3, description="0-3 index into options")
    explanation_short: str = Field(..., max_length=240, description="<=240 chars")

    distractor_misconceptions: List[str] = Field(
        ...,
        min_length=4,
        max_length=4,
        description='List of 4 strings; correct option must be "" and others are misconception labels',
    )

    @field_validator("skill", "bloom_level", "stem")
    @classmethod
    def _strip_nonempty(cls, v: str) -> str:
        v2 = (v or "").strip()
        if not v2:
            raise ValueError("must be non-empty")
        return v2

    @field_validator("options")
    @classmethod
    def _options_uniqueish(cls, v: List[str]) -> List[str]:
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

        # Correct option must have empty misconception label
        if self.distractor_misconceptions[self.correct_index] != "":
            raise ValueError('distractor_misconceptions[correct_index] must be ""')

        # Wrong options should have some label (non-empty) to be diagnostically useful
        for i in range(4):
            if i == self.correct_index:
                continue
            if not (self.distractor_misconceptions[i] or "").strip():
                raise ValueError("wrong option misconception labels must be non-empty")

        return self


class ItemBank(BaseModel):
    items: List[Item] = Field(..., min_length=1)

    @model_validator(mode="after")
    def _ids_unique(self) -> "ItemBank":
        ids = [it.id for it in self.items]
        if len(set(ids)) != len(ids):
            raise ValueError("Item IDs must be unique")
        return self


class ItemBankRoot(RootModel[List[Item]]):
    """
    Some models are more likely to output a raw list.
    We'll accept it and wrap into ItemBank via helper.
    """

    root: List[Item]
