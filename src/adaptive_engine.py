from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from src.schemas import Item, ItemBank
from src.utils import difficulty_to_b, sigmoid, now_timestamp_id


@dataclass
class AnswerEvent:
    item_id: str
    skill: str
    difficulty_label: str
    b: float
    chosen_index: int
    correct_index: int
    was_correct: bool
    theta_before: float
    theta_after: float
    mastery_before: float
    mastery_after: float
    misconception_label: str


@dataclass
class AdaptiveSession:
    session_id: str
    session_name: Optional[str]
    total_questions: int

    theta: float = 0.0
    mastery: Dict[str, float] = field(default_factory=dict)  # per-skill mastery [0,1]
    misconception_counts: Dict[str, int] = field(default_factory=dict)

    asked_item_ids: Set[str] = field(default_factory=set)
    answers: List[AnswerEvent] = field(default_factory=list)
    step_index: int = 0  # increments each answer
    correct_count: int = 0

    ai_insights: Optional[Dict[str, Any]] = None  # {"narrative": str, "recommendations": [..]}

    @property
    def is_finished(self) -> bool:
        return len(self.answers) >= self.total_questions

    def apply_answer(self, item: Item, chosen_index: int, was_correct: bool) -> None:
        """
        Updates theta, mastery, misconception counters.
        Uses a small-step logistic update with step decay as more items answered.
        """
        self.asked_item_ids.add(item.id)

        b = difficulty_to_b(item.difficulty_label)

        # Decaying learning rate
        k = max(1, len(self.answers) + 1)
        base_lr = 0.50
        lr = base_lr * (1.0 / (k**0.5))

        theta_before = self.theta
        p = sigmoid(self.theta - b)

        # Theta update: move toward better calibration
        if was_correct:
            self.theta = self.theta + lr * (1.0 - p)
            self.correct_count += 1
        else:
            self.theta = self.theta - lr * (p)

        # Mastery update per skill in [0,1]
        skill = item.skill
        if skill not in self.mastery:
            self.mastery[skill] = 0.5

        mastery_before = self.mastery[skill]
        m_lr = 0.20 * (1.0 / (k**0.5))

        if was_correct:
            self.mastery[skill] = min(1.0, mastery_before + m_lr * (1.0 - mastery_before))
        else:
            self.mastery[skill] = max(0.0, mastery_before - m_lr * (mastery_before))

        mastery_after = self.mastery[skill]

        # Misconception tracking
        misconception_label = ""
        if not was_correct:
            try:
                misconception_label = (item.distractor_misconceptions[chosen_index] or "").strip()
            except Exception:
                misconception_label = ""
            if misconception_label:
                self.misconception_counts[misconception_label] = self.misconception_counts.get(misconception_label, 0) + 1

        event = AnswerEvent(
            item_id=item.id,
            skill=item.skill,
            difficulty_label=item.difficulty_label,
            b=b,
            chosen_index=chosen_index,
            correct_index=item.correct_index,
            was_correct=was_correct,
            theta_before=theta_before,
            theta_after=self.theta,
            mastery_before=mastery_before,
            mastery_after=mastery_after,
            misconception_label=misconception_label,
        )

        self.answers.append(event)
        self.step_index += 1

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "session_name": self.session_name,
            "total_questions": self.total_questions,
            "theta": self.theta,
            "mastery": self.mastery,
            "misconception_counts": self.misconception_counts,
            "asked_item_ids": sorted(list(self.asked_item_ids)),
            "answers": [event.__dict__ for event in self.answers],
            "step_index": self.step_index,
            "correct_count": self.correct_count,
            "ai_insights": self.ai_insights,
        }

    @classmethod
    def from_json_dict(cls, d: Dict[str, Any]) -> "AdaptiveSession":
        sess = cls(
            session_id=d["session_id"],
            session_name=d.get("session_name"),
            total_questions=int(d["total_questions"]),
            theta=float(d.get("theta", 0.0)),
            mastery=dict(d.get("mastery", {})),
            misconception_counts=dict(d.get("misconception_counts", {})),
            asked_item_ids=set(d.get("asked_item_ids", [])),
            answers=[],
            step_index=int(d.get("step_index", 0)),
            correct_count=int(d.get("correct_count", 0)),
            ai_insights=d.get("ai_insights"),
        )
        for ev in d.get("answers", []):
            sess.answers.append(AnswerEvent(**ev))
        return sess


def build_adaptive_session(item_bank: ItemBank, total_questions: int, session_name: Optional[str] = None) -> AdaptiveSession:
    # Initialize mastery for all skills present to 0.5
    skills = sorted(set(i.skill for i in item_bank.items))
    mastery = {s: 0.5 for s in skills}

    return AdaptiveSession(
        session_id=now_timestamp_id(),
        session_name=session_name,
        total_questions=total_questions,
        theta=0.0,
        mastery=mastery,
        misconception_counts={},
        asked_item_ids=set(),
        answers=[],
        step_index=0,
        correct_count=0,
        ai_insights=None,
    )


def _weakest_skills(mastery: Dict[str, float], top_k: int = 3) -> List[str]:
    return [k for k, _ in sorted(mastery.items(), key=lambda kv: (kv[1], kv[0]))[:top_k]]


def select_next_item(sess: AdaptiveSession, bank: ItemBank) -> Item:
    """
    Next item selection:
      - prioritize weakest skills
      - pick difficulty close to current theta
      - avoid repeats
    """
    weakest = _weakest_skills(sess.mastery, top_k=min(5, len(sess.mastery)))
    theta = sess.theta

    candidates: List[Item] = [it for it in bank.items if it.id not in sess.asked_item_ids]
    if not candidates:
        # fallback (shouldn't happen if bank large enough)
        candidates = bank.items

    # First filter: prioritize weakest skills
    preferred = [it for it in candidates if it.skill in weakest]
    if preferred:
        candidates = preferred

    # Rank by closeness of difficulty to theta; tie-break by lower mastery then stable id
    def score(it: Item) -> Tuple[float, float, str]:
        b = difficulty_to_b(it.difficulty_label)
        diff_gap = abs(b - theta)
        m = sess.mastery.get(it.skill, 0.5)
        return (diff_gap, m, it.id)

    candidates.sort(key=score)
    return candidates[0]
