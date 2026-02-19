from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from src.psychometrics.bkt import bkt_update
from src.psychometrics.irt import online_theta_update, reliability_from_sem, theta_sem
from src.schemas import Item, ItemBank
from src.utils import difficulty_to_b, now_timestamp_id


@dataclass
class AnswerEvent:
    item_id: str
    skill: str
    difficulty_label: str
    chosen_index: int
    correct_index: int
    was_correct: bool

    a: float
    b: float
    c: float

    theta_before: float
    theta_after: float
    sem_after: float

    mastery_before: float
    mastery_after: float

    misconception_label: str


@dataclass
class AdaptiveSession:
    session_id: str
    session_name: Optional[str]
    total_questions: int

    theta: float = 0.0
    theta_sem: float = 3.0
    reliability_heuristic: float = 0.0

    mastery: Dict[str, float] = field(default_factory=dict)  # per-skill P(known)
    misconception_counts: Dict[str, int] = field(default_factory=dict)

    asked_item_ids: Set[str] = field(default_factory=set)
    answers: List[AnswerEvent] = field(default_factory=list)
    step_index: int = 0
    correct_count: int = 0

    ai_insights: Optional[Dict[str, Any]] = None

    @property
    def is_finished(self) -> bool:
        return len(self.answers) >= self.total_questions

    def apply_answer(self, item: Item, chosen_index: int, was_correct: bool) -> None:
        self.asked_item_ids.add(item.id)

        # IRT params: use priors if available, else defaults
        if item.irt:
            a, b, c = float(item.irt.a), float(item.irt.b), float(item.irt.c)
        else:
            a, b, c = 1.0, difficulty_to_b(item.difficulty_label), 0.0

        k = max(1, len(self.answers) + 1)
        lr = 0.35 * (1.0 / (k**0.5))

        theta_before = self.theta
        self.theta = online_theta_update(self.theta, 1 if was_correct else 0, a=a, b=b, c=c, lr=lr)

        # SEM/reliability using administered items so far
        items_params = [(ev.a, ev.b, ev.c) for ev in self.answers] + [(a, b, c)]
        self.theta_sem = theta_sem(self.theta, items_params)
        self.reliability_heuristic = reliability_from_sem(self.theta_sem, prior_var=1.0)

        if was_correct:
            self.correct_count += 1

        # Cognitive mastery update (BKT) per skill
        skill = item.skill
        if skill not in self.mastery:
            self.mastery[skill] = 0.5
        mastery_before = self.mastery[skill]

        # Use IRT guess as BKT guess; slip grows slightly with difficulty
        guess = max(0.05, min(0.30, c if c > 0 else 0.20))
        slip = 0.08 if item.difficulty_label == "easy" else (0.10 if item.difficulty_label == "med" else 0.13)
        learn = 0.12

        self.mastery[skill] = bkt_update(mastery_before, correct=was_correct, slip=slip, guess=guess, learn=learn)
        mastery_after = self.mastery[skill]

        misconception_label = ""
        if not was_correct:
            try:
                misconception_label = (item.distractor_misconceptions[chosen_index] or "").strip()
            except Exception:
                misconception_label = ""
            if misconception_label:
                self.misconception_counts[misconception_label] = self.misconception_counts.get(misconception_label, 0) + 1

        ev = AnswerEvent(
            item_id=item.id,
            skill=item.skill,
            difficulty_label=item.difficulty_label,
            chosen_index=chosen_index,
            correct_index=item.correct_index,
            was_correct=was_correct,
            a=a,
            b=b,
            c=c,
            theta_before=theta_before,
            theta_after=self.theta,
            sem_after=self.theta_sem,
            mastery_before=mastery_before,
            mastery_after=mastery_after,
            misconception_label=misconception_label,
        )

        self.answers.append(ev)
        self.step_index += 1

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "session_name": self.session_name,
            "total_questions": self.total_questions,
            "theta": self.theta,
            "theta_sem": self.theta_sem,
            "reliability_heuristic": self.reliability_heuristic,
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
            theta_sem=float(d.get("theta_sem", 3.0)),
            reliability_heuristic=float(d.get("reliability_heuristic", 0.0)),
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
    skills = sorted(set(i.skill for i in item_bank.items))
    mastery = {s: 0.5 for s in skills}
    return AdaptiveSession(
        session_id=now_timestamp_id(),
        session_name=session_name,
        total_questions=total_questions,
        theta=0.0,
        theta_sem=3.0,
        reliability_heuristic=0.0,
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
    weakest = _weakest_skills(sess.mastery, top_k=min(5, len(sess.mastery)))
    theta = sess.theta

    candidates: List[Item] = [it for it in bank.items if it.id not in sess.asked_item_ids]
    if not candidates:
        candidates = bank.items

    preferred = [it for it in candidates if it.skill in weakest]
    if preferred:
        candidates = preferred

    def score(it: Item) -> Tuple[float, float, str]:
        if it.irt:
            b = float(it.irt.b)
        else:
            b = difficulty_to_b(it.difficulty_label)
        gap = abs(b - theta)
        m = sess.mastery.get(it.skill, 0.5)
        return (gap, m, it.id)

    candidates.sort(key=score)
    return candidates[0]
