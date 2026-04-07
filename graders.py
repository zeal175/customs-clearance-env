"""
Task-specific graders and unified reward helper for CHA environment.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel


def normalize_hs_code(hs: str) -> str:
    """Strip whitespace; keep digits and dots for display comparison."""
    return re.sub(r"\s+", "", (hs or "").strip())


def hs_chapter_digits(hs: str) -> str:
    """First 4 digit characters of HS code (chapter + heading)."""
    digits = re.sub(r"\D", "", normalize_hs_code(hs))
    return digits[:4] if len(digits) >= 4 else digits


def nudge_score(score: float) -> float:
    """
    Validator requires scores strictly in (0, 1).
    Maps [0, 1] -> [0.01, 0.99].
    """
    return round(0.01 + 0.98 * max(0.0, min(1.0, score)), 4)


class ActionForGrading(BaseModel):
    hs_code: str
    flags: list[str]
    recommendation: str
    confidence: float = 0.0
    assessable_value_inr: float | None = None
    duty_amount_inr: float | None = None


def grade_task1(action: ActionForGrading, correct: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    """Exact HS → 1.0; same chapter (first 4 digits) → 0.5; else 0.0."""
    exp = normalize_hs_code(str(correct["hs_code"]))
    got = normalize_hs_code(action.hs_code)
    if got == exp:
        return nudge_score(1.0), {"hs_match": 1.0, "component": "exact"}
    if hs_chapter_digits(got) == hs_chapter_digits(exp) and len(hs_chapter_digits(got)) == 4:
        return nudge_score(0.5), {"hs_match": 0.5, "component": "chapter_only"}
    return nudge_score(0.0), {"hs_match": 0.0, "component": "wrong"}


def grade_task2(action: ActionForGrading, correct: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    """
    Task 2 scoring:
    - Flags contribute 80% (recall with false-flag penalty)
    - Recommendation contributes 20%
    """
    correct_flags = set(correct.get("flags") or [])
    agent_flags = set(action.flags or [])
    correctly_caught = agent_flags & correct_flags
    false_flags = agent_flags - correct_flags
    missed_flags = correct_flags - agent_flags

    if correct_flags:
        recall = len(correctly_caught) / len(correct_flags)
    else:
        recall = 1.0 if not agent_flags else 0.0
    false_penalty = 0.15 * len(false_flags)
    flag_score = max(0.0, min(1.0, recall - false_penalty))
    flag_part = 0.80 * flag_score
    rec_part = (
        0.20
        if (action.recommendation or "").strip() == (correct.get("recommendation") or "").strip()
        else 0.0
    )
    score = nudge_score(flag_part + rec_part)

    breakdown = {
        "correctly_caught": sorted(correctly_caught),
        "missed": sorted(missed_flags),
        "false_flags": sorted(false_flags),
        "recall": round(recall, 4),
        "false_penalty": round(false_penalty, 4),
        "flags_component": round(flag_part, 4),
        "recommendation_component": rec_part,
    }
    return score, breakdown


def _flag_overlap_score(
    agent_flags: set[str], correct_flags: set[str]
) -> float:
    if not correct_flags:
        return 1.0 if not agent_flags else max(0.0, 1.0 - 0.15 * len(agent_flags))
    caught = len(agent_flags & correct_flags)
    return caught / len(correct_flags)


def _within_pct(got: float | None, exp: float, pct: float) -> bool:
    if got is None:
        return False
    if exp == 0:
        return abs(got) < 1e-6
    return abs(got - exp) / abs(exp) <= pct


def grade_task3(action: ActionForGrading, correct: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    """
    HS 0.30, all anomalies (flags) 0.30, recommendation 0.20,
    duty/assessable within 5% 0.20 (split: 0.10 assessable, 0.10 duty).
    """
    exp_hs = normalize_hs_code(str(correct["hs_code"]))
    got_hs = normalize_hs_code(action.hs_code)
    hs_part = 0.0
    if got_hs == exp_hs:
        hs_part = 0.30
    elif hs_chapter_digits(got_hs) == hs_chapter_digits(exp_hs) and len(hs_chapter_digits(got_hs)) == 4:
        hs_part = 0.15

    correct_flags = set(correct.get("flags") or [])
    agent_flags = set(action.flags or [])
    flag_part = 0.30 * _flag_overlap_score(agent_flags, correct_flags)

    rec_part = 0.0
    if (action.recommendation or "").strip() == (correct.get("recommendation") or "").strip():
        rec_part = 0.20

    exp_av = float(correct["assessable_value_inr"])
    exp_duty = float(correct["duty_amount_inr"])
    av_ok = _within_pct(action.assessable_value_inr, exp_av, 0.05)
    duty_ok = _within_pct(action.duty_amount_inr, exp_duty, 0.05)
    duty_part = 0.0
    if av_ok:
        duty_part += 0.10
    if duty_ok:
        duty_part += 0.10

    total = nudge_score(hs_part + flag_part + rec_part + duty_part)

    breakdown = {
        "hs_component": hs_part,
        "flags_component": flag_part,
        "recommendation_component": rec_part,
        "value_duty_component": duty_part,
        "assessable_within_5pct": av_ok,
        "duty_within_5pct": duty_ok,
    }
    return total, breakdown


def calculate_unified_reward(action: ActionForGrading, correct_answer: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    """
    Spec unified reward: HS + flag overlap style + recommendation.
    Used when a single scalar is needed across tasks for comparison experiments.
    """
    exp_hs = normalize_hs_code(str(correct_answer.get("hs_code", "")))
    got_hs = normalize_hs_code(action.hs_code)
    reward = 0.0
    if exp_hs:
        if got_hs == exp_hs:
            reward += 0.40
        elif hs_chapter_digits(got_hs) == hs_chapter_digits(exp_hs) and len(hs_chapter_digits(got_hs)) == 4:
            reward += 0.20

    correct_flags = set(correct_answer.get("flags") or [])
    agent_flags = set(action.flags or [])
    correctly_caught = agent_flags & correct_flags
    false_flags = agent_flags - correct_flags
    missed_flags = correct_flags - agent_flags

    if correct_flags:
        reward += 0.40 * (len(correctly_caught) / len(correct_flags))
    reward -= 0.10 * len(false_flags)
    reward -= 0.05 * len(missed_flags)

    if (action.recommendation or "").strip() == (correct_answer.get("recommendation") or "").strip():
        reward += 0.20

    score = nudge_score(reward)
    breakdown = {
        "unified_score": score,
        "correctly_caught": sorted(correctly_caught),
        "false_flags": sorted(false_flags),
        "missed": sorted(missed_flags),
    }
    return score, breakdown


def grade_for_task(task_id: str, action: ActionForGrading, correct: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    if task_id == "task1":
        return grade_task1(action, correct)
    if task_id == "task2":
        return grade_task2(action, correct)
    if task_id == "task3":
        return grade_task3(action, correct)
    raise ValueError(f"Unknown task_id: {task_id}")
