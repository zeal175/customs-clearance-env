"""
OpenEnv-compatible wrapper around ChaEnvironment.
Adapts our existing environment to the openenv SDK's Environment/Action/Observation interfaces,
enabling the /ws WebSocket endpoint via HTTPEnvServer.register_routes().
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import Any, Optional
from pydantic import Field
from openenv.core.env_server.interfaces import Action, Environment, Observation

from documents import TASK_DOCUMENTS
from graders import ActionForGrading, grade_for_task
from documents import get_document_by_task
import random


# ── Action ────────────────────────────────────────────────────────────────────
class ChaAction(Action):
    """Action for customs clearance tasks."""
    hs_code: str = Field(default="", description="8-digit HS code")
    flags: list[str] = Field(default_factory=list)
    recommendation: str = Field(default="hold", description="clear|hold|query_shipper|refer_to_customs")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    assessable_value_inr: float | None = Field(default=None)
    duty_amount_inr: float | None = Field(default=None)
    task_id: str = Field(default="task1", description="task1|task2|task3")


# ── Observation ───────────────────────────────────────────────────────────────
class ChaObservation(Observation):
    """Observation returned after reset or step."""
    document_type: str = Field(default="")
    document_content: dict[str, Any] = Field(default_factory=dict)
    task_instruction: str = Field(default="")
    episode_id: int = Field(default=0)
    shipment_id: str = Field(default="")
    task_id: str = Field(default="task1")


# ── Environment ───────────────────────────────────────────────────────────────
class ChaOpenEnvEnvironment(Environment[ChaAction, ChaObservation, dict]):
    """OpenEnv wrapper around the customs clearance environment."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._task_id: str = "task1"
        self._current_doc: dict[str, Any] | None = None
        self._done: bool = True
        self._episode_count: int = 0
        self._rng = random.Random()

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: str = "task1",
        **kwargs,
    ) -> ChaObservation:
        if task_id not in TASK_DOCUMENTS:
            task_id = "task1"
        self._task_id = task_id
        self._rng = random.Random(seed if seed is not None else random.randint(0, 2**31 - 1))
        pool = TASK_DOCUMENTS[task_id]
        idx = self._rng.randrange(0, len(pool))
        self._current_doc = get_document_by_task(task_id, idx)
        self._done = False
        self._episode_count += 1
        d = self._current_doc
        return ChaObservation(
            document_type=d["document_type"],
            document_content=d["document_content"],
            task_instruction=d["task_instruction"],
            episode_id=self._episode_count,
            shipment_id=d["id"],
            task_id=self._task_id,
            done=False,
            reward=None,
        )

    def step(self, action: ChaAction, **kwargs) -> ChaObservation:
        if self._done or self._current_doc is None:
            raise RuntimeError("Episode finished; call reset() first.")
        correct = self._current_doc["correct_answer"]
        a = ActionForGrading(
            hs_code=action.hs_code,
            flags=action.flags,
            recommendation=action.recommendation,
            confidence=action.confidence,
            assessable_value_inr=action.assessable_value_inr,
            duty_amount_inr=action.duty_amount_inr,
        )
        score, breakdown = grade_for_task(self._task_id, a, correct)
        self._done = True
        d = self._current_doc
        return ChaObservation(
            document_type=d["document_type"],
            document_content=d["document_content"],
            task_instruction=d["task_instruction"],
            episode_id=self._episode_count,
            shipment_id=d["id"],
            task_id=self._task_id,
            done=True,
            reward=score,
            metadata={"breakdown": breakdown},
        )

    @property
    def state(self) -> dict:
        return {
            "task_id": self._task_id,
            "done": self._done,
            "episode_id": self._episode_count,
            "shipment_id": self._current_doc["id"] if self._current_doc else None,
        }

    def close(self) -> None:
        pass

    