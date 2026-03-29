"""
CHA customs-clearance environment: reset, step, episode state.
"""

from __future__ import annotations

import random
from typing import Any

from documents import TASK_DOCUMENTS, get_document_by_task
from graders import ActionForGrading, grade_for_task


class ChaEnvironment:
    def __init__(self) -> None:
        self._task_id: str = "task1"
        self._doc_index: int = 0
        self._rng: random.Random = random.Random()
        self._done: bool = True
        self._last_reward: float = 0.0
        self._last_breakdown: dict[str, Any] = {}
        self._current_doc: dict[str, Any] | None = None
        self._episode_count: int = 0

    def reset(self, task_id: str, seed: int | None = None) -> dict[str, Any]:
        if task_id not in TASK_DOCUMENTS:
            raise ValueError(f"Unknown task_id: {task_id}")
        self._task_id = task_id
        self._rng = random.Random(seed if seed is not None else random.randint(0, 2**31 - 1))
        pool = TASK_DOCUMENTS[task_id]
        self._doc_index = self._rng.randrange(0, len(pool))
        self._done = False
        self._episode_count += 1
        self._current_doc = get_document_by_task(task_id, self._doc_index)
        self._last_reward = 0.0
        self._last_breakdown = {}
        return self._build_observation()

    def _build_observation(self) -> dict[str, Any]:
        assert self._current_doc is not None
        d = self._current_doc
        return {
            "document_type": d["document_type"],
            "document_content": d["document_content"],
            "task_instruction": d["task_instruction"],
            "episode_id": self._episode_count,
            "shipment_id": d["id"],
        }

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        if self._done:
            raise RuntimeError("Episode finished; call reset() first.")
        assert self._current_doc is not None
        correct = self._current_doc["correct_answer"]
        a = ActionForGrading.model_validate(action)
        score, breakdown = grade_for_task(self._task_id, a, correct)
        self._last_reward = score
        self._last_breakdown = breakdown
        self._done = True
        return {
            "observation": self._build_observation(),
            "reward": score,
            "done": True,
            "info": {
                "breakdown": breakdown,
                "task_id": self._task_id,
                "shipment_id": self._current_doc["id"],
            },
        }

    def state(self) -> dict[str, Any]:
        return {
            "task_id": self._task_id,
            "doc_index": self._doc_index,
            "done": self._done,
            "last_reward": self._last_reward,
            "last_breakdown": self._last_breakdown,
            "episode_id": self._episode_count,
            "shipment_id": self._current_doc["id"] if self._current_doc else None,
        }

    @property
    def current_ground_truth(self) -> dict[str, Any] | None:
        if not self._current_doc:
            return None
        return dict(self._current_doc["correct_answer"])

    @property
    def current_task_id(self) -> str:
        return self._task_id
