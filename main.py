"""
FastAPI server for customs-clearance-env (CHA OpenEnv).
Uses the openenv-core SDK's create_app for standard endpoints (/reset, /step,
/state, /schema, /health, /metadata, /mcp, /ws), then adds domain-specific
routes (/tasks, /grader, /baseline).
"""
from __future__ import annotations

import os
from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel, Field
from openenv.core.env_server.http_server import create_app

from environment_openenv import ChaAction, ChaObservation, ChaOpenEnvEnvironment
from documents import TASK_DOCUMENTS, find_task_for_shipment, get_shipment_by_id, list_task_ids
from graders import ActionForGrading, grade_for_task, nudge_score

app = create_app(
    ChaOpenEnvEnvironment,
    ChaAction,
    ChaObservation,
    env_name="customs-clearance-env",
    max_concurrent_envs=4,
)


# ── Custom domain-specific routes ────────────────────────────────────────────

class GraderAction(BaseModel):
    hs_code: str = Field(..., description="8-digit HS code")
    flags: list[str] = Field(default_factory=list)
    recommendation: str = Field(..., description="clear | hold | query_shipper | refer_to_customs")
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    assessable_value_inr: float | None = None
    duty_amount_inr: float | None = None


class Reward(BaseModel):
    reward: float
    breakdown: dict[str, Any]


class GraderRequest(BaseModel):
    task_id: str | None = None
    shipment_id: str | None = None
    action: GraderAction


@app.get("/")
def root():
    return {"service": "customs-clearance-env", "docs": "/docs"}


@app.get("/tasks")
def list_tasks() -> dict[str, Any]:
    tasks = []
    for tid in list_task_ids():
        pool = TASK_DOCUMENTS[tid]
        difficulty = {"task1": "easy", "task2": "medium", "task3": "hard"}[tid]
        tasks.append({
            "id": tid,
            "name": tid,
            "difficulty": difficulty,
            "num_shipments": len(pool),
            "grader": True,
            "evaluator": True,
        })
    return {"tasks": tasks, "action_schema": ChaAction.model_json_schema()}


@app.post("/grader")
def grader_endpoint(body: GraderRequest) -> Reward:
    """Score an action against ground truth for a specific shipment or the given task."""
    gt: dict[str, Any] | None = None
    tid: str | None = body.task_id
    if body.shipment_id:
        row = get_shipment_by_id(body.shipment_id)
        if not row:
            raise HTTPException(status_code=404, detail="Unknown shipment")
        gt = dict(row["correct_answer"])
        tid = tid or find_task_for_shipment(body.shipment_id)
    if not gt or not tid:
        raise HTTPException(status_code=400, detail="Provide task_id and/or shipment_id.")
    a = ActionForGrading.model_validate(body.action.model_dump())
    reward, breakdown = grade_for_task(tid, a, gt)
    return Reward(reward=reward, breakdown=breakdown)


@app.get("/baseline")
def baseline():
    from openai import OpenAI
    from baseline import evaluate_all_tasks

    base = os.environ.get("ENV_BASE_URL", "http://127.0.0.1:7860")
    token = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY")
    if not token:
        return {"error": "No token"}
    client = OpenAI(
        api_key=token,
        base_url=os.environ.get("API_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
    )
    return evaluate_all_tasks(client, base, os.environ.get("MODEL_NAME", "gpt-4o-mini"))
