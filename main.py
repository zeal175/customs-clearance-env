"""
FastAPI server for customs-clearance-env (CHA OpenEnv).
Uses HTTPEnvServer.register_routes() to add the /ws WebSocket endpoint
required by GenericEnvClient, while keeping all existing REST endpoints.
"""
from __future__ import annotations

import os
from typing import Any

from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel, Field
from openenv.core.env_server.http_server import HTTPEnvServer

from environment_openenv import ChaAction, ChaObservation, ChaOpenEnvEnvironment
from documents import TASK_DOCUMENTS, find_task_for_shipment, get_shipment_by_id, list_task_ids
from environment import ChaEnvironment
from graders import ActionForGrading, grade_for_task
from baseline import BaselineResult, run_baseline_tasks
from inference import evaluate_all_tasks
from openai import OpenAI

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="customs-clearance-env",
    description="CHA (Custom House Agent) sea-freight document processing — OpenEnv-style API",
    version="1.0.0",
)

# ── Register /ws + /health + /schema + /metadata + /reset + /step via SDK ────
_oe_server = HTTPEnvServer(
    env=ChaOpenEnvEnvironment,
    action_cls=ChaAction,
    observation_cls=ChaObservation,
    max_concurrent_envs=4,
)
_oe_server.register_routes(app)  # adds /ws, /health, /schema, /metadata, /reset, /step, /state

# ── Legacy REST env (used by /grader and /baseline) ──────────────────────────
_legacy_env = ChaEnvironment()


# ── Legacy Pydantic models (kept for /grader endpoint) ───────────────────────
class Action(BaseModel):
    hs_code: str = Field(..., description="8-digit style HS code e.g. 8518.30.00")
    flags: list[str] = Field(default_factory=list)
    recommendation: str = Field(..., description="clear | hold | query_shipper | refer_to_customs")
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    assessable_value_inr: float | None = Field(None)
    duty_amount_inr: float | None = Field(None)


class Reward(BaseModel):
    reward: float
    breakdown: dict[str, Any]


class GraderRequest(BaseModel):
    task_id: str | None = None
    shipment_id: str | None = None
    action: Action


# ── Additional endpoints not in SDK ──────────────────────────────────────────
@app.get("/")
def root() -> dict[str, str]:
    return {"service": "customs-clearance-env", "docs": "/docs"}


@app.get("/tasks")
def list_tasks() -> dict[str, Any]:
    tasks = []
    for tid in list_task_ids():
        pool = TASK_DOCUMENTS[tid]
        difficulty = {"task1": "easy", "task2": "medium", "task3": "hard"}[tid]
        name = {
            "task1": "HS Code Classification",
            "task2": "Document Validation",
            "task3": "Full Clearance Decision",
        }[tid]
        tasks.append({
            "id": tid,
            "name": name,
            "difficulty": difficulty,
            "num_shipments": len(pool),
            "evaluator": True,
        })
    return {"tasks": tasks, "action_schema": Action.model_json_schema()}


@app.post("/grader")
def grader_endpoint(body: GraderRequest) -> Reward:
    gt: dict[str, Any] | None = None
    tid: str | None = body.task_id
    if body.shipment_id:
        row = get_shipment_by_id(body.shipment_id)
        if not row:
            raise HTTPException(status_code=404, detail="Unknown shipment_id")
        gt = dict(row["correct_answer"])
        tid = tid or find_task_for_shipment(body.shipment_id)
    else:
        gt = _legacy_env.current_ground_truth
        tid = tid or _legacy_env.current_task_id
    if not gt or not tid:
        raise HTTPException(status_code=400, detail="Provide shipment_id or call /reset first.")
    a = ActionForGrading.model_validate(body.action.model_dump())
    reward, breakdown = grade_for_task(tid, a, gt)
    return Reward(reward=reward, breakdown=breakdown)


@app.get("/baseline", response_model=list[BaselineResult])
def baseline() -> list[BaselineResult]:
    base = os.environ.get("ENV_BASE_URL") or os.environ.get(
        "CHA_BASE_URL", "http://127.0.0.1:7860"
    )
    token = os.environ.get("HF_TOKEN")
    api_base = os.environ.get("API_BASE_URL")
    model = os.environ.get("MODEL_NAME")
    if token and api_base and model:
        try:
            client = OpenAI(api_key=token, base_url=api_base.rstrip("/"))
            return evaluate_all_tasks(client, base, model)
        except Exception as e:
            return [BaselineResult(task_id=t, score=None, error=str(e)) for t in ("task1", "task2", "task3")]
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return [BaselineResult(task_id=t, score=None, error="No API key.") for t in ("task1", "task2", "task3")]
    try:
        return run_baseline_tasks(base_url=base, api_key=api_key)
    except Exception as e:
        return [BaselineResult(task_id=t, score=None, error=str(e)) for t in ("task1", "task2", "task3")]