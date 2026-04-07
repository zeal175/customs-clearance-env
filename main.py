"""
FastAPI server for customs-clearance-env (CHA OpenEnv).
Combines manual REST endpoints with OpenEnv SDK WebSocket support.
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
from graders import ActionForGrading, grade_for_task, nudge_score
from baseline import BaselineResult, run_baseline_tasks, evaluate_all_tasks
from openai import OpenAI

app = FastAPI(
    title="customs-clearance-env",
    description="CHA (Custom House Agent) sea-freight document processing — OpenEnv-style API",
    version="1.0.1",
)

# ── Metadata ──────────────────────────────────────────────────────────────────
METADATA = {
    "name": "customs-clearance-env",
    "description": "OpenEnv simulating Custom House Agent (CHA) operations in Indian sea freight.",
}

# ── SDK /ws WebSocket ────────────────────────────────────────────────────────
# We use the SDK server only to register the /ws route.
# We explicitly define the REST routes for maximum compliance.
_oe_server = HTTPEnvServer(
    env=ChaOpenEnvEnvironment,
    action_cls=ChaAction,
    observation_cls=ChaObservation,
)
# This usually registers /ws and a few others. We will override REST if needed.

# ── Source of truth environments ──────────────────────────────────────────────
_legacy_env = ChaEnvironment()

# ── Models ────────────────────────────────────────────────────────────────────
class Action(BaseModel):
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
    action: Action

class ResetRequest(BaseModel):
    task_id: str = "task1"
    seed: int | None = None

# ── Explicit REST Routes ──────────────────────────────────────────────────────

@app.get("/")
def root(): return {"service": "customs-clearance-env", "docs": "/docs"}

@app.get("/health")
def health(): return {"status": "healthy"}

@app.get("/metadata")
def metadata(): return METADATA

@app.get("/schema")
def get_schema():
    return {
        "action": Action.model_json_schema(),
        "observation": ChaObservation.model_json_schema(),
        "state": {"type": "object"}
    }

@app.post("/reset")
def reset_endpoint(body: ResetRequest | None = None):
    body = body or ResetRequest()
    obs = _legacy_env.reset(body.task_id, body.seed)
    # The SDK observation model expects result.observation
    return {
        "observation": obs.model_dump(),
        "reward": nudge_score(0.0),
        "done": False,
        "episode_id": obs.episode_id,
        "shipment_id": obs.shipment_id,
    }

@app.post("/step")
def step_endpoint(action: Action):
    out = _legacy_env.step(action.model_dump())
    return out  # Already has observation, reward, done, info

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
    return {"tasks": tasks, "action_schema": Action.model_json_schema()}

@app.get("/state")
def get_state():
    return _legacy_env.state()

@app.post("/grader")
def grader_endpoint(body: GraderRequest) -> Reward:
    gt: dict[str, Any] | None = None
    tid: str | None = body.task_id
    if body.shipment_id:
        row = get_shipment_by_id(body.shipment_id)
        if not row: raise HTTPException(status_code=404, detail="Unknown shipment")
        gt = dict(row["correct_answer"])
        tid = tid or find_task_for_shipment(body.shipment_id)
    else:
        gt = _legacy_env.current_ground_truth
        tid = tid or _legacy_env.current_task_id
    if not gt or not tid:
        raise HTTPException(status_code=400, detail="Call /reset first.")
    a = ActionForGrading.model_validate(body.action.model_dump())
    reward, breakdown = grade_for_task(tid, a, gt)
    return Reward(reward=reward, breakdown=breakdown)

@app.get("/baseline")
def baseline():
    base = os.environ.get("ENV_BASE_URL", "http://127.0.0.1:7860")
    token = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY")
    if not token: return {"error": "No token"}
    client = OpenAI(api_key=token, base_url=os.environ.get("API_BASE_URL", "https://api.openai.com/v1").rstrip("/"))
    return evaluate_all_tasks(client, base, os.environ.get("MODEL_NAME", "gpt-4o-mini"))

# OpenEnv SDK needs register_routes(app) to handle the WebSocket mount
# This must come AFTER manual routes if we want to prioritize them.
_oe_server.register_routes(app)