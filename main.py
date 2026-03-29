"""
FastAPI server for customs-clearance-env (CHA OpenEnv).
"""

from __future__ import annotations

import os
from typing import Any

from baseline import BaselineResult, run_baseline_tasks
from inference import evaluate_all_tasks
from openai import OpenAI
from documents import TASK_DOCUMENTS, find_task_for_shipment, get_shipment_by_id, list_task_ids
from environment import ChaEnvironment
from fastapi import Body, FastAPI, HTTPException
from graders import ActionForGrading, grade_for_task
from pydantic import BaseModel, Field

env = ChaEnvironment()

app = FastAPI(
    title="customs-clearance-env",
    description="CHA (Custom House Agent) sea-freight document processing — OpenEnv-style API",
    version="1.0.0",
)


class Observation(BaseModel):
    document_type: str
    document_content: dict[str, Any]
    task_instruction: str


class EpisodeObservation(BaseModel):
    """Shape returned by POST /reset (OpenEnv /schema observation contract)."""

    document_type: str
    document_content: dict[str, Any]
    task_instruction: str
    episode_id: int
    shipment_id: str


class EnvStateSchema(BaseModel):
    """Shape returned by GET /state (OpenEnv /schema state contract)."""

    task_id: str
    doc_index: int
    done: bool
    last_reward: float
    last_breakdown: dict[str, Any]
    episode_id: int
    shipment_id: str | None = None


class Action(BaseModel):
    hs_code: str = Field(..., description="8-digit style HS code e.g. 8518.30.00")
    flags: list[str] = Field(default_factory=list)
    recommendation: str = Field(
        ...,
        description="clear | hold | query_shipper | refer_to_customs",
    )
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    assessable_value_inr: float | None = Field(
        None,
        description="Task 3: estimated assessable value in INR",
    )
    duty_amount_inr: float | None = Field(
        None,
        description="Task 3: estimated duty amount in INR",
    )


class Reward(BaseModel):
    score: float
    breakdown: dict[str, Any]


class ResetRequest(BaseModel):
    task_id: str = "task1"
    seed: int | None = None


class StepResponse(BaseModel):
    """OpenEnv contract: step returns observation, reward, done, info."""

    observation: EpisodeObservation
    reward: float
    done: bool
    info: dict[str, Any]


class GraderRequest(BaseModel):
    task_id: str | None = None
    shipment_id: str | None = None
    action: Action


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "customs-clearance-env", "docs": "/docs"}


@app.get("/health")
def health() -> dict[str, str]:
    """OpenEnv runtime contract: GET /health."""
    return {"status": "healthy"}


@app.get("/metadata")
def metadata() -> dict[str, str]:
    """OpenEnv runtime contract: GET /metadata."""
    return {
        "name": "customs-clearance-env",
        "description": (
            "OpenEnv simulating Custom House Agent (CHA) operations in Indian sea freight. "
            "Agents classify goods (HS), validate shipping documents, and recommend clearance."
        ),
    }


@app.get("/schema")
def openenv_schema() -> dict[str, Any]:
    """OpenEnv runtime contract: GET /schema (JSON Schema objects)."""
    return {
        "action": Action.model_json_schema(),
        "observation": EpisodeObservation.model_json_schema(),
        "state": EnvStateSchema.model_json_schema(),
    }


@app.post("/mcp")
def mcp_stub(payload: Any = Body(None)) -> dict[str, Any]:
    """
    OpenEnv runtime contract: POST /mcp returns JSON-RPC 2.0 envelope.
    Minimal stub for validation; extend for real MCP tools if needed.
    """
    if not isinstance(payload, dict):
        payload = {}
    return {
        "jsonrpc": "2.0",
        "id": payload.get("id"),
        "result": {"ok": True},
    }


@app.post("/reset")
def reset_episode(body: ResetRequest) -> dict[str, Any]:
    try:
        obs = env.reset(body.task_id, body.seed)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return obs


@app.post("/step", response_model=StepResponse)
def step_episode(action: Action) -> StepResponse:
    try:
        out = env.step(action.model_dump())
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return StepResponse(**out)


@app.get("/state")
def get_state() -> dict[str, Any]:
    return env.state()


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
        tasks.append(
            {
                "id": tid,
                "name": name,
                "difficulty": difficulty,
                "num_shipments": len(pool),
            }
        )
    return {
        "tasks": tasks,
        "action_schema": Action.model_json_schema(),
    }


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
        gt = env.current_ground_truth
        tid = tid or env.current_task_id

    if not gt or not tid:
        raise HTTPException(
            status_code=400,
            detail="Provide shipment_id or call /reset first so ground truth exists.",
        )

    a = ActionForGrading.model_validate(body.action.model_dump())
    score, breakdown = grade_for_task(tid, a, gt)
    return Reward(score=score, breakdown=breakdown)


@app.get("/baseline", response_model=list[BaselineResult])
def baseline() -> list[BaselineResult]:
    """
    LLM eval: submission env (API_BASE_URL, MODEL_NAME, HF_TOKEN) or legacy OPENAI_API_KEY.
    Customs API URL: ENV_BASE_URL or CHA_BASE_URL (default http://127.0.0.1:7860).
    """
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
        except Exception as e:  # noqa: BLE001
            return [
                BaselineResult(task_id=t, score=None, error=str(e))
                for t in ("task1", "task2", "task3")
            ]

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return [
            BaselineResult(
                task_id=t,
                score=None,
                error="Set HF_TOKEN+API_BASE_URL+MODEL_NAME or OPENAI_API_KEY; baseline skipped.",
            )
            for t in ("task1", "task2", "task3")
        ]

    try:
        return run_baseline_tasks(base_url=base, api_key=api_key)
    except Exception as e:  # noqa: BLE001 — surface baseline failures
        return [
            BaselineResult(task_id=t, score=None, error=str(e))
            for t in ("task1", "task2", "task3")
        ]
