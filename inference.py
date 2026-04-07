"""
Hackathon inference entrypoint — customs-clearance-env
Uses GenericEnvClient (openenv SDK) to connect via WebSocket.

Stdout format (mandatory):
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from typing import Any, List, Optional

from openai import OpenAI
from openenv.core.generic_client import GenericEnvClient

# ── env vars ──────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or "placeholder"
ENV_BASE_URL = (
    os.getenv("ENV_BASE_URL")
    or os.getenv("CHA_BASE_URL")
    or "https://zealowo-customs-clearance.hf.space"
)

BENCHMARK = "customs-clearance-env"
TASKS     = ("task1", "task2", "task3")

FALLBACK_ACTION: dict[str, Any] = {
    "hs_code": "8471.30.00",
    "flags": [],
    "recommendation": "hold",
    "confidence": 0.5,
    "assessable_value_inr": 100000.0,
    "duty_amount_inr": 10000.0,
    "task_id": "task1",
}


# ── stdout helpers ─────────────────────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── LLM helpers ───────────────────────────────────────────────────────────────
def _extract_json(text: str) -> dict[str, Any]:
    m = re.search(r"\{[\s\S]*\}", text.strip())
    if not m:
        raise ValueError("No JSON object in model response")
    return json.loads(m.group(0))


def _llm_action(client: OpenAI, obs: dict, task_id: str) -> dict[str, Any]:
    schema_hint = (
        "Return a single JSON object with keys: "
        '"hs_code" (string), "flags" (array of strings), '
        '"recommendation" (one of: clear, hold, query_shipper, refer_to_customs), '
        '"confidence" (0-1). For task3 also include '
        '"assessable_value_inr" and "duty_amount_inr" (numbers).'
    )
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": (
                "You are an expert Indian sea-freight Custom House Agent (CHA). "
                "Respond with ONLY valid JSON, no markdown fences."
            )},
            {"role": "user", "content": json.dumps(obs, indent=2) + "\n\n" + schema_hint},
        ],
        temperature=0.2,
    )
    data = _extract_json(response.choices[0].message.content or "")
    return {
        "hs_code": str(data.get("hs_code", "")),
        "flags": list(data.get("flags") or []),
        "recommendation": str(data.get("recommendation", "hold")),
        "confidence": float(data.get("confidence", 0.5)),
        "assessable_value_inr": float(data["assessable_value_inr"]) if "assessable_value_inr" in data else None,
        "duty_amount_inr": float(data["duty_amount_inr"]) if "duty_amount_inr" in data else None,
        "task_id": task_id,
        "metadata": {},
    }


# ── per-task episode ───────────────────────────────────────────────────────────
async def run_task(client: OpenAI, task_id: str) -> float:
    async with GenericEnvClient(
        base_url=ENV_BASE_URL,
        connect_timeout_s=30.0,
        message_timeout_s=120.0,
    ) as env:
        result = await env.reset(task_id=task_id)
        obs = result.observation  # dict

        try:
            action = _llm_action(client, obs, task_id)
        except Exception:
            action = dict(FALLBACK_ACTION)
            action["task_id"] = task_id

        result = await env.step(action)
        reward = result.reward or 0.01
        return float(reward)


# ── main ───────────────────────────────────────────────────────────────────────
async def amain() -> None:
    client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL.rstrip("/"))

    for task_id in TASKS:
        log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
        rewards: List[float] = []
        error_msg: Optional[str] = None

        try:
            reward = await run_task(client, task_id)
            rewards.append(reward)
            log_step(step=1, action=task_id, reward=reward, done=True, error=None)
        except Exception as e:
            reward = 0.01
            error_msg = str(e)
            rewards.append(reward)
            log_step(step=1, action=task_id, reward=reward, done=True, error=error_msg)

        score = min(max(rewards[0], 0.0), 1.0)
        log_end(success=score > 0.0, steps=1, score=score, rewards=rewards)


def main() -> None:
    try:
        asyncio.run(amain())
    except Exception as e:
        print(f"[DEBUG] fatal: {e}", file=sys.stderr, flush=True)
        print("[END] success=false steps=0 score=0.010 rewards=0.01,0.01,0.01", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
