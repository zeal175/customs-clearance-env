"""
Hackathon inference entrypoint: OpenAI-compatible client using env vars:
  API_BASE_URL     — LLM API base (e.g. https://api.openai.com/v1)
  MODEL_NAME       — model id
  OPENAI_API_KEY   — API key (checklist name; used as api_key for the client)
  ENV_BASE_URL     — customs-clearance-env HTTP API (default http://127.0.0.1:7860)
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any

import requests
from openai import OpenAI
from pydantic import BaseModel


class BaselineResult(BaseModel):
    task_id: str
    score: float | None = None
    error: str | None = None


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError("No JSON object in model response")
    return json.loads(m.group(0))


def parse_action(content: str) -> dict[str, Any]:
    data = _extract_json(content)
    out: dict[str, Any] = {
        "hs_code": str(data.get("hs_code", "")),
        "flags": list(data.get("flags") or []),
        "recommendation": str(data.get("recommendation", "hold")),
        "confidence": float(data.get("confidence", 0.5)),
    }
    if "assessable_value_inr" in data:
        out["assessable_value_inr"] = float(data["assessable_value_inr"])
    if "duty_amount_inr" in data:
        out["duty_amount_inr"] = float(data["duty_amount_inr"])
    return out


def run_task(client: OpenAI, env_base_url: str, task_id: str, model: str) -> float:
    r = requests.post(
        f"{env_base_url.rstrip('/')}/reset", json={"task_id": task_id}, timeout=120
    )
    r.raise_for_status()
    obs = r.json()
    user = json.dumps(obs, indent=2)
    schema_hint = (
        "Return a single JSON object with keys: "
        '"hs_code" (string), "flags" (array of strings), '
        '"recommendation" (one of: clear, hold, query_shipper, refer_to_customs), '
        '"confidence" (0-1). For task3 also include "assessable_value_inr" and "duty_amount_inr" (numbers).'
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert Indian sea-freight Custom House Agent (CHA). "
                    "Respond with ONLY valid JSON, no markdown fences."
                ),
            },
            {"role": "user", "content": user + "\n\n" + schema_hint},
        ],
        temperature=0.2,
    )
    raw = response.choices[0].message.content or ""
    action = parse_action(raw)
    sr = requests.post(
        f"{env_base_url.rstrip('/')}/step", json=action, timeout=120
    )
    sr.raise_for_status()
    result = sr.json()
    return float(result["reward"])


def evaluate_all_tasks(client: OpenAI, env_base_url: str, model: str) -> list[BaselineResult]:
    out: list[BaselineResult] = []
    for task_id in ("task1", "task2", "task3"):
        try:
            score = run_task(client, env_base_url, task_id, model)
            out.append(BaselineResult(task_id=task_id, score=score, error=None))
        except Exception as e:  # noqa: BLE001
            out.append(BaselineResult(task_id=task_id, score=None, error=str(e)))
    return out


def main() -> None:
    try:
        api_base = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
        model = os.environ.get("MODEL_NAME", "gpt-4o-mini")
        token = os.environ["OPENAI_API_KEY"]
    except KeyError as e:
        print(
            "Missing required env var:",
            e.args[0],
            file=sys.stderr,
        )
        print(
            "Required: OPENAI_API_KEY. Optional: API_BASE_URL, MODEL_NAME, ENV_BASE_URL (default http://127.0.0.1:7860).",
            file=sys.stderr,
        )
        raise SystemExit(1) from e

    env_base = os.environ.get("ENV_BASE_URL") or os.environ.get(
        "CHA_BASE_URL", "http://127.0.0.1:7860"
    )
    client = OpenAI(api_key=token, base_url=api_base.rstrip("/"))
    for row in evaluate_all_tasks(client, env_base, model):
        print(f"{row.task_id}: score={row.score} error={row.error}")


if __name__ == "__main__":
    main()
