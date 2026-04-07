"""
Legacy baseline helpers used by FastAPI GET /baseline.
This file is self-contained and provides sync REST-based evaluation.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

import requests
from openai import OpenAI
from pydantic import BaseModel


class BaselineResult(BaseModel):
    task_id: str
    score: float | None = None
    error: str | None = None


def _extract_json(text: str) -> dict[str, Any]:
    m = re.search(r"\{[\s\S]*\}", text.strip())
    if not m:
        raise ValueError("No JSON object in model response")
    return json.loads(m.group(0))


def _llm_action(client: OpenAI, obs: dict, model: str, task_id: str) -> dict[str, Any]:
    schema_hint = (
        "Return a single JSON object with keys: "
        '"hs_code" (string), "flags" (array of strings), '
        '"recommendation" (one of: clear, hold, query_shipper, refer_to_customs), '
        '"confidence" (0-1). For task3 also include '
        '"assessable_value_inr" and "duty_amount_inr" (numbers).'
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": (
                "You are an expert Indian sea-freight Custom House Agent (CHA). "
                "Respond with ONLY valid JSON, no markdown fences."
            )},
            {"role": "user", "content": json.dumps(obs, indent=2) + "\n\n" + schema_hint},
        ],
        temperature=0.2,
    )
    raw = response.choices[0].message.content or ""
    data = _extract_json(raw)
    
    # Minimal validation/parsing
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


def evaluate_all_tasks(client: OpenAI, env_base_url: str, model: str) -> list[BaselineResult]:
    """Sync REST fallback used by main.py /baseline."""
    out: list[BaselineResult] = []
    for task_id in ("task1", "task2", "task3"):
        try:
            r = requests.post(f"{env_base_url.rstrip('/')}/reset", json={"task_id": task_id}, timeout=120)
            r.raise_for_status()
            obs = r.json()
            
            action = _llm_action(client, obs, model, task_id)
            
            sr = requests.post(f"{env_base_url.rstrip('/')}/step", json=action, timeout=120)
            sr.raise_for_status()
            score = float(sr.json()["reward"])
            out.append(BaselineResult(task_id=task_id, score=score))
        except Exception as e:
            out.append(BaselineResult(task_id=task_id, score=None, error=str(e)))
    return out


def run_baseline_tasks(base_url: str, api_key: str) -> list[BaselineResult]:
    """Build OpenAI client from OPENAI_API_KEY path (optional API_BASE_URL)."""
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    api_base = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
    client = OpenAI(api_key=api_key, base_url=api_base.rstrip("/"))
    return evaluate_all_tasks(client, base_url, model)


if __name__ == "__main__":
    base = os.environ.get("CHA_BASE_URL", "http://127.0.0.1:7860")
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        print("Set OPENAI_API_KEY and ensure the API is running at", base)
        raise SystemExit(1)
    for row in run_baseline_tasks(base_url=base, api_key=key):
        print(f"{row.task_id}: score={row.score} error={row.error}")
