"""
Legacy baseline helpers used by FastAPI GET /baseline.
Prefer root `inference.py` for submission (API_BASE_URL, MODEL_NAME, HF_TOKEN).
"""

from __future__ import annotations

import os

from openai import OpenAI

from inference import BaselineResult, evaluate_all_tasks


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
