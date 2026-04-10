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
    "step_kind": "final_submission",
    "requested_fields": [],
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


# ── Domain-specific prompts ──────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert Indian sea-freight Custom House Agent (CHA) with deep knowledge of:
- The Indian Customs Tariff (Harmonized System 8-digit codes)
- Document verification for Bills of Lading, Invoices, and Packing Lists
- CIF valuation: Assessable Value = FOB + Freight (~4%) + Insurance (~1.125%)
- Common duty rates: Electronics 10-20%, Textiles 25-35%, Chemicals 10-15%, Food 0-45%
- INR conversion: ~83 INR per USD
- Red flags: quantity mismatches between documents, missing country of origin,
  goods description inconsistencies, suspiciously low declared values, weight
  discrepancies, invoice number mismatches, consignee name typos

HS code structure: Chapter (2 digits) . Heading (2 digits) . Subheading (2 digits) . Tariff item (2 digits)
Example: 8518.30.00 = Ch85 Electrical equipment > 18 Sound equipment > 30 Headphones

Respond with ONLY valid JSON, no markdown fences or explanation."""

TASK1_USER_TEMPLATE = """\
TASK: Assign the correct 8-digit HS code for the goods described in this invoice.

Think step by step:
1. Identify the goods from the description
2. Determine the HS chapter (first 2 digits) based on material/function
3. Narrow to heading (4 digits), then subheading and tariff item (8 digits)
4. Use the standard Indian Customs Tariff format: XXXX.XX.XX

INVOICE DATA:
{obs_json}

Return JSON: {{"hs_code": "XXXX.XX.XX", "flags": [], "recommendation": "clear", "confidence": 0.0-1.0}}"""

TASK2_USER_TEMPLATE = """\
TASK: Review the shipment documents below. Compare Invoice, Packing List, and Bill of Lading.
Flag every compliance issue you find.

Check systematically:
1. Do quantities match across all documents?
2. Is the country of origin declared on the invoice?
3. Does the B/L invoice number match the invoice?
4. Do goods descriptions match across documents?
5. Do weights match between packing list and B/L?
6. Is the consignee name consistent?
7. Is there a notify party on the B/L?
8. Is the declared value suspiciously low for the goods/quantity?

Known flag vocabulary (use exact strings):
- quantity_mismatch
- missing_country_of_origin
- weight_mismatch_packing_vs_bl
- invoice_number_mismatch_bl_vs_invoice
- missing_invoice_number_on_bl
- goods_description_mismatch_invoice_vs_packing_list
- consignee_name_mismatch
- missing_notify_party
- suspected_undervaluation

SHIPMENT DOCUMENTS:
{obs_json}

Return JSON: {{"hs_code": "", "flags": ["flag1", ...], "recommendation": "clear|hold|query_shipper|refer_to_customs", "confidence": 0.0-1.0}}"""

TASK3_USER_TEMPLATE = """\
TASK: Full clearance decision. Classify goods, flag all anomalies, recommend action,
and estimate assessable value and duty in INR.

Valuation formula:
  Assessable Value (INR) = Declared USD * 83.0 * (1 + 0.04 + 0.0125)
  Duty (INR) = Assessable Value * duty_rate
  Common rates: Electronics ~10-20%, Textiles ~25-35%, Chemicals ~10-15%, Hardware ~10-15%

Check for: document mismatches, origin vs loading port inconsistency, vague descriptions,
undervaluation, dual-use chemicals, high-value shipments, textile origin issues.

Additional flag vocabulary for task3:
- vague_goods_description
- origin_loading_mismatch
- high_value_shipment
- mixed_consignment_requires_classification
- textile_declaration_review
- multilingual_document_review
- dual_use_or_controlled_chemical_risk
- ambiguous_end_use
(Plus all task2 flags above)

{revealed_section}

SHIPMENT DOCUMENTS:
{obs_json}

Return JSON: {{
  "hs_code": "XXXX.XX.XX",
  "flags": ["flag1", ...],
  "recommendation": "clear|hold|query_shipper|refer_to_customs",
  "confidence": 0.0-1.0,
  "assessable_value_inr": <number>,
  "duty_amount_inr": <number>
}}"""

TASK_TEMPLATES = {
    "task1": TASK1_USER_TEMPLATE,
    "task2": TASK2_USER_TEMPLATE,
    "task3": TASK3_USER_TEMPLATE,
}


# ── LLM helpers ───────────────────────────────────────────────────────────────
def _extract_json(text: str) -> dict[str, Any]:
    m = re.search(r"\{[\s\S]*\}", text.strip())
    if not m:
        raise ValueError("No JSON object in model response")
    return json.loads(m.group(0))


def _llm_action(
    client: OpenAI,
    obs: dict,
    task_id: str,
    revealed: dict[str, Any] | None = None,
) -> dict[str, Any]:
    template = TASK_TEMPLATES[task_id]

    revealed_section = ""
    if revealed:
        revealed_section = "ADDITIONAL INFORMATION REVEALED:\n" + json.dumps(revealed, indent=2)

    user_content = template.format(
        obs_json=json.dumps(obs, indent=2),
        revealed_section=revealed_section,
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.15,
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
        "step_kind": "final_submission",
        "requested_fields": [],
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
        obs = result.observation
        max_steps = obs.get("max_steps", 1)
        revealed: dict[str, Any] = {}

        if max_steps > 1 and task_id == "task3":
            request_action = {
                "step_kind": "request_information",
                "requested_fields": [
                    "detailed_goods_description",
                    "duty_rate_schedule",
                    "exchange_rate",
                    "certificate_of_origin",
                ],
                "hs_code": "",
                "flags": [],
                "recommendation": "hold",
                "task_id": task_id,
                "metadata": {},
            }
            result = await env.step(request_action)
            revealed = result.observation.get("revealed_content", {})

        try:
            action = _llm_action(client, obs, task_id, revealed=revealed or None)
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
