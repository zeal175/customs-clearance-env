---
title: customs-clearance-env
emoji: 🚢
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
short_description: "OpenEnv CHA — sea freight HS, docs, clearance (0–1)"
tags:
  - openenv
---

# customs-clearance-env

**OpenEnv-style environment — Custom House Agent (CHA), Indian sea freight**

**Author:** Rakesh Karthikeyan  
**Context:** Scaler School of Technology × Meta × PyTorch Hackathon (2026)

This repository simulates work a **Custom House Agent** does on **import/export sea freight**: reading shipping documents, assigning **HS codes**, spotting **compliance and consistency issues**, and recommending whether a file should **clear**, **hold**, **query the shipper**, or **refer to customs**. The domain is underrepresented in agent benchmarks; trade compliance is document-heavy, rule-driven, and high-stakes in the real world.

---

## Judge checklist — live Space

Automated checks often hit your **Hugging Face Space URL first**. Deploy the Space and verify:

```bash
curl -s -X POST "https://YOUR_USERNAME-YOUR_SPACE_NAME.hf.space/reset" \
  -H "Content-Type: application/json" \
  -d '{"task_id":"task1"}' | python3 -m json.tool
```

You must get **valid JSON** (observation), not an error page. Replace the URL with your actual Space URL (HF may use `.hf.space` or a custom subdomain; use the URL shown on your Space page).

---

## Episode loop

1. **`POST /reset`** — start an episode; receive an **observation** (documents + instruction).  
2. **`POST /step`** — submit an **action**; receive **observation** (echo of the current episode observation), **reward** in `[0.0, 1.0]`, **`done: true`**, and **info** (including per-task **breakdown**).  
3. Repeat from step 1 for a new shipment.

Optional: **`POST /grader`** scores an action against ground truth (same logic as `/step`) using the current episode or a given `shipment_id`.

---

## Observation space

After `POST /reset`, the API returns a JSON object with the following fields.

| Field | Type | Description |
|--------|------|-------------|
| `document_type` | string | e.g. `invoice` (task 1) or `shipment_file` (tasks 2–3). |
| `document_content` | object | Structured fields for the shipment(s): invoice lines, packing list, bill of lading, etc. Task 2–3 may nest multiple documents under keys such as `documents`. |
| `task_instruction` | string | What the agent must do this episode. |
| `episode_id` | integer | Monotonic counter for the server process. |
| `shipment_id` | string | Stable id for this scenario in `documents.py` (useful for `/grader` and debugging). |

**Contract:** `document_content` is always a JSON object; its inner shape depends on the task and scenario. The agent should treat unknown keys gracefully.

---

## Action space

`POST /step` and `POST /grader` accept a JSON body with:

| Field | Type | Required | Description |
|--------|------|----------|-------------|
| `hs_code` | string | yes | HS-style code, typically 8 digits with dots (e.g. `8518.30.00`). Required for all tasks; task 2 focuses on flags/recommendation but still accepts `hs_code`. |
| `flags` | array of string | no (default `[]`) | Compliance / anomaly labels the agent asserts (exact strings compared to reference sets in the dataset). |
| `recommendation` | string | yes | One of: `clear`, `hold`, `query_shipper`, `refer_to_customs`. |
| `confidence` | number | no | `0.0`–`1.0`; informational for baselines, not used in scoring. |
| `assessable_value_inr` | number or null | no | **Task 3:** estimated assessable value in **INR**. |
| `duty_amount_inr` | number or null | no | **Task 3:** estimated duty in **INR**. |

**Recommendation enum (exact strings):**

- `clear` — proceed subject to normal checks.  
- `hold` — do not clear until discrepancies resolved.  
- `query_shipper` — request clarification or documents from shipper.  
- `refer_to_customs` — escalate (e.g. control, valuation, or classification risk).

---

## Reward and step response

`POST /step` returns:

| Field | Type | Description |
|--------|------|-------------|
| `observation` | object | Same shape as after `/reset` (episode documents + ids). |
| `reward` | number | Final score in `[0.0, 1.0]`. |
| `done` | boolean | Always `true` after one step (single-step episodes). |
| `info` | object | Includes `task_id`, `shipment_id`, and `breakdown` (per-task components). |

---

## Tasks (prose)

### Task 1 — HS Code Classification (easy)

**`task_id`:** `task1`

The agent receives a **single clean commercial invoice**: shipper, consignee, goods description, quantity, value, origin, ports. The instruction asks for the correct **HS code** for the goods. Ground truth is one reference HS code per scenario. Scoring rewards **exact** code match, with **partial credit** if the first four digit positions (chapter/heading) match.

### Task 2 — Document Validation (medium)

**`task_id`:** `task2`

The agent receives a **small shipment file** (e.g. invoice + packing list + bill of lading) with **planted inconsistencies**: quantity mismatches, missing mandatory fields, undervaluation suspicion, consignee typos, weight mismatches, etc. The agent must **list all relevant flags** and choose the **correct recommendation**. Scoring combines **flag recall** with a penalty for **false flags**, plus **recommendation** accuracy (see below).

### Task 3 — Full Clearance Decision (hard)

**`task_id`:** `task3`

The agent receives a **messier file**: vague goods descriptions, cross-document mismatches, suspicious valuation, origin vs loading port issues, multilingual or partial foreign-language content, and higher-risk scenarios. The agent must propose an **HS code**, **flags**, **recommendation**, and **numeric estimates** of **assessable value** and **duty** in INR. Scoring uses multiple components (HS, flags, recommendation, value/duty within tolerance).

---

## Scoring summary (deterministic)

| Task | Main idea |
|------|-----------|
| **task1** | Exact HS → full score; same first 4 digit run → half score; else zero. `hs_code` only drives the score. |
| **task2** | **80%** from flags: recall on expected flags minus `0.15` per false flag, clamped; **20%** from matching `recommendation`. |
| **task3** | **0.30** HS (half if same chapter/heading prefix); **0.30** flag overlap vs expected; **0.20** recommendation; **0.20** value/duty (**0.10** each) if within **5%** of reference INR amounts. |

Full logic: `graders.py`. Dataset and labels: `documents.py`.

---

## Dataset

- **24** synthetic shipments: **8** per task (`documents.py`).  
- Each row has a `correct_answer` used by the grader (not sent to the agent).

---

## API reference

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service id and link to `/docs`. |
| GET | `/health` | OpenEnv contract: `{"status":"healthy"}`. |
| GET | `/metadata` | OpenEnv contract: `name`, `description`. |
| GET | `/schema` | OpenEnv contract: JSON Schemas for `action`, `observation`, `state`. |
| POST | `/mcp` | OpenEnv contract: JSON-RPC 2.0 stub (extensible). |
| POST | `/reset` | Body: `{"task_id":"task1"\|"task2"\|"task3", "seed": optional}`. Returns observation. |
| POST | `/step` | Body: action JSON. Returns reward + `done` + `info`. |
| GET | `/state` | Current episode metadata (task, shipment, last reward). |
| GET | `/tasks` | Task list + JSON Schema for the action. |
| POST | `/grader` | Body: `{"action": {...}, "shipment_id": optional, "task_id": optional}`. Scores vs ground truth. |
| GET | `/baseline` | Runs OpenAI baseline over all three tasks if `OPENAI_API_KEY` is set (see below). |

Interactive docs: `http://localhost:7860/docs` when running locally.

---

## Local setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 7860
```

---

## Docker

```bash
docker build -t customs-clearance-env .
docker run --rm -p 7860:7860 customs-clearance-env
```

Then use `http://127.0.0.1:7860` in the same way as local `uvicorn`.

---

## Inference script (submission checklist)

Root file **`inference.py`** runs the LLM against this environment using the **OpenAI** Python client and these variables:

| Variable | Purpose |
|----------|---------|
| `API_BASE_URL` | LLM API base URL (e.g. `https://api.openai.com/v1` for OpenAI). |
| `MODEL_NAME` | Model id passed to `chat.completions.create`. |
| `HF_TOKEN` | API key for the client (`api_key=`; name follows hackathon spec). |
| `ENV_BASE_URL` | Base URL of **this** customs API (default `http://127.0.0.1:7860`). Optional alias: `CHA_BASE_URL`. |

Example (local env + OpenAI):

```bash
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export HF_TOKEN=sk-...
export ENV_BASE_URL=http://127.0.0.1:7860
python inference.py
```

The script prints one line per task (`task1`–`task3`) with `score` and `error`. It should finish well under typical **20 minute** limits.

## Baseline (HTTP)

With the API running, `GET /baseline` uses the same env trio **`HF_TOKEN` + `API_BASE_URL` + `MODEL_NAME`** if all are set; otherwise it falls back to **`OPENAI_API_KEY`** + optional **`OPENAI_MODEL`** / **`API_BASE_URL`**.

```bash
export OPENAI_API_KEY=sk-...
export CHA_BASE_URL=http://127.0.0.1:7860
export OPENAI_MODEL=gpt-4o-mini
curl -s http://127.0.0.1:7860/baseline | python3 -m json.tool
```

Legacy CLI: `python baseline.py` (uses `OPENAI_API_KEY`).

### Baseline scores (fill after you run)

| Task | Model | Score (0–1) | Notes |
|------|--------|-------------|--------|
| task1 | | *TBD* | |
| task2 | | *TBD* | |
| task3 | | *TBD* | |

Replace `*TBD*` with scores from `inference.py` or `GET /baseline`. Document the **date** and **model** (`MODEL_NAME` or `OPENAI_MODEL`).

---

## OpenEnv CLI and validation

**Install the official Meta OpenEnv CLI** (not the unrelated PyPI package named `openenv`):

```bash
pip install openenv-core
```

**Runtime validation** (what automated checks use against a live server):

```bash
# Terminal A
uvicorn main:app --host 0.0.0.0 --port 7860

# Terminal B
openenv validate --url http://127.0.0.1:7860
```

After Hugging Face deploy:

```bash
openenv validate --url https://YOUR_SPACE_URL
```

The validator expects, among other things: `GET /health` (`{"status":"healthy"}`), `GET /metadata` (`name`, `description`), `GET /schema` (`action`, `observation`, `state` JSON Schema objects), `POST /mcp` (JSON-RPC `2.0` envelope), plus `POST /reset`, `POST /step`, `GET /state` in simulation mode.

**Local directory validation** (`openenv validate` with no URL) may require a full scaffold (e.g. `pyproject.toml`); this repo is optimized for Docker + FastAPI. If the hackathon only requires a passing `--url` check, use the commands above.

Metadata file: `openenv.yaml`.

---

## Repository layout

| Path | Role |
|------|------|
| `main.py` | FastAPI app and route definitions. |
| `environment.py` | Reset/step state machine. |
| `documents.py` | Synthetic shipments + ground truth. |
| `graders.py` | Task-specific scoring. |
| `inference.py` | Submission LLM driver (`API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`). |
| `baseline.py` | Legacy wrapper + `/baseline` helper using `OPENAI_API_KEY`. |
| `openenv.yaml` | Environment metadata. |
| `Dockerfile` | Container for HF Spaces / local. |

---

## License / attribution

Built for the **Scaler School of Technology × Meta × PyTorch Hackathon 2026**.
