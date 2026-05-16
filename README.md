# SHL Assessment Recommender

A FastAPI service that recommends SHL assessments using hybrid retrieval and a single LLM call. The service is stateless: each request provides the full conversation history and returns a structured response.

## Features

- Hybrid retrieval (BM25 + FAISS) with Reciprocal Rank Fusion
- Single-call LLM with strict JSON output schema
- Post-generation URL validation against the SHL catalog
- Turn-cap awareness and behavior guardrails
- FastAPI endpoints for health and chat

## API

### GET /health

Response:

```json
{"status": "ok"}
```

### POST /chat

Request:

```json
{
  "messages": [
    {"role": "user", "content": "We need a solution for senior leadership."}
  ]
}
```

Response:

```json
{
  "reply": "...",
  "recommendations": [
    {"name": "...", "url": "...", "test_type": "K"}
  ],
  "end_of_conversation": false
}
```

## Setup

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

2. Create a .env file based on .env.example and set keys:

```
GROQ_API_KEY=...
OPENROUTER_API_KEY=...
GEMINI_API_KEY=...
LLM_PROVIDER=groq
```

3. Run the API:

```bash
.venv\Scripts\python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

## Evaluation

Run the evaluation script (rate-limited to avoid API throttling):

```bash
set EVAL_BASE_URL=http://127.0.0.1:8000
set EVAL_TURN_DELAY=12
.venv\Scripts\python evaluate.py
```

Outputs:

- eval_output.txt
- evaluation_results.json

## Deployment

### Render (Docker)

- Connect the GitHub repository
- Render reads render.yaml and Dockerfile
- Set environment variables in the Render dashboard:
  - GROQ_API_KEY
  - OPENROUTER_API_KEY
  - GEMINI_API_KEY
  - LLM_PROVIDER

### Railway (Docker)

- Connect the GitHub repository
- Railway detects the Dockerfile
- Set the same environment variables

## Notes

- Do not commit .env
- Secrets are stored only in the deployment environment variables
