# Citera AI

Upload PDFs, ask questions in plain English, and get answers that come only from your documents, with the page numbers they came from. If the answer is not in your documents, it says so instead of guessing.

The step-by-step explanation of how it is built is in the project notes document. This file is only for running it.

## What is inside
FastAPI web app, Postgres with pgvector (text, details and search vectors in one database), two small AI models that run on your computer (one finds passages by meaning, one re-ranks them), Groq for the chat AI, LangGraph for the question flow, and a plain HTML/CSS/JavaScript page.

## Run it for development
You need Python 3.12 and Docker Desktop.

```bash
python -m venv .venv
.venv\Scripts\activate                       # Windows (on Mac/Linux: source .venv/bin/activate)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt
copy .env.example .env                       # then put your GROQ_API_KEY in .env
docker compose up -d                         # starts the database
uvicorn app.main:app --reload                # open http://localhost:8000
```

The first start downloads two small AI models (about 200 MB) and the first PDF may take a little longer.

## Run the whole stack in Docker
Web app and background worker as separate containers:

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"    # copy the result into .env as JWT_SECRET=...
docker compose --profile full up -d --build                     # open http://localhost:8000
docker compose --profile full logs -f worker
```

## Check the code
```bash
ruff check .
pytest -q                                          # fast: the AI parts are faked
RUN_EVAL=1 pytest -q tests/test_eval_gate.py       # slow: the search quality gate with the real models
python -m eval.retrieval_eval                      # search quality report
python -m eval.e2e_eval                            # full-answer report (uses your Groq key)
```

## Useful addresses
`/` the web page, `/docs` the API documentation, `/health` is it running, `/health/ready` database, job queue and AI status, `/metrics` Prometheus metrics (set `METRICS_TOKEN` to protect them).

## Before you put it on the internet
Set a real `JWT_SECRET`, use `ENVIRONMENT=production`, put it behind HTTPS (a reverse proxy), set `METRICS_TOKEN` or keep `/metrics` private, cap request body size at the proxy, and do not publish the database port.
