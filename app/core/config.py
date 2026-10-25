from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Citera AI"
    environment: str = "dev"
    log_level: str = "INFO"
    log_format: str = "text"  # "text" for people, "json" for log tools (use json in production)

    # Metrics (Prometheus format) at /metrics. If METRICS_TOKEN is set, callers must send it
    # as "Authorization: Bearer <token>". Without one, keep /metrics off the public internet.
    metrics_enabled: bool = True
    metrics_token: str = ""

    database_url: str = "postgresql://citera:citera@localhost:5432/citera"

    data_dir: Path = Path("data")

    # Security. In "dev" a missing JWT_SECRET gets a random one per run (logins end on
    # restart); in any other environment the app refuses to start without one.
    jwt_secret: str = ""
    jwt_expire_minutes: int = 60 * 24
    cors_origins: str = "http://localhost:8000,http://127.0.0.1:8000"
    rate_limit_per_minute: int = 30  # per signed-in user, for upload and chat
    auth_rate_limit_per_minute: int = 10  # per client address, for register and login

    # Chat
    history_turns: int = 4  # earlier question/answer pairs the AI may see

    # Background processing. In production run workers separately (`python -m app.worker`)
    # and set EMBEDDED_WORKER=false; the default keeps `uvicorn` alone enough for development.
    embedded_worker: bool = True
    worker_poll_seconds: float = 2.0  # how long an idle worker sleeps between checks
    job_max_attempts: int = 3
    job_retry_backoff_seconds: float = 30.0  # wait grows with each failed attempt
    job_stale_after_seconds: float = 900.0  # a "running" job older than this is presumed dead
    job_orphan_min_age_seconds: float = 120.0  # processing document with no job, this old, gets one
    job_recovery_interval_seconds: float = 60.0  # how often a worker runs its recovery checks

    # Uploads
    max_upload_mb: int = 15
    max_pdf_pages: int = 500

    # Chunking (characters)
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # Embeddings: the dimension must match the model AND the vector(N) column in the DB.
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    embedding_batch_size: int = 32

    # Retrieval
    retrieval_candidates: int = 20  # how many chunks the merged list passes to the reranker
    retrieval_top_k: int = 5  # how many chunks we finally keep for the LLM
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # Cited chunk with a reranker score at or above this = "strong" evidence, else "moderate".
    strong_relevance_score: float = 0.8
    # A chunk is kept if its re-ranker score (0 to 1) is at least min_relevance_score OR its
    # meaning-search similarity is at least min_similarity. Chosen with the Phase 8 evaluation
    # (python -m eval.retrieval_eval): the re-ranker alone lost 27% of real answers because
    # its absolute scores are low for everyday wording; the two together lost none.
    min_relevance_score: float = 0.2
    min_similarity: float = 0.3

    groq_api_key: str = ""
    # Groq retires models over time (llama-3.3-70b-versatile stopped working for us).
    # Check what your key can use: https://console.groq.com/docs/models
    groq_model: str = "openai/gpt-oss-120b"
    # For reasoning models such as gpt-oss: "low", "medium" or "high" (how long it thinks before
    # answering). Empty = the model's own default. Must stay empty for models without reasoning.
    groq_reasoning_effort: str = ""
    # If the main model fails (outage, rate limit, retired), try this one. Empty = no fallback.
    groq_fallback_model: str = "openai/gpt-oss-20b"
    # A model that keeps failing is skipped for a while instead of being tried (and timing out)
    # on every question: after this many failures in a row it is "open" for this many seconds.
    llm_circuit_failure_threshold: int = 3
    llm_circuit_reset_seconds: float = 30.0
    # Prices per million tokens, copied from your provider's price page. 0 = cost not estimated.
    llm_input_price_per_million: float = 0.0
    llm_output_price_per_million: float = 0.0
    llm_timeout_seconds: int = 45
    llm_max_retries: int = 2  # the library retries failed calls itself (timeouts, rate limits)


    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
