"""Application settings.

All configuration comes from environment variables (or a local `.env` file).
Nothing secret is ever hard-coded here. See `.env.example` for the full list.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = the folder that contains `app/`, `frontend/`, `data/`.
BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Typed settings. Environment variable names are the field names in UPPER_CASE."""

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM (Groq) ---------------------------------------------------------
    groq_api_key: str = ""
    # Verify availability at https://console.groq.com/docs/models (catalogue changes).
    groq_model: str = "llama-3.3-70b-versatile"
    # Optional smaller/faster model for question classification. Empty = use groq_model.
    groq_classifier_model: str = ""
    llm_timeout_seconds: int = 45

    # --- Embeddings ---------------------------------------------------------
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Chunking -----------------------------------------------------------
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # --- Retrieval ----------------------------------------------------------
    retrieval_top_k: int = 5
    # Scores are cosine similarity (-1..1). Chunks below this are treated as "not relevant".
    min_relevance_score: float = 0.25
    # Top score at or above this is reported as "strong" evidence.
    strong_relevance_score: float = 0.45

    # --- Uploads ------------------------------------------------------------
    max_upload_mb: int = 15
    max_pdf_pages: int = 500
    max_documents_per_user: int = 20

    # --- Chat ---------------------------------------------------------------
    history_turns: int = 4  # how many previous user/assistant pairs the LLM may see

    # --- Server -------------------------------------------------------------
    data_dir: Path = BASE_DIR / "data"
    cors_origins: str = "http://localhost:8000,http://127.0.0.1:8000"
    rate_limit_per_minute: int = 30
    log_level: str = "INFO"

    # Derived paths ----------------------------------------------------------
    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def indexes_dir(self) -> Path:
        return self.data_dir / "indexes"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "collegemate.db"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    def ensure_directories(self) -> None:
        """Create the data folders if they do not exist yet."""
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.indexes_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Create the settings object once and reuse it."""
    return Settings()
