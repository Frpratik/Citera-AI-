"""Integration tests using the REAL embedding model and the REAL Groq API.

Skipped by default. To run them:
    1. put a valid GROQ_API_KEY in .env
    2. RUN_INTEGRATION=1 pytest -m integration        (Windows PowerShell: $env:RUN_INTEGRATION=1)
The first run downloads the embedding model (~90 MB).
"""

import io
import os

import pytest

from app.api.deps import build_services
from app.core.config import Settings
from tests.helpers import ATTENDANCE_TEXT, make_pdf

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.getenv("RUN_INTEGRATION") != "1", reason="set RUN_INTEGRATION=1 to run"),
]


def test_real_models_end_to_end(tmp_path):
    settings = Settings(data_dir=tmp_path / "data")  # reads GROQ_API_KEY from .env / environment
    assert settings.groq_api_key, "GROQ_API_KEY is required for integration tests"
    services = build_services(settings)

    owner = "integration-owner-1"
    doc = services.documents.upload(owner, "attendance.pdf", io.BytesIO(make_pdf([ATTENDANCE_TEXT])))
    assert doc.status == "ready"

    response = services.chat.answer(owner, "What is the minimum attendance requirement?")
    assert response.answer_type == "document_grounded"
    assert "75" in response.answer
    assert response.sources[0].document_name == "attendance.pdf"

    general = services.chat.answer(owner, "Explain artificial intelligence in simple words.")
    assert general.answer_type == "general"
