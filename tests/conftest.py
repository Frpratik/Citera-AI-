"""Shared pytest fixtures. Everything here runs offline (fake embeddings + fake LLM)."""

import pytest
from fastapi.testclient import TestClient

from app.api.deps import build_services, get_services
from app.core.config import Settings
from app.main import app
from tests.helpers import FakeEmbeddings, FakeLLM

CLIENT_A = "client-aaaaaaaa-0001"
CLIENT_B = "client-bbbbbbbb-0002"


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        groq_api_key="test-key",
        min_relevance_score=0.15,  # the fake embeddings give lower cosine scores than real ones
        strong_relevance_score=0.4,
        rate_limit_per_minute=1000,
        _env_file=None,  # ignore any real .env on the developer's machine
    )


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def services(settings, fake_llm):
    return build_services(settings, embeddings=FakeEmbeddings(), llm=fake_llm)


@pytest.fixture
def client(services):
    app.dependency_overrides[get_services] = lambda: services
    # Not using `with TestClient(...)` so the real lifespan (which builds real services) is skipped.
    yield TestClient(app)
    app.dependency_overrides.clear()


def headers(client_id: str = CLIENT_A) -> dict:
    return {"X-Client-ID": client_id}
