import psycopg
import pytest
from fastapi.testclient import TestClient
from psycopg.conninfo import make_conninfo

from app.api.deps import build_services
from app.auth.passwords import PasswordService
from app.core.config import Settings, get_settings
from app.db.migrate import run_migrations
from app.db.pool import create_pool
from app.main import create_app
from tests.fakes import (
    FakeEmbedder,
    FakeGenerator,
    FakePlanner,
    FakeReranker,
    FakeResponder,
)

TEST_DB = "citera_test"


@pytest.fixture(scope="session")
def _test_pool():
    """A pool on a throw-away database, so tests never touch your dev data."""
    base_url = get_settings().database_url
    try:
        with psycopg.connect(base_url, autocommit=True, connect_timeout=3) as admin:
            admin.execute(f"DROP DATABASE IF EXISTS {TEST_DB} WITH (FORCE)")
            admin.execute(f"CREATE DATABASE {TEST_DB}")
    except psycopg.OperationalError:
        pytest.skip("Postgres is not reachable (run: docker compose up -d)")
    test_url = make_conninfo(base_url, dbname=TEST_DB)
    run_migrations(test_url)
    pool = create_pool(test_url, max_size=4)
    yield pool
    pool.close()


@pytest.fixture
def db_pool(_test_pool):
    """The test pool with all tables emptied before each test."""
    with _test_pool.connection() as conn:
        conn.execute("TRUNCATE documents, users, messages, chat_traces RESTART IDENTITY CASCADE")
    return _test_pool


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def make_client(db_pool, embedder, tmp_path):
    """Builds a test client on the real test database with every AI part faked.

    `make_client(max_upload_mb=1)` overrides settings. The fakes are reachable as
    `client.planner`, `client.generator` and `client.services`.
    """

    def make(**overrides) -> TestClient:
        settings = Settings(
            data_dir=tmp_path, jwt_secret="test-secret-" + "x" * 40, _env_file=None, **overrides
        )
        planner, generator, responder = FakePlanner(), FakeGenerator(), FakeResponder()
        services = build_services(
            settings, db_pool, embedder=embedder, reranker=FakeReranker(),
            planner=planner, generator=generator, responder=responder,
            passwords=PasswordService(time_cost=1, memory_cost=8, parallelism=1),  # fast, test-only
        )
        client = TestClient(create_app(services))
        client.planner, client.generator, client.responder, client.services = planner, generator, responder, services
        return client

    return make


@pytest.fixture
def client(make_client):
    return make_client()
