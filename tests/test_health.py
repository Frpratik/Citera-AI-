from fastapi.testclient import TestClient

from app.main import app


def test_health_ok_and_request_id_header():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert "x-request-id" in response.headers


def test_ready_checks_the_database_and_reports_whether_the_ai_is_configured(client):
    body = client.get("/health/ready").json()
    assert body["status"] == "ready" and body["llm_configured"] is False
    assert set(body["jobs"]) == {"queued", "running", "done", "failed"}
    assert set(body["llm_circuits"].values()) == {"closed"}  # nothing has failed


def test_responses_carry_basic_safety_headers(client):
    headers = client.get("/health").headers
    assert headers["x-content-type-options"] == "nosniff"
    assert headers["x-frame-options"] == "DENY"


def test_unknown_route_returns_404():
    client = TestClient(app)
    response = client.get("/does-not-exist")
    assert response.status_code == 404
