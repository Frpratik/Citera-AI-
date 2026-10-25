from app.core.errors import LLMUnavailableError
from tests.api_helpers import signup, upload


def user_id(client, headers) -> str:
    return client.get("/api/auth/me", headers=headers).json()["id"]


# ---------------------------------------------------------------- /metrics
def test_metrics_count_requests_by_route_pattern_not_by_real_path(client):
    headers = signup(client)
    client.get("/api/documents/3f2a9c1e-5b7d-4e8a-9c21-7d4e5f6a8b90", headers=headers)  # 404, an id in the path

    text = client.get("/metrics").text

    assert 'http_requests_total{method="GET",route="/api/documents/{document_id}",status="404"} 1.0' in text
    assert "3f2a9c1e" not in text  # ids never become labels


def test_metrics_include_chat_outcomes_and_the_job_queue(client):
    headers = signup(client)
    upload(client, headers, process=False)
    client.post("/api/chat", json={"message": "refund?"}, headers=headers)

    text = client.get("/metrics").text

    assert 'chat_outcomes_total{outcome="insufficient_evidence"} 1.0' in text  # no ready document yet
    assert 'ingest_jobs{status="queued"} 1.0' in text


def test_metrics_can_be_protected_with_a_token(make_client):
    client = make_client(metrics_token="s3cret-token")
    assert client.get("/metrics").status_code == 401
    assert client.get("/metrics", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/metrics", headers={"Authorization": "Bearer s3cret-token"}).status_code == 200


def test_metrics_can_be_switched_off(make_client):
    assert make_client(metrics_enabled=False).get("/metrics").status_code == 404


# ---------------------------------------------------------------- traces
def test_every_question_leaves_a_trace_with_timings_and_no_text(client):
    headers = signup(client)
    upload(client, headers)
    client.post("/api/chat", json={"message": "refund deadline?"}, headers=headers)

    [row] = client.services.traces.recent(user_id(client, headers))

    assert row["outcome"] == "document_grounded" and row["route"] == "document"
    names = [s["name"] for s in row["spans"]]
    assert {"plan_question", "retrieve_documents", "answer_from_documents"} <= set(names)
    assert {"retrieval.embed_question", "retrieval.meaning_search", "retrieval.keyword_search", "retrieval.rerank"} <= set(names)
    assert "refund" not in str(row)  # the question and answer are never stored in the trace


def test_a_failed_question_is_traced_with_its_error_type(client):
    client.planner.route = "greeting"
    client.responder.error = LLMUnavailableError("down")
    headers = signup(client)
    client.post("/api/chat", json={"message": "hello"}, headers=headers)

    [row] = client.services.traces.recent(user_id(client, headers))
    assert row["outcome"] == "error:LLMUnavailableError"


def test_a_streamed_question_is_traced_too(client):
    headers = signup(client)
    upload(client, headers)
    client.post("/api/chat/stream", json={"message": "refund deadline?"}, headers=headers)

    [row] = client.services.traces.recent(user_id(client, headers))
    assert row["outcome"] == "document_grounded"
    assert "retrieval.rerank" in [s["name"] for s in row["spans"]]  # spans survived the stream's threads


def test_a_broken_trace_store_never_breaks_the_chat(client):
    def broken(*args, **kwargs):
        raise RuntimeError("database trouble")

    client.services.traces.add = broken
    headers = signup(client)
    assert client.post("/api/chat", json={"message": "hi"}, headers=headers).status_code == 200


# ---------------------------------------------------------------- /api/usage
def test_usage_adds_up_the_users_questions(client):
    headers = signup(client)
    for _ in range(2):
        client.post("/api/chat", json={"message": "hi there"}, headers=headers)

    usage = client.get("/api/usage", headers=headers).json()

    assert usage["questions"] == 2 and usage["hours"] == 24
    assert usage["estimated_cost"] is None  # no prices configured
    assert usage["avg_response_ms"] >= 0 and usage["fallbacks"] == 0


def test_usage_is_private_to_each_user_and_needs_a_login(client):
    alice, bob = signup(client, "alice@example.com"), signup(client, "bob@example.com")
    client.post("/api/chat", json={"message": "hi"}, headers=alice)

    assert client.get("/api/usage", headers=bob).json()["questions"] == 0
    assert client.get("/api/usage").status_code == 401
    assert client.get("/api/usage?hours=0", headers=alice).status_code == 422


def test_usage_estimates_cost_when_prices_are_set(make_client):
    client = make_client(llm_input_price_per_million=0.5, llm_output_price_per_million=1.5)
    headers = signup(client)
    client.services.traces.add(
        user_id(client, headers), "req-1", "document", "document_grounded",
        {"total_ms": 900, "llm_calls": 2, "input_tokens": 1_000_000, "output_tokens": 500_000,
         "fallback_used": True, "spans": [], "events": []},
    )

    usage = client.get("/api/usage", headers=headers).json()

    assert usage["input_tokens"] == 1_000_000 and usage["fallbacks"] == 1
    assert usage["estimated_cost"] == 1.25  # 1M x 0.5 + 0.5M x 1.5 per million
