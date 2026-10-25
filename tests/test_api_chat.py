from app.core.errors import LLMUnavailableError
from tests.api_helpers import parse_sse, signup, upload


def test_chat_without_documents_says_so(client):
    headers = signup(client)
    body = client.post("/api/chat", json={"message": "What is the refund deadline?"}, headers=headers).json()
    assert "haven't uploaded" in body["answer"]
    assert body["answer_type"] == "insufficient_evidence" and body["sources"] == []


def test_chat_answers_from_the_users_document_with_a_source(client):
    headers = signup(client)
    upload(client, headers)
    body = client.post("/api/chat", json={"message": "What is the refund deadline?"}, headers=headers).json()

    assert body["answer"] == "Within 14 days [1]."
    assert body["answer_type"] == "document_grounded" and body["evidence_status"] == "strong"
    assert body["route"] == "document"
    assert [(s["document_name"], s["page"]) for s in body["sources"]] == [("terms.pdf", 1)]


def test_a_greeting_gets_a_general_reply_without_documents(client):
    client.planner.route = "greeting"
    headers = signup(client)
    body = client.post("/api/chat", json={"message": "hello"}, headers=headers).json()
    assert body["answer_type"] == "general" and body["sources"] == []


def test_chat_needs_a_login_and_a_sensible_message(client):
    assert client.post("/api/chat", json={"message": "hi"}).status_code == 401
    headers = signup(client)
    assert client.post("/api/chat", json={"message": "   "}, headers=headers).status_code == 422
    assert client.post("/api/chat", json={"message": "x" * 2001}, headers=headers).status_code == 422


def test_history_is_saved_returned_in_order_and_given_to_the_next_question(client):
    headers = signup(client)
    client.post("/api/chat", json={"message": "first question"}, headers=headers)
    client.post("/api/chat", json={"message": "second question"}, headers=headers)

    history = client.get("/api/chat/history", headers=headers).json()["messages"]
    assert [(m["role"], m["content"]) for m in history][:2] == [
        ("user", "first question"), ("assistant", history[1]["content"])
    ]
    assert [m["content"] for m in history if m["role"] == "user"] == ["first question", "second question"]
    # the second question was planned with the first exchange as context
    assert [m["content"] for m in client.planner.seen_history if m["role"] == "user"] == ["first question"]


def test_history_belongs_to_each_user_and_can_be_cleared(client):
    alice, bob = signup(client, "alice@example.com"), signup(client, "bob@example.com")
    client.post("/api/chat", json={"message": "alice only"}, headers=alice)

    assert client.get("/api/chat/history", headers=bob).json()["messages"] == []
    assert client.delete("/api/chat/history", headers=alice).json() == {"cleared": 2}
    assert client.get("/api/chat/history", headers=alice).json()["messages"] == []


def test_when_the_answering_ai_is_down_the_user_still_gets_the_matching_passages(client):
    headers = signup(client)
    upload(client, headers)
    client.generator.error = LLMUnavailableError("The language model is temporarily unavailable.")

    response = client.post("/api/chat", json={"message": "refund?"}, headers=headers)

    assert response.status_code == 200
    body = response.json()
    assert body["answer_type"] == "passages_only" and "temporarily unavailable" in body["answer"]
    assert "Refund requests must be filed within 14 days" in body["answer"]  # the passage itself
    assert [(s["document_name"], s["page"]) for s in body["sources"]] == [("terms.pdf", 1)]
    assert len(client.get("/api/chat/history", headers=headers).json()["messages"]) == 2  # still saved


def test_when_the_chat_ai_is_down_and_there_is_nothing_to_show_the_error_is_a_503(client):
    client.planner.route = "greeting"  # small talk has no passages to fall back on
    client.responder.error = LLMUnavailableError("The language model could not respond right now.")
    headers = signup(client)

    response = client.post("/api/chat", json={"message": "hello"}, headers=headers)

    assert response.status_code == 503 and response.json()["error"]["code"] == "llm_unavailable"
    assert client.get("/api/chat/history", headers=headers).json()["messages"] == []  # nothing saved


def test_chat_is_rate_limited_per_user(make_client):
    client = make_client(rate_limit_per_minute=2)
    headers = signup(client)
    codes = [client.post("/api/chat", json={"message": "hi"}, headers=headers).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


# ---------------------------------------------------------------- streaming
def test_stream_reports_progress_then_the_checked_answer_then_done(client):
    headers = signup(client)
    upload(client, headers)

    response = client.post("/api/chat/stream", json={"message": "refund deadline?"}, headers=headers)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(response.text)
    assert [name for name, _ in events] == ["step", "step", "step", "answer", "done"]
    assert [d["step"] for n, d in events if n == "step"] == [
        "plan_question", "retrieve_documents", "answer_from_documents",
    ]
    assert events[0][1]["label"] == "Understanding your question"
    answer = events[3][1]
    assert answer["answer_type"] == "document_grounded" and answer["sources"][0]["page"] == 1

    assert len(client.get("/api/chat/history", headers=headers).json()["messages"]) == 2  # saved too


def test_stream_for_a_greeting_skips_the_search_step(client):
    client.planner.route = "greeting"
    headers = signup(client)
    events = parse_sse(client.post("/api/chat/stream", json={"message": "hi"}, headers=headers).text)
    assert [d["step"] for n, d in events if n == "step"] == ["plan_question", "answer_generally"]


def test_stream_failure_arrives_as_an_error_event_and_saves_nothing(client):
    client.planner.route = "greeting"
    client.responder.error = LLMUnavailableError("The language model is down.")
    headers = signup(client)

    response = client.post("/api/chat/stream", json={"message": "hello"}, headers=headers)

    assert response.status_code == 200  # the stream had already started
    events = parse_sse(response.text)
    assert events[-1] == ("error", {"code": "llm_unavailable", "message": "The language model is down."})
    assert "answer" not in [name for name, _ in events]
    assert client.get("/api/chat/history", headers=headers).json()["messages"] == []


def test_stream_with_the_answering_ai_down_still_ends_with_the_passages(client):
    headers = signup(client)
    upload(client, headers)
    client.generator.error = LLMUnavailableError("The language model is down.")

    events = parse_sse(client.post("/api/chat/stream", json={"message": "refund?"}, headers=headers).text)

    assert [name for name, _ in events][-2:] == ["answer", "done"]
    assert events[-2][1]["answer_type"] == "passages_only"


def test_stream_needs_a_login(client):
    assert client.post("/api/chat/stream", json={"message": "hi"}).status_code == 401
