"""API-level tests, including the end-to-end flow:
upload PDF -> retrieve -> answer -> sources -> delete."""

from tests.conftest import CLIENT_A, CLIENT_B, headers
from tests.helpers import ATTENDANCE_TEXT, FEES_TEXT, make_pdf


def _upload(client, client_id=CLIENT_A, name="attendance.pdf", pages=None, data=None):
    payload = data if data is not None else make_pdf(pages or [ATTENDANCE_TEXT])
    return client.post(
        "/api/documents/upload",
        files={"file": (name, payload, "application/pdf")},
        headers=headers(client_id),
    )


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# ------------------------------------------------------------------ end to end
def test_end_to_end_upload_ask_sources_delete(client):
    # 1. Upload two documents.
    upload = _upload(client, name="attendance.pdf", pages=[ATTENDANCE_TEXT])
    assert upload.status_code == 201
    document = upload.json()
    assert document["status"] == "ready"
    assert document["chunk_count"] >= 1
    assert _upload(client, name="fees.pdf", pages=[FEES_TEXT]).status_code == 201

    listed = client.get("/api/documents", headers=headers()).json()["documents"]
    assert {d["filename"] for d in listed} == {"attendance.pdf", "fees.pdf"}

    # 2. Ask a document question: grounded answer + source attribution.
    chat = client.post("/api/chat", json={"message": "What is the minimum attendance requirement?"},
                       headers=headers())
    assert chat.status_code == 200
    body = chat.json()
    assert body["answer_type"] == "document_grounded"
    assert "75%" in body["answer"]
    assert body["sources"][0]["document_name"] == "attendance.pdf"
    assert body["sources"][0]["page"] == 1
    assert body["sources"][0]["document_id"] == document["id"]

    # 3. General chat still works and has no sources.
    general = client.post("/api/chat", json={"message": "Hi"}, headers=headers()).json()
    assert general["answer_type"] == "general"
    assert general["sources"] == []

    # 4. Delete the document: it can no longer be used as a source.
    assert client.delete(f"/api/documents/{document['id']}", headers=headers()).status_code == 200
    after = client.post("/api/chat", json={"message": "What is the minimum attendance requirement?"},
                        headers=headers()).json()
    assert after["answer_type"] == "insufficient_evidence"
    assert after["sources"] == []


# ------------------------------------------------------------------ validation and errors
def test_missing_client_id_is_rejected(client):
    response = client.get("/api/documents")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "missing_client_id"


def test_invalid_client_id_is_rejected(client):
    response = client.get("/api/documents", headers={"X-Client-ID": "../../etc"})
    assert response.status_code == 400


def test_non_pdf_extension_is_rejected(client):
    response = _upload(client, name="notes.txt", data=b"hello")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_upload"


def test_fake_pdf_content_is_rejected(client):
    response = _upload(client, name="fake.pdf", data=b"not really a pdf")
    assert response.status_code == 400


def test_oversized_upload_is_rejected(client):
    big = b"%PDF-1.4\n" + b"x" * (16 * 1024 * 1024)  # default limit is 15 MB
    response = _upload(client, name="big.pdf", data=big)
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "file_too_large"


def test_scanned_pdf_is_returned_as_failed_document(client):
    response = _upload(client, name="blank.pdf", pages=["   "])
    assert response.status_code == 201
    assert response.json()["status"] == "failed"
    assert response.json()["error"]


def test_empty_message_is_rejected(client):
    response = client.post("/api/chat", json={"message": "   "}, headers=headers())
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_too_long_message_is_rejected(client):
    response = client.post("/api/chat", json={"message": "x" * 2001}, headers=headers())
    assert response.status_code == 422


def test_bad_document_id_format_is_rejected(client):
    response = client.delete("/api/documents/not-a-valid-id", headers=headers())
    assert response.status_code == 422


def test_unknown_document_returns_404(client):
    response = client.get(f"/api/documents/{'f' * 32}/status", headers=headers())
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "document_not_found"


# ------------------------------------------------------------------ isolation
def test_other_clients_cannot_see_or_delete_my_documents(client):
    document = _upload(client, CLIENT_A).json()

    assert client.get("/api/documents", headers=headers(CLIENT_B)).json()["documents"] == []
    assert client.delete(f"/api/documents/{document['id']}", headers=headers(CLIENT_B)).status_code == 404
    assert client.get(f"/api/documents/{document['id']}/status", headers=headers(CLIENT_B)).status_code == 404

    answer = client.post("/api/chat", json={"message": "What is the minimum attendance?"},
                         headers=headers(CLIENT_B)).json()
    assert answer["sources"] == []
    assert answer["answer_type"] == "insufficient_evidence"


def test_reprocess_and_status(client):
    document = _upload(client).json()
    status = client.get(f"/api/documents/{document['id']}/status", headers=headers()).json()
    assert status["status"] == "ready"
    again = client.post(f"/api/documents/{document['id']}/reprocess", headers=headers())
    assert again.status_code == 200
    assert again.json()["status"] == "ready"


# ------------------------------------------------------------------ chat history + errors
def test_history_is_saved_and_can_be_cleared(client):
    client.post("/api/chat", json={"message": "Hi"}, headers=headers())
    messages = client.get("/api/chat/history", headers=headers()).json()["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant"]

    assert client.delete("/api/chat/history", headers=headers()).json()["cleared"] == 2
    assert client.get("/api/chat/history", headers=headers()).json()["messages"] == []


def test_history_is_private_per_client(client):
    client.post("/api/chat", json={"message": "Hi"}, headers=headers(CLIENT_A))
    assert client.get("/api/chat/history", headers=headers(CLIENT_B)).json()["messages"] == []


def test_llm_not_configured_returns_503(client, services):
    from app.core.errors import LLMUnavailableError

    def boom(*_args, **_kwargs):
        raise LLMUnavailableError("The language model is not configured.")

    services.llm.classify = boom
    response = client.post("/api/chat", json={"message": "Hi"}, headers=headers())
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "llm_unavailable"


def test_rate_limit(client, services):
    from app.api.deps import RateLimiter

    services.rate_limiter = RateLimiter(2)
    codes = [
        client.post("/api/chat", json={"message": "Hi"}, headers=headers()).status_code
        for _ in range(3)
    ]
    assert codes == [200, 200, 429]
