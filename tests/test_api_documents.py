from tests.api_helpers import signup, upload
from tests.pdf_factory import make_pdf


def test_upload_returns_at_once_then_the_worker_makes_it_ready(client):
    headers = signup(client)
    created = upload(client, headers, process=False)

    assert created.status_code == 202  # Accepted: saved and queued, not finished
    doc = created.json()
    assert doc["status"] == "processing" and doc["chunk_count"] is None
    assert "owner_id" not in doc
    assert client.get(f"/api/documents/{doc['id']}", headers=headers).json()["status"] == "processing"

    client.services.worker.drain()  # what the background worker does

    ready = client.get(f"/api/documents/{doc['id']}", headers=headers).json()
    assert ready["status"] == "ready" and ready["chunk_count"] == 1 and ready["filename"] == "terms.pdf"
    listed = client.get("/api/documents", headers=headers).json()["documents"]
    assert [d["id"] for d in listed] == [doc["id"]]


def test_uploading_needs_a_login(client):
    response = client.post("/api/documents", files={"file": ("a.pdf", make_pdf(["x"]), "application/pdf")})
    assert response.status_code == 401


def test_another_user_cannot_see_get_or_delete_my_document(client):
    alice, bob = signup(client, "alice@example.com"), signup(client, "bob@example.com")
    doc_id = upload(client, alice).json()["id"]

    assert client.get("/api/documents", headers=bob).json()["documents"] == []
    assert client.get(f"/api/documents/{doc_id}", headers=bob).status_code == 404
    assert client.delete(f"/api/documents/{doc_id}", headers=bob).status_code == 404
    assert client.get(f"/api/documents/{doc_id}", headers=alice).status_code == 200  # still mine


def test_bad_uploads_are_rejected_at_once_with_a_clear_reason(client):
    headers = signup(client)
    wrong_name = upload(client, headers, name="notes.txt")
    assert wrong_name.status_code == 400 and wrong_name.json()["error"]["code"] == "invalid_upload"

    fake = client.post(
        "/api/documents", files={"file": ("fake.pdf", b"MZ not a pdf", "application/pdf")}, headers=headers
    )
    assert fake.status_code == 400
    assert client.get("/api/documents", headers=headers).json()["documents"] == []  # nothing left behind


def test_a_pdf_without_text_ends_as_failed_with_a_message(client):
    headers = signup(client)
    doc_id = upload(client, headers, name="scan.pdf", pages=("", "")).json()["id"]
    doc = client.get(f"/api/documents/{doc_id}", headers=headers).json()
    assert doc["status"] == "failed" and "No text" in doc["error"]


def test_a_failed_document_can_be_reprocessed_but_only_its_owner_and_only_if_failed(client):
    alice, bob = signup(client, "alice@example.com"), signup(client, "bob@example.com")
    failed_id = upload(client, alice, name="scan.pdf", pages=("", "")).json()["id"]
    ready_id = upload(client, alice).json()["id"]

    assert client.post(f"/api/documents/{failed_id}/reprocess", headers=bob).status_code == 404
    refused = client.post(f"/api/documents/{ready_id}/reprocess", headers=alice)
    assert refused.status_code == 400 and refused.json()["error"]["code"] == "invalid_request"

    again = client.post(f"/api/documents/{failed_id}/reprocess", headers=alice)
    assert again.status_code == 202 and again.json()["status"] == "processing"
    assert again.json()["error"] is None


def test_oversized_uploads_are_turned_away_before_being_read(make_client):
    client = make_client(max_upload_mb=1)
    headers = signup(client)
    big = b"%PDF-" + b"x" * (2 * 1024 * 1024)
    response = client.post(
        "/api/documents", files={"file": ("big.pdf", big, "application/pdf")}, headers=headers
    )
    assert response.status_code == 413 and response.json()["error"]["code"] == "file_too_large"


def test_a_malformed_document_id_is_rejected(client):
    headers = signup(client)
    assert client.get("/api/documents/not-a-uuid", headers=headers).status_code == 422


def test_delete_removes_the_document_its_chunks_its_job_and_its_file(client, tmp_path):
    headers = signup(client)
    doc_id = upload(client, headers).json()["id"]
    saved_file = tmp_path / "uploads" / f"{doc_id}.pdf"
    assert saved_file.exists()

    assert client.delete(f"/api/documents/{doc_id}", headers=headers).json() == {"id": doc_id, "deleted": True}

    assert client.get("/api/documents", headers=headers).json()["documents"] == []
    assert not saved_file.exists()
    with client.services.pool.connection() as conn:
        assert conn.execute("SELECT count(*) AS n FROM chunks").fetchone()["n"] == 0
        assert conn.execute("SELECT count(*) AS n FROM ingest_jobs").fetchone()["n"] == 0


def test_uploads_are_rate_limited_per_user(make_client):
    client = make_client(rate_limit_per_minute=2)
    headers = signup(client)
    codes = [upload(client, headers, process=False).status_code for _ in range(3)]
    assert codes == [202, 202, 429]
