"""Embedding + FAISS integration (with fake embeddings): search, persistence, deletion, isolation."""

import io

import pytest

from tests.helpers import ATTENDANCE_TEXT, FEES_TEXT, make_pdf

OWNER_A = "owner-aaaaaaaa-01"
OWNER_B = "owner-bbbbbbbb-02"


def _upload(services, owner, name, pages):
    return services.documents.upload(owner, name, io.BytesIO(make_pdf(pages)))


def _retrieve(services, owner, question):
    return services.retrieval.retrieve(owner, question)


def test_retrieves_the_right_document_and_page(services):
    attendance = _upload(services, OWNER_A, "attendance.pdf", [ATTENDANCE_TEXT])
    _upload(services, OWNER_A, "fees.pdf", [FEES_TEXT])

    result = _retrieve(services, OWNER_A, "What is the minimum attendance requirement?")

    assert result.document_count == 2
    assert result.chunks
    best = result.chunks[0]
    assert best.document_id == attendance.id
    assert best.document_name == "attendance.pdf"
    assert best.page == 1
    assert "75%" in best.text


def test_unrelated_question_returns_no_relevant_chunks(services):
    _upload(services, OWNER_A, "attendance.pdf", [ATTENDANCE_TEXT])
    result = _retrieve(services, OWNER_A, "zebra giraffe savanna migration")
    assert result.chunks == []


def test_deleted_document_is_never_retrieved(services):
    doc = _upload(services, OWNER_A, "attendance.pdf", [ATTENDANCE_TEXT])
    assert _retrieve(services, OWNER_A, "minimum attendance").chunks

    services.documents.delete(OWNER_A, doc.id)

    assert _retrieve(services, OWNER_A, "minimum attendance").chunks == []
    assert not services.vector_store.index_exists(OWNER_A, doc.id)


def test_documents_are_isolated_between_owners(services):
    _upload(services, OWNER_A, "attendance.pdf", [ATTENDANCE_TEXT])
    result = _retrieve(services, OWNER_B, "minimum attendance")
    assert result.chunks == []
    assert result.document_count == 0


def test_index_survives_restart(services, settings):
    """A new service stack pointing at the same data folder can still search old documents."""
    from app.api.deps import build_services
    from tests.helpers import FakeEmbeddings, FakeLLM

    _upload(services, OWNER_A, "attendance.pdf", [ATTENDANCE_TEXT])

    restarted = build_services(settings, embeddings=FakeEmbeddings(), llm=FakeLLM())
    restarted.documents.recover_after_restart()

    assert _retrieve(restarted, OWNER_A, "minimum attendance").chunks


def test_recovery_marks_documents_with_missing_index_as_failed(services):
    doc = _upload(services, OWNER_A, "attendance.pdf", [ATTENDANCE_TEXT])
    services.vector_store.delete_index(OWNER_A, doc.id)  # simulate lost files

    services.documents.recover_after_restart()

    assert services.documents.get(OWNER_A, doc.id).status == "failed"
    # ...and Reprocess repairs it.
    assert services.documents.reprocess(OWNER_A, doc.id).status == "ready"


def test_failed_pdf_is_recorded_and_can_be_deleted(services):
    doc = _upload(services, OWNER_A, "scanned.pdf", ["   "])  # valid PDF, no text
    assert doc.status == "failed"
    assert doc.error
    services.documents.delete(OWNER_A, doc.id)
    with pytest.raises(Exception):
        services.documents.get(OWNER_A, doc.id)
