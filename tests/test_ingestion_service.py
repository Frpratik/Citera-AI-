import io

import pytest

from app.core.config import Settings
from app.core.errors import InvalidUploadError, NotFoundError
from app.repositories.chunks import ChunkRepository
from app.repositories.documents import DocumentRepository
from app.repositories.jobs import JobRepository
from app.services.ingestion_service import IngestionService
from tests.pdf_factory import make_pdf


@pytest.fixture
def service(db_pool, embedder, tmp_path):
    settings = Settings(data_dir=tmp_path, _env_file=None)
    return IngestionService(
        DocumentRepository(db_pool), ChunkRepository(db_pool), JobRepository(db_pool), embedder, settings
    )


def test_ingest_now_makes_a_document_ready_and_searchable(service, db_pool, embedder, tmp_path):
    pdf = make_pdf(["Library fines are charged per day.", "Hostel curfew is at ten pm."])
    record = service.ingest_now("alice-owner", "rules.pdf", io.BytesIO(pdf))

    assert record.status == "ready" and record.page_count == 2 and record.chunk_count == 2
    assert (tmp_path / "uploads" / f"{record.id}.pdf").exists()

    query = embedder.embed_query("hostel curfew")
    hits = ChunkRepository(db_pool).vector_search("alice-owner", query, 1)
    assert hits[0].page == 2 and hits[0].document_name == "rules.pdf"


def test_accept_saves_the_file_and_queues_a_job_but_does_not_process(service, db_pool, tmp_path):
    record = service.accept("alice-owner", "rules.pdf", io.BytesIO(make_pdf(["Some rule text."])))

    assert record.status == "processing" and record.chunk_count is None
    assert (tmp_path / "uploads" / f"{record.id}.pdf").exists()
    assert JobRepository(db_pool).latest_for_document(record.id)["status"] == "queued"
    assert ChunkRepository(db_pool).vector_search("alice-owner", [1.0] + [0.0] * 383, 5) == []


def test_get_list_and_delete_only_work_for_the_owner(service, db_pool, tmp_path):
    record = service.ingest_now("alice-owner", "rules.pdf", io.BytesIO(make_pdf(["Some rule text."])))
    saved_file = tmp_path / "uploads" / f"{record.id}.pdf"

    assert [d.id for d in service.list("alice-owner")] == [record.id]
    assert service.list("bob-owner") == []
    with pytest.raises(NotFoundError):
        service.get("bob-owner", record.id)
    with pytest.raises(NotFoundError):
        service.delete("bob-owner", record.id)
    assert saved_file.exists()  # Bob's failed delete changed nothing

    service.delete("alice-owner", record.id)
    assert service.list("alice-owner") == [] and not saved_file.exists()


def test_rejected_upload_leaves_no_document_behind(service, db_pool):
    with pytest.raises(InvalidUploadError):
        service.accept("alice-owner", "fake.pdf", io.BytesIO(b"this is not a pdf"))
    assert DocumentRepository(db_pool).list_for_owner("alice-owner") == []


def test_pdf_without_text_is_marked_failed_with_a_message(service, db_pool, embedder):
    record = service.ingest_now("alice-owner", "scan.pdf", io.BytesIO(make_pdf(["", ""])))
    assert record.status == "failed" and "No text" in record.error
    query = embedder.embed_query("anything")
    assert ChunkRepository(db_pool).vector_search("alice-owner", query, 5) == []
