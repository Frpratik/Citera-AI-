from typing import Annotated

from fastapi import APIRouter, File, Path, UploadFile

from app.api.deps import CurrentUser, RateLimitedUser, ServicesDep
from app.api.schemas import DeleteOut, DocumentListOut, DocumentOut

router = APIRouter(prefix="/api/documents", tags=["documents"])

# Ids are UUIDs; rejecting anything else early keeps junk out of queries and paths.
DocumentId = Annotated[
    str,
    Path(
        pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
        description="Document id",
    ),
]


@router.post("", response_model=DocumentOut, status_code=202)
def upload_document(file: Annotated[UploadFile, File()], user: RateLimitedUser, services: ServicesDep):
    """Upload a PDF. It is saved and queued; the reply (202 Accepted) comes straight away with
    status "processing". Poll GET /api/documents/{id} until the status is "ready" or
    "failed" (a failed document carries an `error` message explaining why)."""
    return DocumentOut.model_validate(services.ingestion.accept(user.id, file.filename, file.file))


@router.post("/{document_id}/reprocess", response_model=DocumentOut, status_code=202)
def reprocess_document(document_id: DocumentId, user: RateLimitedUser, services: ServicesDep):
    """Try a failed document again."""
    return DocumentOut.model_validate(services.ingestion.reprocess(user.id, document_id))


@router.get("", response_model=DocumentListOut)
def list_documents(user: CurrentUser, services: ServicesDep):
    records = services.ingestion.list(user.id)
    return DocumentListOut(documents=[DocumentOut.model_validate(r) for r in records])


@router.get("/{document_id}", response_model=DocumentOut)
def get_document(document_id: DocumentId, user: CurrentUser, services: ServicesDep):
    return DocumentOut.model_validate(services.ingestion.get(user.id, document_id))


@router.delete("/{document_id}", response_model=DeleteOut)
def delete_document(document_id: DocumentId, user: CurrentUser, services: ServicesDep):
    services.ingestion.delete(user.id, document_id)
    return DeleteOut(id=document_id, deleted=True)
