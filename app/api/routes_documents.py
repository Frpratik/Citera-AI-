"""Document endpoints. Handlers stay thin: validate, call a service, return a model."""

from fastapi import APIRouter, Depends, File, Path, UploadFile

from app.api.deps import Services, enforce_rate_limit, get_client_id, get_services
from app.schemas.documents import DeleteOut, DocumentListOut, DocumentOut, DocumentStatusOut

router = APIRouter(prefix="/api/documents", tags=["documents"])

# Document ids are 32 hex characters; rejecting anything else early keeps junk out of paths.
DocumentId = Path(pattern=r"^[0-9a-f]{32}$", description="Document id")


@router.post("/upload", response_model=DocumentOut, status_code=201)
def upload_document(
    file: UploadFile = File(...),
    client_id: str = Depends(enforce_rate_limit),
    services: Services = Depends(get_services),
):
    """Upload a PDF and process it. If processing fails the document is returned with
    status "failed" and an `error` message, so it can be reprocessed or deleted."""
    record = services.documents.upload(client_id, file.filename, file.file)
    return DocumentOut.model_validate(record)


@router.get("", response_model=DocumentListOut)
def list_documents(
    client_id: str = Depends(get_client_id), services: Services = Depends(get_services)
):
    records = services.documents.list(client_id)
    return DocumentListOut(documents=[DocumentOut.model_validate(r) for r in records])


@router.get("/{document_id}/status", response_model=DocumentStatusOut)
def document_status(
    document_id: str = DocumentId,
    client_id: str = Depends(get_client_id),
    services: Services = Depends(get_services),
):
    record = services.documents.get(client_id, document_id)
    return DocumentStatusOut(id=record.id, status=record.status, error=record.error)


@router.post("/{document_id}/reprocess", response_model=DocumentOut)
def reprocess_document(
    document_id: str = DocumentId,
    client_id: str = Depends(enforce_rate_limit),
    services: Services = Depends(get_services),
):
    record = services.documents.reprocess(client_id, document_id)
    return DocumentOut.model_validate(record)


@router.delete("/{document_id}", response_model=DeleteOut)
def delete_document(
    document_id: str = DocumentId,
    client_id: str = Depends(get_client_id),
    services: Services = Depends(get_services),
):
    services.documents.delete(client_id, document_id)
    return DeleteOut(id=document_id, deleted=True)
