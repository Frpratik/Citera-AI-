"""Chat endpoints."""

from fastapi import APIRouter, Depends

from app.api.deps import Services, enforce_rate_limit, get_client_id, get_services
from app.schemas.chat import ChatRequest, ChatResponse, ClearHistoryOut, HistoryMessage, HistoryOut

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat(
    body: ChatRequest,
    client_id: str = Depends(enforce_rate_limit),
    services: Services = Depends(get_services),
):
    """Ask a question. The LangGraph workflow decides whether to search the documents."""
    return services.chat.answer(client_id, body.message)


@router.get("/history", response_model=HistoryOut)
def get_history(
    client_id: str = Depends(get_client_id), services: Services = Depends(get_services)
):
    return HistoryOut(messages=[HistoryMessage(**m) for m in services.chat.history(client_id)])


@router.delete("/history", response_model=ClearHistoryOut)
def clear_history(
    client_id: str = Depends(get_client_id), services: Services = Depends(get_services)
):
    return ClearHistoryOut(cleared=services.chat.clear_history(client_id))
