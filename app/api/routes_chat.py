import json
import logging

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.answering.schemas import ChatResult
from app.api.deps import CurrentUser, RateLimitedUser, ServicesDep
from app.api.schemas import ChatRequest, ClearHistoryOut, HistoryMessage, HistoryOut
from app.core.errors import AppError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])

# Friendly labels for the progress events of the streaming endpoint.
STEP_LABELS = {
    "plan_question": "Understanding your question",
    "retrieve_documents": "Searching your documents",
    "answer_from_documents": "Writing the answer from your documents",
    "answer_generally": "Writing a reply",
}


def _sse(event: str, data: dict) -> str:
    """One Server-Sent Event: the browser reads these one by one as they arrive."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.post("", response_model=ChatResult)
def chat(body: ChatRequest, user: RateLimitedUser, services: ServicesDep):
    """Ask a question. The workflow decides whether to search the user's documents."""
    return services.chat.ask(user.id, body.message)


@router.post("/stream")
def chat_stream(body: ChatRequest, user: RateLimitedUser, services: ServicesDep):
    """Same as POST /api/chat, but as a stream of events: `step` while working, then the
    CHECKED `answer`, then `done`. (The answer arrives whole on purpose; see the notes.)"""

    def events():
        try:
            for name, data in services.chat.ask_stream(user.id, body.message):
                if name == "step":
                    data = {**data, "label": STEP_LABELS.get(data["step"], data["step"])}
                yield _sse(name, data)
        except AppError as exc:
            # The response has already started, so the status code can no longer change.
            yield _sse("error", {"code": exc.code, "message": exc.message})
        except Exception as exc:
            logger.error("Unhandled error in stream: %s", type(exc).__name__, exc_info=exc)
            yield _sse("error", {"code": "internal_error", "message": "Something went wrong on the server."})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/history", response_model=HistoryOut)
def get_history(user: CurrentUser, services: ServicesDep):
    return HistoryOut(messages=[HistoryMessage.model_validate(m) for m in services.chat.history(user.id)])


@router.delete("/history", response_model=ClearHistoryOut)
def clear_history(user: CurrentUser, services: ServicesDep):
    return ClearHistoryOut(cleared=services.chat.clear_history(user.id))
