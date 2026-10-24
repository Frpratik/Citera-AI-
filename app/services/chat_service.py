"""Chat service: runs the LangGraph workflow and keeps the chat history."""

import logging

from app.repositories.document_repository import DocumentRepository
from app.schemas.chat import ChatResponse, SourceOut

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self, workflow, repository: DocumentRepository, history_turns: int):
        self._workflow = workflow  # compiled LangGraph graph
        self._repository = repository
        self._history_messages = history_turns * 2  # a "turn" = user message + reply

    def answer(self, owner_id: str, question: str) -> ChatResponse:
        history = [
            {"role": item["role"], "content": item["content"]}
            for item in self._repository.recent_messages(owner_id, self._history_messages)
        ]

        final_state = self._workflow.invoke(
            {"question": question, "owner_id": owner_id, "history": history}
        )

        response = ChatResponse(
            answer=final_state["answer"],
            answer_type=final_state["answer_type"],
            sources=[SourceOut(**source) for source in final_state.get("sources", [])],
            evidence_status=final_state.get("evidence_status", "not_applicable"),
            route=final_state.get("route", "unknown"),
            warnings=final_state.get("warnings", []),
        )

        # Save only after success, so failed requests do not pollute the history.
        self._repository.add_message(owner_id, "user", question)
        self._repository.add_message(owner_id, "assistant", response.answer, response.answer_type)
        logger.info("Chat answered: route=%s type=%s", response.route, response.answer_type)
        return response

    def history(self, owner_id: str, limit: int = 100) -> list[dict]:
        return self._repository.recent_messages(owner_id, limit)

    def clear_history(self, owner_id: str) -> int:
        return self._repository.clear_messages(owner_id)
