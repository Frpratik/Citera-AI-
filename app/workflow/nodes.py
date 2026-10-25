from collections.abc import Callable

from app.answering.schemas import AnswerResult
from app.answering.service import AnswerService
from app.retrieval.service import RetrievalService
from app.workflow.planner import QuestionPlanner
from app.workflow.responder import GeneralResponder
from app.workflow.state import GraphState

NO_DOCUMENTS_MESSAGE = (
    "I can't answer that yet because you haven't uploaded any ready documents. "
    "Upload a PDF and ask again."
)
GENERAL_FALLBACK_NOTE = (
    "(I couldn't find this in your uploaded documents, so this is a general answer.)\n\n"
)


class WorkflowNodes:
    """The steps of the graph. Each takes the state and returns only the keys it changes."""

    def __init__(
        self,
        planner: QuestionPlanner,
        retrieval: RetrievalService,
        answerer: AnswerService,
        responder: GeneralResponder,
        count_ready_documents: Callable[[str], int],
    ):
        self._planner = planner
        self._retrieval = retrieval
        self._answerer = answerer
        self._responder = responder
        self._count_ready_documents = count_ready_documents

    def plan_question(self, state: GraphState) -> dict:
        plan = self._planner.plan(state["question"], state.get("history", []))
        return {"route": plan.route, "search_question": plan.search_question}

    def retrieve_documents(self, state: GraphState) -> dict:
        owner_id = state["owner_id"]
        count = self._count_ready_documents(owner_id)
        # No ready documents: skip the search entirely.
        chunks = self._retrieval.retrieve(owner_id, state["search_question"]) if count else []
        return {"chunks": chunks, "document_count": count}

    def answer_from_documents(self, state: GraphState) -> dict:
        if state["document_count"] == 0:
            return {
                "result": AnswerResult(
                    answer=NO_DOCUMENTS_MESSAGE,
                    answer_type="insufficient_evidence",
                    evidence_status="none",
                )
            }
        return {"result": self._answerer.answer(state["search_question"], state["chunks"])}

    def answer_generally(self, state: GraphState) -> dict:
        text = self._responder.respond(state["question"], state.get("history", []))
        if state.get("route") == "uncertain":
            text = GENERAL_FALLBACK_NOTE + text
        return {
            "result": AnswerResult(
                answer=text.strip(), answer_type="general", evidence_status="not_applicable"
            )
        }
