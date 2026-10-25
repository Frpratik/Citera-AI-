from typing import TypedDict

from app.answering.schemas import AnswerResult
from app.repositories.chunks import SearchHit


class GraphState(TypedDict, total=False):
    """The one dictionary passed through every step. Each step returns only what it changes.

    `total=False` because early steps run before later keys (like `result`) exist.
    """

    # inputs
    owner_id: str
    question: str  # the user's latest message, exactly as typed
    history: list[dict]  # recent [{"role": "user" | "assistant", "content": "..."}]

    # set by plan_question
    route: str  # greeting | general_knowledge | document | uncertain
    search_question: str  # the question rewritten to stand alone

    # set by retrieve_documents
    chunks: list[SearchHit]
    document_count: int  # how many ready documents the user has

    # final output
    result: AnswerResult
