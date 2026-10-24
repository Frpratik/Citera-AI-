"""Graph state.

`TypedDict` is a plain Python dict whose keys and value types are declared, so
editors and type checkers can catch typos like state["anwser"]. At runtime it is
still just a dict. LangGraph passes this one dict through every node; each node
reads what it needs and returns only the keys it wants to change, and LangGraph
merges those updates into the state.

`total=False` means keys are optional, because early nodes run before later keys
(e.g. `answer`) exist.
"""

from typing import TypedDict

from app.services.retrieval_service import RetrievedChunk


class GraphState(TypedDict, total=False):
    # --- inputs (set by ChatService) ---
    question: str  # the user's latest message
    owner_id: str  # whose documents we may search
    history: list[dict]  # recent [{"role": "user"|"assistant", "content": "..."}]

    # --- set by classify_question ---
    route: str  # greeting | general_knowledge | document | college_document | uncertain
    route_reason: str

    # --- set by retrieve_documents ---
    chunks: list[RetrievedChunk]  # relevant chunks (already above the score threshold)
    document_count: int  # how many ready documents the user has

    # --- set by generate_rag_answer (raw LLM output, not yet verified) ---
    sufficient_evidence: bool
    used_source_ids: list[int]
    conflict_detected: bool

    # --- final output (set by validate_answer / generate_general_answer) ---
    answer: str
    answer_type: str  # document_grounded | general | insufficient_evidence | conflicting_information
    sources: list[dict]
    evidence_status: str  # strong | moderate | none | not_applicable
    warnings: list[str]
