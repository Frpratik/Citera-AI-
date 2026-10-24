"""Routing functions used by LangGraph conditional edges.

A conditional edge is a function that looks at the current state and returns a
label; LangGraph maps that label to the next node. This is how the graph decides
"retrieve documents" vs "answer directly".

Routing strategy (layered, not just keywords):
  1. An LLM classifier with structured output picks one of four routes.
  2. 'uncertain' goes to retrieval, because wrongly answering a fee/date question
     from general knowledge is a hallucination risk, while retrieval can safely
     come back empty.
  3. If an 'uncertain' question finds no relevant document text, it falls back to a
     general answer that clearly says it did not come from the documents.

Limitation: any classifier makes mistakes, especially on very short or follow-up
messages ("and for hostel?"). We bias toward the safe path and rely on the
relevance threshold and answer validation as a second line of defence.
"""

from app.graph.state import GraphState

RETRIEVAL_ROUTES = {"college_document", "uncertain"}


def route_after_classification(state: GraphState) -> str:
    """After classify_question: go retrieve, or answer directly."""
    return "retrieve" if state.get("route") in RETRIEVAL_ROUTES else "general"


def route_after_retrieval(state: GraphState) -> str:
    """After retrieve_documents: grounded answer, or general fallback for uncertain cases."""
    if state.get("route") == "uncertain" and not state.get("chunks"):
        return "general"
    return "rag"
