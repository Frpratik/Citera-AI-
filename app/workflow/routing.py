"""Decisions for the graph's forks. Each looks at the state and returns a label."""

from app.workflow.state import GraphState

# Routes that need the user's documents. "uncertain" goes here too: guessing from general
# knowledge about a fee or a date is a made-up answer, while searching can safely find nothing.
DOCUMENT_ROUTES = {"document", "uncertain"}


def route_after_plan(state: GraphState) -> str:
    return "retrieve" if state["route"] in DOCUMENT_ROUTES else "general"


def route_after_retrieval(state: GraphState) -> str:
    # An unclear question that found nothing in the documents becomes a general reply
    # (clearly labelled). A clear document question with nothing found is answered
    # "not found" instead, and never from general knowledge.
    if state["route"] == "uncertain" and not state["chunks"]:
        return "general"
    return "answer"
