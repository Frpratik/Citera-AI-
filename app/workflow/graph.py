"""Builds the LangGraph workflow.

        START
          |
     plan_question  (route + rewritten question)
       |        \\
  document/      greeting / general_knowledge
  uncertain            |
       |               |
  retrieve_documents   |
    |         \\        |
  chunks or    uncertain + nothing found
  document        |    |
    |             |    |
  answer_from_documents      answer_generally
    |                              |
   END                            END

Why LangGraph and not one big function or a free-roaming agent? We decide the allowed
steps and forks ourselves (plain, testable Python), and the AI only works INSIDE a step.
For an assistant that must not invent document facts, predictable beats autonomous.
"""

from collections.abc import Callable, Iterator

from langgraph.graph import END, START, StateGraph

from app.answering.schemas import ChatResult
from app.answering.service import AnswerService
from app.observability.metrics import Metrics
from app.observability.tracing import span
from app.retrieval.service import RetrievalService
from app.workflow.nodes import WorkflowNodes
from app.workflow.planner import QuestionPlanner
from app.workflow.responder import GeneralResponder
from app.workflow.routing import route_after_plan, route_after_retrieval
from app.workflow.state import GraphState


class ChatWorkflow:
    """The compiled graph with a plain method, so callers never touch LangGraph."""

    def __init__(self, compiled_graph):
        self._graph = compiled_graph

    def ask(self, owner_id: str, question: str, history: list[dict] | None = None) -> ChatResult:
        final = self._graph.invoke(
            {"owner_id": owner_id, "question": question, "history": history or []}
        )
        return ChatResult(**final["result"].model_dump(), route=final["route"])

    def stream(
        self, owner_id: str, question: str, history: list[dict] | None = None
    ) -> Iterator[tuple[str, object]]:
        """Same as `ask`, but reports progress: ("step", node_name) after each finished
        step, then ("result", ChatResult) at the end."""
        state: dict = {}
        inputs = {"owner_id": owner_id, "question": question, "history": history or []}
        for update in self._graph.stream(inputs, stream_mode="updates"):
            for node_name, changes in update.items():
                state.update(changes)
                yield "step", node_name
        yield "result", ChatResult(**state["result"].model_dump(), route=state["route"])


def _timed(name: str, step: Callable[[GraphState], dict], metrics: Metrics | None):
    """The same step, but timed: the time goes into the question's trace and the metrics."""

    def wrapper(state: GraphState) -> dict:
        with span(name, metrics):
            return step(state)

    return wrapper


def build_chat_workflow(
    planner: QuestionPlanner,
    retrieval: RetrievalService,
    answerer: AnswerService,
    responder: GeneralResponder,
    count_ready_documents: Callable[[str], int],
    metrics: Metrics | None = None,
) -> ChatWorkflow:
    nodes = WorkflowNodes(planner, retrieval, answerer, responder, count_ready_documents)

    graph = StateGraph(GraphState)

    # Nodes: the steps.
    graph.add_node("plan_question", _timed("plan_question", nodes.plan_question, metrics))
    graph.add_node("retrieve_documents", _timed("retrieve_documents", nodes.retrieve_documents, metrics))
    graph.add_node("answer_from_documents", _timed("answer_from_documents", nodes.answer_from_documents, metrics))
    graph.add_node("answer_generally", _timed("answer_generally", nodes.answer_generally, metrics))

    # Edges: the allowed moves. Conditional edges ask a routing function which way to go.
    graph.add_edge(START, "plan_question")
    graph.add_conditional_edges(
        "plan_question",
        route_after_plan,
        {"retrieve": "retrieve_documents", "general": "answer_generally"},
    )
    graph.add_conditional_edges(
        "retrieve_documents",
        route_after_retrieval,
        {"answer": "answer_from_documents", "general": "answer_generally"},
    )
    graph.add_edge("answer_from_documents", END)
    graph.add_edge("answer_generally", END)

    return ChatWorkflow(graph.compile())
