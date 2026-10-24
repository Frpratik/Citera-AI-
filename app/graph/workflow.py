"""Builds the LangGraph workflow.

        START
          |
   classify_question
      |            \\
  (document/        (greeting/general)
   uncertain)            |
      |                  |
 retrieve_documents      |
   |          \\          |
 (chunks     (uncertain + no chunks)
  found or        |
  document)       |
   |              |
 generate_rag_answer   generate_general_answer
   |                          |
 validate_answer              |
   |                          |
  END                        END

Why LangGraph instead of a plain LangChain chain or an agent?
  * A chain is a fixed pipeline: A -> B -> C. It has no built-in branching.
  * An agent lets the LLM decide freely which tools to call and when to stop, which
    is flexible but hard to predict and test.
  * LangGraph lets *us* define the allowed steps and branches explicitly (nodes and
    edges) over a shared typed state. The LLM is used inside nodes, but the control
    flow is plain, testable Python. For a support-style assistant that must not
    hallucinate document facts, predictability matters more than autonomy.
"""

from langgraph.graph import END, START, StateGraph

from app.core.config import Settings
from app.graph.nodes import GraphNodes
from app.graph.routing import route_after_classification, route_after_retrieval
from app.graph.state import GraphState
from app.services.llm_service import LLMService
from app.services.retrieval_service import RetrievalService


def build_workflow(llm: LLMService, retrieval: RetrievalService, settings: Settings):
    """Create and compile the question-answering graph."""
    nodes = GraphNodes(llm, retrieval, settings)

    graph = StateGraph(GraphState)

    # Nodes: the steps.
    graph.add_node("classify_question", nodes.classify_question)
    graph.add_node("retrieve_documents", nodes.retrieve_documents)
    graph.add_node("generate_rag_answer", nodes.generate_rag_answer)
    graph.add_node("generate_general_answer", nodes.generate_general_answer)
    graph.add_node("validate_answer", nodes.validate_answer)

    # Edges: the allowed transitions.
    graph.add_edge(START, "classify_question")
    graph.add_conditional_edges(
        "classify_question",
        route_after_classification,
        {"retrieve": "retrieve_documents", "general": "generate_general_answer"},
    )
    graph.add_conditional_edges(
        "retrieve_documents",
        route_after_retrieval,
        {"rag": "generate_rag_answer", "general": "generate_general_answer"},
    )
    graph.add_edge("generate_rag_answer", "validate_answer")
    graph.add_edge("validate_answer", END)
    graph.add_edge("generate_general_answer", END)

    return graph.compile()
