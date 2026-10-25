"""The whole chat flow on a real database: only the AI parts are faked."""

from app.answering.schemas import RagAnswer
from app.answering.service import AnswerService
from app.repositories.chunks import ChunkRepository
from app.repositories.documents import DocumentRepository
from app.retrieval.service import RetrievalService
from app.workflow.graph import build_chat_workflow
from app.workflow.nodes import NO_DOCUMENTS_MESSAGE
from app.workflow.planner import QuestionPlan
from tests.fakes import FakeReranker
from tests.helpers import store_document

TEXT = "Refund requests must be filed within 14 days of purchase."


class Planner:
    def __init__(self, route="document", search_question="refund requests filed within 14 days"):
        self.plan_result = QuestionPlan(route=route, search_question=search_question, reason="x")

    def plan(self, question, history):
        return self.plan_result


class Generator:
    def generate(self, question, chunks):
        return RagAnswer(answer="Within 14 days [1].", sufficient_evidence=True, used_source_ids=[1])


class Responder:
    def respond(self, question, history):
        return "Hi!"


def workflow(db_pool, embedder, planner=None):
    retrieval = RetrievalService(ChunkRepository(db_pool), embedder, FakeReranker())
    return build_chat_workflow(
        planner or Planner(), retrieval, AnswerService(Generator()), Responder(),
        DocumentRepository(db_pool).count_ready,
    )


def test_a_question_travels_from_the_database_to_a_cited_answer(db_pool, embedder):
    store_document(db_pool, embedder, "alice-owner", "terms.pdf", [TEXT, "Warranty lasts one year."])
    result = workflow(db_pool, embedder).ask("alice-owner", "refund deadline?")

    assert result.answer_type == "document_grounded" and result.route == "document"
    assert [(s.document_name, s.page) for s in result.sources] == [("terms.pdf", 1)]


def test_another_users_documents_are_never_used(db_pool, embedder):
    store_document(db_pool, embedder, "alice-owner", "terms.pdf", [TEXT])
    result = workflow(db_pool, embedder).ask("bob-owner", "refund deadline?")
    assert result.answer == NO_DOCUMENTS_MESSAGE and result.sources == []


def test_documents_still_processing_do_not_count(db_pool, embedder):
    store_document(db_pool, embedder, "alice-owner", "terms.pdf", [TEXT], ready=False)
    result = workflow(db_pool, embedder).ask("alice-owner", "refund deadline?")
    assert result.answer == NO_DOCUMENTS_MESSAGE


def test_greeting_works_with_no_documents_at_all(db_pool, embedder):
    result = workflow(db_pool, embedder, Planner("greeting")).ask("alice-owner", "hello")
    assert result.answer == "Hi!" and result.answer_type == "general"
