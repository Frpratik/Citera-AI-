from app.answering.schemas import AnswerResult, RagAnswer
from app.answering.service import NO_EVIDENCE_MESSAGE, AnswerService
from app.observability.metrics import Metrics
from app.observability.tracing import Trace, use_trace
from app.repositories.chunks import SearchHit
from app.workflow.graph import build_chat_workflow
from app.workflow.nodes import GENERAL_FALLBACK_NOTE, NO_DOCUMENTS_MESSAGE
from app.workflow.planner import QuestionPlan
from app.workflow.routing import route_after_plan, route_after_retrieval

CHUNK = SearchHit("c1", "d1", "terms.pdf", 3, "Refund requests must be filed within 14 days.", 0.95)
GROUNDED = AnswerResult(
    answer="Within 14 days [1].", answer_type="document_grounded", evidence_status="strong"
)


class FakePlanner:
    def __init__(self, route, search_question="rewritten question"):
        self.plan_result = QuestionPlan(route=route, search_question=search_question, reason="x")
        self.seen = None

    def plan(self, question, history):
        self.seen = (question, history)
        return self.plan_result


class FakeRetrieval:
    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = []

    def retrieve(self, owner_id, question):
        self.calls.append((owner_id, question))
        return self.chunks


class FakeAnswerer:
    def __init__(self):
        self.calls = []

    def answer(self, question, chunks):
        self.calls.append((question, chunks))
        return GROUNDED


class FakeResponder:
    def __init__(self):
        self.seen = None

    def respond(self, question, history):
        self.seen = (question, history)
        return "A general reply."


def build(route, chunks=(), documents=1, answerer=None, search_question="rewritten question"):
    planner = FakePlanner(route, search_question)
    retrieval = FakeRetrieval(list(chunks))
    answerer = answerer or FakeAnswerer()
    responder = FakeResponder()
    workflow = build_chat_workflow(planner, retrieval, answerer, responder, lambda owner: documents)
    return workflow, planner, retrieval, answerer, responder


# ---------------------------------------------------------------- the forks, as pure functions
def test_plan_routes_split_documents_from_general():
    assert route_after_plan({"route": "document"}) == "retrieve"
    assert route_after_plan({"route": "uncertain"}) == "retrieve"
    assert route_after_plan({"route": "greeting"}) == "general"
    assert route_after_plan({"route": "general_knowledge"}) == "general"


def test_only_an_unclear_question_with_nothing_found_falls_back_to_general():
    assert route_after_retrieval({"route": "uncertain", "chunks": []}) == "general"
    assert route_after_retrieval({"route": "uncertain", "chunks": [CHUNK]}) == "answer"
    assert route_after_retrieval({"route": "document", "chunks": []}) == "answer"


# ---------------------------------------------------------------- the whole graph
def test_greeting_skips_the_documents_entirely():
    workflow, _, retrieval, answerer, responder = build("greeting")
    result = workflow.ask("alice-owner", "hello", [{"role": "user", "content": "earlier"}])

    assert result.answer == "A general reply." and result.answer_type == "general"
    assert result.evidence_status == "not_applicable" and result.route == "greeting"
    assert retrieval.calls == [] and answerer.calls == []
    assert responder.seen == ("hello", [{"role": "user", "content": "earlier"}])


def test_document_question_searches_with_the_rewritten_question():
    workflow, planner, retrieval, answerer, _ = build(
        "document", chunks=[CHUNK], search_question="What is the refund deadline?"
    )
    result = workflow.ask("alice-owner", "and the deadline?", [])

    assert planner.seen == ("and the deadline?", [])
    assert retrieval.calls == [("alice-owner", "What is the refund deadline?")]
    assert answerer.calls == [("What is the refund deadline?", [CHUNK])]
    assert result.answer == "Within 14 days [1]." and result.route == "document"


def test_no_ready_documents_gives_a_clear_message_and_no_search():
    workflow, _, retrieval, answerer, _ = build("document", documents=0)
    result = workflow.ask("alice-owner", "what is the fee?")

    assert result.answer == NO_DOCUMENTS_MESSAGE
    assert result.answer_type == "insufficient_evidence"
    assert retrieval.calls == [] and answerer.calls == []


def test_document_question_with_nothing_found_says_not_found_and_never_calls_the_ai():
    class ExplodingGenerator:
        def generate(self, question, chunks):
            raise AssertionError("the AI must not be called without evidence")

    workflow, *_ = build("document", chunks=[], answerer=AnswerService(ExplodingGenerator()))
    result = workflow.ask("alice-owner", "what is the fee?")

    assert result.answer == NO_EVIDENCE_MESSAGE
    assert result.answer_type == "insufficient_evidence"


def test_unclear_question_with_nothing_found_becomes_a_labelled_general_reply():
    workflow, _, retrieval, answerer, _ = build("uncertain", chunks=[])
    result = workflow.ask("alice-owner", "tell me about it")

    assert len(retrieval.calls) == 1  # it did search first
    assert answerer.calls == []
    assert result.answer.startswith(GENERAL_FALLBACK_NOTE.strip())
    assert result.answer_type == "general" and result.route == "uncertain"


def test_unclear_question_that_finds_chunks_is_answered_from_them():
    workflow, _, _, answerer, _ = build("uncertain", chunks=[CHUNK])
    result = workflow.ask("alice-owner", "refund?")
    assert answerer.calls and result.answer_type == "document_grounded"


def test_stream_reports_each_step_then_the_result():
    workflow, *_ = build("document", chunks=[CHUNK])
    events = list(workflow.stream("alice-owner", "refund?"))

    assert [(kind, name) for kind, name in events[:-1]] == [
        ("step", "plan_question"), ("step", "retrieve_documents"), ("step", "answer_from_documents"),
    ]
    kind, result = events[-1]
    assert kind == "result" and result.answer == "Within 14 days [1]." and result.route == "document"


def test_each_step_is_timed_into_the_questions_trace():
    metrics, trace = Metrics(), Trace()
    workflow = build_chat_workflow(
        FakePlanner("document"), FakeRetrieval([CHUNK]), FakeAnswerer(), FakeResponder(), lambda owner: 1, metrics
    )
    with use_trace(trace):
        workflow.ask("alice-owner", "refund?")
    assert [s["name"] for s in trace.spans] == ["plan_question", "retrieve_documents", "answer_from_documents"]
    assert 'chat_stage_seconds_count{stage="plan_question"} 1.0' in metrics.render()[0].decode()


def test_a_real_answer_service_works_inside_the_graph():
    class Generator:
        def generate(self, question, chunks):
            return RagAnswer(answer="14 days [1].", sufficient_evidence=True, used_source_ids=[1])

    workflow, *_ = build("document", chunks=[CHUNK], answerer=AnswerService(Generator()))
    result = workflow.ask("alice-owner", "refund deadline")
    assert result.answer_type == "document_grounded"
    assert [(s.document_name, s.page) for s in result.sources] == [("terms.pdf", 3)]
