from types import SimpleNamespace

import pytest

from app.core.errors import LLMUnavailableError
from app.workflow.planner import GroqQuestionPlanner, QuestionPlan
from app.workflow.responder import GroqGeneralResponder
from tests.fakes import FakeChain, FakeLLM

HISTORY = [
    {"role": "user", "content": "What is the hostel curfew?"},
    {"role": "assistant", "content": "Ten pm."},
]


# ---------------------------------------------------------------- planner
def test_planner_returns_the_ais_decision_and_sees_the_history():
    plan = QuestionPlan(route="document", search_question="What is the library fine?", reason="fact")
    chain = FakeChain([plan])
    llm = FakeLLM(chain)

    result = GroqQuestionPlanner(llm).plan("and the library?", HISTORY)

    assert result == plan
    assert llm.fake_model.schema is QuestionPlan
    human = chain.seen_messages[1].content
    assert "hostel curfew" in human and "and the library?" in human


def test_planner_falls_back_to_uncertain_with_the_original_words():
    chain = FakeChain([ValueError("bad json")])
    result = GroqQuestionPlanner(FakeLLM(chain)).plan("fee for exam", [])
    assert result.route == "uncertain" and result.search_question == "fee for exam"


def test_planner_never_returns_an_empty_search_question():
    blank = QuestionPlan(route="document", search_question="   ", reason="x")
    result = GroqQuestionPlanner(FakeLLM(FakeChain([blank]))).plan("exam fee", [])
    assert result.search_question == "exam fee"


# ---------------------------------------------------------------- general responder
def test_responder_sends_history_then_the_question_and_returns_text():
    chain = FakeChain([SimpleNamespace(content="Hello there!")])
    llm = FakeLLM(chain)

    reply = GroqGeneralResponder(llm).respond("hi", HISTORY)

    assert reply == "Hello there!"
    assert llm.temperature == 0.5
    roles = [type(m).__name__ for m in chain.seen_messages]
    assert roles == ["SystemMessage", "HumanMessage", "AIMessage", "HumanMessage"]
    assert chain.seen_messages[-1].content == "hi"


def test_responder_failure_becomes_a_clear_error():
    chain = FakeChain([RuntimeError("boom")])
    with pytest.raises(LLMUnavailableError, match="could not respond"):
        GroqGeneralResponder(FakeLLM(chain)).respond("hi", [])
