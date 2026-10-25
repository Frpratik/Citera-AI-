import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from app.answering.generator import GroqAnswerGenerator
from app.answering.schemas import RagAnswer
from app.core.config import Settings
from app.core.errors import LLMUnavailableError
from app.llm.client import LLMClient
from app.repositories.chunks import SearchHit
from tests.fakes import FakeChain, FakeLLM

CHUNKS = [SearchHit("c1", "d1", "terms.pdf", 3, "Refunds within 14 days.", 0.9)]
GOOD = RagAnswer(answer="14 days [1].", sufficient_evidence=True, used_source_ids=[1])


def test_sends_rules_then_numbered_sources_and_the_question():
    chain = FakeChain([GOOD])
    llm = FakeLLM(chain)
    result = GroqAnswerGenerator(llm).generate("What is the deadline?", CHUNKS)

    assert result == GOOD
    assert llm.fake_model.schema is RagAnswer  # the AI is told to fill in our form
    assert llm.temperature == 0.0  # facts only
    system, human = chain.seen_messages
    assert isinstance(system, SystemMessage) and "ONLY" in system.content
    assert isinstance(human, HumanMessage)
    assert '<source id="1"' in human.content and "What is the deadline?" in human.content


def test_a_reply_that_is_a_plain_dict_is_validated_into_the_form():
    chain = FakeChain([{"answer": "ok", "sufficient_evidence": False}])
    result = GroqAnswerGenerator(FakeLLM(chain)).generate("q", CHUNKS)
    assert isinstance(result, RagAnswer) and result.sufficient_evidence is False


def test_one_failed_attempt_is_retried():
    chain = FakeChain([ValueError("bad json"), GOOD])
    assert GroqAnswerGenerator(FakeLLM(chain)).generate("q", CHUNKS) == GOOD


def test_two_failed_attempts_give_a_clear_error():
    chain = FakeChain([ValueError("bad"), ValueError("bad again")])
    with pytest.raises(LLMUnavailableError, match="could not produce an answer"):
        GroqAnswerGenerator(FakeLLM(chain)).generate("q", CHUNKS)


def test_an_outage_is_not_retried_because_asking_again_at_once_cannot_help():
    chain = FakeChain([LLMUnavailableError("down"), GOOD])
    with pytest.raises(LLMUnavailableError, match="down"):
        GroqAnswerGenerator(FakeLLM(chain)).generate("q", CHUNKS)
    assert len(chain.replies) == 1  # the second (good) reply was never asked for


def test_a_missing_key_error_is_not_swallowed():
    llm = LLMClient(Settings(groq_api_key="", _env_file=None))
    with pytest.raises(LLMUnavailableError, match="GROQ_API_KEY"):
        GroqAnswerGenerator(llm).generate("q", CHUNKS)
