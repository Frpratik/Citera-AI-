import pytest

from app.core.config import Settings
from app.core.errors import LLMUnavailableError
from app.llm.client import LLMClient


def client(api_key: str, **options) -> LLMClient:
    return LLMClient(Settings(groq_api_key=api_key, _env_file=None, **options))


def test_not_configured_without_a_key():
    assert client("").is_configured is False


def test_asking_for_a_model_without_a_key_gives_a_clear_error():
    with pytest.raises(LLMUnavailableError, match="GROQ_API_KEY"):
        client("").model()


def test_configured_with_a_key_and_model_is_reused_per_temperature():
    llm = client("test-key-not-real")
    assert llm.is_configured is True
    first = llm.model(0.0)  # building the model makes no network call
    assert llm.model(0.0) is first  # same temperature: same object
    assert llm.model(0.5) is not first  # different temperature: its own object


def test_model_uses_the_settings():
    llm = client(
        "test-key-not-real", groq_model="some-model", llm_timeout_seconds=7, llm_max_retries=1
    )
    primary = llm.model().primary
    assert primary.model_name == "some-model"
    assert primary.request_timeout == 7
    assert primary.max_retries == 1


def test_reasoning_effort_is_sent_only_when_set():
    assert client("test-key-not-real").model().primary.reasoning_effort is None
    assert client("test-key-not-real", groq_reasoning_effort="low").model().primary.reasoning_effort == "low"


def test_a_fallback_model_is_tried_after_the_main_one():
    llm = client("test-key-not-real", groq_model="main-model", groq_fallback_model="backup-model")
    assert llm.model().names == ["main-model", "backup-model"]


@pytest.mark.parametrize("fallback", ["", "main-model"])  # none set, or the same as the main one
def test_no_fallback_when_there_is_nothing_different_to_fall_back_to(fallback):
    llm = client("test-key-not-real", groq_model="main-model", groq_fallback_model=fallback)
    assert llm.model().names == ["main-model"]


def test_each_model_reports_its_own_name_to_the_trace():
    llm = client("test-key-not-real", groq_model="main-model", groq_fallback_model="backup-model")
    slots = llm.model()._slots
    assert [slot.model.callbacks[0]._model_name for slot in slots] == ["main-model", "backup-model"]


def test_circuit_states_start_closed_and_are_shared_across_temperatures():
    llm = client("test-key-not-real", groq_model="main-model", groq_fallback_model="backup-model")
    assert llm.circuit_states() == {"main-model": "closed", "backup-model": "closed"}
    llm.model(0.0), llm.model(0.5)
    assert len(llm._breakers) == 2  # one breaker per model name, not per temperature
