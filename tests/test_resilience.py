import pytest

from app.core.errors import LLMUnavailableError
from app.llm.resilience import (
    CircuitBreaker,
    ModelSlot,
    ResilientModel,
    is_provider_failure,
)
from app.observability.metrics import Metrics
from app.observability.tracing import Trace, use_trace


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class ProviderError(Exception):
    def __init__(self, status_code: int | None = None):
        super().__init__("boom")
        self.status_code = status_code


# ---------------------------------------------------------------- is it the provider's fault?
@pytest.mark.parametrize("status", [404, 408, 429, 500, 502, 503])
def test_these_statuses_count_as_provider_problems(status):
    assert is_provider_failure(ProviderError(status)) is True


@pytest.mark.parametrize("status", [400, 401, 403, 422])
def test_our_own_mistakes_are_not_provider_problems(status):
    assert is_provider_failure(ProviderError(status)) is False


def test_timeouts_and_connection_errors_count_but_parsing_errors_do_not():
    assert is_provider_failure(TimeoutError()) and is_provider_failure(ConnectionError())
    assert is_provider_failure(type("APIConnectionError", (Exception,), {})())
    assert not is_provider_failure(ValueError("could not parse the reply"))


# ---------------------------------------------------------------- the circuit breaker
def test_breaker_opens_after_repeated_failures_then_tests_the_water():
    clock = Clock()
    breaker = CircuitBreaker(failure_threshold=3, reset_seconds=30, clock=clock)
    assert breaker.state == "closed" and breaker.allow()

    assert breaker.record_failure() is False and breaker.record_failure() is False
    assert breaker.record_failure() is True  # the third one switches it on
    assert breaker.state == "open" and breaker.allow() is False  # calls are refused

    clock.now = 31  # the pause is over
    assert breaker.state == "half_open"
    assert breaker.allow() is True  # ONE trial call is let through...
    assert breaker.allow() is False  # ...and nobody else

    breaker.record_success()
    assert breaker.state == "closed" and breaker.allow()


def test_a_failed_trial_closes_the_door_again():
    clock = Clock()
    breaker = CircuitBreaker(failure_threshold=1, reset_seconds=30, clock=clock)
    breaker.record_failure()
    clock.now = 31
    assert breaker.allow()  # the trial
    breaker.record_failure()  # it failed
    assert breaker.state == "open" and not breaker.allow()
    clock.now = 62
    assert breaker.state == "half_open"


def test_a_success_wipes_the_failure_count():
    breaker = CircuitBreaker(failure_threshold=3, reset_seconds=30, clock=Clock())
    breaker.record_failure(), breaker.record_failure()
    breaker.record_success()
    breaker.record_failure(), breaker.record_failure()
    assert breaker.state == "closed"  # 2 + 2 failures, but never 3 in a row


# ---------------------------------------------------------------- failing over between models
class ScriptedModel:
    """Replies (or raises) from a script, and counts how often it was asked."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def _next(self):
        self.calls += 1
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def invoke(self, messages):
        return self._next()

    def with_structured_output(self, schema):
        return self


def resilient(primary, backup=None, metrics=None, threshold=2):
    clock = Clock()
    slots = [ModelSlot("main", primary, CircuitBreaker(threshold, 30, clock))]
    if backup:
        slots.append(ModelSlot("backup", backup, CircuitBreaker(threshold, 30, clock)))
    return ResilientModel(slots, metrics), slots


def test_the_main_model_is_used_when_it_works():
    primary, backup = ScriptedModel("main answer"), ScriptedModel("backup answer")
    model, _ = resilient(primary, backup)
    trace = Trace()
    with use_trace(trace):
        assert model.invoke(["hi"]) == "main answer"
    assert backup.calls == 0 and not trace.fallback_used


def test_a_provider_problem_moves_the_call_to_the_fallback_model():
    metrics = Metrics()
    primary, backup = ScriptedModel(ProviderError(503)), ScriptedModel("backup answer")
    model, _ = resilient(primary, backup, metrics)
    trace = Trace()
    with use_trace(trace):
        assert model.invoke(["hi"]) == "backup answer"
    assert trace.fallback_used and "fallback:backup" in trace.events
    assert 'llm_failovers_total{from_model="main",to_model="backup"} 1.0' in metrics.render()[0].decode()


def test_our_own_errors_are_not_retried_on_another_model():
    primary, backup = ScriptedModel(ValueError("bad reply format")), ScriptedModel("backup answer")
    model, slots = resilient(primary, backup)
    with pytest.raises(ValueError):
        model.invoke(["hi"])
    assert backup.calls == 0 and slots[0].breaker.state == "closed"  # not counted as an outage


def test_when_every_model_fails_the_error_is_clear():
    model, _ = resilient(ScriptedModel(ProviderError(500)), ScriptedModel(ProviderError(429)))
    with pytest.raises(LLMUnavailableError, match="temporarily unavailable"):
        model.invoke(["hi"])


def test_a_model_that_keeps_failing_is_skipped_instead_of_asked_again():
    metrics = Metrics()
    primary, backup = ScriptedModel(ProviderError(503)), ScriptedModel("backup answer")
    model, slots = resilient(primary, backup, metrics, threshold=2)

    model.invoke(["1"]), model.invoke(["2"])  # two failures in a row open the breaker
    assert slots[0].breaker.state == "open" and primary.calls == 2

    model.invoke(["3"])  # now the main model is not even tried
    assert primary.calls == 2 and backup.calls == 3
    assert 'llm_circuit_opened_total{model="main"} 1.0' in metrics.render()[0].decode()


def test_structured_output_gets_the_same_protection():
    model, _ = resilient(ScriptedModel(ProviderError(503)), ScriptedModel("structured backup"))
    assert model.with_structured_output(dict).invoke(["hi"]) == "structured backup"
