"""Keeping answers flowing when the model provider has a bad day.

  CircuitBreaker   stop hammering a model that keeps failing; check again after a pause
  ResilientModel   try the main model, and on a PROVIDER problem move to the fallback model

Only provider problems (down, slow, rate-limited, model retired) trigger a failover. A bad
request or a reply we could not parse is our problem, and trying another model would not fix it.
"""

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from app.core.errors import LLMUnavailableError
from app.observability.metrics import Metrics
from app.observability.tracing import current_trace

logger = logging.getLogger(__name__)

_FAILOVER_STATUS = {404, 408, 429}  # model not found, timeout, rate limited (and anything 5xx)
_CONNECTION_ERRORS = {
    "APIConnectionError", "APITimeoutError", "ConnectError", "ConnectTimeout", "ReadTimeout", "TimeoutException",
}


def is_provider_failure(exc: Exception) -> bool:
    """Is this the provider's problem (so another model may well work)?"""
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status in _FAILOVER_STATUS or status >= 500
    return isinstance(exc, (TimeoutError, ConnectionError)) or type(exc).__name__ in _CONNECTION_ERRORS


class CircuitBreaker:
    """closed: normal. open: failing, calls are refused for `reset_seconds`. half-open: after
    the pause ONE trial call is allowed; if it works the breaker closes, if not it re-opens."""

    def __init__(self, failure_threshold: int = 3, reset_seconds: float = 30.0, clock: Callable[[], float] = time.monotonic):
        self._threshold = failure_threshold
        self._reset = reset_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._failures = 0
        self._opened_at: float | None = None
        self._trial_running = False

    @property
    def state(self) -> str:
        with self._lock:
            if self._opened_at is None:
                return "closed"
            return "open" if self._clock() - self._opened_at < self._reset else "half_open"

    def allow(self) -> bool:
        with self._lock:
            if self._opened_at is None:
                return True
            if self._clock() - self._opened_at >= self._reset and not self._trial_running:
                self._trial_running = True  # this caller is the one trial
                return True
            return False

    def record_success(self) -> None:
        with self._lock:
            self._failures, self._opened_at, self._trial_running = 0, None, False

    def record_failure(self) -> bool:
        """Returns True if this failure just switched the breaker on (opened it)."""
        with self._lock:
            self._failures += 1
            was_closed = self._opened_at is None
            if self._trial_running or self._failures >= self._threshold:
                self._opened_at, self._trial_running = self._clock(), False
                return was_closed
            return False


@dataclass
class ModelSlot:
    name: str
    model: object
    breaker: CircuitBreaker


class _StructuredView:
    """`model.with_structured_output(schema)` for a ResilientModel: same failover, structured reply."""

    def __init__(self, owner: "ResilientModel", schema):
        self._owner = owner
        self._schema = schema

    def invoke(self, messages):
        return self._owner.run(lambda model: model.with_structured_output(self._schema).invoke(messages))


class ResilientModel:
    def __init__(self, slots: list[ModelSlot], metrics: Metrics | None = None):
        self._slots = slots
        self._metrics = metrics

    @property
    def primary(self):
        return self._slots[0].model

    @property
    def names(self) -> list[str]:
        """The models in the order they are tried."""
        return [slot.name for slot in self._slots]

    def invoke(self, messages):
        return self.run(lambda model: model.invoke(messages))

    def with_structured_output(self, schema):
        return _StructuredView(self, schema)

    def run(self, call: Callable[[object], object]):
        last_error: Exception | None = None
        for index, slot in enumerate(self._slots):
            if not slot.breaker.allow():
                logger.info("Skipping model %s: its circuit is open", slot.name)
                continue
            try:
                result = call(slot.model)
            except Exception as exc:
                if not is_provider_failure(exc):
                    slot.breaker.record_success()  # the provider answered; the problem is ours
                    raise
                last_error = exc
                if slot.breaker.record_failure() and self._metrics:
                    self._metrics.llm_circuit_opened.labels(slot.name).inc()
                logger.warning("Model %s failed (%s)", slot.name, type(exc).__name__)
                continue
            slot.breaker.record_success()
            if index > 0:
                self._note_failover(self._slots[0].name, slot.name)
            return result
        raise LLMUnavailableError(
            "The language model is temporarily unavailable. Please try again in a moment."
        ) from last_error

    def _note_failover(self, from_model: str, to_model: str) -> None:
        logger.warning("Answered with fallback model %s instead of %s", to_model, from_model)
        if self._metrics:
            self._metrics.llm_failovers.labels(from_model, to_model).inc()
        trace = current_trace.get()
        if trace:
            trace.event(f"fallback:{to_model}")
