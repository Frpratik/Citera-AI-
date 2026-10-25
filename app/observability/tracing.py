"""A trace follows ONE chat question: how long each step took, which model calls it made,
how many tokens they used, and whether a fallback was needed. It never stores the question
or the answer text.

The current trace lives in a ContextVar, so code deep inside a call (a model call, say)
can add to it without every function having to pass it along.
"""

import contextlib
import contextvars
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

from app.observability.metrics import Metrics


@dataclass
class LLMCall:
    model: str
    stage: str  # which step of the flow made the call
    seconds: float
    input_tokens: int
    output_tokens: int
    outcome: str  # "ok" or "error"


class Trace:
    def __init__(self):
        self._started = time.monotonic()
        self._lock = threading.Lock()
        self._open: list[str] = []  # steps currently running (innermost last)
        self.spans: list[dict] = []
        self.llm_calls: list[LLMCall] = []
        self.events: list[str] = []

    # -- recording -------------------------------------------------------------------
    def enter(self, name: str) -> None:
        with self._lock:
            self._open.append(name)

    def leave(self, name: str, seconds: float) -> None:
        with self._lock:
            if name in self._open:
                self._open.remove(name)
            self.spans.append({"name": name, "ms": round(seconds * 1000)})

    def add_llm_call(self, call: LLMCall) -> None:
        with self._lock:
            self.llm_calls.append(call)

    def event(self, text: str) -> None:
        with self._lock:
            self.events.append(text)

    @property
    def current_stage(self) -> str:
        with self._lock:
            return self._open[-1] if self._open else "none"

    # -- totals ----------------------------------------------------------------------
    @property
    def total_seconds(self) -> float:
        return time.monotonic() - self._started

    @property
    def input_tokens(self) -> int:
        return sum(c.input_tokens for c in self.llm_calls)

    @property
    def output_tokens(self) -> int:
        return sum(c.output_tokens for c in self.llm_calls)

    @property
    def fallback_used(self) -> bool:
        return any(e.startswith("fallback:") for e in self.events)

    def summary(self) -> dict:
        return {
            "total_ms": round(self.total_seconds * 1000),
            "llm_calls": len(self.llm_calls),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "fallback_used": self.fallback_used,
            "spans": self.spans,
            "events": self.events,
        }


current_trace: contextvars.ContextVar[Trace | None] = contextvars.ContextVar("current_trace", default=None)


@contextlib.contextmanager
def use_trace(trace: Trace) -> Iterator[Trace]:
    token = current_trace.set(trace)
    try:
        yield trace
    finally:
        current_trace.reset(token)


@contextlib.contextmanager
def span(name: str, metrics: Metrics | None = None) -> Iterator[None]:
    """Time a step. Recorded in the current trace (if any) and in the metrics (if given)."""
    trace = current_trace.get()
    if trace:
        trace.enter(name)
    started = time.monotonic()
    try:
        yield
    finally:
        seconds = time.monotonic() - started
        if trace:
            trace.leave(name, seconds)
        if metrics:
            metrics.stage_seconds.labels(name).observe(seconds)


class TraceCallback(BaseCallbackHandler):
    """Plugged into every model: LangChain calls it when a model call starts and ends, and it
    reads the token counts from the reply. This is how usage is captured even when the model
    is asked for structured output."""

    def __init__(self, metrics: Metrics | None = None, model_name: str | None = None):
        self._metrics = metrics
        self._model_name = model_name  # each model gets its own callback, so we simply know the name
        self._started: dict[UUID, tuple[float, str, str]] = {}

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        params = kwargs.get("invocation_params") or {}
        model = self._model_name or params.get("model") or params.get("model_name") or "unknown"
        trace = current_trace.get()
        self._started[run_id] = (time.monotonic(), model, trace.current_stage if trace else "none")

    def on_llm_end(self, response, *, run_id, **kwargs):
        input_tokens = output_tokens = 0
        for generations in response.generations:
            for generation in generations:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None) or {}
                input_tokens += usage.get("input_tokens", 0)
                output_tokens += usage.get("output_tokens", 0)
        if not (input_tokens or output_tokens):  # some providers report it only here
            usage = (response.llm_output or {}).get("token_usage") or {}
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
        self._finish(run_id, "ok", input_tokens, output_tokens)

    def on_llm_error(self, error, *, run_id, **kwargs):
        self._finish(run_id, "error", 0, 0)

    def _finish(self, run_id: UUID, outcome: str, input_tokens: int, output_tokens: int) -> None:
        started, model, stage = self._started.pop(run_id, (time.monotonic(), "unknown", "none"))
        seconds = time.monotonic() - started
        trace = current_trace.get()
        if trace:
            trace.add_llm_call(LLMCall(model, stage, round(seconds, 3), input_tokens, output_tokens, outcome))
        if self._metrics:
            self._metrics.llm_calls.labels(model, outcome).inc()
            self._metrics.llm_seconds.labels(model).observe(seconds)
            self._metrics.llm_tokens.labels(model, "input").inc(input_tokens)
            self._metrics.llm_tokens.labels(model, "output").inc(output_tokens)
