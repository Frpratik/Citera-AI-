import contextvars
import logging
from collections.abc import Iterator

from app.answering.schemas import ChatResult
from app.core.logging import request_id_var
from app.observability.metrics import Metrics
from app.observability.tracing import Trace, current_trace, use_trace
from app.repositories.messages import MessageRecord, MessageRepository
from app.repositories.traces import TraceRepository
from app.workflow.graph import ChatWorkflow

logger = logging.getLogger(__name__)


class ChatService:
    """Runs the workflow with the user's recent history, saves the exchange, and records a
    trace of how the question went (time per step, tokens, fallbacks)."""

    def __init__(
        self,
        workflow: ChatWorkflow,
        messages: MessageRepository,
        history_turns: int,
        traces: TraceRepository | None = None,
        metrics: Metrics | None = None,
    ):
        self._workflow = workflow
        self._messages = messages
        self._history_messages = history_turns * 2  # one turn = a question + its answer
        self._traces = traces
        self._metrics = metrics

    def _history(self, owner_id: str) -> list[dict]:
        recent = self._messages.recent(owner_id, self._history_messages)
        return [{"role": m.role, "content": m.content} for m in recent]

    def _save(self, owner_id: str, question: str, result: ChatResult) -> None:
        # Saved only after success, so a failed request leaves no half-finished history.
        self._messages.add_exchange(owner_id, question, result.answer, result.answer_type)

    def _record(self, owner_id: str, trace: Trace, route: str | None, outcome: str) -> None:
        summary = trace.summary()
        if self._metrics:
            self._metrics.chat_outcomes.labels(outcome).inc()
        logger.info("chat_trace", extra={"trace": {**summary, "route": route, "outcome": outcome}})
        if self._traces is None:
            return
        try:
            self._traces.add(owner_id, request_id_var.get(), route, outcome, summary)
        except Exception:
            logger.exception("Could not save the trace")

    def ask(self, owner_id: str, question: str) -> ChatResult:
        trace, route, outcome = Trace(), None, "error"
        try:
            with use_trace(trace):
                result = self._workflow.ask(owner_id, question, self._history(owner_id))
            self._save(owner_id, question, result)
            route, outcome = result.route, result.answer_type
            return result
        except Exception as exc:
            outcome = f"error:{type(exc).__name__}"
            raise
        finally:
            self._record(owner_id, trace, route, outcome)

    def ask_stream(self, owner_id: str, question: str) -> Iterator[tuple[str, dict]]:
        """Events for the streaming endpoint: ("step", ...) while working, then ("answer", ...)
        with the final CHECKED answer, then ("done", {})."""
        trace, route, outcome = Trace(), None, "abandoned"  # "abandoned" = the client left early
        # A generator is resumed from different threads, so a plain "set the trace" would be
        # lost between steps. Every step is run inside ONE saved context that holds the trace.
        context = contextvars.copy_context()
        context.run(current_trace.set, trace)
        try:
            events = self._workflow.stream(owner_id, question, self._history(owner_id))
            while True:
                try:
                    kind, payload = context.run(next, events)
                except StopIteration:
                    break
                if kind == "step":
                    yield "step", {"step": payload}
                else:
                    self._save(owner_id, question, payload)
                    route, outcome = payload.route, payload.answer_type
                    yield "answer", payload.model_dump(mode="json")
            yield "done", {}
        except Exception as exc:
            outcome = f"error:{type(exc).__name__}"
            raise
        finally:
            self._record(owner_id, trace, route, outcome)

    def history(self, owner_id: str, limit: int = 100) -> list[MessageRecord]:
        return self._messages.recent(owner_id, limit)

    def clear_history(self, owner_id: str) -> int:
        return self._messages.clear(owner_id)
