"""Counters and timers in Prometheus format, served at /metrics.

Each app gets its OWN registry (not the global one), so creating the app twice, as the tests
do, never trips over "metric already registered".
"""

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

_SECONDS = (0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 45, 90)


class Metrics:
    def __init__(self):
        self.registry = CollectorRegistry()
        r = self.registry
        self.http_requests = Counter(
            "http_requests_total", "HTTP requests", ["method", "route", "status"], registry=r
        )
        self.http_seconds = Histogram(
            "http_request_seconds", "HTTP request time", ["method", "route"], buckets=_SECONDS, registry=r
        )
        self.stage_seconds = Histogram(
            "chat_stage_seconds", "Time per step of the chat flow", ["stage"], buckets=_SECONDS, registry=r
        )
        self.chat_outcomes = Counter(
            "chat_outcomes_total", "Chat questions by how they ended", ["outcome"], registry=r
        )
        self.llm_calls = Counter(
            "llm_calls_total", "Calls to the language model", ["model", "outcome"], registry=r
        )
        self.llm_seconds = Histogram(
            "llm_call_seconds", "Language model call time", ["model"], buckets=_SECONDS, registry=r
        )
        self.llm_tokens = Counter(
            "llm_tokens_total", "Tokens used", ["model", "kind"], registry=r
        )
        self.llm_failovers = Counter(
            "llm_failovers_total", "Times a call moved to the fallback model", ["from_model", "to_model"], registry=r
        )
        self.llm_circuit_opened = Counter(
            "llm_circuit_opened_total", "Times a model was switched off after repeated failures", ["model"], registry=r
        )
        self.jobs = Gauge("ingest_jobs", "Background jobs by status", ["status"], registry=r)

    def render(self) -> tuple[bytes, str]:
        return generate_latest(self.registry), CONTENT_TYPE_LATEST
