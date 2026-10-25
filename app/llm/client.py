import logging
import threading

from app.core.config import Settings
from app.core.errors import LLMUnavailableError
from app.llm.resilience import CircuitBreaker, ModelSlot, ResilientModel
from app.observability.metrics import Metrics
from app.observability.tracing import TraceCallback

logger = logging.getLogger(__name__)


class LLMClient:
    """The one place that knows how to reach the chat AI (Groq).

    The rest of the app asks for a model with `model(temperature)` and never sees the API key
    or the library details. What it gets back is a ResilientModel: the main model plus an
    optional fallback, each behind a circuit breaker, each reporting usage to the trace.
    """

    def __init__(self, settings: Settings, metrics: Metrics | None = None):
        self._settings = settings
        self._metrics = metrics
        self._models: dict[float, ResilientModel] = {}
        self._breakers: dict[str, CircuitBreaker] = {}  # one per model NAME, shared by all temperatures
        self._lock = threading.Lock()

    @property
    def is_configured(self) -> bool:
        return bool(self._settings.groq_api_key)

    def circuit_states(self) -> dict[str, str]:
        """For the readiness check: {"model name": "closed" | "open" | "half_open"}."""
        names = [self._settings.groq_model]
        if self._settings.groq_fallback_model:
            names.append(self._settings.groq_fallback_model)
        with self._lock:
            return {name: self._breakers[name].state if name in self._breakers else "closed" for name in names}

    def model(self, temperature: float = 0.0) -> ResilientModel:
        """Temperature = how creative the AI is. 0 = stick to the facts, same answer each time."""
        if not self.is_configured:
            # The app still starts without a key; only asking a question fails, clearly.
            raise LLMUnavailableError(
                "The language model is not configured. Set GROQ_API_KEY in the .env file and restart."
            )
        with self._lock:
            if temperature not in self._models:
                names = [self._settings.groq_model]
                fallback = self._settings.groq_fallback_model
                if fallback and fallback != self._settings.groq_model:
                    names.append(fallback)
                self._models[temperature] = ResilientModel(
                    [self._slot(name, temperature) for name in names], self._metrics
                )
            return self._models[temperature]

    def _slot(self, name: str, temperature: float) -> ModelSlot:
        # Imported here so app start-up and unit tests stay fast.
        from langchain_groq import ChatGroq

        s = self._settings
        breaker = self._breakers.setdefault(
            name, CircuitBreaker(s.llm_circuit_failure_threshold, s.llm_circuit_reset_seconds)
        )
        options = {"reasoning_effort": s.groq_reasoning_effort} if s.groq_reasoning_effort else {}
        logger.info("Creating Groq model %s (temperature %s)", name, temperature)
        chat = ChatGroq(
            model=name,
            temperature=temperature,
            api_key=s.groq_api_key,
            timeout=s.llm_timeout_seconds,
            max_retries=s.llm_max_retries,  # the library already retries transient errors itself
            callbacks=[TraceCallback(self._metrics, name)],
            **options,
        )
        return ModelSlot(name, chat, breaker)
