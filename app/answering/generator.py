import logging
from typing import Protocol

from langchain_core.messages import HumanMessage, SystemMessage

from app.answering.prompts import RAG_SYSTEM_PROMPT, format_sources
from app.answering.schemas import RagAnswer
from app.core.errors import LLMUnavailableError
from app.llm.client import LLMClient
from app.repositories.chunks import SearchHit

logger = logging.getLogger(__name__)


class AnswerGenerator(Protocol):
    """Anything that can draft an answer from a question and chunks (tests use a fake)."""

    def generate(self, question: str, chunks: list[SearchHit]) -> RagAnswer: ...


class GroqAnswerGenerator:
    def __init__(self, llm: LLMClient, attempts: int = 2):
        self._llm = llm
        self._attempts = attempts

    def generate(self, question: str, chunks: list[SearchHit]) -> RagAnswer:
        # Temperature 0: stick to the facts. A missing key raises its own clear error here.
        model = self._llm.model(0.0).with_structured_output(RagAnswer)
        messages = [
            SystemMessage(RAG_SYSTEM_PROMPT),
            HumanMessage(f"{format_sources(chunks)}\n\nQuestion: {question}"),
        ]
        last_error: Exception | None = None
        for attempt in range(1, self._attempts + 1):
            try:
                result = model.invoke(messages)
                return result if isinstance(result, RagAnswer) else RagAnswer.model_validate(result)
            except LLMUnavailableError:
                raise  # the model (and its fallback) is down: asking again at once will not help
            except Exception as exc:  # noqa: BLE001 - the library raises many different error types
                # Structured replies occasionally fail to parse; trying again usually works.
                last_error = exc
                logger.warning("Answer attempt %d failed (%s)", attempt, type(exc).__name__)
        raise LLMUnavailableError(
            "The language model could not produce an answer right now. Please try again."
        ) from last_error
