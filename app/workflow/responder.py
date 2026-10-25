import logging
from typing import Protocol

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.core.errors import LLMUnavailableError
from app.llm.client import LLMClient

logger = logging.getLogger(__name__)

GENERAL_SYSTEM_PROMPT = """You are a friendly assistant for a document question-answering app. \
Answer greetings and general questions naturally and clearly. You are NOT looking at any \
uploaded document for this reply, so do not state specific policies, fees, figures, dates or \
clauses from the user's documents; if asked about those, say you would need to check the \
uploaded documents. Keep answers reasonably short unless asked for detail."""


class GeneralResponder(Protocol):
    def respond(self, question: str, history: list[dict]) -> str: ...


class GroqGeneralResponder:
    def __init__(self, llm: LLMClient):
        self._llm = llm

    def respond(self, question: str, history: list[dict]) -> str:
        model = self._llm.model(0.5)  # a little creativity is fine for small talk
        messages: list[BaseMessage] = [SystemMessage(GENERAL_SYSTEM_PROMPT)]
        for item in history:
            content = item["content"][:1500]  # keep the prompt small
            messages.append(HumanMessage(content) if item["role"] == "user" else AIMessage(content))
        messages.append(HumanMessage(question))
        try:
            reply = model.invoke(messages).content
        except Exception as exc:
            logger.warning("General reply failed (%s)", type(exc).__name__)
            raise LLMUnavailableError(
                "The language model could not respond right now. Please try again."
            ) from exc
        return reply if isinstance(reply, str) else str(reply)
