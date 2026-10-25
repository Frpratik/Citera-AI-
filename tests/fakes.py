import hashlib
import re

import numpy as np

from app.answering.schemas import RagAnswer
from app.core.config import Settings
from app.llm.client import LLMClient
from app.workflow.planner import QuestionPlan


class FakeChain:
    """Plays the part of `model.with_structured_output(...)`: replies from a script."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.seen_messages = None

    def invoke(self, messages):
        self.seen_messages = messages
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


class FakeModel:
    def __init__(self, chain):
        self.chain = chain
        self.schema = None

    def with_structured_output(self, schema):
        self.schema = schema
        return self.chain

    def invoke(self, messages):  # plain (unstructured) calls
        return self.chain.invoke(messages)


class FakeLLM(LLMClient):
    """An LLMClient whose model replies from a script, so no network is used."""

    def __init__(self, chain):
        super().__init__(Settings(groq_api_key="x", _env_file=None))
        self.fake_model = FakeModel(chain)

    def model(self, temperature=0.0):
        self.temperature = temperature
        return self.fake_model


class FakePlanner:
    """Routes every question the same way (changeable per test) and remembers what it saw."""

    def __init__(self, route="document", search_question="refund requests filed within 14 days"):
        self.route = route
        self.search_question = search_question
        self.seen_history = None

    def plan(self, question, history):
        self.seen_history = history
        return QuestionPlan(route=self.route, search_question=self.search_question, reason="fake")


class FakeGenerator:
    """Always answers by citing source 1, unless told to fail."""

    def __init__(self):
        self.error = None

    def generate(self, question, chunks):
        if self.error:
            raise self.error
        return RagAnswer(answer="Within 14 days [1].", sufficient_evidence=True, used_source_ids=[1])


class FakeResponder:
    def __init__(self):
        self.error = None

    def respond(self, question, history):
        if self.error:
            raise self.error
        return "Hello! How can I help?"


class FakeReranker:
    """Scores a chunk by the share of the question's words it contains (0 to 1)."""

    def warm_up(self) -> None:
        pass

    def score(self, question: str, texts: list[str]) -> list[float]:
        wanted = set(re.findall(r"\w+", question.lower()))
        return [
            len(wanted & set(re.findall(r"\w+", text.lower()))) / max(len(wanted), 1)
            for text in texts
        ]


class FakeEmbedder:
    """Deterministic bag-of-words embedder: texts sharing words get higher cosine similarity.

    Lets tests check ranking and plumbing without downloading or running a real model.
    """

    dim = 384

    def warm_up(self) -> None:
        pass

    def _vec(self, text: str) -> list[float]:
        v = np.zeros(self.dim)
        for word in re.findall(r"\w+", text.lower()):
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1
        norm = np.linalg.norm(v)
        return (v / norm if norm else v).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class FlakyEmbedder(FakeEmbedder):
    """Fails the first `failures` times it is asked to embed documents, then works."""

    def __init__(self, failures: int):
        self.failures = failures

    def embed_documents(self, texts):
        if self.failures > 0:
            self.failures -= 1
            raise RuntimeError("model hiccup")
        return super().embed_documents(texts)
