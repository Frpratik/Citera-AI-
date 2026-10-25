import logging
import math
import threading
from typing import Protocol

logger = logging.getLogger(__name__)


class Reranker(Protocol):
    """Gives each text a score for how well it answers the question (higher = better)."""

    def score(self, question: str, texts: list[str]) -> list[float]: ...

    def warm_up(self) -> None:
        """Load the model now, so the first real request does not pay for it."""
        ...


class CrossEncoderReranker:
    """A cross-encoder reads the question and one chunk TOGETHER and outputs one score.

    Slower than the embedding search (it cannot be prepared in advance), so we only run
    it on the ~20 candidates that the fast searches already picked.
    """

    def __init__(self, model_name: str):
        self._model_name = model_name
        self._model = None
        self._lock = threading.Lock()

    def _get_model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import CrossEncoder

                    logger.info("Loading reranker model %s", self._model_name)
                    self._model = CrossEncoder(self._model_name)
        return self._model

    def warm_up(self) -> None:
        self.score("warm up", ["warm up"])  # load the model and run one tiny scoring

    def score(self, question: str, texts: list[str]) -> list[float]:
        """Scores between 0 and 1. The model itself outputs raw numbers (about -11 for
        unrelated text, up to +9 for a direct answer); we squash them with a sigmoid."""
        if not texts:
            return []
        pairs = [(question, text) for text in texts]
        raw_scores = self._get_model().predict(pairs, show_progress_bar=False)
        return [sigmoid(float(s)) for s in raw_scores]


def sigmoid(x: float) -> float:
    """Map any number to (0, 1): 0 -> 0.5, large positive -> near 1, large negative -> near 0."""
    x = max(min(x, 60.0), -60.0)  # keep math.exp from overflowing
    return 1.0 / (1.0 + math.exp(-x))
