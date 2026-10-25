import logging
import threading
from typing import Protocol

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    """What the rest of the app needs from an embedding model. Tests pass a fake."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...

    def warm_up(self) -> None:
        """Load the model now, so the first real request does not pay for it."""
        ...


class SentenceTransformerEmbedder:
    """Local embedding model, loaded lazily on first use (the first run downloads it)."""

    def __init__(self, model_name: str, batch_size: int = 32):
        self._model_name = model_name
        self._batch_size = batch_size
        self._model = None
        self._lock = threading.Lock()

    def _get_model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    # Imported here so app start-up and unit tests do not pay for torch.
                    from sentence_transformers import SentenceTransformer

                    logger.info("Loading embedding model %s", self._model_name)
                    self._model = SentenceTransformer(self._model_name)
        return self._model

    def warm_up(self) -> None:
        # Loading is not enough: the first real encoding call sets up more machinery.
        self.embed_documents(["warm up"])

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._get_model().encode(
            texts,
            batch_size=self._batch_size,
            # Unit-length vectors make cosine similarity a plain dot product.
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
