"""Embedding model wrapper.

An embedding turns text into a list of numbers (a vector) so that texts with
similar *meaning* end up close together, even without sharing words:

    "last date to pay exam fee"  ~  "examination fee payment deadline"

The model is loaded lazily (first use) because loading it takes a few seconds and
the first run downloads ~90 MB from Hugging Face.
"""

import logging
import threading

from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self, model_name: str, embeddings: Embeddings | None = None):
        self._model_name = model_name
        self._embeddings = embeddings  # tests inject a fake here
        self._lock = threading.Lock()

    @property
    def embeddings(self) -> Embeddings:
        if self._embeddings is None:
            with self._lock:
                if self._embeddings is None:
                    # Imported here so app start-up stays fast and tests need no torch.
                    from langchain_huggingface import HuggingFaceEmbeddings

                    logger.info("Loading embedding model %s", self._model_name)
                    self._embeddings = HuggingFaceEmbeddings(
                        model_name=self._model_name,
                        # Normalised vectors make inner product == cosine similarity (-1..1).
                        encode_kwargs={"normalize_embeddings": True},
                    )
        return self._embeddings
