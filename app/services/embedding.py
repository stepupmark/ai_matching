"""
Embedding service wrapping sentence-transformers.

Generates normalized embeddings for any text input.
The model is loaded once and cached for the lifetime of the service.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Generates and caches text embeddings using sentence-transformers."""

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None) -> None:
        self._model_name = model_name or settings.embedding_model_name
        self._device = device or settings.embedding_device
        self._model: Optional[SentenceTransformer] = None
        self._dimension: int = settings.embedding_dimension
        self._model_version: str = settings.embedding_model_version

    # ── Lazy-loaded model ──────────────────────────────────────────

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info(
                "Loading embedding model '%s' on %s ...",
                self._model_name,
                self._device,
            )
            self._model = SentenceTransformer(
                self._model_name,
                device=self._device,
            )
            logger.info("Embedding model loaded (dim=%d).", self._dimension)
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def model_name(self) -> str:
        return self._model_name

    # ── Embedding methods ──────────────────────────────────────────

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string → normalized 1-D numpy array."""
        vec = self.model.encode(text, normalize_embeddings=True)
        return np.asarray(vec, dtype=np.float32)

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Embed a batch of texts → shape (N, dim) normalized numpy array."""
        vecs = self.model.encode(
            texts,
            batch_size=batch_size or settings.embedding_batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)
