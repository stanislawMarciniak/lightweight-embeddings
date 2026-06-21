from __future__ import annotations

import logging
from typing import List

import numpy as np

from app.config import get_settings
from app.services.semantic_model import get_encoder, load_encoder, resolve_model_path
from app.utils.glove_loader import get_word_embedding
from app.utils.text_splitter import tokenize

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Produces sentence embeddings for queries, FAQ and document chunks.

    Backend is selectable via settings.EMBEDDING_BACKEND:
      * "custom" -> CompactSimilarityModel encoder (128-d, WordPiece + BERT 768-d lookup)
      * "glove"  -> GloVe-100 mean pooling (legacy, 100-d)
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.backend = self.settings.EMBEDDING_BACKEND
        self._glove_dim = 100

    def _encoder(self):
        enc = get_encoder()
        if enc is None:
            enc = load_encoder(resolve_model_path(self.settings))
        return enc

    @property
    def dim(self) -> int:
        if self.backend == "custom":
            return self._encoder().output_dim
        return self._glove_dim

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if self.backend == "custom":
            return self._encoder().encode(texts)
        return np.stack([self._glove_mean(t) for t in texts], axis=0)

    def get_sentence_embedding(self, sentence: str) -> np.ndarray:
        if self.backend == "custom":
            return self._encoder().encode_one(sentence)
        return self._glove_mean(sentence)

    def _glove_mean(self, sentence: str) -> np.ndarray:
        tokens = tokenize(sentence)
        vectors: List[np.ndarray] = []
        for token in tokens:
            vec = get_word_embedding(token)
            if vec is not None:
                vectors.append(vec)
        if not vectors:
            return np.zeros(self._glove_dim, dtype=np.float32)
        mean = np.stack(vectors, axis=0).mean(axis=0)
        norm = np.linalg.norm(mean)
        return (mean / norm) if norm > 0 else mean
