"""Frozen BERT token-embedding table (30522 x 768) for torch-free inference."""

from __future__ import annotations

import logging
import os
from functools import lru_cache

import numpy as np

logger = logging.getLogger(__name__)

TOKEN_EMBED_DIM = 768
VOCAB_SIZE = 30522
DEFAULT_BERT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "models", "bert_token_embeddings.npz"
)


def _resolve_bert_path(path: str | None = None) -> str:
    if path:
        return path
    env = os.environ.get("BERT_EMBEDDINGS_PATH")
    if env:
        return env
    try:
        from app.config import get_settings

        return get_settings().BERT_EMBEDDINGS_PATH
    except Exception:  # noqa: BLE001 - tests / export without full settings
        return DEFAULT_BERT_PATH


@lru_cache(maxsize=1)
def load_bert_token_embeddings(path: str | None = None) -> np.ndarray:
    """Return embedding matrix of shape (30522, 768)."""
    path = _resolve_bert_path(path)
    if os.path.exists(path):
        data = np.load(path)
        matrix = data["embedding"] if "embedding" in data else data[data.files[0]]
        matrix = matrix.astype(np.float32, copy=False)
        if matrix.shape != (VOCAB_SIZE, TOKEN_EMBED_DIM):
            raise ValueError(f"Expected BERT embeddings {(VOCAB_SIZE, TOKEN_EMBED_DIM)}, got {matrix.shape}")
        logger.info("Loaded BERT token embeddings from %s", path)
        return matrix

    logger.warning("BERT embeddings file missing at %s; downloading from HuggingFace", path)
    from transformers import AutoModel

    from app.utils.wordpiece import HF_CACHE_DIR, HF_MODEL_NAME

    bert = AutoModel.from_pretrained(HF_MODEL_NAME, cache_dir=HF_CACHE_DIR)
    matrix = bert.get_input_embeddings().weight.detach().cpu().numpy().astype(np.float32)
    del bert
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, embedding=matrix)
    logger.info("Saved BERT token embeddings to %s", path)
    return matrix


def lookup_token_embeddings(input_ids: np.ndarray, table: np.ndarray | None = None) -> np.ndarray:
    """(batch, seq) int64 -> (batch, seq, 768)."""
    table = table if table is not None else load_bert_token_embeddings()
    return table[input_ids]
