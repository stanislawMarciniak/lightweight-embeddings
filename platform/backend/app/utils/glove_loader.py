from __future__ import annotations

import logging
from functools import lru_cache
from typing import Dict

import numpy as np

from app.config import get_settings


logger = logging.getLogger(__name__)


@lru_cache()
def load_glove_embeddings() -> Dict[str, np.ndarray]:
    """
    Load GloVe 100d embeddings from the configured file.

    The file must be downloaded separately from:
    https://nlp.stanford.edu/projects/glove/
    (glove.6B.100d.txt) and placed at backend/app/models/glove.6B.100d.txt
    """
    settings = get_settings()
    path = settings.GLOVE_PATH
    embeddings: Dict[str, np.ndarray] = {}

    try:
        with open(path, "r", encoding="utf8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                word = parts[0]
                try:
                    vector = np.asarray([float(x) for x in parts[1:]], dtype=np.float32)
                    if vector.shape[0] != 100:
                        continue
                    embeddings[word] = vector
                except ValueError:
                    continue
    except FileNotFoundError:
        logger.error("GloVe embeddings file not found at %s", path)
        raise

    logger.info("Loaded %d GloVe word vectors from %s", len(embeddings), path)
    return embeddings


def get_word_embedding(word: str) -> np.ndarray | None:
    """Return the embedding for a single word, or None if not found."""
    word = word.lower()
    embeddings = load_glove_embeddings()
    return embeddings.get(word)

