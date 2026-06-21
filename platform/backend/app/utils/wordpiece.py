"""bert-base-uncased WordPiece tokenization (matches experiments/preprocessing/tokenizer.py)."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import List, Tuple

import numpy as np

HF_MODEL_NAME = "bert-base-uncased"
PAD_TOKEN_ID = 0
MAX_LEN = 128
HF_CACHE_DIR = os.environ.get("HF_HOME") or os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "models", "hf_cache"
)


@lru_cache(maxsize=1)
def get_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        HF_MODEL_NAME,
        use_fast=True,
        cache_dir=HF_CACHE_DIR,
    )


def encode_texts(texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Tokenize + dynamic pad -> (input_ids, attention_mask) as numpy arrays."""
    if not texts:
        return (
            np.zeros((0, 1), dtype=np.int64),
            np.zeros((0, 1), dtype=np.float32),
        )
    tok = get_tokenizer()
    encoded = tok(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="np",
    )
    return encoded["input_ids"].astype(np.int64), encoded["attention_mask"].astype(np.float32)
