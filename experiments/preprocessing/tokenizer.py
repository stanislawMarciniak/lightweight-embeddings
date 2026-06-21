"""WordPiece tokenization built on the bert-base-uncased fast tokenizer.

This replaces the old GloVe / whitespace tokenizer. The tokenizer provides
lowercase + unicode normalization, WordPiece subwords, [CLS]/[SEP] specials and
an attention_mask. Padding is dynamic (per-batch) — never ``padding="max_length"``.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict, List

import torch
from transformers import AutoTokenizer

HF_MODEL_NAME = "bert-base-uncased"
VOCAB_SIZE = 30522
PAD_TOKEN_ID = 0
MAX_LEN = 128
# Keep HF downloads inside the workspace so the pipeline works in sandboxed envs.
HF_CACHE_DIR = os.environ.get("HF_HOME") or os.path.join("data", "hf_cache")


@lru_cache(maxsize=1)
def get_tokenizer():
    """Cached bert-base-uncased fast (WordPiece) tokenizer."""
    return AutoTokenizer.from_pretrained(
        HF_MODEL_NAME,
        use_fast=True,
        cache_dir=HF_CACHE_DIR,
    )


def tokenize(sentences: List[str]) -> List[torch.Tensor]:
    """Tokenize sentences into *unpadded* input_id tensors (one per sentence).

    Dynamic padding is applied later, per batch, by ``pad_dynamic`` so each batch
    is only padded to its own longest sequence.
    """
    tok = get_tokenizer()
    encoded = tok(
        sentences,
        truncation=True,
        max_length=MAX_LEN,
        padding=False,
        add_special_tokens=True,
    )
    return [torch.tensor(ids, dtype=torch.long) for ids in encoded["input_ids"]]


def pad_dynamic(seqs: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Right-pad a list of variable-length id tensors to the batch max length.

    Returns ``input_ids`` and the derived ``attention_mask`` (1 for real tokens,
    0 for padding). Equivalent to ``DataCollatorWithPadding`` for a single stream.
    """
    input_ids = torch.nn.utils.rnn.pad_sequence(
        seqs, batch_first=True, padding_value=PAD_TOKEN_ID
    )
    attention_mask = (input_ids != PAD_TOKEN_ID).to(torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def encode_for_inference(sentences: List[str]) -> Dict[str, torch.Tensor]:
    """One-shot tokenize + dynamic pad for ad-hoc inference (e.g. latency probes)."""
    return pad_dynamic(tokenize(sentences))
