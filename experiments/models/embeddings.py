"""Standalone token-embedding provider, intentionally *outside* the encoder.

The encoder (`CompactSimilarityModel`) only ever sees ``(batch, seq_len, 768)``
float tensors. Turning ``input_ids`` into those tensors is the job of this
module, so the encoder parameter count stays ~296k and the embedding table
(~23M params) is never trained/quantized as part of the encoder.

The table is the pretrained bert-base-uncased word-embedding matrix (frozen),
which is what makes the 768-d WordPiece vectors semantically meaningful.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, Optional

import torch
import torch.nn as nn

from preprocessing.tokenizer import HF_CACHE_DIR, HF_MODEL_NAME, PAD_TOKEN_ID, VOCAB_SIZE

EMBED_DIM = 768


class TokenEmbedding(nn.Module):
    """``nn.Embedding(30522, 768, padding_idx=0)`` initialized from BERT, frozen."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embedding_dim: int = EMBED_DIM,
        padding_idx: int = PAD_TOKEN_ID,
        pretrained: bool = True,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        if pretrained:
            self._load_pretrained()
        if freeze:
            self.embedding.weight.requires_grad_(False)

    def _load_pretrained(self) -> None:
        from transformers import AutoModel

        bert = AutoModel.from_pretrained(HF_MODEL_NAME, cache_dir=HF_CACHE_DIR)
        weight = bert.get_input_embeddings().weight.detach().clone()
        with torch.no_grad():
            self.embedding.weight.copy_(weight)
        del bert

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """(batch, seq_len) -> (batch, seq_len, 768)."""
        return self.embedding(input_ids)

    @torch.no_grad()
    def embed_pair(self, batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
        """Build the encoder inputs (token_embeddings + masks) from a tokenized batch."""
        ids1 = batch["input_ids_1"].to(device)
        ids2 = batch["input_ids_2"].to(device)
        return {
            "token_embeddings_1": self.embedding(ids1),
            "attention_mask_1": batch["attention_mask_1"].to(device),
            "token_embeddings_2": self.embedding(ids2),
            "attention_mask_2": batch["attention_mask_2"].to(device),
        }


@lru_cache(maxsize=1)
def get_token_embedding() -> TokenEmbedding:
    """Process-wide shared embedding table (built once, reused everywhere)."""
    return TokenEmbedding()


def shared_token_embedding(device: Optional[torch.device] = None) -> TokenEmbedding:
    emb = get_token_embedding()
    if device is not None:
        emb.to(device)
    return emb
