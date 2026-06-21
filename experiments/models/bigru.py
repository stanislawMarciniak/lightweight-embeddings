from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


EMBED_DIM = 768


class BiGRUEncoder(nn.Module):
    """
    Standard Bi-GRU encoder, hidden 256.
    """

    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
        hidden_size: int = 256,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            embed_dim,
            hidden_size,
            batch_first=True,
            bidirectional=bidirectional,
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.scorer = nn.Sequential(
            nn.Linear(2 * out_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, 1),
            nn.Sigmoid(),
        )
        self.out_dim = out_dim

    def encode(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        packed_output, h = self.gru(token_embeddings)
        if self.gru.bidirectional:
            h_cat = torch.cat([h[-2], h[-1]], dim=-1)
        else:
            h_cat = h[-1]
        return nn.functional.normalize(h_cat, p=2, dim=-1)

    def forward(
        self,
        token_embeddings_1: torch.Tensor,
        attention_mask_1: torch.Tensor,
        token_embeddings_2: torch.Tensor,
        attention_mask_2: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        z1 = self.encode(token_embeddings_1, attention_mask_1)
        z2 = self.encode(token_embeddings_2, attention_mask_2)
        pair = torch.cat([z1, z2], dim=-1)
        score = self.scorer(pair).squeeze(-1)
        return {"z1": z1, "z2": z2, "score": score}


class BiGRUEncoderOptimized(BiGRUEncoder):
    """
    Optimized Bi-GRU with hidden size 128, quantization-ready.
    """

    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
        hidden_size: int = 128,
        bidirectional: bool = True,
    ) -> None:
        super().__init__(
            embed_dim=embed_dim,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
        )

