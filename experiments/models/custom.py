from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(2, dim=-1, keepdim=True)
        return x / (norm + self.eps) * self.scale


class FastAttentionPooling(nn.Module):
    """Masked softmax attention pooling over tokens."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.attention = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.attention(x).squeeze(-1)
        scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = weights.nan_to_num(0.0)
        return (x * weights.unsqueeze(-1)).sum(dim=1)


EMBED_DIM = 100


class CustomHybridModel(nn.Module):
    """
    Fast attention-pooled encoder with gated projection and rich pair scoring.

    Architecture keeps the essentials that drive Pearson correlation while
    staying close to DAN-level inference speed:
        1. Single Conv1d (kernel=3) captures local bigram context cheaply.
        2. Attention pooling learns which tokens matter for similarity.
        3. Gated projection compresses the representation expressively.
        4. Scorer receives rich pair features (z1, z2, |diff|, prod, cos).

    ~70K parameters, single-model inference (no ensemble overhead).
    """

    def __init__(self, vocab_size: int, embed_dim: int = EMBED_DIM) -> None:
        super().__init__()
        hidden = 128

        self.conv = nn.Conv1d(embed_dim, hidden, kernel_size=3, padding=1)
        self.norm = RMSNorm(hidden)
        self.drop = nn.Dropout(0.15)

        self.attn_pool = FastAttentionPooling(hidden)

        sent_dim = 64
        self.gate = nn.Linear(hidden, sent_dim * 2)
        self.sent_norm = RMSNorm(sent_dim)

        pair_dim = sent_dim * 4 + 1
        self.scorer = nn.Sequential(
            nn.Linear(pair_dim, 96),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(96, 1),
            nn.Sigmoid(),
        )

    def encode(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.conv(token_embeddings.transpose(1, 2)).transpose(1, 2)
        x = F.gelu(x)
        x = self.norm(x)
        x = self.drop(x)

        z = self.attn_pool(x, attention_mask)

        gate_out = self.gate(z)
        h, g = gate_out.chunk(2, dim=-1)
        z = h * torch.sigmoid(g)
        z = self.sent_norm(z)
        return F.normalize(z, p=2, dim=-1)

    def forward(
        self,
        token_embeddings_1: torch.Tensor,
        attention_mask_1: torch.Tensor,
        token_embeddings_2: torch.Tensor,
        attention_mask_2: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        z1 = self.encode(token_embeddings_1, attention_mask_1)
        z2 = self.encode(token_embeddings_2, attention_mask_2)
        diff = (z1 - z2).abs()
        prod = z1 * z2
        cos = F.cosine_similarity(z1, z2).unsqueeze(-1)
        pair = torch.cat([z1, z2, diff, prod, cos], dim=-1)
        score = self.scorer(pair).squeeze(-1)
        return {"z1": z1, "z2": z2, "score": score}
