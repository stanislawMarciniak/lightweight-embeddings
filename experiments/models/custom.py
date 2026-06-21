from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

EMBED_DIM = 768
SENT_DIM = 128


class HybridAttentionPooling(nn.Module):
    """O(n) sigmoid-gated pooling blended with masked mean (no softmax).

        scores = linear(x)
        scores = scores * alpha + beta
        weights = sigmoid(scores)            # masked to ignore padding
        weighted_sum = sum(weights * x)
        mean = masked_mean(x)
        output = weighted_sum + gamma * mean

    alpha, beta, gamma are learnable scalars (gamma initialized to 0.08).
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(dim, 1)
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.gamma = nn.Parameter(torch.full((1,), 0.08))

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(x.dtype)  # (B, T, 1)

        scores = self.score(x)                  # (B, T, 1)
        scores = scores * self.alpha + self.beta
        weights = torch.sigmoid(scores) * mask  # zero weight on padding
        weighted_sum = (weights * x).sum(dim=1)  # (B, D)

        mean = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return weighted_sum + self.gamma * mean


class GatedProjection(nn.Module):
    """Gated projection with residual, using three independent Linear layers.

        h = fc(x)
        g = sigmoid(gate(x))
        r = residual(x)
        output = h * g + r
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.gate = nn.Linear(in_dim, out_dim)
        self.residual = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc(x)
        g = torch.sigmoid(self.gate(x))
        r = self.residual(x)
        return h * g + r


class ScaledL2Normalization(nn.Module):
    """L2-normalize then rescale by a learnable per-dimension vector (init ones).

        y = h / (||h|| + eps)
        y = y * scale
    """

    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        norm = h.norm(2, dim=-1, keepdim=True)
        return h / (norm + self.eps) * self.scale


class CompactSimilarityModel(nn.Module):
    """Compact STS encoder over 768-d WordPiece token embeddings.

    The token-embedding lookup lives outside this module: ``encode`` receives
    ``token_embeddings`` of shape ``(batch, seq_len, 768)`` (never input_ids).

        token_embeddings (B, T, 768)
            -> HybridAttentionPooling   -> (B, 768)
            -> GatedProjection 768->128 -> (B, 128)
            -> ScaledL2Normalization    -> (B, 128)

    A lightweight pair scorer reads [z1, z2, |z1-z2|, z1*z2, cos]. Encoder core
    (pooling + projection + norm) is ~296k parameters.
    """

    def __init__(self, embed_dim: int = EMBED_DIM, sent_dim: int = SENT_DIM) -> None:
        super().__init__()
        self.pool = HybridAttentionPooling(embed_dim)
        self.project = GatedProjection(embed_dim, sent_dim)
        self.norm = ScaledL2Normalization(sent_dim)

        pair_dim = sent_dim * 4 + 1
        self.scorer = nn.Sequential(
            nn.Linear(pair_dim, 96),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(96, 1),
            nn.Sigmoid(),
        )

    def encode(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # token_embeddings: (batch, seq_len, 768)
        pooled = self.pool(token_embeddings, attention_mask)
        z = self.project(pooled)
        z = self.norm(z)
        return z

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


# Backwards-compatible alias: the registry / framework instantiate CustomHybridModel().
CustomHybridModel = CompactSimilarityModel
