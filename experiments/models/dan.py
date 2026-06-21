from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


EMBED_DIM = 768


class DeepAveragingNetwork(nn.Module):
    """
    Standard DAN with 3 layers, hidden 768.
    """

    def __init__(self, embed_dim: int = EMBED_DIM) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim, 768),
            nn.GELU(),
            nn.Linear(768, 768),
            nn.GELU(),
            nn.Linear(768, 768),
            nn.GELU(),
        )
        self.scorer = nn.Sequential(
            nn.Linear(2 * 768, 768),
            nn.GELU(),
            nn.Linear(768, 1),
            nn.Sigmoid(),
        )

    def encode(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # token_embeddings: (B, T, D), attention_mask: (B, T)
        mask = attention_mask.unsqueeze(-1).float()
        summed = (token_embeddings * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        mean = summed / counts
        h = self.encoder(mean)
        return nn.functional.normalize(h, p=2, dim=-1)

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


class DeepAveragingNetworkOptimized(nn.Module):
    """
    Optimized DAN with 2 layers, hidden 128 and INT8-ready.
    """

    def __init__(self, embed_dim: int = EMBED_DIM) -> None:
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.act = nn.GELU()
        self.scorer = nn.Sequential(
            nn.Linear(2 * 128, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def encode(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        summed = (token_embeddings * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        mean = summed / counts
        h = self.act(self.fc1(mean))
        h = self.act(self.fc2(h))
        return nn.functional.normalize(h, p=2, dim=-1)

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

