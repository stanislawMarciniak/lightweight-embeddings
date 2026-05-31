from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn


EMBED_DIM = 100


class TextCNN(nn.Module):
    """
    Standard 1D CNN text encoder.
    filter_sizes: [2,3,4], channels: 256
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = EMBED_DIM,
        filter_sizes: List[int] | None = None,
        num_channels: int = 256,
    ) -> None:
        super().__init__()
        filter_sizes = filter_sizes or [2, 3, 4]
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embed_dim,
                    out_channels=num_channels,
                    kernel_size=fs,
                )
                for fs in filter_sizes
            ]
        )
        self.scorer = nn.Sequential(
            nn.Linear(2 * num_channels * len(filter_sizes), 512),
            nn.GELU(),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    def encode(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # token_embeddings: (B, T, D)
        emb = token_embeddings.transpose(1, 2)  # (B, D, T)
        conv_outs = []
        for conv in self.convs:
            x = torch.relu(conv(emb))  # (B, C, T')
            x = torch.max_pool1d(x, kernel_size=x.size(2)).squeeze(2)
            conv_outs.append(x)
        h = torch.cat(conv_outs, dim=1)
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


class TextCNNOptimized(nn.Module):
    """
    Optimized 1D CNN with filter_sizes [2,3], channels 64, INT8-ready.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = EMBED_DIM,
        filter_sizes: List[int] | None = None,
        num_channels: int = 64,
    ) -> None:
        super().__init__()
        filter_sizes = filter_sizes or [2, 3]
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embed_dim,
                    out_channels=num_channels,
                    kernel_size=fs,
                )
                for fs in filter_sizes
            ]
        )
        self.scorer = nn.Sequential(
            nn.Linear(2 * num_channels * len(filter_sizes), 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def encode(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        emb = token_embeddings.transpose(1, 2)
        conv_outs = []
        for conv in self.convs:
            x = torch.relu(conv(emb))
            x = torch.max_pool1d(x, kernel_size=x.size(2)).squeeze(2)
            conv_outs.append(x)
        h = torch.cat(conv_outs, dim=1)
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

