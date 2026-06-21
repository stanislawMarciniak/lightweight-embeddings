from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class SimpleSRUCell(nn.Module):
    """
    Lightweight SRU-like recurrent unit.
    """

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.fc = nn.Linear(input_size + hidden_size, 3 * hidden_size)
        # Init forget-gate bias to 1 so the cell initially retains state (better gradient flow).
        with torch.no_grad():
            self.fc.bias[0:hidden_size].fill_(1.0)

    def forward(
        self, x_t: torch.Tensor, h_prev: torch.Tensor
    ) -> torch.Tensor:
        combined = torch.cat([x_t, h_prev], dim=-1)
        f, r, u = self.fc(combined).chunk(3, dim=-1)
        f = torch.sigmoid(f)
        r = torch.sigmoid(r)
        u = torch.tanh(u)
        h_t = f * h_prev + (1 - f) * u
        h_t = r * h_t
        return h_t


EMBED_DIM = 768


class SRUEncoder(nn.Module):
    """
    SRU-style encoder.
    Standard: hidden 256.
    """

    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
        hidden_size: int = 256,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = SimpleSRUCell(embed_dim, hidden_size)
        self.scorer = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

    def encode(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = token_embeddings.size()
        h = torch.zeros(
            batch_size, self.hidden_size, device=token_embeddings.device, dtype=token_embeddings.dtype
        )
        mask = attention_mask.to(token_embeddings.dtype).unsqueeze(-1)  # (B, T, 1)
        for t in range(seq_len):
            h_new = self.cell(token_embeddings[:, t, :], h)
            # Only update h on non-padding positions; keep previous h on padding.
            m_t = mask[:, t]  # (B, 1)
            h = h_new * m_t + h * (1.0 - m_t)
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


class SRUEncoderOptimized(SRUEncoder):
    """
    Optimized SRU encoder with hidden 128, quantization-ready.
    """

    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
        hidden_size: int = 128,
    ) -> None:
        super().__init__(embed_dim=embed_dim, hidden_size=hidden_size)

