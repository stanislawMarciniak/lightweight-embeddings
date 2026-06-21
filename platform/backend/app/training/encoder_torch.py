"""PyTorch CompactSimilarityModel encoder (training/export only).

Matches experiments/models/custom.py encode path:
  (B, T, 768) -> HybridAttentionPooling -> GatedProjection -> ScaledL2Normalization -> (B, 128)

Production serving uses the torch-free numpy runtime in semantic_model.py.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ENCODER_PREFIXES = ("pool.", "project.", "norm.")
EMBED_DIM = 768
SENT_DIM = 128


class HybridAttentionPooling(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(dim, 1)
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.gamma = nn.Parameter(torch.full((1,), 0.08))

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(x.dtype)
        scores = self.score(x) * self.alpha + self.beta
        weights = torch.sigmoid(scores) * mask
        weighted_sum = (weights * x).sum(dim=1)
        mean = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return weighted_sum + self.gamma * mean


class GatedProjection(nn.Module):
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
    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        norm = h.norm(2, dim=-1, keepdim=True)
        return h / (norm + self.eps) * self.scale


class CustomEncoder(nn.Module):
    """Sentence encoder over 768-d WordPiece token embeddings -> 128-d output."""

    def __init__(self, embed_dim: int = EMBED_DIM, sent_dim: int = SENT_DIM) -> None:
        super().__init__()
        self.pool = HybridAttentionPooling(embed_dim)
        self.project = GatedProjection(embed_dim, sent_dim)
        self.norm = ScaledL2Normalization(sent_dim)

    def forward(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(token_embeddings, attention_mask)
        z = self.project(pooled)
        return self.norm(z)


def load_weights(model: CustomEncoder, path: str) -> CustomEncoder:
    """Load encoder weights from a torch .pt checkpoint or a numpy .npz export."""
    if path.endswith(".npz"):
        data = np.load(path)
        sd = {k: torch.from_numpy(data[k]) for k in data.files if not k.startswith("meta_")}
        model.load_state_dict(sd, strict=True)
    else:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        enc_sd = {k: v for k, v in sd.items() if k.startswith(ENCODER_PREFIXES)}
        model.load_state_dict(enc_sd, strict=True)
    return model


def export_npz(model: CustomEncoder, path: str, best_val_pearson: float = float("nan")) -> None:
    """Export encoder weights to the torch-free .npz consumed by semantic_model.py."""
    sd = model.state_dict()
    arrays = {k: v.detach().cpu().float().numpy() for k, v in sd.items()}
    meta = {
        "meta_embed_dim": np.int64(EMBED_DIM),
        "meta_sent_dim": np.int64(arrays["norm.scale"].shape[0]),
        "meta_best_val_pearson": np.float64(best_val_pearson),
    }
    np.savez(path, **arrays, **meta)


def export_onnx(model: CustomEncoder, path: str, opset: int = 17) -> None:
    """Export encode() to ONNX with dynamic batch/seq axes."""
    model.eval()
    dummy_emb = torch.randn(1, 6, EMBED_DIM)
    dummy_mask = torch.ones(1, 6)
    kwargs = dict(
        input_names=["token_embeddings", "attention_mask"],
        output_names=["embedding"],
        dynamic_axes={
            "token_embeddings": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "embedding": {0: "batch"},
        },
        opset_version=opset,
    )
    try:
        torch.onnx.export(model, (dummy_emb, dummy_mask), path, dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(model, (dummy_emb, dummy_mask), path, **kwargs)
