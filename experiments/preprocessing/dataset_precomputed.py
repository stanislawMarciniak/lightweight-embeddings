from __future__ import annotations

from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset


class PrecomputedSTSDataset(Dataset):
    """
    STS dataset backed by precomputed token embeddings.

    Each .pt file is expected to contain:
        {
            "token_embeddings_1": Tensor (N, T, D),
            "attention_mask_1": Tensor (N, T),
            "token_embeddings_2": Tensor (N, T, D),
            "attention_mask_2": Tensor (N, T),
            "labels": Tensor (N),
            "genre": List[str],
            "sentence1": List[str],
            "sentence2": List[str],
        }
    """

    def __init__(self, path: str) -> None:
        data: Dict[str, Any] = torch.load(path, map_location="cpu")
        self.token_embeddings_1: torch.Tensor = data["token_embeddings_1"]
        self.attention_mask_1: torch.Tensor = data["attention_mask_1"]
        self.token_embeddings_2: torch.Tensor = data["token_embeddings_2"]
        self.attention_mask_2: torch.Tensor = data["attention_mask_2"]
        self.labels: torch.Tensor = data["labels"]
        self.genre: List[str] = data["genre"]
        self.sentence1: List[str] = data["sentence1"]
        self.sentence2: List[str] = data["sentence2"]

    def __len__(self) -> int:
        return int(self.labels.size(0))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "token_embeddings_1": self.token_embeddings_1[idx],
            "attention_mask_1": self.attention_mask_1[idx],
            "token_embeddings_2": self.token_embeddings_2[idx],
            "attention_mask_2": self.attention_mask_2[idx],
            "label": self.labels[idx],
            "genre": self.genre[idx],
            "sentence1": self.sentence1[idx],
            "sentence2": self.sentence2[idx],
        }

