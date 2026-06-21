from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from datasets import DatasetDict, load_dataset
from torch.utils.data import Dataset

from .tokenizer import PAD_TOKEN_ID, tokenize

STS_FEATURES = ["sentence1", "sentence2", "score", "genre"]


def load_stsbenchmark() -> DatasetDict:
    ds: DatasetDict = load_dataset("mteb/stsbenchmark-sts")
    for split in ds.keys():
        ds[split] = ds[split].remove_columns(
            [c for c in ds[split].column_names if c not in STS_FEATURES]
        )
    return ds


class STSDataset(Dataset):
    """STS sentence-pair dataset that yields WordPiece token ids.

    Sentences are tokenized once up-front (no padding); dynamic per-batch padding
    happens in ``sts_collate_fn``. No embedding lookup occurs here — the dataset
    only produces ``input_ids`` / ``attention_mask`` (plus raw sentences, which
    the pretrained baselines and latency probes consume).
    """

    def __init__(self, hf_split, _tokenizer: Optional[object] = None) -> None:
        self.sentence1: List[str] = [row["sentence1"] for row in hf_split]
        self.sentence2: List[str] = [row["sentence2"] for row in hf_split]
        self.genre: List[str] = [row.get("genre", "") for row in hf_split]
        self.labels: torch.Tensor = torch.tensor(
            [float(row["score"]) / 5.0 for row in hf_split], dtype=torch.float32
        )
        # Pre-tokenize once (fast tokenizer); store unpadded id tensors.
        self.ids1: List[torch.Tensor] = tokenize(self.sentence1)
        self.ids2: List[torch.Tensor] = tokenize(self.sentence2)

    def __len__(self) -> int:
        return int(self.labels.size(0))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "input_ids_1": self.ids1[idx],
            "attention_mask_1": (self.ids1[idx] != PAD_TOKEN_ID).to(torch.long),
            "input_ids_2": self.ids2[idx],
            "attention_mask_2": (self.ids2[idx] != PAD_TOKEN_ID).to(torch.long),
            "label": self.labels[idx],
            "genre": self.genre[idx],
            "sentence1": self.sentence1[idx],
            "sentence2": self.sentence2[idx],
        }


def sts_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Dynamic padding: every batch is padded only to its own longest sequence.

    Produces ``(batch_size, dynamic_seq_len)`` tensors where ``dynamic_seq_len``
    varies between batches.
    """
    def pad(key: str) -> torch.Tensor:
        return torch.nn.utils.rnn.pad_sequence(
            [item[key] for item in batch], batch_first=True, padding_value=PAD_TOKEN_ID
        )

    input_ids_1 = pad("input_ids_1")
    input_ids_2 = pad("input_ids_2")
    return {
        "input_ids_1": input_ids_1,
        "attention_mask_1": (input_ids_1 != PAD_TOKEN_ID).to(torch.long),
        "input_ids_2": input_ids_2,
        "attention_mask_2": (input_ids_2 != PAD_TOKEN_ID).to(torch.long),
        "label": torch.stack([item["label"] for item in batch]),
        "sentence1": [item["sentence1"] for item in batch],
        "sentence2": [item["sentence2"] for item in batch],
    }
