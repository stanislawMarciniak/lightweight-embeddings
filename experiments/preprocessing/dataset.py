from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from datasets import DatasetDict, load_dataset
from torch.utils.data import Dataset

from .tokenizer import BertSentenceTokenizer, get_default_tokenizer


STS_FEATURES = ["sentence1", "sentence2", "score", "genre"]


def load_stsbenchmark() -> DatasetDict:
    ds: DatasetDict = load_dataset("mteb/stsbenchmark-sts")
    # Keep only required features
    for split in ds.keys():
        ds[split] = ds[split].remove_columns(
            [c for c in ds[split].column_names if c not in STS_FEATURES]
        )
    return ds


@dataclass
class STSExample:
    input_ids_1: torch.Tensor
    attention_mask_1: torch.Tensor
    input_ids_2: torch.Tensor
    attention_mask_2: torch.Tensor
    label: torch.Tensor
    genre: str


class STSDataset(Dataset):
    def __init__(
        self,
        hf_split,
        tokenizer: Optional[BertSentenceTokenizer] = None,
    ) -> None:
        self.data = hf_split
        self.tokenizer = tokenizer or get_default_tokenizer()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.data[idx]
        s1: str = row["sentence1"]
        s2: str = row["sentence2"]
        score: float = float(row["score"])
        genre: str = row["genre"]

        encoded = self.tokenizer.encode_pair_batch([s1], [s2])
        # Remove batch dim
        ex = STSExample(
            input_ids_1=encoded["input_ids_1"].squeeze(0),
            attention_mask_1=encoded["attention_mask_1"].squeeze(0),
            input_ids_2=encoded["input_ids_2"].squeeze(0),
            attention_mask_2=encoded["attention_mask_2"].squeeze(0),
            label=torch.tensor(score / 5.0, dtype=torch.float32),
            genre=genre,
        )
        return {
            "input_ids_1": ex.input_ids_1,
            "attention_mask_1": ex.attention_mask_1,
            "input_ids_2": ex.input_ids_2,
            "attention_mask_2": ex.attention_mask_2,
            "label": ex.label,
            "genre": ex.genre,
            "sentence1": s1,
            "sentence2": s2,
        }


def sts_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Dynamic padding per batch, right padding
    input_ids_1 = [item["input_ids_1"] for item in batch]
    attention_mask_1 = [item["attention_mask_1"] for item in batch]
    input_ids_2 = [item["input_ids_2"] for item in batch]
    attention_mask_2 = [item["attention_mask_2"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])
    genres = [item["genre"] for item in batch]
    sentence1 = [item["sentence1"] for item in batch]
    sentence2 = [item["sentence2"] for item in batch]

    input_ids_1 = torch.nn.utils.rnn.pad_sequence(
        input_ids_1, batch_first=True, padding_value=0
    )
    attention_mask_1 = torch.nn.utils.rnn.pad_sequence(
        attention_mask_1, batch_first=True, padding_value=0
    )
    input_ids_2 = torch.nn.utils.rnn.pad_sequence(
        input_ids_2, batch_first=True, padding_value=0
    )
    attention_mask_2 = torch.nn.utils.rnn.pad_sequence(
        attention_mask_2, batch_first=True, padding_value=0
    )

    return {
        "input_ids_1": input_ids_1,
        "attention_mask_1": attention_mask_1,
        "input_ids_2": input_ids_2,
        "attention_mask_2": attention_mask_2,
        "label": labels,
        "sentence1": sentence1,
        "sentence2": sentence2,
    }

