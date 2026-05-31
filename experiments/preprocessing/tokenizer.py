from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from transformers import BertTokenizerFast


class BertSentenceTokenizer:
    """
    Thin wrapper around bert-base-uncased tokenizer with dynamic padding.
    """

    def __init__(self, model_name: str = "bert-base-uncased") -> None:
        self.tokenizer = BertTokenizerFast.from_pretrained(
            model_name,
            do_lower_case=True,
        )

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id

    @property
    def cls_token_id(self) -> int:
        return self.tokenizer.cls_token_id

    @property
    def sep_token_id(self) -> int:
        return self.tokenizer.sep_token_id

    def encode_batch(
        self, sentences: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return encoded["input_ids"], encoded["attention_mask"]

    def encode_pair_batch(
        self, sents1: List[str], sents2: List[str]
    ) -> Dict[str, torch.Tensor]:
        ids1, mask1 = self.encode_batch(sents1)
        ids2, mask2 = self.encode_batch(sents2)
        return {
            "input_ids_1": ids1,
            "attention_mask_1": mask1,
            "input_ids_2": ids2,
            "attention_mask_2": mask2,
        }


def get_default_tokenizer() -> BertSentenceTokenizer:
    return BertSentenceTokenizer()

