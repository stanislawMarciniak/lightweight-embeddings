"""Thesis baseline encoders for STS benchmarking.

Three deterministic reference encoders with a unified interface:

* ``GloveMeanBaseline`` — O(n) embedding lookup + masked arithmetic mean (100-d).
* ``Model2VecBaseline`` — O(n) static 256-d vocabulary + Zipf-weighted sum.
* ``MiniLMBaseline`` — 6-layer MiniLM transformer + masked mean pooling (384-d).

Each class exposes ``encode`` / ``encode_single`` / ``similarity`` and a
``forward`` compatible with the STS benchmark (sentence pairs -> z1, z2, score).
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from preprocessing.tokenizer import HF_CACHE_DIR, HF_MODEL_NAME, MAX_LEN, VOCAB_SIZE, get_tokenizer

GLOVE_PATH = "data/glove.6B.100d.txt"
GLOVE_VOCAB_LIMIT = 50_000
GLOVE_DIM = 100
MODEL2VEC_DIM = 256
MODEL2VEC_MATRIX_PATH = "data/model2vec_static_256.pt"
MODEL2VEC_HF_NAME = "minishlab/potion-base-8M"
MINILM_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MINILM_DIM = 384


def _cosine_rows(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Pairwise cosine similarity, (B, D) x (B, D) -> (B,)."""
    return F.cosine_similarity(z1, z2, dim=-1)


def _sts_score_from_cosine(cos: torch.Tensor) -> torch.Tensor:
    """Map cosine in [-1, 1] to STS score in [0, 1]."""
    return (cos + 1.0) * 0.5


class BaselineEncoder(nn.Module, ABC):
    """Shared interface and STS benchmark adapter for all baselines."""

    embedding_dim: int

    @abstractmethod
    def encode(self, sentences: List[str]) -> torch.Tensor:
        """Return sentence embeddings of shape ``(batch_size, embedding_dim)``."""

    def encode_single(self, sentence: str) -> torch.Tensor:
        return self.encode([sentence])[0]

    @torch.no_grad()
    def similarity(self, sentence1: str, sentence2: str) -> float:
        z1 = self.encode_single(sentence1)
        z2 = self.encode_single(sentence2)
        return float(F.cosine_similarity(z1.unsqueeze(0), z2.unsqueeze(0)).item())

    def forward(
        self,
        input_ids_1: torch.Tensor,
        attention_mask_1: torch.Tensor,
        input_ids_2: torch.Tensor,
        attention_mask_2: torch.Tensor,
        sentence1: List[str],
        sentence2: List[str],
        **kwargs: object,
    ) -> Dict[str, torch.Tensor]:
        del input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, kwargs
        z1 = self.encode(sentence1)
        z2 = self.encode(sentence2)
        cos = _cosine_rows(z1, z2)
        return {"z1": z1, "z2": z2, "score": _sts_score_from_cosine(cos)}


class GloveMeanBaseline(BaselineEncoder):
    """Mean-pooling GloVe-100d baseline — absolute computational lower bound.

    Pipeline: whitespace tokenization -> O(1) lookup -> masked arithmetic mean.

    * **Embedding dimension:** 100
    * **Trainable parameters:** 0 (static lookup table stored as buffer)
    * **Table size:** 50,001 x 100 (~19 MB); index 0 is the zero vector for OOV/pad
    * **Complexity:** O(n) per sentence (n = token count)
    * **Expected latency:** lowest among baselines (lookup + sum only)
    * **Memory:** ~19 MB static table + batch buffers
    """

    embedding_dim = GLOVE_DIM

    def __init__(self, glove_path: str = GLOVE_PATH, vocab_limit: int = GLOVE_VOCAB_LIMIT) -> None:
        super().__init__()
        table, word2idx = self._build_table(glove_path, vocab_limit)
        self.register_buffer("embedding_table", table)
        self.word2idx = word2idx

    @staticmethod
    def _build_table(path: str, vocab_limit: int) -> tuple[torch.Tensor, Dict[str, int]]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"GloVe file not found at {path}. Download glove.6B.100d.txt to data/."
            )
        # Row 0: zero vector for OOV / padding.
        table = torch.zeros(vocab_limit + 1, GLOVE_DIM, dtype=torch.float32)
        word2idx: Dict[str, int] = {}
        loaded = 0
        with open(path, "r", encoding="utf8") as f:
            for line in f:
                if loaded >= vocab_limit:
                    break
                parts = line.strip().split()
                if len(parts) != GLOVE_DIM + 1:
                    continue
                word = parts[0]
                loaded += 1
                word2idx[word] = loaded  # indices 1..vocab_limit
                table[loaded] = torch.tensor([float(x) for x in parts[1:]], dtype=torch.float32)
        return table, word2idx

    def _tokenize(self, sentences: List[str]) -> torch.Tensor:
        rows: List[List[int]] = []
        for s in sentences:
            words = s.lower().split()
            ids = [self.word2idx.get(w, 0) for w in words]
            rows.append(ids or [0])
        max_len = max(len(r) for r in rows)
        padded = [r + [0] * (max_len - len(r)) for r in rows]
        device = self.embedding_table.device
        return torch.tensor(padded, dtype=torch.long, device=device)

    @torch.no_grad()
    def encode(self, sentences: List[str]) -> torch.Tensor:
        ids = self._tokenize(sentences)                       # (B, T)
        emb = F.embedding(ids, self.embedding_table)          # (B, T, 100)
        mask = (ids != 0).unsqueeze(-1).to(emb.dtype)         # (B, T, 1)
        summed = (emb * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        return summed / counts                                # (B, 100) — no normalization


class Model2VecBaseline(BaselineEncoder):
    """Distilled static embedding baseline (Model2Vec).

    Primary path: load a pretrained ``model2vec.StaticModel`` (default:
    ``minishlab/potion-base-8M``, 256-d). This is a real distilled static
    encoder — O(n) token lookup + weighted pooling, no transformer forward pass.

    Offline fallback (no Hub / import failure): PCA-compress frozen BERT token
    embeddings to 256-d, then Zipf-weighted pooling over ``bert-base-uncased``
    WordPiece tokens (~0.59 test Pearson vs ~0.55 for a random projection).

    * **Embedding dimension:** 256
    * **Trainable parameters:** 0
    * **Expected latency:** low (static lookup only)
    """

    embedding_dim = MODEL2VEC_DIM

    def __init__(
        self,
        model_name: str = MODEL2VEC_HF_NAME,
        matrix_path: str = MODEL2VEC_MATRIX_PATH,
        zipf_t: float = 1.0,
        cache_dir: str = HF_CACHE_DIR,
    ) -> None:
        super().__init__()
        self.register_buffer("_device_anchor", torch.zeros(0))
        self._backend: str
        try:
            from model2vec import StaticModel

            self.static_model = StaticModel.from_pretrained(model_name)
            self._backend = "static_model"
            self.embedding_dim = self.static_model.dim
        except Exception:
            self._backend = "bert_pca"
            self.tokenizer = get_tokenizer()
            matrix = self._load_pca_matrix(matrix_path, cache_dir)
            self.register_buffer("embedding_table", matrix)
            self.register_buffer("_zipf_alpha", self._build_zipf_alphas(VOCAB_SIZE, zipf_t))

    @staticmethod
    def _load_pca_matrix(path: str, cache_dir: str) -> torch.Tensor:
        if os.path.exists(path):
            data = torch.load(path, map_location="cpu", weights_only=True)
            if isinstance(data, dict) and "embedding" in data:
                data = data["embedding"]
            matrix = data.float()
            if matrix.shape != (VOCAB_SIZE, MODEL2VEC_DIM):
                raise ValueError(
                    f"Expected matrix shape ({VOCAB_SIZE}, {MODEL2VEC_DIM}), got {tuple(matrix.shape)}"
                )
            return matrix

        from transformers import AutoModel

        bert = AutoModel.from_pretrained(HF_MODEL_NAME, cache_dir=cache_dir)
        weight = bert.get_input_embeddings().weight.detach().float()
        del bert
        _, _, V = torch.pca_lowrank(weight, q=MODEL2VEC_DIM, center=True, niter=4)
        matrix = weight @ V[:, :MODEL2VEC_DIM]
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(matrix, path)
        return matrix

    @staticmethod
    def _build_zipf_alphas(vocab_size: int, t: float) -> torch.Tensor:
        ids = torch.arange(vocab_size, dtype=torch.float32)
        freq = torch.clamp(vocab_size - ids, min=1.0)
        alpha = torch.clamp(1.0 - torch.sqrt(torch.tensor(t, dtype=torch.float32) / freq), min=0.0)
        alpha[0] = 0.0
        return alpha

    def _tokenize_batch(self, sentences: List[str]) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        device = self.embedding_table.device
        return encoded["input_ids"].to(device), encoded["attention_mask"].to(device)

    @torch.no_grad()
    def encode(self, sentences: List[str]) -> torch.Tensor:
        device = self._device_anchor.device
        if self._backend == "static_model":
            vecs = self.static_model.encode(sentences, max_length=MAX_LEN)
            return torch.as_tensor(vecs, dtype=torch.float32, device=device)

        ids, mask = self._tokenize_batch(sentences)
        emb = F.embedding(ids, self.embedding_table)
        alpha = self._zipf_alpha[ids] * mask.to(emb.dtype)
        weighted = (emb * alpha.unsqueeze(-1)).sum(dim=1)
        return weighted


class MiniLMBaseline(BaselineEncoder):
    """MiniLM-L6-v2 transformer baseline — expensive upper bound.

    Pipeline: WordPiece tokenization -> 6-layer MiniLM -> last hidden state
    -> masked mean pooling (no CLS-only pooling).

        mean = sum(mask * token) / sum(mask)

    * **Embedding dimension:** 384
    * **Trainable parameters:** 0 (frozen pretrained transformer)
    * **Parameter count:** ~22.7M (all frozen)
    * **Complexity:** O(n * L * d^2) with L=6 layers; quadratic in d per layer
    * **Expected latency:** highest among baselines
    * **Memory:** ~90 MB model weights + activations per batch
    """

    embedding_dim = MINILM_DIM

    def __init__(self, model_name: str = MINILM_MODEL_NAME, cache_dir: str = HF_CACHE_DIR) -> None:
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, cache_dir=cache_dir)
        self.transformer = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        self.transformer.eval()
        for param in self.transformer.parameters():
            param.requires_grad = False

    def _tokenize_batch(self, sentences: List[str]) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        device = next(self.transformer.parameters()).device
        return encoded["input_ids"].to(device), encoded["attention_mask"].to(device)

    @torch.no_grad()
    def encode(self, sentences: List[str]) -> torch.Tensor:
        ids, mask = self._tokenize_batch(sentences)
        hidden = self.transformer(input_ids=ids, attention_mask=mask).last_hidden_state  # (B,T,384)
        m = mask.unsqueeze(-1).to(hidden.dtype)
        summed = (hidden * m).sum(dim=1)
        counts = m.sum(dim=1).clamp(min=1e-9)
        return summed / counts  # (B, 384) masked mean, no extra normalization
