from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA


class GloveMeanEncoder(nn.Module):
    """
    GloVe-100d mean pooling encoder.
    """

    def __init__(self, glove_path: str = "data/glove.6B.100d.txt") -> None:
        super().__init__()
        self.glove_dim = 100
        self.word2idx: Dict[str, int] = {}
        embeddings = self._load_glove(glove_path)
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embeddings, dtype=torch.float32),
            freeze=True,
        )

    def _load_glove(self, path: str) -> np.ndarray:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"GloVe file not found at {path}. Please download glove.6B.100d."
            )
        vectors = []
        self.word2idx = {}
        with open(path, "r", encoding="utf8") as f:
            for idx, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) != self.glove_dim + 1:
                    continue
                word = parts[0]
                vec = np.asarray(parts[1:], dtype=np.float32)
                self.word2idx[word] = idx
                vectors.append(vec)
        embeddings = np.stack(vectors)
        return embeddings

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids assumed to be word indices in GloVe vocab; for simplicity,
        # treat them as indices and mean-pool.
        emb = self.embedding(token_ids)  # (B, T, 100)
        mask = (token_ids != 0).float().unsqueeze(-1)
        summed = (emb * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        mean = summed / counts
        return nn.functional.normalize(mean, p=2, dim=-1)


class SentenceTransformerEncoder(nn.Module):
    """
    Wrapper for sentence-transformers models.
    """

    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.model = SentenceTransformer(model_name)
        self.out_dim = self.model.get_sentence_embedding_dimension()

    def forward(self, sentences: Tuple[str, ...]) -> torch.Tensor:
        with torch.no_grad():
            emb = self.model.encode(
                list(sentences),
                convert_to_tensor=True,
                normalize_embeddings=True,
            )
        return emb


class Model2VecBGEEncoder(nn.Module):
    """
    Compressed BGE-based encoder with optional PCA and INT8 simulation.

    PCA is fitted once on the full training set via `fit_pca_on_training`
    and then reused for all encoder instances.
    """

    _shared_pca: PCA | None = None

    @classmethod
    def fit_pca_on_training(
        cls,
        sentences: List[str],
        model_name: str = "sentence-transformers/BAAI/bge-small-en-v1.5",
        target_dim: int = 256,
    ) -> None:
        """
        Fit PCA on sentence embeddings computed over the full training set.
        """
        base = SentenceTransformer(model_name)
        with torch.no_grad():
            emb = base.encode(list(sentences), convert_to_numpy=True)
        base_dim = emb.shape[1]
        if target_dim >= base_dim:
            cls._shared_pca = None
            return
        pca = PCA(n_components=target_dim)
        pca.fit(emb)
        cls._shared_pca = pca

    def __init__(
        self,
        model_name: str = "sentence-transformers/BAAI/bge-small-en-v1.5",
        target_dim: int = 256,
        simulate_int8: bool = True,
    ) -> None:
        super().__init__()
        self.base = SentenceTransformer(model_name)
        base_dim = self.base.get_sentence_embedding_dimension()
        self.simulate_int8 = simulate_int8

        if target_dim < base_dim:
            if Model2VecBGEEncoder._shared_pca is None:
                raise RuntimeError(
                    "PCA has not been fitted for Model2VecBGEEncoder. "
                    "Call Model2VecBGEEncoder.fit_pca_on_training(...) before use."
                )
            self.pca = Model2VecBGEEncoder._shared_pca
            self.out_dim = self.pca.n_components
        else:
            self.pca = None
            self.out_dim = base_dim

    def forward(self, sentences: Tuple[str, ...]) -> torch.Tensor:
        with torch.no_grad():
            emb = self.base.encode(list(sentences), convert_to_numpy=True)
        if self.pca is not None:
            emb = self.pca.transform(emb)
        emb_t = torch.tensor(emb, dtype=torch.float32)
        if self.simulate_int8:
            scale = 127.0 / (emb_t.abs().max(dim=-1, keepdim=True).values + 1e-6)
            q = torch.clamp((emb_t * scale).round(), -128, 127).to(torch.int8)
            emb_t = q.float() / scale
        emb_t = nn.functional.normalize(emb_t, p=2, dim=-1)
        return emb_t


def _cosine_similarity(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Cosine similarity; z1, z2 assumed L2-normalized -> (batch,) in [-1, 1]."""
    return (z1 * z2).sum(dim=-1)


def _sts_score_from_cosine(cos: torch.Tensor) -> torch.Tensor:
    """Map cosine in [-1, 1] to score in [0, 1] (STS label range)."""
    return (cos + 1.0) * 0.5


class STSGloveMeanWrapper(nn.Module):
    """
    Wraps GloveMeanEncoder for STS benchmark: batch of sentence pairs -> z1, z2, score.
    Uses word-level tokenization and GloVe vocab; OOV/padding -> 0.
    """

    def __init__(self, glove_path: str = "data/glove.6B.100d.txt") -> None:
        super().__init__()
        self.encoder = GloveMeanEncoder(glove_path=glove_path)
        self.word2idx = self.encoder.word2idx

    def _sentences_to_ids(self, sentences: List[str], device: torch.device) -> torch.Tensor:
        rows = []
        for s in sentences:
            words = s.lower().split()
            ids = [self.word2idx.get(w, 0) for w in words]
            if not ids:
                ids = [0]
            rows.append(ids)
        max_len = max(len(r) for r in rows)
        padded = [r + [0] * (max_len - len(r)) for r in rows]
        return torch.tensor(padded, dtype=torch.long, device=device)

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
        device = next(self.encoder.parameters()).device
        ids_1 = self._sentences_to_ids(sentence1, device)
        ids_2 = self._sentences_to_ids(sentence2, device)
        z1 = self.encoder(ids_1)
        z2 = self.encoder(ids_2)
        cos = _cosine_similarity(z1, z2)
        score = _sts_score_from_cosine(cos)
        return {"z1": z1, "z2": z2, "score": score}


class STSSentenceTransformerWrapper(nn.Module):
    """
    Wraps SentenceTransformerEncoder for STS: sentence pairs -> z1, z2, score.
    """

    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.encoder = SentenceTransformerEncoder(model_name)

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
        z1 = self.encoder(tuple(sentence1))
        z2 = self.encoder(tuple(sentence2))
        cos = _cosine_similarity(z1, z2)
        score = _sts_score_from_cosine(cos)
        return {"z1": z1, "z2": z2, "score": score}


class STSModel2VecBGEWrapper(nn.Module):
    """
    Wraps Model2VecBGEEncoder for STS: sentence pairs -> z1, z2, score.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/BAAI/bge-small-en-v1.5",
        target_dim: int = 256,
        simulate_int8: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = Model2VecBGEEncoder(
            model_name=model_name,
            target_dim=target_dim,
            simulate_int8=simulate_int8,
        )

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
        z1 = self.encoder(tuple(sentence1))
        z2 = self.encoder(tuple(sentence2))
        cos = _cosine_similarity(z1, z2)
        score = _sts_score_from_cosine(cos)
        return {"z1": z1, "z2": z2, "score": score}

