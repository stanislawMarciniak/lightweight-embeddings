"""Deployment footprint estimates for benchmark reporting.

Methodology — two axes that must never be mixed:

(A) **learned_encoder_size_mb** (research / architecture axis)
    Trainable encoder checkpoint weights only. Excludes tokenizers, GloVe/BERT/
    Model2Vec lookup tables, and any external pipeline components.

(B) **system_size_mb** (engineering / deployment axis)
    Full standalone footprint: learned encoder + tokenizer + embedding tables.

Component breakdown (CSV transparency only):
  * ``tokenizer_size_mb`` — tokenizer vocab files (system only)
  * ``embedding_table_size_mb`` — frozen lookup tables (system only)
  * ``model_size_mb`` / ``total_size_mb`` — legacy aliases kept for compatibility
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch.nn as nn

from models.baselines import (
    GloveMeanBaseline,
    MINILM_MODEL_NAME,
    MODEL2VEC_HF_NAME,
    MiniLMBaseline,
    Model2VecBaseline,
)
from preprocessing.tokenizer import HF_CACHE_DIR, HF_MODEL_NAME
from registry import is_baseline
from utils.quantization import model_size_mb

_TOKENIZER_FILENAMES = frozenset(
    {
        "tokenizer.json",
        "vocab.txt",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "merges.txt",
    }
)


@dataclass(frozen=True, slots=True)
class DeploymentSize:
    """Size breakdown for one model precision block."""

    learned_encoder_size_mb: float
    tokenizer_size_mb: float
    embedding_table_size_mb: float
    requires_bundled_tokenizer: bool

    @property
    def system_size_mb(self) -> float:
        return (
            self.learned_encoder_size_mb
            + self.tokenizer_size_mb
            + self.embedding_table_size_mb
        )

    # Legacy aliases (do not use in new plots).
    @property
    def model_size_mb(self) -> float:
        return self.learned_encoder_size_mb

    @property
    def total_size_mb(self) -> float:
        return self.system_size_mb

    @property
    def learned_size_mb(self) -> float:
        return self.learned_encoder_size_mb

    @property
    def core_size_mb(self) -> float:
        return self.learned_encoder_size_mb + self.embedding_table_size_mb


def _bytes_to_mb(n: int) -> float:
    return n / (1024**2)


def _hf_cache_roots() -> list[str]:
    roots: list[str] = []
    if HF_CACHE_DIR:
        roots.append(HF_CACHE_DIR)
        roots.append(os.path.join(HF_CACHE_DIR, "hub"))
    hub = os.environ.get("HF_HUB_CACHE")
    if hub:
        roots.append(hub)
    roots.append(os.path.expanduser("~/.cache/huggingface/hub"))
    seen: set[str] = set()
    unique: list[str] = []
    for r in roots:
        if r not in seen:
            seen.add(r)
            unique.append(r)
    return unique


def _hf_repo_cache_dir(repo_id: str) -> str | None:
    slugs = {
        repo_id.replace("/", "--"),
        repo_id.replace("/", "--").lower(),
    }
    for slug in slugs:
        for root in _hf_cache_roots():
            path = os.path.join(root, f"models--{slug}")
            if os.path.isdir(path):
                return path
    return None


def _hf_tokenizer_files_mb(repo_id: str) -> float:
    repo_dir = _hf_repo_cache_dir(repo_id)
    if repo_dir is None:
        return 0.0
    total = 0
    for root, _, files in os.walk(repo_dir):
        for name in files:
            if name in _TOKENIZER_FILENAMES:
                total += os.path.getsize(os.path.join(root, name))
    return _bytes_to_mb(total)


def _bert_embedding_table_mb() -> float:
    from models.embeddings import get_token_embedding

    weight = get_token_embedding().embedding.weight
    return _bytes_to_mb(weight.numel() * weight.element_size())


def _glove_table_mb(model: GloveMeanBaseline) -> float:
    if hasattr(model, "embedding_table"):
        t = model.embedding_table
        return _bytes_to_mb(t.numel() * t.element_size())
    return 0.0


def _model2vec_table_mb(model: Model2VecBaseline) -> float:
    if getattr(model, "_backend", None) != "static_model":
        if hasattr(model, "embedding_table"):
            t = model.embedding_table
            return _bytes_to_mb(t.numel() * t.element_size())
        return 0.0
    sm = model.static_model
    total = sm.embedding.nbytes
    if sm.weights is not None:
        total += sm.weights.nbytes
    if sm.token_mapping is not None:
        total += sm.token_mapping.nbytes
    return _bytes_to_mb(total)


def _model2vec_tokenizer_mb(model: Model2VecBaseline) -> float:
    if getattr(model, "_backend", None) != "static_model":
        return _hf_tokenizer_files_mb(HF_MODEL_NAME)
    return _hf_tokenizer_files_mb(MODEL2VEC_HF_NAME)


def estimate_deployment_size(name: str, model: nn.Module) -> DeploymentSize:
    """Return deployment size breakdown for ``name`` / ``model``."""
    if not is_baseline(name):
        learned = model_size_mb(model)
        return DeploymentSize(
            learned_encoder_size_mb=learned,
            tokenizer_size_mb=_hf_tokenizer_files_mb(HF_MODEL_NAME),
            embedding_table_size_mb=_bert_embedding_table_mb(),
            requires_bundled_tokenizer=False,
        )

    if name == "glove_mean":
        assert isinstance(model, GloveMeanBaseline)
        return DeploymentSize(
            learned_encoder_size_mb=0.0,
            tokenizer_size_mb=0.0,
            embedding_table_size_mb=_glove_table_mb(model),
            requires_bundled_tokenizer=False,
        )

    if name == "model2vec_static":
        assert isinstance(model, Model2VecBaseline)
        return DeploymentSize(
            learned_encoder_size_mb=0.0,
            tokenizer_size_mb=_model2vec_tokenizer_mb(model),
            embedding_table_size_mb=_model2vec_table_mb(model),
            requires_bundled_tokenizer=getattr(model, "_backend", None) == "static_model",
        )

    if name == "minilm":
        assert isinstance(model, MiniLMBaseline)
        return DeploymentSize(
            learned_encoder_size_mb=model_size_mb(model),
            tokenizer_size_mb=_hf_tokenizer_files_mb(MINILM_MODEL_NAME),
            embedding_table_size_mb=0.0,
            requires_bundled_tokenizer=True,
        )

    learned = model_size_mb(model)
    return DeploymentSize(
        learned_encoder_size_mb=learned,
        tokenizer_size_mb=0.0,
        embedding_table_size_mb=0.0,
        requires_bundled_tokenizer=False,
    )


def with_encoder_size(deployment: DeploymentSize, encoder_mb: float) -> DeploymentSize:
    """Replace learned encoder size (e.g. INT8 checkpoint); system size recomputed."""
    return DeploymentSize(
        learned_encoder_size_mb=encoder_mb,
        tokenizer_size_mb=deployment.tokenizer_size_mb,
        embedding_table_size_mb=deployment.embedding_table_size_mb,
        requires_bundled_tokenizer=deployment.requires_bundled_tokenizer,
    )


def size_block(deployment: DeploymentSize) -> dict[str, float | bool]:
    return {
        "learned_encoder_size_mb": deployment.learned_encoder_size_mb,
        "system_size_mb": deployment.system_size_mb,
        "tokenizer_size_mb": deployment.tokenizer_size_mb,
        "embedding_table_size_mb": deployment.embedding_table_size_mb,
        # Legacy fields for backward-compatible CSV readers.
        "learned_size_mb": deployment.learned_encoder_size_mb,
        "model_size_mb": deployment.learned_encoder_size_mb,
        "total_size_mb": deployment.system_size_mb,
        "core_size_mb": deployment.core_size_mb,
        "requires_bundled_tokenizer": deployment.requires_bundled_tokenizer,
    }


def block_learned_encoder_size_mb(block: dict[str, Any]) -> float:
    """Research axis: trainable encoder weights only."""
    if "learned_encoder_size_mb" in block:
        return float(block["learned_encoder_size_mb"])
    if "learned_size_mb" in block:
        return float(block["learned_size_mb"])
    return float(block.get("model_size_mb", 0.0))


def block_system_size_mb(block: dict[str, Any]) -> float:
    """Engineering axis: full deployment footprint."""
    if "system_size_mb" in block:
        return float(block["system_size_mb"])
    if "total_size_mb" in block:
        return float(block["total_size_mb"])
    return (
        block_learned_encoder_size_mb(block)
        + float(block.get("tokenizer_size_mb", 0.0))
        + float(block.get("embedding_table_size_mb", 0.0))
    )


# Backward-compatible alias.
block_learned_size_mb = block_learned_encoder_size_mb
