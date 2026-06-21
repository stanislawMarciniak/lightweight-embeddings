"""Torch-free numpy inference for the CompactSimilarityModel sentence encoder.

Pipeline (matches experiments/):
  text -> bert-base-uncased WordPiece -> input_ids
       -> BERT token embedding lookup (768-d, frozen)
       -> HybridAttentionPooling -> GatedProjection -> ScaledL2Normalization
       -> 128-d sentence embedding

Encoder weights live in ``custom_hybrid_encoder.npz``; BERT token embeddings in
``bert_token_embeddings.npz``. Tokenization uses the same HuggingFace tokenizer
as the training pipeline.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import List, Optional

import numpy as np

from app.utils.bert_embeddings import TOKEN_EMBED_DIM, load_bert_token_embeddings, lookup_token_embeddings
from app.utils.wordpiece import encode_texts

logger = logging.getLogger(__name__)

OUTPUT_DIM = 128


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def build_token_inputs(texts: List[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Texts -> (token_embeddings (N,T,768), mask (N,T), input_ids (N,T))."""
    input_ids, mask = encode_texts(texts)
    table = load_bert_token_embeddings()
    token_emb = lookup_token_embeddings(input_ids, table)
    return token_emb.astype(np.float32, copy=False), mask, input_ids


class SemanticEncoder:
    """Numpy implementation of CustomEncoder.forward()."""

    def __init__(self, npz_path: str) -> None:
        w = np.load(npz_path)
        self.W = {k: w[k] for k in w.files if not k.startswith("meta_")}
        self.output_dim = int(w["meta_sent_dim"]) if "meta_sent_dim" in w.files else OUTPUT_DIM
        self.best_val_pearson = float(w.get("meta_best_val_pearson", float("nan")))

    def forward(self, token_embeddings: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """(N, T, 768), (N, T) -> (N, 128)."""
        W = self.W
        x = token_embeddings
        m = mask[..., None]

        scores = x @ W["pool.score.weight"].T + W["pool.score.bias"]
        scores = scores * W["pool.alpha"] + W["pool.beta"]
        weights = _sigmoid(scores) * m
        weighted_sum = (weights * x).sum(axis=1)
        mean = (x * m).sum(axis=1) / np.clip(m.sum(axis=1), 1.0, None)
        pooled = weighted_sum + W["pool.gamma"] * mean

        h = pooled @ W["project.fc.weight"].T + W["project.fc.bias"]
        g = _sigmoid(pooled @ W["project.gate.weight"].T + W["project.gate.bias"])
        r = pooled @ W["project.residual.weight"].T + W["project.residual.bias"]
        z = h * g + r

        norm = np.linalg.norm(z, axis=-1, keepdims=True)
        return z / (norm + 1e-8) * W["norm.scale"]

    def encode(self, texts: List[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return np.zeros((0, self.output_dim), dtype=np.float32)
        emb, mask, _ = build_token_inputs(texts)
        return self.forward(emb, mask).astype(np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


class OnnxEncoder:
    """ONNX Runtime encoder; tokenization + BERT lookup stay in numpy."""

    def __init__(self, onnx_path: str) -> None:
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        self.session = ort.InferenceSession(onnx_path, sess_options=so, providers=["CPUExecutionProvider"])
        self._in_emb = self.session.get_inputs()[0].name
        self._in_mask = self.session.get_inputs()[1].name
        self._out = self.session.get_outputs()[0].name
        out_shape = self.session.get_outputs()[0].shape
        self.output_dim = int(out_shape[-1]) if isinstance(out_shape[-1], int) else OUTPUT_DIM
        self.best_val_pearson = float("nan")

    def encode(self, texts: List[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return np.zeros((0, self.output_dim), dtype=np.float32)
        emb, mask, _ = build_token_inputs(texts)
        out = self.session.run([self._out], {self._in_emb: emb, self._in_mask: mask})[0]
        return out.astype(np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


_ENCODER = None
_LOCK = threading.Lock()


def _build_encoder(path: str):
    if path.endswith(".onnx"):
        return OnnxEncoder(path)
    return SemanticEncoder(path)


def load_encoder(path: str):
    """Load (once) and cache the encoder."""
    global _ENCODER
    with _LOCK:
        if _ENCODER is None:
            t0 = time.perf_counter()
            load_bert_token_embeddings()
            _ENCODER = _build_encoder(path)
            logger.info(
                "Semantic encoder loaded (%s, dim=%d) in %.1f ms",
                "onnx" if path.endswith(".onnx") else "numpy",
                _ENCODER.output_dim,
                (time.perf_counter() - t0) * 1000.0,
            )
    return _ENCODER


def reload_encoder(path: str):
    """Hot-reload encoder weights from disk."""
    global _ENCODER
    t0 = time.perf_counter()
    new_enc = _build_encoder(path)
    with _LOCK:
        _ENCODER = new_enc
    logger.info("Semantic encoder hot-reloaded from %s in %.1f ms", path, (time.perf_counter() - t0) * 1000.0)
    return _ENCODER


def get_encoder():
    return _ENCODER


def resolve_model_path(settings) -> str:
    if getattr(settings, "EMBEDDING_RUNTIME", "numpy") == "onnx":
        return settings.SEMANTIC_ONNX_PATH
    return settings.SEMANTIC_MODEL_PATH
