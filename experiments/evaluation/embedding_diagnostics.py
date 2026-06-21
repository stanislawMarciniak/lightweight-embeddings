"""Diagnostics for *embedding* quality, independent of the STS pairwise scorer.

These tools exist to answer a single question: does training actually produce
good sentence embeddings (the thing a FAQ-Retrieval / RAG system uses via cosine
similarity), or does it merely fit the pairwise regressor `score = scorer(pair)`?

Nothing here touches the model or the training loop; everything operates on
already-computed embedding matrices, so it is safe to run on any checkpoint.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def _as_float_matrix(emb: np.ndarray) -> np.ndarray:
    emb = np.asarray(emb, dtype=np.float64)
    if emb.ndim != 2:
        raise ValueError(f"expected (N, D) matrix, got shape {emb.shape}")
    return emb


def embedding_quality(emb: np.ndarray, sample: int = 2000, seed: int = 0) -> Dict[str, float]:
    """Geometry/health metrics for an (N, D) embedding matrix.

    Returns:
        mean_norm / std_norm: norm statistics (≈1 for L2-normalized models).
        mean_cosine / std_cosine: off-diagonal cosine distribution. A high
            ``mean_cosine`` (→1) is the classic *embedding collapse* signal — all
            sentences look alike, retrieval becomes impossible.
        mean_distance: 1 - mean_cosine, the average angular spread.
        variance: mean per-dimension variance of the (centered) embeddings.
        top1_explained_var: share of variance on the largest principal axis
            (→1 means an anisotropic, near-degenerate space).
        effective_rank: exp(entropy of normalized eigenvalues) — how many
            dimensions are actually used.
        isotropy: effective_rank / D in [0, 1]; 1 = perfectly isotropic.
    """
    emb = _as_float_matrix(emb)
    n, d = emb.shape
    rng = np.random.default_rng(seed)

    norms = np.linalg.norm(emb, axis=1)

    # Covariance spectrum -> variance, anisotropy, effective rank, isotropy.
    cov = np.cov(emb, rowvar=False)
    eigvals = np.clip(np.linalg.eigvalsh(cov), 0.0, None)
    total = float(eigvals.sum()) or 1.0
    p = eigvals / total
    p_nz = p[p > 0]
    eff_rank = float(np.exp(-(p_nz * np.log(p_nz)).sum())) if p_nz.size else 0.0

    # Off-diagonal cosine on a random subsample (collapse / diversity).
    idx = rng.choice(n, size=min(sample, n), replace=False)
    sub = emb[idx]
    sub_unit = sub / (np.linalg.norm(sub, axis=1, keepdims=True) + 1e-12)
    cos = sub_unit @ sub_unit.T
    m = ~np.eye(cos.shape[0], dtype=bool)
    off = cos[m]

    return {
        "n": float(n),
        "dim": float(d),
        "mean_norm": float(norms.mean()),
        "std_norm": float(norms.std()),
        "mean_cosine": float(off.mean()),
        "std_cosine": float(off.std()),
        "mean_distance": float(1.0 - off.mean()),
        "variance": float(np.var(emb - emb.mean(0), axis=0).mean()),
        "top1_explained_var": float(eigvals.max() / total),
        "effective_rank": eff_rank,
        "isotropy": float(eff_rank / d),
    }


def retrieval_metrics(
    query_emb: np.ndarray,
    corpus_emb: np.ndarray,
    gold_idx: Sequence[int],
    ks: Sequence[int] = (1, 5, 10),
) -> Dict[str, float]:
    """Single-positive retrieval metrics from query/corpus embedding matrices.

    For each query i, ``gold_idx[i]`` is the index of its one relevant document
    in the corpus. Ranking is by cosine similarity (the RAG retrieval operation).

    Returns Recall@k for each k, plus MRR and nDCG@max(ks). With exactly one
    relevant item per query, Recall@1 == Precision@1 == Hit@1.
    """
    q = _as_float_matrix(query_emb)
    c = _as_float_matrix(corpus_emb)
    gold = np.asarray(gold_idx, dtype=np.int64)

    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    c = c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-12)
    sims = q @ c.T  # (Q, C)

    # Rank of the gold item = how many corpus items score strictly higher.
    gold_scores = sims[np.arange(len(gold)), gold]
    ranks = (sims > gold_scores[:, None]).sum(axis=1)  # 0-based rank

    out: Dict[str, float] = {}
    for k in ks:
        out[f"recall@{k}"] = float((ranks < k).mean())
    out["mrr"] = float((1.0 / (ranks + 1.0)).mean())
    kmax = max(ks)
    dcg = np.where(ranks < kmax, 1.0 / np.log2(ranks + 2.0), 0.0)  # IDCG=1 (single positive)
    out[f"ndcg@{kmax}"] = float(dcg.mean())
    return out


def format_report(title: str, metrics: Dict[str, float]) -> str:
    lines = [f"--- {title} ---"]
    for k, v in metrics.items():
        lines.append(f"  {k:>20}: {v:.4f}")
    return "\n".join(lines)
