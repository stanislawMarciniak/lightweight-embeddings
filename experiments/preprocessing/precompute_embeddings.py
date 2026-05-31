from __future__ import annotations

import os
from typing import Dict, List, Tuple

import torch

from models.baselines import GloveMeanEncoder
from preprocessing.dataset import STS_FEATURES, load_stsbenchmark


MAX_LEN = 64


def _sentences_to_ids(
    sentences: List[str],
    word2idx: Dict[str, int],
) -> List[List[int]]:
    ids_list: List[List[int]] = []
    for s in sentences:
        words = s.lower().split()
        ids = [word2idx.get(w, 0) for w in words][:MAX_LEN]
        if not ids:
            ids = [0]
        if len(ids) < MAX_LEN:
            ids = ids + [0] * (MAX_LEN - len(ids))
        ids_list.append(ids)
    return ids_list


def _build_split_tensors(
    split,
    glove: GloveMeanEncoder,
) -> Dict[str, torch.Tensor | List[str]]:
    sentences1: List[str] = []
    sentences2: List[str] = []
    scores: List[float] = []
    genres: List[str] = []

    for row in split:
        sentences1.append(row["sentence1"])
        sentences2.append(row["sentence2"])
        scores.append(float(row["score"]))
        genres.append(row["genre"])

    word2idx = glove.word2idx

    ids1 = _sentences_to_ids(sentences1, word2idx)
    ids2 = _sentences_to_ids(sentences2, word2idx)

    ids1_t = torch.tensor(ids1, dtype=torch.long)
    ids2_t = torch.tensor(ids2, dtype=torch.long)

    with torch.no_grad():
        emb1 = glove.embedding(ids1_t)  # (N, T, D)
        emb2 = glove.embedding(ids2_t)

    attention_mask_1 = (ids1_t != 0).to(torch.long)
    attention_mask_2 = (ids2_t != 0).to(torch.long)

    labels = torch.tensor([s / 5.0 for s in scores], dtype=torch.float32)

    return {
        "token_embeddings_1": emb1,
        "attention_mask_1": attention_mask_1,
        "token_embeddings_2": emb2,
        "attention_mask_2": attention_mask_2,
        "labels": labels,
        "genre": genres,
        "sentence1": sentences1,
        "sentence2": sentences2,
    }


def precompute_and_save_embeddings(
    out_dir: str = "data/precomputed_embeddings",
    glove_path: str = "data/glove.6B.100d.txt",
) -> None:
    """
    Precompute GloVe-based token embeddings for all STS splits and save to disk.
    """
    os.makedirs(out_dir, exist_ok=True)

    ds = load_stsbenchmark()
    glove = GloveMeanEncoder(glove_path=glove_path)

    for split_name in ["train", "validation", "test"]:
        print(f"Precomputing embeddings for split={split_name}")
        tensors = _build_split_tensors(ds[split_name], glove)
        out_path = os.path.join(out_dir, f"{split_name}_token_embeddings.pt")
        torch.save(tensors, out_path)
        print(f"Saved {split_name} embeddings to {out_path}")


if __name__ == "__main__":
    precompute_and_save_embeddings()

