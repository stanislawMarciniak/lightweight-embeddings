"""Embedding-quality + retrieval audit for the STS models.

Answers: does training improve *embeddings* (what RAG uses, via cosine) or mostly
the pairwise scorer `score = scorer(pair)`? Produces evidence for the audit:

  * STS Pearson via the scorer vs via plain cosine(z1, z2),
  * embedding geometry (collapse / isotropy / variance),
  * a retrieval benchmark (Recall@k, MRR, nDCG) simulated from STS pairs,
  * the same numbers for an untrained (random-init) encoder for reference.

Usage:
    python scripts/audit_embeddings.py            # train custom_hybrid, then audit
    python scripts/audit_embeddings.py --no-train # audit existing checkpoint only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DEFAULT_CONFIG  # noqa: E402
from evaluation.embedding_diagnostics import (  # noqa: E402
    embedding_quality,
    format_report,
    retrieval_metrics,
)
from evaluation.metrics import compute_pearson, compute_spearman  # noqa: E402
from models.embeddings import get_token_embedding  # noqa: E402
from preprocessing.dataset import sts_collate_fn  # noqa: E402
from registry import build_model, load_datasets, training_config_for  # noqa: E402


@torch.no_grad()
def extract(model: torch.nn.Module, loader: DataLoader, device: torch.device, embedder=None):
    """Run a model over the loader and collect z1, z2, scores, labels."""
    import inspect

    model.to(device).eval()
    params = set(inspect.signature(model.forward).parameters.keys())
    sentence_input = "sentence1" in params
    if not sentence_input and embedder is not None:
        embedder.to(device)
    z1s, z2s, scores, labels = [], [], [], []
    for batch in loader:
        y = batch["label"]
        if sentence_input:
            bsz = y.size(0)
            inp = {
                "input_ids_1": torch.zeros(bsz, 1, dtype=torch.long, device=device),
                "attention_mask_1": torch.ones(bsz, 1, dtype=torch.long, device=device),
                "input_ids_2": torch.zeros(bsz, 1, dtype=torch.long, device=device),
                "attention_mask_2": torch.ones(bsz, 1, dtype=torch.long, device=device),
                "sentence1": batch["sentence1"],
                "sentence2": batch["sentence2"],
            }
        else:
            inp = embedder.embed_pair(batch, device)
        out = model(**inp)
        z1s.append(out["z1"].cpu())
        z2s.append(out["z2"].cpu())
        scores.append(out["score"].cpu())
        labels.append(y)
    return (
        torch.cat(z1s).numpy(),
        torch.cat(z2s).numpy(),
        torch.cat(scores).numpy(),
        torch.cat(labels).numpy(),
    )


def audit_one(name: str, z1, z2, score, labels, pos_thresh: float = 0.8) -> Dict:
    cos = (z1 * z2).sum(1) / (
        np.linalg.norm(z1, axis=1) * np.linalg.norm(z2, axis=1) + 1e-12
    )
    sts = {
        "pearson_scorer": compute_pearson(score, labels),
        "spearman_scorer": compute_spearman(score, labels),
        "pearson_cosine": compute_pearson(cos, labels),
        "spearman_cosine": compute_spearman(cos, labels),
    }
    emb = embedding_quality(np.vstack([z1, z2]))

    pos = np.where(labels >= pos_thresh)[0]
    retr: Dict[str, float] = {}
    if len(pos) >= 5:
        # query = sentence1 of a high-similarity pair; corpus = ALL sentence2;
        # the gold doc is that pair's own sentence2.
        retr = retrieval_metrics(z1[pos], z2, gold_idx=pos, ks=(1, 5, 10))
        retr["n_queries"] = float(len(pos))
        retr["corpus_size"] = float(len(z2))
    return {"sts": sts, "embedding": emb, "retrieval": retr}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-train", action="store_true", help="Audit existing checkpoint only.")
    ap.add_argument("--epochs", type=int, default=60, help="Max epochs when training custom_hybrid.")
    args = ap.parse_args()

    device = torch.device("cpu")
    train_data, val_data, test_data = load_datasets()
    cfg = DEFAULT_CONFIG.training

    ckpt = "results/custom_hybrid/model.pt"
    history = None
    if not args.no_train or not os.path.exists(ckpt):
        from training.trainer import Trainer, create_dataloader

        print(f"=== Training custom_hybrid (<= {args.epochs} epochs) to capture curve ===")
        tcfg = training_config_for("custom_hybrid", replace(cfg, num_epochs=args.epochs, device="cpu"))
        tr = create_dataloader(train_data, tcfg.batch_size, 0, shuffle=True, collate_fn=sts_collate_fn)
        vl = create_dataloader(val_data, tcfg.batch_size, 0, shuffle=False, collate_fn=sts_collate_fn)
        model = build_model("custom_hybrid")
        history = Trainer(model, tr, vl, tcfg, experiment_dir="results/custom_hybrid").fit()

    model = build_model("custom_hybrid")
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model_state_dict"], strict=True)

    embedder = get_token_embedding().to(device)
    loader = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=0, collate_fn=sts_collate_fn)

    print("\n=== Extracting embeddings: custom_hybrid ===")
    z1, z2, score, labels = extract(model, loader, device, embedder)
    custom = audit_one("custom_hybrid", z1, z2, score, labels)

    print("=== Extracting embeddings: untrained custom_hybrid (random-init baseline) ===")
    gz1, gz2, gscore, glabels = extract(build_model("custom_hybrid"), loader, device, embedder)
    glove = audit_one("untrained", gz1, gz2, gscore, glabels)

    report = {"custom_hybrid": custom, "untrained": glove}
    if history is not None:
        report["history"] = {
            "train_loss": history["train_loss"],
            "val_loss": history["val_loss"],
            "val_pearson": history["val_pearson"],
        }

    os.makedirs("results", exist_ok=True)
    with open("results/audit_embeddings.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n############### EMBEDDING / RETRIEVAL AUDIT ###############")
    for mname, r in (("custom_hybrid", custom), ("untrained", glove)):
        print(f"\n##### {mname} #####")
        print(format_report("STS (scorer vs cosine)", r["sts"]))
        print(format_report("Embedding geometry", r["embedding"]))
        if r["retrieval"]:
            print(format_report("Retrieval (query->paraphrase)", r["retrieval"]))
    if history is not None:
        h = report["history"]
        print("\n##### Training curve (custom_hybrid) #####")
        n = len(h["val_pearson"])
        idxs = list(range(min(5, n))) + (["..."] if n > 10 else []) + list(range(max(5, n - 5), n))
        for i in idxs:
            if i == "...":
                print("  ...")
                continue
            print(
                f"  ep{i+1:>3}: train_loss={h['train_loss'][i]:.4f} "
                f"val_loss={h['val_loss'][i]:.4f} val_pearson={h['val_pearson'][i]:.4f}"
            )
    print("\nSaved: results/audit_embeddings.json")


if __name__ == "__main__":
    main()
