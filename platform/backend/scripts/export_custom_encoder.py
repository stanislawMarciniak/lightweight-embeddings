"""Export CompactSimilarityModel encoder weights + BERT token embeddings for production.

Run with an environment that has torch + transformers (e.g. experiments venv):

    cd platform/backend
    ../../experiments/venv/bin/python scripts/export_custom_encoder.py \
        --ckpt app/models/custom_hybrid.pt \
        --out  app/models/custom_hybrid_encoder.npz \
        --bert-out app/models/bert_token_embeddings.npz
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from app.training.encoder_torch import ENCODER_PREFIXES, CustomEncoder, export_npz, export_onnx, load_weights

HF_MODEL = "bert-base-uncased"
HF_CACHE = os.path.join(HERE, "app", "models", "hf_cache")


def export_bert_embeddings(out_path: str) -> None:
    from transformers import AutoModel

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path):
        print(f"BERT embeddings already exist at {out_path}, skipping")
        return
    bert = AutoModel.from_pretrained(HF_MODEL, cache_dir=HF_CACHE)
    matrix = bert.get_input_embeddings().weight.detach().cpu().numpy().astype(np.float32)
    del bert
    np.savez(out_path, embedding=matrix)
    print(f"Exported BERT token embeddings {matrix.shape} -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "app", "models", "custom_hybrid.pt"))
    ap.add_argument("--out", default=os.path.join(HERE, "app", "models", "custom_hybrid_encoder.npz"))
    ap.add_argument("--onnx-out", default=os.path.join(HERE, "app", "models", "custom_hybrid_encoder.onnx"))
    ap.add_argument("--bert-out", default=os.path.join(HERE, "app", "models", "bert_token_embeddings.npz"))
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = CustomEncoder()
    load_weights(model, args.ckpt)
    export_npz(model, args.out, best_val_pearson=float(ckpt.get("best_val_pearson", float("nan"))))
    export_onnx(model, args.onnx_out)
    export_bert_embeddings(args.bert_out)

    sd = ckpt["model_state_dict"]
    n_enc = sum(1 for k in sd if k.startswith(ENCODER_PREFIXES))
    print(f"Exported encoder ({n_enc} tensors) -> {args.out}")
    print(f"Exported ONNX -> {args.onnx_out}")


if __name__ == "__main__":
    main()
