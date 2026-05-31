from __future__ import annotations

import argparse
import os
from typing import Dict, Set

import torch
from torch.utils.data import ConcatDataset

from config import DEFAULT_CONFIG, ExperimentConfig
from models.baselines import (
    Model2VecBGEEncoder,
    STSGloveMeanWrapper,
    STSModel2VecBGEWrapper,
    STSSentenceTransformerWrapper,
)
from models.bigru import BiGRUEncoder, BiGRUEncoderOptimized
from models.cnn import TextCNN, TextCNNOptimized
from models.custom import CustomHybridModel
from models.dan import DeepAveragingNetwork, DeepAveragingNetworkOptimized
from preprocessing.dataset_precomputed import PrecomputedSTSDataset
from evaluation.benchmark import run_benchmark


def build_model_factories(vocab_size: int) -> Dict[str, callable]:
    return {
        "dan_standard": lambda: DeepAveragingNetwork(vocab_size=vocab_size),
        "dan_optimized": lambda: DeepAveragingNetworkOptimized(vocab_size=vocab_size),
        "cnn_standard": lambda: TextCNN(vocab_size=vocab_size),
        "cnn_optimized": lambda: TextCNNOptimized(vocab_size=vocab_size),
        "bigru_standard": lambda: BiGRUEncoder(vocab_size=vocab_size),
        "bigru_optimized": lambda: BiGRUEncoderOptimized(vocab_size=vocab_size),
        "sru_standard": lambda: __import__("models.sru", fromlist=["SRUEncoder"]).SRUEncoder(
            vocab_size=vocab_size
        ),
        "sru_optimized": lambda: __import__("models.sru", fromlist=["SRUEncoderOptimized"]).SRUEncoderOptimized(
            vocab_size=vocab_size
        ),
        "custom_hybrid": lambda: CustomHybridModel(vocab_size=vocab_size),
        # Baselines (eval-only, no training / INT8)
        "glove_mean": lambda: STSGloveMeanWrapper(glove_path="data/glove.6B.100d.txt"),
        "sentence_transformer": lambda: STSSentenceTransformerWrapper(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        ),
        "bge_small": lambda: STSModel2VecBGEWrapper(
            model_name="BAAI/bge-small-en-v1.5",
            target_dim=256,
            simulate_int8=True,
        ),
    }


def baseline_model_names() -> Set[str]:
    return {"glove_mean", "sentence_transformer", "bge_small"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run embedding model experiments with optional selective training/eval."
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model names to run (default: all). Example: dan_standard,cnn_standard",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training; load checkpoints and evaluate. Ignores baselines (they are always eval-only).",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use train+validation as one training set (no holdout; validation loader still uses original val for monitoring).",
    )
    return parser.parse_args()


def main(config: ExperimentConfig | None = None, args: argparse.Namespace | None = None) -> None:
    config = config or DEFAULT_CONFIG
    args = args or parse_args()

    train_data = PrecomputedSTSDataset("data/precomputed_embeddings/train_token_embeddings.pt")
    val_data = PrecomputedSTSDataset("data/precomputed_embeddings/validation_token_embeddings.pt")
    test_data = PrecomputedSTSDataset("data/precomputed_embeddings/test_token_embeddings.pt")

    if args.full:
        train_data = ConcatDataset([train_data, val_data])
        print("Full mode: training on train+validation combined.")

    # Fit PCA for BGE encoder on full training sentences before any evaluation.
    if args.full:
        d0, d1 = train_data.datasets[0], train_data.datasets[1]
        train_sentences = list(d0.sentence1) + list(d0.sentence2) + list(d1.sentence1) + list(d1.sentence2)
    else:
        train_sentences = list(train_data.sentence1) + list(train_data.sentence2)
    if train_sentences:
        Model2VecBGEEncoder.fit_pca_on_training(
            train_sentences,
            model_name="BAAI/bge-small-en-v1.5",
            target_dim=256,
        )

    # vocab_size is unused in models now but kept for API compatibility.
    model_builders = build_model_factories(vocab_size=0)
    all_model_names = list(model_builders.keys())

    models_filter: Set[str] | None = None
    if args.models:
        requested = {n.strip() for n in args.models.split(",") if n.strip()}
        invalid = requested - set(all_model_names)
        if invalid:
            raise SystemExit(f"Unknown model(s): {invalid}. Available: {sorted(all_model_names)}")
        models_filter = requested

    # Use a separate results subdir for full mode so standard and full outputs don't override each other.
    if args.full:
        config = ExperimentConfig(
            training=config.training,
            quantization=config.quantization,
            results_dir=os.path.join(config.results_dir, "full"),
        )

    run_benchmark(
        model_builders=model_builders,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        config=config,
        eval_only=baseline_model_names(),
        models_filter=models_filter,
        skip_training=args.eval_only,
        all_model_names=all_model_names,
        full_mode=args.full,
    )


if __name__ == "__main__":
    main(args=parse_args())

