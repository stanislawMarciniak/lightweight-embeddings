"""High-level orchestration: train models and benchmark models.

This is the only place that wires together the registry, the trainer, and the
benchmark utilities. ``main.py`` is a thin CLI on top of these two functions.
"""

from __future__ import annotations

import os
import time
from typing import List

import torch

from config import DEFAULT_CONFIG, ExperimentConfig
from evaluation.benchmark import benchmark_one, plot_results, write_results
from registry import (
    build_model,
    is_baseline,
    load_datasets,
    training_config_for,
)


def _device(config: ExperimentConfig) -> torch.device:
    return torch.device(config.training.device if torch.cuda.is_available() else "cpu")


def run_train(names: List[str], config: ExperimentConfig | None = None) -> None:
    """Train each trainable model once and save results/<name>/model.pt."""
    config = config or DEFAULT_CONFIG
    from training.trainer import Trainer, create_dataloader

    trainable = [n for n in names if not is_baseline(n)]
    skipped = [n for n in names if is_baseline(n)]
    if skipped:
        print(f"Skipping pretrained baselines (nothing to train): {skipped}")
    if not trainable:
        print("No trainable models selected.")
        return

    train_data, val_data, _ = load_datasets()
    cfg = config.training
    from preprocessing.dataset import sts_collate_fn

    train_loader = create_dataloader(
        train_data, cfg.batch_size, cfg.num_workers, shuffle=True, collate_fn=sts_collate_fn
    )
    val_loader = create_dataloader(
        val_data, cfg.batch_size, cfg.num_workers, shuffle=False, collate_fn=sts_collate_fn
    )

    from utils.seed import set_global_seed

    for name in trainable:
        print(f"\n=== Training {name} ===")
        model_cfg = training_config_for(name, cfg)
        # Seed *before* building the model so weight init is part of the seed,
        # making a chosen seed fully reproducible (not just dropout/shuffle).
        set_global_seed(model_cfg.seed)
        model = build_model(name)
        exp_dir = os.path.join(config.results_dir, name)
        Trainer(
            model,
            train_loader,
            val_loader,
            model_cfg,
            experiment_dir=exp_dir,
        ).fit()
        print(f"Saved checkpoint: {os.path.join(exp_dir, 'model.pt')}")


def run_benchmark(
    names: List[str],
    config: ExperimentConfig | None = None,
    verbose: bool = False,
) -> None:
    """Evaluate selected models (FP32 + INT8), measure timings, write results & plots.

    Logs high-level progress always; ``verbose`` adds per-stage messages and the
    per-stage timing breakdown for each model. Results are saved incrementally
    after every model so a long run can be interrupted without losing progress.
    """
    config = config or DEFAULT_CONFIG

    print("Loading STS test set...", flush=True)
    train_data, _, test_data = load_datasets()
    device = _device(config)

    # The shared 768-d WordPiece embedding table is needed to turn input_ids into
    # token_embeddings for every trainable encoder (eval + latency). Build once.
    embedder = None
    if any(not is_baseline(n) for n in names):
        from models.embeddings import get_token_embedding

        embedder = get_token_embedding().to(device)

    total = len(names)
    print(f"Found {total} models to benchmark", flush=True)

    entries = []
    ordered = []
    completed = 0
    for i, name in enumerate(names, 1):
        print(f"\n[{i}/{total}] Benchmarking {name}...", flush=True)
        start = time.perf_counter()
        entry = benchmark_one(
            name,
            results_dir=config.results_dir,
            test_dataset=test_data,
            eval_batch_size=config.eval_batch_size,
            num_workers=0,  # in-memory tensors: workers only add IPC overhead
            device=device,
            embedder=embedder,
            verbose=verbose,
        )
        elapsed = time.perf_counter() - start
        if entry is None:
            continue
        entries.append(entry)
        if verbose:
            print("  Saving benchmark results...", flush=True)
        ordered = write_results(config.results_dir, entries)  # incremental persist
        completed += 1
        print(f"Completed {name} in {elapsed:.1f}s", flush=True)
        print(f"Progress: {completed}/{total} models completed", flush=True)

    if not entries:
        print("No models benchmarked; nothing written.")
        return

    plot_results(ordered, config.results_dir)
    print(f"\nResults written to {config.results_dir}/results.json and results.csv")
