from __future__ import annotations

import inspect
import json
import os
from dataclasses import asdict, replace
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import ExperimentConfig, TrainingConfig
from evaluation.metrics import (
    compute_cosine_error,
    compute_mse,
    compute_pearson,
    compute_spearman,
)
from models.baselines import (
    STSGloveMeanWrapper,
    STSModel2VecBGEWrapper,
    STSSentenceTransformerWrapper,
)
from utils.quantization import (
    count_parameters,
    measure_inference_time,
    model_size_mb,
    quantize_model_int8,
)


def plot_training_curves(
    history: Dict[str, List[float]],
    exp_dir: str,
    full_mode: bool,
    model_name: str = "",
) -> None:
    """Plot training loss and validation/test loss over epochs. Saves to exp_dir/training_curves.png."""
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train loss", color="C0")
    if full_mode and history.get("test_loss"):
        plt.plot(epochs, history["test_loss"], label="Test loss", color="C2")
    else:
        plt.plot(epochs, history["val_loss"], label="Validation loss", color="C1")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training curves{f' — {model_name}' if model_name else ''}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "training_curves.png"))
    plt.close()


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    model.to(device)
    model.eval()
    preds: List[float] = []
    targets: List[float] = []
    all_z1: List[torch.Tensor] = []
    all_z2: List[torch.Tensor] = []
    forward_params = set(inspect.signature(model.forward).parameters.keys())
    baseline_types = (STSGloveMeanWrapper, STSSentenceTransformerWrapper, STSModel2VecBGEWrapper)
    is_baseline = isinstance(model, baseline_types)

    with torch.no_grad():
        for batch in dataloader:
            labels = batch["label"].to(device)
            if is_baseline:
                batch_size = labels.size(0)
                dummy_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
                dummy_mask = torch.ones(batch_size, 1, dtype=torch.long, device=device)
                inputs: Dict[str, Any] = {
                    "input_ids_1": dummy_ids,
                    "attention_mask_1": dummy_mask,
                    "input_ids_2": dummy_ids,
                    "attention_mask_2": dummy_mask,
                    "sentence1": batch["sentence1"],
                    "sentence2": batch["sentence2"],
                }
            else:
                inputs = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                    if k != "label" and k in forward_params
                }

            out = model(**inputs)
            score = out["score"]

            preds.extend(score.cpu().numpy())
            targets.extend(labels.cpu().numpy())
            all_z1.append(out["z1"].cpu())
            all_z2.append(out["z2"].cpu())

    y_pred = np.array(preds)
    y_true = np.array(targets)
    z1_cat = torch.cat(all_z1)
    z2_cat = torch.cat(all_z2)

    metrics = {
        "pearson": compute_pearson(y_pred, y_true),
        "spearman": compute_spearman(y_pred, y_true),
        "mse": compute_mse(y_pred, y_true),
        "cosine_error": compute_cosine_error(z1_cat, z2_cat, y_true),
    }
    info = {"n_samples": len(y_true)}
    return metrics, info


def run_benchmark(
    model_builders: Dict[str, Any],
    train_data,
    val_data,
    test_data,
    config: ExperimentConfig,
    eval_only: set[str] | None = None,
    models_filter: set[str] | None = None,
    skip_training: bool = False,
    all_model_names: List[str] | None = None,
    full_mode: bool = False,
) -> None:
    from training.trainer import Trainer, create_dataloader

    os.makedirs(config.results_dir, exist_ok=True)
    eval_only = eval_only or set()
    models_filter = models_filter or set(model_builders.keys())

    # Filter to requested models
    model_builders = {
        k: v for k, v in model_builders.items()
        if k in models_filter
    }
    if not model_builders:
        print("No models to run (empty filter).")
        return

    # Load existing results for merging (when running a subset we update only those models)
    json_path = os.path.join(config.results_dir, "results.json")
    existing_results: Dict[str, Dict[str, Any]] = {}
    if os.path.exists(json_path):
        with open(json_path, encoding="utf8") as f:
            for r in json.load(f):
                existing_results[r["model"]] = r

    results_json: List[Dict[str, Any]] = []
    device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")

    baseline_types = (STSGloveMeanWrapper, STSSentenceTransformerWrapper, STSModel2VecBGEWrapper)

    for name, builder in model_builders.items():
        print(f"=== Running experiment for {name} ===")
        try:
            model = builder()
        except (FileNotFoundError, OSError, RuntimeError) as e:
            print(f"  Skipping {name}: failed to build model: {e}")
            print("  (For sentence_transformers, try: pip install --upgrade transformers sentence-transformers)")
            continue

        train_loader = create_dataloader(
            train_data, batch_size=config.training.batch_size, num_workers=config.training.num_workers
        )
        val_loader = create_dataloader(
            val_data, batch_size=config.training.batch_size, num_workers=config.training.num_workers, shuffle=False,
        )
        test_loader = DataLoader(
            test_data,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.training.num_workers,
            collate_fn=None,
        )

        exp_dir = os.path.join(config.results_dir, name)
        os.makedirs(exp_dir, exist_ok=True)

        checkpoint_path = os.path.join(exp_dir, "model.pt")
        do_train = name not in eval_only and not skip_training

        if do_train:
            training_cfg: TrainingConfig = config.training
            if name == "custom_hybrid":
                from torch.utils.data import ConcatDataset
                from training.trainer import cross_validate

                training_cfg = replace(
                    config.training,
                    lr=min(config.training.lr, 1e-3),
                    w_pearson=0.4,
                    w_contrastive=0.4,
                )
                if full_mode:
                    combined = train_data
                else:
                    combined = ConcatDataset([train_data, val_data])
                model, cv_info = cross_validate(
                    model_factory=builder,
                    combined_data=combined,
                    config=training_cfg,
                    experiment_dir=exp_dir,
                    n_folds=5,
                )
                if cv_info.get("avg_history"):
                    plot_training_curves(
                        cv_info["avg_history"], exp_dir, False, model_name=name,
                    )
            else:
                trainer = Trainer(
                    model,
                    train_loader,
                    val_loader,
                    training_cfg,
                    experiment_dir=exp_dir,
                    full_mode=full_mode,
                    test_loader=test_loader if full_mode else None,
                )
                history = trainer.fit()
                plot_training_curves(history, exp_dir, full_mode, model_name=name)
        elif name not in eval_only and skip_training:
            if not os.path.exists(checkpoint_path):
                print(f"  Skipping {name}: no checkpoint at {checkpoint_path}")
                continue
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            model.load_state_dict(ckpt["model_state_dict"], strict=True)

        # FP32 evaluation
        metrics_fp32, _ = evaluate_model(model, test_loader, device=device)
        batch = next(iter(test_loader))
        # For baselines, add dummy ids/masks so quantization timing uses compatible inputs.
        if isinstance(model, baseline_types):
            bsz = batch["label"].size(0)
            dummy_ids = torch.zeros(bsz, 1, dtype=torch.long)
            dummy_mask = torch.ones(bsz, 1, dtype=torch.long)
            timing_batch = {
                **batch,
                "input_ids_1": dummy_ids,
                "attention_mask_1": dummy_mask,
                "input_ids_2": dummy_ids,
                "attention_mask_2": dummy_mask,
            }
        else:
            timing_batch = batch
        avg_batch, avg_sample = measure_inference_time(model, timing_batch, device=device)
        size_mb = model_size_mb(model)
        n_params = count_parameters(model)

        if name not in eval_only:
            # Quantized evaluation (INT8 runs on CPU only; no CUDA impl for quantized::linear_dynamic)
            model_cpu = model.cpu()
            qmodel = quantize_model_int8(model_cpu)
            device_cpu = torch.device("cpu")

            metrics_int8, _ = evaluate_model(qmodel, test_loader, device=device_cpu)

            q_batch, q_sample = measure_inference_time(qmodel, timing_batch, device=device_cpu)

            q_size_mb = model_size_mb(qmodel)
            q_n_params = count_parameters(qmodel)
        else:
            # Eval-only models (e.g. baselines): no INT8; reuse FP32 for CSV/plots
            metrics_int8 = metrics_fp32
            q_batch, q_sample = avg_batch, avg_sample
            q_size_mb = size_mb
            q_n_params = n_params

        result_entry = {
            "model": name,
            "fp32": {
                "metrics": metrics_fp32,
                "avg_batch_time": avg_batch,
                "avg_sample_time": avg_sample,
                "throughput": 1.0 / avg_sample,
                "model_size_mb": size_mb,
                "n_params": n_params,
            },
            "int8": {
                "metrics": metrics_int8,
                "avg_batch_time": q_batch,
                "avg_sample_time": q_sample,
                "throughput": 1.0 / q_sample,
                "model_size_mb": q_size_mb,
                "n_params": q_n_params,
            },
        }
        results_json.append(result_entry)
        existing_results[name] = result_entry

    # Merge with existing results and use canonical order
    all_names = all_model_names or list(model_builders.keys())
    ordered_results = [existing_results[n] for n in all_names if n in existing_results]
    for n in sorted(existing_results):
        if n not in all_names:
            ordered_results.append(existing_results[n])

    with open(json_path, "w", encoding="utf8") as f:
        json.dump(ordered_results, f, indent=2)

    # CSV summary
    import csv

    csv_path = os.path.join(config.results_dir, "results.csv")
    with open(csv_path, "w", newline="", encoding="utf8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model",
                "precision",
                "pearson",
                "spearman",
                "mse",
                "cosine_error",
                "avg_sample_time",
                "model_size_mb",
                "n_params",
            ]
        )
        for r in ordered_results:
            for prec in ["fp32", "int8"]:
                writer.writerow(
                    [
                        r["model"],
                        prec,
                        r[prec]["metrics"]["pearson"],
                        r[prec]["metrics"]["spearman"],
                        r[prec]["metrics"]["mse"],
                        r[prec]["metrics"]["cosine_error"],
                        r[prec]["avg_sample_time"],
                        r[prec]["model_size_mb"],
                        r[prec]["n_params"],
                    ]
                )

    # Visualization
    plot_results(ordered_results, config.results_dir)


def plot_results(results: List[Dict[str, Any]], out_dir: str) -> None:
    if not results:
        return
    # Pearson vs model size
    plt.figure()
    for r in results:
        size = r["fp32"]["model_size_mb"]
        pearson = r["fp32"]["metrics"]["pearson"]
        plt.scatter(size, pearson, label=r["model"])
    plt.xlabel("Model size (MB)")
    plt.ylabel("Pearson correlation (FP32)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pearson_vs_model_size.png"))
    plt.close()

    # Inference time vs accuracy
    plt.figure()
    for r in results:
        t = r["fp32"]["avg_sample_time"]
        pearson = r["fp32"]["metrics"]["pearson"]
        plt.scatter(t, pearson, label=r["model"])
    plt.xlabel("Avg inference time per sample (ms, FP32)")
    plt.ylabel("Pearson correlation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "time_vs_accuracy.png"))
    plt.close()

    # FP32 vs INT8 Pearson comparison
    plt.figure()
    x = np.arange(len(results))
    width = 0.35
    pearson_fp32 = [r["fp32"]["metrics"]["pearson"] for r in results]
    pearson_int8 = [r["int8"]["metrics"]["pearson"] for r in results]
    labels = [r["model"] for r in results]
    plt.bar(x - width / 2, pearson_fp32, width, label="FP32")
    plt.bar(x + width / 2, pearson_int8, width, label="INT8")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Pearson correlation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fp32_vs_int8_pearson.png"))
    plt.close()

    # Pearson vs Spearman (FP32)
    plt.figure()
    for r in results:
        pearson = float(r["fp32"]["metrics"]["pearson"])
        spearman = float(r["fp32"]["metrics"]["spearman"])
        plt.scatter(spearman, pearson, label=r["model"], zorder=2)
    plt.plot([-1, 1], [-1, 1], "k--", alpha=0.5, label="y=x", zorder=1)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel("Spearman correlation")
    plt.ylabel("Pearson correlation")
    plt.title("Pearson vs Spearman (FP32 models)")
    plt.legend(loc="lower right", fontsize=7)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pearson_vs_spearman.png"))
    plt.close()