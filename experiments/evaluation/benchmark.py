from __future__ import annotations

import csv
import inspect
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch
from torch.utils.data import DataLoader

from evaluation.deployment_size import (
    block_learned_encoder_size_mb,
    block_system_size_mb,
    estimate_deployment_size,
    size_block,
    with_encoder_size,
)
from evaluation.metrics import (
    compute_cosine_error,
    compute_mse,
    compute_pearson,
    compute_spearman,
)
from registry import (
    all_model_names,
    assert_learned_size_plot_eligible,
    build_model,
    in_learned_size_plot,
    in_lightweight_latency_plot,
    is_baseline,
)
from utils.quantization import (
    _is_quantized,
    count_parameters,
    measure_inference_time,
    model_size_mb,
    quantize_model_int8,
    single_core_cpu,
)

# Fair latency: single-core CPU; batch size matches eval_batch_size (128).
LATENCY_BATCH_SIZE = 128


def _vlog(verbose: bool, msg: str) -> None:
    if verbose:
        print(f"  {msg}", flush=True)


def reported_precision(name: str) -> str:
    """Which precision is the headline result for a model (plots / single-row models).

    Convention: '*_optimized' -> INT8; everything else -> FP32.
    ``custom_hybrid`` is written as *both* FP32 and INT8 rows in ``results.csv``.
    """
    return "int8" if name.endswith("_optimized") else "fp32"


def reported_block(r: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    prec = reported_precision(r["model"])
    return prec, r[prec]


def csv_rows(r: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """Precision blocks to emit in ``results.csv``."""
    name = r["model"]
    if name == "custom_hybrid":
        return [("fp32", r["fp32"]), ("int8", r["int8"])]
    if is_baseline(name):
        return [("fp32", r["fp32"])]
    prec = reported_precision(name)
    return [(prec, r[prec])]

def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    embedder: Optional[torch.nn.Module] = None,
    verbose: bool = False,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Evaluate a model on the loader. Returns (metrics, stage_timings_seconds).

    Trainable encoders consume ``token_embeddings`` of shape (B, T, 768); the
    integer ``input_ids`` from the loader are turned into those via ``embedder``
    (the shared, frozen WordPiece embedding table). Pretrained baselines consume
    raw sentences and ignore the embedder.
    """
    model.to(device)
    model.eval()
    preds: List[float] = []
    targets: List[float] = []
    all_z1: List[torch.Tensor] = []
    all_z2: List[torch.Tensor] = []
    forward_params = set(inspect.signature(model.forward).parameters.keys())
    is_sentence_input = "sentence1" in forward_params
    if not is_sentence_input and embedder is not None:
        embedder.to(device)

    _vlog(verbose, "Encoding sentence pairs..." if is_sentence_input else "Generating embeddings...")
    gen_start = time.perf_counter()
    with torch.inference_mode():
        for batch in dataloader:
            labels = batch["label"].to(device)
            if is_sentence_input:
                bsz = labels.size(0)
                dummy_ids = torch.zeros(bsz, 1, dtype=torch.long, device=device)
                dummy_mask = torch.ones(bsz, 1, dtype=torch.long, device=device)
                inputs: Dict[str, Any] = {
                    "input_ids_1": dummy_ids,
                    "attention_mask_1": dummy_mask,
                    "input_ids_2": dummy_ids,
                    "attention_mask_2": dummy_mask,
                    "sentence1": batch["sentence1"],
                    "sentence2": batch["sentence2"],
                }
            else:
                inputs = embedder.embed_pair(batch, device)

            out = model(**inputs)
            preds.extend(out["score"].cpu().numpy())
            targets.extend(labels.cpu().numpy())
            all_z1.append(out["z1"].cpu())
            all_z2.append(out["z2"].cpu())
    gen_time = time.perf_counter() - gen_start

    y_pred = np.array(preds)
    y_true = np.array(targets)
    z1_cat = torch.cat(all_z1)
    z2_cat = torch.cat(all_z2)

    metric_start = time.perf_counter()
    _vlog(verbose, "Computing Pearson...")
    pearson = compute_pearson(y_pred, y_true)
    _vlog(verbose, "Computing Spearman...")
    spearman = compute_spearman(y_pred, y_true)
    _vlog(verbose, "Computing retrieval metrics...")
    mse = compute_mse(y_pred, y_true)
    cosine_error = compute_cosine_error(z1_cat, z2_cat, y_true)
    metric_time = time.perf_counter() - metric_start

    metrics = {"pearson": pearson, "spearman": spearman, "mse": mse, "cosine_error": cosine_error}
    return metrics, {"generate": gen_time, "metrics": metric_time}


def measure_end_to_end_time(
    model: torch.nn.Module,
    sentence1: List[str],
    sentence2: List[str],
    embedder,
    device: torch.device,
    repeats: int = 10,
    warmup: int = 3,
) -> Tuple[float, float]:
    """End-to-end latency from raw sentences: tokenization + preprocessing + forward.

    For sentence-transformers baselines the forward already consumes raw strings
    (tokenization happens inside). For trainable models we reproduce the real user
    path: WordPiece tokenization -> dynamic padding -> 768-d embedding lookup ->
    forward. Returns (avg_batch_ms, avg_sample_ms).
    """
    from preprocessing.tokenizer import tokenize, pad_dynamic

    if _is_quantized(model) and device.type == "cuda":
        device = torch.device("cpu")
    model.eval()
    model.to(device)

    forward_params = set(inspect.signature(model.forward).parameters.keys())
    is_sentence_input = "sentence1" in forward_params
    bsz = len(sentence1)
    if not is_sentence_input and embedder is not None:
        embedder.to(device)

    def run() -> None:
        if is_sentence_input:
            dummy_ids = torch.zeros(bsz, 1, dtype=torch.long, device=device)
            dummy_mask = torch.ones(bsz, 1, dtype=torch.long, device=device)
            model(
                input_ids_1=dummy_ids,
                attention_mask_1=dummy_mask,
                input_ids_2=dummy_ids,
                attention_mask_2=dummy_mask,
                sentence1=sentence1,
                sentence2=sentence2,
            )
        else:
            # Real user path: tokenize -> dynamic pad -> embed -> forward.
            pad1 = pad_dynamic(tokenize(sentence1))
            pad2 = pad_dynamic(tokenize(sentence2))
            batch = {
                "input_ids_1": pad1["input_ids"],
                "attention_mask_1": pad1["attention_mask"],
                "input_ids_2": pad2["input_ids"],
                "attention_mask_2": pad2["attention_mask"],
            }
            model(**embedder.embed_pair(batch, device))

    with torch.no_grad():
        for _ in range(warmup):
            run()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times: List[float] = []
        for _ in range(repeats):
            start = time.perf_counter()
            run()
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

    avg_batch = (sum(times) / len(times)) * 1000  # ms per batch
    return avg_batch, avg_batch / bsz


def _measure_latency(
    model: torch.nn.Module,
    batch: Dict[str, Any],
    embedder: Optional[torch.nn.Module],
) -> Tuple[float, float]:
    """Forward-only and end-to-end latency on single-core CPU.

    ``fwd`` times encoder forward on pre-computed token embeddings (trainable
    models) or model forward (baselines). ``e2e`` includes tokenization and BERT
    lookup where applicable. Reported values are per STS sentence pair (batch
    time / batch size).
    """
    bsz = LATENCY_BATCH_SIZE
    s1 = batch["sentence1"][:bsz]
    s2 = batch["sentence2"][:bsz]
    forward_params = set(inspect.signature(model.forward).parameters.keys())
    is_sentence_input = "sentence1" in forward_params

    with single_core_cpu() as cpu:
        model.cpu().eval()
        if is_sentence_input:
            timing_batch = {
                "input_ids_1": torch.zeros(bsz, 1, dtype=torch.long),
                "attention_mask_1": torch.ones(bsz, 1, dtype=torch.long),
                "input_ids_2": torch.zeros(bsz, 1, dtype=torch.long),
                "attention_mask_2": torch.ones(bsz, 1, dtype=torch.long),
                "sentence1": s1,
                "sentence2": s2,
            }
        else:
            assert embedder is not None
            embedder.to(cpu)
            mini = {
                "input_ids_1": batch["input_ids_1"][:bsz],
                "attention_mask_1": batch["attention_mask_1"][:bsz],
                "input_ids_2": batch["input_ids_2"][:bsz],
                "attention_mask_2": batch["attention_mask_2"][:bsz],
            }
            timing_batch = embedder.embed_pair(mini, cpu)
        _, fwd = measure_inference_time(model, timing_batch, device=cpu)
        _, e2e = measure_end_to_end_time(model, s1, s2, embedder, device=cpu)
    return fwd, e2e


def benchmark_one(
    name: str,
    results_dir: str,
    test_dataset,
    eval_batch_size: int,
    num_workers: int,
    device: torch.device,
    embedder,
    verbose: bool = False,
) -> Optional[Dict[str, Any]]:
    """Run the full benchmark for a single model with staged logging + per-stage timing.

    Returns the result entry, or None if the model could not be built / has no
    checkpoint. All stage messages and timings are printed only when ``verbose``.
    """
    baseline = is_baseline(name)
    timings: Dict[str, float] = {}

    # --- Load model (+ checkpoint) ---
    _vlog(verbose, "Loading model...")
    s = time.perf_counter()
    try:
        model = build_model(name)
    except (FileNotFoundError, OSError, RuntimeError) as e:
        print(f"  Skipping {name}: failed to build ({e})", flush=True)
        return None
    timings["Load model"] = time.perf_counter() - s

    if not baseline:
        ckpt_path = os.path.join(results_dir, name, "model.pt")
        if not os.path.exists(ckpt_path):
            print(f"  Skipping {name}: no checkpoint at {ckpt_path} (train it first)", flush=True)
            return None
        _vlog(verbose, "Loading checkpoint...")
        s = time.perf_counter()
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        timings["Load checkpoint"] = time.perf_counter() - s

    # --- Prepare dataset ---
    _vlog(verbose, "Preparing test dataloader...")
    s = time.perf_counter()
    from preprocessing.dataset import sts_collate_fn

    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=sts_collate_fn,
    )
    batch = next(iter(test_loader))
    timings["Prepare dataset"] = time.perf_counter() - s

    # --- FP32 evaluation ---
    metrics_fp32, eval_timings = evaluate_model(
        model, test_loader, device=device, embedder=embedder, verbose=verbose
    )
    timings["Generate embeddings"] = eval_timings["generate"]
    timings["Compute metrics"] = eval_timings["metrics"]

    # --- Latency benchmark (single-core CPU) ---
    _vlog(verbose, f"Running latency benchmark (single-core CPU, batch={LATENCY_BATCH_SIZE})...")
    s = time.perf_counter()
    fwd_sample, e2e_sample = _measure_latency(model, batch, embedder)
    timings["Latency benchmark"] = time.perf_counter() - s
    deployment = estimate_deployment_size(name, model)
    n_params = count_parameters(model)

    # --- INT8 (trainable models only) ---
    if not baseline:
        _vlog(verbose, "Quantizing to INT8...")
        s = time.perf_counter()
        qmodel = quantize_model_int8(model.cpu())
        timings["Quantize INT8"] = time.perf_counter() - s
        cpu = torch.device("cpu")
        metrics_int8, int8_eval_timings = evaluate_model(
            qmodel, test_loader, device=cpu, embedder=embedder, verbose=verbose
        )
        timings["INT8 eval"] = int8_eval_timings["generate"] + int8_eval_timings["metrics"]
        q_sample, q_e2e_sample = _measure_latency(qmodel, batch, embedder)
        q_deployment = with_encoder_size(deployment, model_size_mb(qmodel))
        # Quantization changes weight dtype/storage, not the parameter *count*.
        q_n_params = n_params
    else:
        metrics_int8 = metrics_fp32
        q_sample = fwd_sample
        q_e2e_sample = e2e_sample
        q_deployment = deployment
        q_n_params = n_params

    fp32_sizes = size_block(deployment)
    int8_sizes = size_block(q_deployment)

    if verbose:
        for stage, dt in timings.items():
            print(f"    {stage}: {dt:.1f}s", flush=True)

    return {
        "model": name,
        "requires_bundled_tokenizer": deployment.requires_bundled_tokenizer,
        "fp32": {
            "metrics": metrics_fp32,
            "avg_sample_time": fwd_sample,
            "e2e_sample_time": e2e_sample,
            "n_params": n_params,
            **fp32_sizes,
        },
        "int8": {
            "metrics": metrics_int8,
            "avg_sample_time": q_sample,
            "e2e_sample_time": q_e2e_sample,
            "n_params": q_n_params,
            **int8_sizes,
        },
    }


def write_results(results_dir: str, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge new entries into results.json/results.csv and return the ordered list."""
    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, "results.json")

    existing: Dict[str, Dict[str, Any]] = {}
    if os.path.exists(json_path):
        with open(json_path, encoding="utf8") as f:
            for r in json.load(f):
                existing[r["model"]] = r
    for e in entries:
        existing[e["model"]] = e

    order = all_model_names()
    ordered = [existing[n] for n in order if n in existing]

    with open(json_path, "w", encoding="utf8") as f:
        json.dump(ordered, f, indent=2)

    csv_path = os.path.join(results_dir, "results.csv")
    with open(csv_path, "w", newline="", encoding="utf8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model", "precision", "pearson", "spearman", "mse", "cosine_error",
                "fwd_sample_time_ms", "e2e_sample_time_ms",
                "learned_encoder_size_mb", "system_size_mb",
                "tokenizer_size_mb", "embedding_table_size_mb", "n_params",
            ]
        )
        for r in ordered:
            for prec, block in csv_rows(r):
                writer.writerow(
                    [
                        r["model"], prec,
                        block["metrics"]["pearson"],
                        block["metrics"]["spearman"],
                        block["metrics"]["mse"],
                        block["metrics"]["cosine_error"],
                        block["avg_sample_time"],
                        block.get("e2e_sample_time", ""),
                        block_learned_encoder_size_mb(block),
                        block_system_size_mb(block),
                        block["tokenizer_size_mb"],
                        block["embedding_table_size_mb"],
                        block["n_params"],
                    ]
                )
    return ordered


_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "p", "d", "8"]


def _model_styles(names: List[str]) -> Dict[str, Tuple[Any, str]]:
    """Assign each model a unique (color, marker) pair.

    The default matplotlib color cycle only has 10 colors, so with >10 models the
    colors wrap around and become ambiguous. We sample a large continuous colormap
    and pair it with a rotating marker set, which keeps every model distinguishable
    even when two share a similar hue.
    """
    cmap = plt.get_cmap("turbo")
    n = max(len(names), 1)
    styles: Dict[str, Tuple[Any, str]] = {}
    for i, name in enumerate(names):
        color = cmap((i + 0.5) / n)
        styles[name] = (color, _MARKERS[i % len(_MARKERS)])
    return styles


def _scatter_models(
    ax,
    rows: List[Tuple[str, float, float]],
    styles: Dict[str, Tuple[Any, str]],
) -> None:
    for name, x, y in rows:
        color, marker = styles[name]
        ax.scatter(x, y, color=color, marker=marker, s=70, edgecolors="black",
                   linewidths=0.4, label=name, zorder=3)


def _legend_outside(ax) -> None:
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5),
              borderaxespad=0.0, frameon=True)


def _format_mb_tick(value: float, _pos: int) -> str:
    if value >= 10:
        return f"{int(round(value))}"
    if value >= 1:
        return f"{value:g}"
    return f"{value:.2g}"


def _configure_learned_size_axis(ax, sizes_mb: List[float]) -> None:
    """Log-scale MB axis with power-of-ten tick labels (10^-1, 10^0, …)."""
    positive = [s for s in sizes_mb if s > 0]
    if not positive:
        return

    lo, hi = min(positive), max(positive)
    ax.set_xscale("log")
    exp_lo = int(np.floor(np.log10(lo)))
    exp_hi = int(np.ceil(np.log10(hi)))
    ticks = [10.0 ** exp for exp in range(exp_lo, exp_hi + 1)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([rf"$10^{{{exp}}}$" for exp in range(exp_lo, exp_hi + 1)])
    ax.set_xlim(10.0 ** (exp_lo - 0.15), 10.0 ** (exp_hi + 0.15))


def _configure_size_axis(ax, sizes_mb: List[float]) -> None:
    """Readable MB axis: linear when values span <10x, log with explicit ticks otherwise."""
    positive = [s for s in sizes_mb if s > 0]
    if not positive:
        return

    lo, hi = min(positive), max(positive)
    if hi / lo < 10:
        ax.set_xscale("linear")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(_format_mb_tick))
        ax.set_xlim(max(0, lo * 0.9), hi * 1.05)
        return

    ax.set_xscale("log")
    exp_lo = int(np.floor(np.log10(lo)))
    exp_hi = int(np.ceil(np.log10(hi)))
    ticks: List[float] = []
    for exp in range(exp_lo, exp_hi + 1):
        for mant in (1, 2, 5):
            t = mant * (10 ** exp)
            if lo * 0.85 <= t <= hi * 1.15:
                ticks.append(t)
    if not ticks:
        ticks = [lo, hi]
    ax.set_xticks(ticks)
    ax.set_xticklabels([_format_mb_tick(t, 0) for t in ticks])
    ax.set_xlim(lo * 0.85, hi * 1.15)


def plot_results(results: List[Dict[str, Any]], out_dir: str) -> None:
    """Generate benchmark plots.

    Methodology (never mix the two axes):
      (A) ``learned_encoder_size_mb`` — trainable encoder weights only.
          Architecture comparison: pearson_vs_model_size.
      (B) ``system_size_mb`` — tokenizer + embedding tables + encoder.
          Deployment comparison: system_cost_vs_accuracy.

    Tokenizers and lookup tables affect only axis (B).
    """
    if not results:
        return

    names = [r["model"] for r in results]
    styles = _model_styles(names)

    # --- (A) Research axis: learned encoder size vs Pearson ---
    fig, ax = plt.subplots(figsize=(8, 5))
    rows = []
    for r in results:
        if not in_learned_size_plot(r["model"]):
            continue
        assert_learned_size_plot_eligible(r["model"])
        block = reported_block(r)[1]
        rows.append(
            (r["model"], block_learned_encoder_size_mb(block), block["metrics"]["pearson"])
        )
    if rows:
        _scatter_models(ax, rows, {n: styles[n] for n, _, _ in rows})
        _configure_learned_size_axis(ax, [x for _, x, _ in rows])
        ax.set_xlabel("Learned encoder size (MB, weights only)")
        ax.set_ylabel("Pearson correlation")
        ax.set_title("Pearson vs learned encoder size (architecture comparison)")
        ax.grid(True, alpha=0.3)
        _legend_outside(ax)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "pearson_vs_model_size.png"), bbox_inches="tight")
    plt.close(fig)

    # --- (B) Engineering axis: full system footprint vs Pearson (all models) ---
    fig, ax = plt.subplots(figsize=(8, 5))
    rows = [
        (
            r["model"],
            block_system_size_mb(reported_block(r)[1]),
            reported_block(r)[1]["metrics"]["pearson"],
        )
        for r in results
    ]
    _scatter_models(ax, rows, styles)
    _configure_size_axis(ax, [x for _, x, _ in rows])
    ax.set_xlabel("Full system size (MB, incl. tokenizer + embeddings)")
    ax.set_ylabel("Pearson correlation")
    ax.set_title("System cost vs accuracy (deployment view)")
    ax.grid(True, alpha=0.3)
    _legend_outside(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "system_cost_vs_accuracy.png"), bbox_inches="tight")
    plt.close(fig)

    # Forward time vs accuracy — lightweight encoders + static baselines (no transformers).
    fig, ax = plt.subplots(figsize=(8, 5))
    rows = [
        (r["model"], reported_block(r)[1]["avg_sample_time"], reported_block(r)[1]["metrics"]["pearson"])
        for r in results
        if in_lightweight_latency_plot(r["model"])
    ]
    if rows:
        _scatter_models(ax, rows, {n: styles[n] for n, _, _ in rows})
        ax.set_xlabel("Avg forward time per pair (ms)")
        ax.set_ylabel("Pearson correlation")
        ax.set_title("Forward latency vs accuracy (lightweight encoders)")
        ax.grid(True, alpha=0.3)
        _legend_outside(ax)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "time_vs_accuracy_no_sentence_transformers.png"),
                    bbox_inches="tight")
    plt.close(fig)

    # End-to-end latency (tokenization + preprocessing + inference) — all models.
    fig, ax = plt.subplots(figsize=(8, 5))
    rows = []
    for r in results:
        e2e = reported_block(r)[1].get("e2e_sample_time")
        if e2e is None:
            continue
        rows.append((r["model"], e2e, reported_block(r)[1]["metrics"]["pearson"]))
    if rows:
        _scatter_models(ax, rows, {n: styles[n] for n, _, _ in rows})
        ax.set_xlabel("Avg end-to-end time per pair incl. tokenization (ms)")
        ax.set_ylabel("Pearson correlation")
        ax.set_title("End-to-end latency vs accuracy")
        ax.grid(True, alpha=0.3)
        _legend_outside(ax)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "time_vs_accuracy_with_tokenization.png"),
                    bbox_inches="tight")
    plt.close(fig)

    # FP32 vs INT8 Pearson comparison (trainable models only show a real delta).
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(results))
    width = 0.4
    ax.bar(x - width / 2, [r["fp32"]["metrics"]["pearson"] for r in results], width, label="FP32")
    ax.bar(x + width / 2, [r["int8"]["metrics"]["pearson"] for r in results], width, label="INT8")
    ax.set_xticks(x)
    ax.set_xticklabels([r["model"] for r in results], rotation=45, ha="right")
    ax.set_ylabel("Pearson correlation")
    ax.set_title("FP32 vs INT8 Pearson")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fp32_vs_int8_pearson.png"))
    plt.close(fig)

    # Pearson vs Spearman (reported precision per model).
    fig, ax = plt.subplots(figsize=(8, 5))
    rows = [(r["model"], float(reported_block(r)[1]["metrics"]["spearman"]),
             float(reported_block(r)[1]["metrics"]["pearson"])) for r in results]
    _scatter_models(ax, rows, styles)
    lo = min([v for _, sp, pe in rows for v in (sp, pe)] + [0.0])
    ax.plot([lo, 1], [lo, 1], "k--", alpha=0.5, zorder=1, label="y=x")
    ax.set_xlabel("Spearman correlation")
    ax.set_ylabel("Pearson correlation")
    ax.set_title("Pearson vs Spearman")
    ax.grid(True, alpha=0.3)
    _legend_outside(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pearson_vs_spearman.png"), bbox_inches="tight")
    plt.close(fig)
