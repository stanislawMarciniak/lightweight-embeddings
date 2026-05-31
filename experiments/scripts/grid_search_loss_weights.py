from __future__ import annotations

import csv
import json
import os
import sys
from dataclasses import asdict
from typing import Dict, List, Set, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
from torch.utils.data import DataLoader

from config import ExperimentConfig, TrainingConfig
from evaluation.benchmark import evaluate_model
from models.custom import CustomHybridModel
from preprocessing.dataset_precomputed import PrecomputedSTSDataset
from training.losses import HybridLoss
from training.trainer import Trainer, create_dataloader
from utils.seed import set_global_seed


CURRENT_W_PEARSON = 0.7 / 1.1
CURRENT_W_SPEARMAN = 0.2 / 1.1
CURRENT_W_CONTRASTIVE = 0.2 / 1.1


PROBE_EPOCHS = 40

TOP_K_REFINE = 5

REFINE_STEP = 0.05

COARSE_STEP = 0.2


def _round3(t: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return (round(t[0], 4), round(t[1], 4), round(t[2], 4))


def coarse_grid() -> List[Tuple[float, float, float]]:
    """Uniform simplex grid at COARSE_STEP + fixed baselines + current config."""
    seen: Set[Tuple[float, float, float]] = set()

    def add(p: float, s: float, c: float) -> None:
        t = _round3((p, s, c))
        if t not in seen and c >= -1e-9:
            seen.add(_round3((p, s, max(c, 0.0))))

    # Baselines
    add(1.0, 0.0, 0.0)   # pearson only
    add(0.0, 1.0, 0.0)   # spearman only
    add(0.0, 0.0, 1.0)   # contrastive only

    add(0.5, 0.5, 0.0)
    add(0.8, 0.2, 0.0)

    # Centre of simplex
    add(1/3, 1/3, 1/3)

    # Current default (normalized)
    add(CURRENT_W_PEARSON, CURRENT_W_SPEARMAN, CURRENT_W_CONTRASTIVE)

    # Uniform grid
    steps = [round(i * COARSE_STEP, 4) for i in range(6)]
    for w_p in steps:
        for w_s in steps:
            w_c = round(1.0 - w_p - w_s, 4)
            add(w_p, w_s, w_c)

    return sorted(seen)


def neighbourhood(
    center: Tuple[float, float, float],
    step: float,
    visited: Set[Tuple[float, float, float]],
) -> List[Tuple[float, float, float]]:

    wp, ws, wc = center
    candidates = []
    deltas = [
        (+step, -step,  0.0),
        (-step, +step,  0.0),
        (+step,  0.0, -step),
        (-step,  0.0, +step),
        ( 0.0, +step, -step),
        ( 0.0, -step, +step),
    ]
    for dp, ds, dc in deltas:
        t = _round3((wp + dp, ws + ds, wc + dc))
        if all(v >= -1e-9 for v in t) and t not in visited:
            clamped = _round3((max(t[0], 0.0), max(t[1], 0.0), max(t[2], 0.0)))
            candidates.append(clamped)
    return candidates

def run_one(
    w_pearson: float,
    w_spearman: float,
    w_contrastive: float,
    train_data: PrecomputedSTSDataset,
    val_data: PrecomputedSTSDataset,
    test_data: PrecomputedSTSDataset,
    base_config: ExperimentConfig,
    exp_subdir: str,
    probe_epochs: int,
) -> Dict:

    set_global_seed(base_config.training.seed)

    training_cfg_dict = asdict(base_config.training)
    training_cfg_dict["num_epochs"] = probe_epochs
    training = TrainingConfig(**training_cfg_dict)

    model = CustomHybridModel(vocab_size=0)

    train_loader = create_dataloader(
        train_data,
        batch_size=training.batch_size,
        num_workers=training.num_workers,
        shuffle=True,
    )
    val_loader = create_dataloader(
        val_data,
        batch_size=training.batch_size,
        num_workers=training.num_workers,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=training.batch_size,
        shuffle=False,
        num_workers=training.num_workers,
        collate_fn=None,
    )

    exp_dir = os.path.join(base_config.results_dir, "grid_search_loss_weights", exp_subdir)
    os.makedirs(exp_dir, exist_ok=True)

    trainer = Trainer(model, train_loader, val_loader, training, experiment_dir=exp_dir)

    trainer.criterion = HybridLoss(
        w_pearson=w_pearson,
        w_spearman=w_spearman,
        w_contrastive=w_contrastive,
        tau_spearman=training.tau_spearman,
        margin=training.margin,
    )

    history = trainer.fit()
    best_val_pearson = max(history["val_pearson"]) if history["val_pearson"] else float("-inf")

    device = torch.device(training.device if torch.cuda.is_available() else "cpu")
    metrics, _ = evaluate_model(model, test_loader, device=device)

    return {
        "w_pearson": w_pearson,
        "w_spearman": w_spearman,
        "w_contrastive": w_contrastive,
        "val_pearson": round(best_val_pearson, 6),
        "test_pearson": round(metrics["pearson"], 6),
        "phase": None,  # set by caller
    }


def _save(results: List[Dict], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, "grid_search_loss_weights.json")
    with open(json_path, "w", encoding="utf8") as f:
        json.dump(results, f, indent=2)

    csv_path = os.path.join(out_dir, "grid_search_loss_weights.csv")
    with open(csv_path, "w", newline="", encoding="utf8") as f:
        writer = csv.writer(f)
        writer.writerow(["w_pearson", "w_spearman", "w_contrastive",
                         "val_pearson", "test_pearson", "phase"])
        for r in results:
            writer.writerow([r["w_pearson"], r["w_spearman"], r["w_contrastive"],
                             r["val_pearson"], r["test_pearson"], r["phase"]])


def _subdir(wp: float, ws: float, wc: float) -> str:
    return f"p{wp:.3f}_s{ws:.3f}_c{wc:.3f}".replace(".", "_")


def main() -> None:
    base_config = ExperimentConfig()
    base_config.results_dir = os.path.join(PROJECT_ROOT, "results")

    data_dir = os.path.join(PROJECT_ROOT, "data", "precomputed_embeddings")
    train_data = PrecomputedSTSDataset(os.path.join(data_dir, "train_token_embeddings.pt"))
    val_data   = PrecomputedSTSDataset(os.path.join(data_dir, "validation_token_embeddings.pt"))
    test_data  = PrecomputedSTSDataset(os.path.join(data_dir, "test_token_embeddings.pt"))

    out_dir = os.path.join(base_config.results_dir, "grid_search_loss_weights")
    visited: Set[Tuple[float, float, float]] = set()
    results: List[Dict] = []

    combos = coarse_grid()
    print(f"\n{'='*65}")
    print(f"PHASE 1 — coarse grid ({len(combos)} combinations, {PROBE_EPOCHS} probe epochs each)")
    print(f"{'='*65}\n")

    for i, (wp, ws, wc) in enumerate(combos, 1):
        key = _round3((wp, ws, wc))
        visited.add(key)
        print(f"  [{i:>2}/{len(combos)}] p={wp:.3f} s={ws:.3f} c={wc:.3f}", end="  ", flush=True)

        row = run_one(wp, ws, wc, train_data, val_data, test_data,
                      base_config, _subdir(wp, ws, wc), PROBE_EPOCHS)
        row["phase"] = "coarse"
        results.append(row)

        print(f"val={row['val_pearson']:.4f}  test={row['test_pearson']:.4f}")
        _save(results, out_dir)

    top_k = sorted(results, key=lambda r: r["val_pearson"], reverse=True)[:TOP_K_REFINE]

    print(f"\n{'='*65}")
    print(f"PHASE 2 — refining around top-{TOP_K_REFINE} coarse results (step={REFINE_STEP})")
    print(f"{'='*65}")
    for r in top_k:
        print(f"  seed: p={r['w_pearson']:.3f} s={r['w_spearman']:.3f} c={r['w_contrastive']:.3f}"
              f"  val={r['val_pearson']:.4f}")
    print()

    refine_queue: List[Tuple[float, float, float]] = []
    for r in top_k:
        center = _round3((r["w_pearson"], r["w_spearman"], r["w_contrastive"]))
        refine_queue.extend(neighbourhood(center, REFINE_STEP, visited))

    # Deduplicate queue preserving order
    seen_queue: Set[Tuple[float, float, float]] = set()
    unique_queue = []
    for pt in refine_queue:
        if pt not in seen_queue and pt not in visited:
            seen_queue.add(pt)
            unique_queue.append(pt)

    print(f"  {len(unique_queue)} new refinement points to evaluate\n")

    for i, (wp, ws, wc) in enumerate(unique_queue, 1):
        visited.add(_round3((wp, ws, wc)))
        print(f"  [{i:>2}/{len(unique_queue)}] p={wp:.3f} s={ws:.3f} c={wc:.3f}", end="  ", flush=True)

        row = run_one(wp, ws, wc, train_data, val_data, test_data,
                      base_config, _subdir(wp, ws, wc), PROBE_EPOCHS)
        row["phase"] = "refine"
        results.append(row)

        print(f"val={row['val_pearson']:.4f}  test={row['test_pearson']:.4f}")
        _save(results, out_dir)


    best = max(results, key=lambda r: r["val_pearson"])

    print(f"\n{'='*65}")
    print(f"GRID SEARCH COMPLETE — {len(results)} combinations evaluated")
    print(f"{'='*65}")
    print(f"\nTop 5 by val Pearson:")
    print(f"  {'w_p':>6} {'w_s':>6} {'w_c':>6}  {'val':>8}  {'test':>8}  phase")
    print(f"  {'-'*55}")
    for r in sorted(results, key=lambda r: r["val_pearson"], reverse=True)[:5]:
        print(f"  {r['w_pearson']:>6.3f} {r['w_spearman']:>6.3f} {r['w_contrastive']:>6.3f}"
              f"  {r['val_pearson']:>8.4f}  {r['test_pearson']:>8.4f}  {r['phase']}")

    print(f"\nBest: w_pearson={best['w_pearson']}, w_spearman={best['w_spearman']}, "
          f"w_contrastive={best['w_contrastive']}  →  val={best['val_pearson']:.4f}")
    print(f"Results saved to: {out_dir}/")


if __name__ == "__main__":
    main()