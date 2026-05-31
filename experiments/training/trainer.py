from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from config import TrainingConfig
from evaluation.metrics import compute_pearson, compute_spearman
from training.losses import HybridLoss
from training.optimizer import create_optimizer, create_plateau_scheduler, create_warmup_scheduler
from utils.seed import set_global_seed


class EarlyStopping:
    def __init__(
        self,
        window: int = 5,
        min_improvement: float = 1e-4,
        patience_epochs: int = 20,
    ) -> None:
        self.window = window
        self.min_improvement = min_improvement
        self.patience_epochs = patience_epochs
        self.history: list[float] = []
        self.best: float | None = None
        self.no_improve_epochs = 0

    def step(self, value: float) -> bool:
        self.history.append(value)
        if self.best is None or value > self.best + self.min_improvement:
            self.best = value
            self.no_improve_epochs = 0
        else:
            self.no_improve_epochs += 1
        return self.no_improve_epochs >= self.patience_epochs


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        experiment_dir: str,
        warmup_epochs: int = 3,
        full_mode: bool = False,
        test_loader: Optional[DataLoader] = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.warmup_epochs = warmup_epochs
        self.full_mode = full_mode
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.criterion = HybridLoss(
            w_pearson=config.w_pearson,
            w_spearman=config.w_spearman,
            w_contrastive=config.w_contrastive,
            tau_spearman=config.tau_spearman,
            margin=config.margin,
        )
        self.optimizer = create_optimizer(
            self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        self.warmup_scheduler = create_warmup_scheduler(self.optimizer, warmup_epochs)
        self.plateau_scheduler = create_plateau_scheduler(self.optimizer)
        self.early_stopping = EarlyStopping(
            window=config.early_stopping_ma_window,
            min_improvement=config.early_stopping_min_improvement,
            patience_epochs=config.early_stopping_patience_epochs,
        )
        self.experiment_dir = experiment_dir
        os.makedirs(self.experiment_dir, exist_ok=True)
        set_global_seed(config.seed)

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.device.type == "cuda")

    def _run_epoch(
        self, train: bool = True, loader: Optional[DataLoader] = None
    ) -> Tuple[float, float]:
        if loader is None:
            loader = self.train_loader if train else self.val_loader
        self.model.train(train)
        total_loss = 0.0
        total_counted = 0
        all_preds: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []

        for batch in loader:
            inputs = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            labels = inputs.pop("label")

            with torch.set_grad_enabled(train), \
                 torch.autocast(device_type=self.device.type, enabled=(self.device.type == "cuda" and train)):
                out = self.model(**inputs)
                score = out["score"]
                loss, _ = self.criterion(score, labels, out["z1"], out["z2"])

            if train:
                if torch.isfinite(loss):
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    continue

            batch_loss = loss.item()
            if math.isfinite(batch_loss):
                total_loss += batch_loss * labels.size(0)
                total_counted += labels.size(0)
            all_preds.append(score.detach().cpu())
            all_targets.append(labels.detach().cpu())

        avg_loss = total_loss / max(total_counted, 1)
        preds = torch.cat(all_preds)
        targets = torch.cat(all_targets)
        pearson = compute_pearson(preds.numpy(), targets.numpy())
        return avg_loss, pearson

    def fit(self) -> Dict[str, Any]:
        history: Dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_pearson": [],
            "test_loss": [],
        }

        # In full_mode we optimize train loss (minimize); otherwise val Pearson (maximize).
        # Use a single "score" that we maximize: score = -train_loss in full_mode, else val_pearson.
        best_score: float = float("-inf")
        best_state: Optional[Dict[str, Any]] = None
        best_val_pearson: float = float("-inf")

        for epoch in range(self.config.num_epochs):
            start = time.perf_counter()
            train_loss, _ = self._run_epoch(train=True)
            val_loss, val_pearson = self._run_epoch(train=False)
            if self.full_mode and self.test_loader is not None:
                test_loss, _ = self._run_epoch(train=False, loader=self.test_loader)
                history["test_loss"].append(test_loss)
            elapsed = time.perf_counter() - start

            val_pearson_safe = -1.0 if math.isnan(val_pearson) else val_pearson
            if self.full_mode:
                score = -train_loss
            elif val_pearson_safe == -1.0:
                score = -val_loss
            else:
                score = val_pearson_safe

            if epoch < self.warmup_epochs:
                self.warmup_scheduler.step()
            else:
                self.plateau_scheduler.step(score)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_pearson"].append(val_pearson)

            if score > best_score:
                best_score = score
                best_val_pearson = val_pearson_safe
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

            should_stop = self.early_stopping.step(score)

            best_display = f"best_loss={-best_score:.4f}" if self.full_mode else f"best={best_val_pearson:.4f}"
            print(
                f"Epoch {epoch+1}/{self.config.num_epochs} "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_pearson={val_pearson:.4f} {best_display} "
                f"time={elapsed:.1f}s"
            )

            if should_stop:
                print("Early stopping triggered.")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "config": asdict(self.config),
                "best_val_pearson": best_val_pearson,
            },
            os.path.join(self.experiment_dir, "model.pt"),
        )
        # Persist history for plotting (strip empty test_loss if not used)
        history_export = {k: v for k, v in history.items() if v}
        with open(
            os.path.join(self.experiment_dir, "history.json"),
            "w",
            encoding="utf8",
        ) as f:
            json.dump(history_export, f, indent=2)
        return history


def create_dataloader(
    dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=None,
    )


# ---------------------------------------------------------------------------
# Cross-validation utilities
# ---------------------------------------------------------------------------


class PairSwapDataset(Dataset):
    """Returns each sample twice: original ordering and with swapped sentence pairs.

    Since STS similarity is symmetric (sim(s1,s2) == sim(s2,s1)) but the scorer
    concatenates [z1, z2, ...] positionally, exposing both orderings during
    training effectively doubles the data and teaches the scorer symmetry.
    """

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset) * 2  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        real_idx = idx // 2
        item = self.dataset[real_idx]
        if idx % 2 == 1:
            return {
                "token_embeddings_1": item["token_embeddings_2"],
                "attention_mask_1": item["attention_mask_2"],
                "token_embeddings_2": item["token_embeddings_1"],
                "attention_mask_2": item["attention_mask_1"],
                "label": item["label"],
                "genre": item.get("genre", ""),
                "sentence1": item.get("sentence2", ""),
                "sentence2": item.get("sentence1", ""),
            }
        return item


class EnsembleWrapper(nn.Module):
    """Averages predictions from multiple fold models."""

    def __init__(self, models: List[nn.Module]) -> None:
        super().__init__()
        self.models = nn.ModuleList(models)

    def encode(
        self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        encodings = [m.encode(token_embeddings, attention_mask) for m in self.models]
        return torch.stack(encodings).mean(dim=0)

    def forward(
        self,
        token_embeddings_1: torch.Tensor,
        attention_mask_1: torch.Tensor,
        token_embeddings_2: torch.Tensor,
        attention_mask_2: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        all_scores: List[torch.Tensor] = []
        all_z1: List[torch.Tensor] = []
        all_z2: List[torch.Tensor] = []
        for m in self.models:
            out = m(
                token_embeddings_1, attention_mask_1,
                token_embeddings_2, attention_mask_2,
            )
            all_scores.append(out["score"])
            all_z1.append(out["z1"])
            all_z2.append(out["z2"])
        return {
            "score": torch.stack(all_scores).mean(dim=0),
            "z1": torch.stack(all_z1).mean(dim=0),
            "z2": torch.stack(all_z2).mean(dim=0),
        }


def _k_fold_indices(
    n: int, k: int, seed: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Simple k-fold split without sklearn dependency."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    fold_size = n // k
    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(k):
        start = i * fold_size
        end = (start + fold_size) if i < k - 1 else n
        val_idx = perm[start:end]
        train_idx = np.concatenate([perm[:start], perm[end:]])
        folds.append((train_idx, val_idx))
    return folds


def cross_validate(
    model_factory,
    combined_data: Dataset,
    config: TrainingConfig,
    experiment_dir: str,
    n_folds: int = 5,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """K-fold cross-validation with pair-swap augmentation and model ensembling.

    Returns an EnsembleWrapper (averages fold-model predictions) and a dict
    with CV metrics and averaged training history.
    """
    os.makedirs(experiment_dir, exist_ok=True)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    n = len(combined_data)  # type: ignore[arg-type]
    folds = _k_fold_indices(n, n_folds, seed=config.seed)

    oof_preds = np.zeros(n)
    oof_targets = np.zeros(n)
    all_histories: List[Dict[str, list]] = []
    best_fold_pearson = -1.0
    best_fold_idx = 0
    best_fold_epochs = 0

    for fold_i, (train_idx, val_idx) in enumerate(folds):
        fold_dir = os.path.join(experiment_dir, f"fold_{fold_i}")
        print(
            f"\n--- Fold {fold_i + 1}/{n_folds} "
            f"(train={len(train_idx)}, val={len(val_idx)}) ---"
        )

        train_subset = Subset(combined_data, train_idx.tolist())
        val_subset = Subset(combined_data, val_idx.tolist())
        train_aug = PairSwapDataset(train_subset)

        train_loader = create_dataloader(
            train_aug, config.batch_size, config.num_workers, shuffle=True,
        )
        val_loader = create_dataloader(
            val_subset, config.batch_size, config.num_workers, shuffle=False,
        )

        fold_config = replace(config, seed=config.seed + fold_i)
        set_global_seed(fold_config.seed)
        model = model_factory()

        trainer = Trainer(
            model, train_loader, val_loader, fold_config, experiment_dir=fold_dir,
        )
        history = trainer.fit()
        all_histories.append(history)

        # Collect out-of-fold predictions
        model.eval()
        model.to(device)
        preds_list: List[np.ndarray] = []
        targets_list: List[np.ndarray] = []
        with torch.no_grad():
            for batch in val_loader:
                inputs = {
                    k: v.to(device)
                    for k, v in batch.items()
                    if isinstance(v, torch.Tensor)
                }
                labels = inputs.pop("label")
                out = model(**inputs)
                preds_list.append(out["score"].cpu().numpy())
                targets_list.append(labels.cpu().numpy())

        fold_preds = np.concatenate(preds_list)
        fold_targets = np.concatenate(targets_list)
        oof_preds[val_idx] = fold_preds
        oof_targets[val_idx] = fold_targets

        fold_pearson = compute_pearson(fold_preds, fold_targets)
        print(f"  Fold {fold_i + 1} val Pearson: {fold_pearson:.4f}")

        if fold_pearson > best_fold_pearson:
            best_fold_pearson = fold_pearson
            best_fold_idx = fold_i
            vp = history["val_pearson"]
            best_fold_epochs = int(np.argmax(vp)) + 1

        model.cpu()
        del model

    cv_pearson = compute_pearson(oof_preds, oof_targets)
    cv_spearman = compute_spearman(oof_preds, oof_targets)
    print(f"\n=== CV Pearson: {cv_pearson:.4f}, CV Spearman: {cv_spearman:.4f} ===")
    print(f"  Best fold: {best_fold_idx + 1} (Pearson {best_fold_pearson:.4f}, {best_fold_epochs} epochs)")

    # Retrain a single model on ALL combined data using the best fold's seed
    # and epoch count. This gives the model maximum data exposure.
    final_seed = config.seed + best_fold_idx
    final_config = replace(config, seed=final_seed, num_epochs=best_fold_epochs)
    set_global_seed(final_seed)
    final_model = model_factory()

    full_aug = PairSwapDataset(combined_data)
    full_loader = create_dataloader(
        full_aug, config.batch_size, config.num_workers, shuffle=True,
    )
    final_dir = os.path.join(experiment_dir, "final")
    print(f"\n--- Retraining on all data (seed={final_seed}, epochs={best_fold_epochs}) ---")
    final_trainer = Trainer(
        final_model, full_loader, full_loader, final_config,
        experiment_dir=final_dir, full_mode=True,
    )
    final_history = final_trainer.fit()

    min_epochs = min(len(h["train_loss"]) for h in all_histories)
    avg_history: Dict[str, list] = {
        "train_loss": [
            float(np.mean([h["train_loss"][e] for h in all_histories]))
            for e in range(min_epochs)
        ],
        "val_loss": [
            float(np.mean([h["val_loss"][e] for h in all_histories]))
            for e in range(min_epochs)
        ],
        "val_pearson": [
            float(np.mean([h["val_pearson"][e] for h in all_histories]))
            for e in range(min_epochs)
        ],
    }

    cv_info: Dict[str, Any] = {
        "cv_pearson": cv_pearson,
        "cv_spearman": cv_spearman,
        "best_fold_pearson": best_fold_pearson,
        "best_fold_idx": best_fold_idx,
        "best_fold_epochs": best_fold_epochs,
        "n_folds": n_folds,
        "avg_history": avg_history,
    }
    with open(
        os.path.join(experiment_dir, "cv_info.json"), "w", encoding="utf8"
    ) as f:
        json.dump(cv_info, f, indent=2)

    torch.save(
        {
            "model_state_dict": final_model.state_dict(),
            "cv_pearson": cv_pearson,
            "best_fold_pearson": best_fold_pearson,
        },
        os.path.join(experiment_dir, "model.pt"),
    )

    return final_model, cv_info