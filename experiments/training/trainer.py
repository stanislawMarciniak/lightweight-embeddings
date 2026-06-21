from __future__ import annotations

import math
import os
import time
from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from config import TrainingConfig
from evaluation.metrics import compute_pearson
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
        self.best: float | None = None
        self.no_improve_epochs = 0

    def step(self, value: float) -> bool:
        if self.best is None or value > self.best + self.min_improvement:
            self.best = value
            self.no_improve_epochs = 0
        else:
            self.no_improve_epochs += 1
        return self.no_improve_epochs >= self.patience_epochs


class Trainer:
    """Single, uniform training loop for all trainable models.

    Trains on the train split, early-stops on validation Pearson, keeps the best
    checkpoint, and saves it to ``experiment_dir/model.pt``. Intentionally does
    not persist training curves / history — only the artifact needed downstream
    (the checkpoint) is written.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        experiment_dir: str,
        warmup_epochs: int = 3,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.warmup_epochs = warmup_epochs
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        # Shared, frozen 768-d WordPiece embedding table lives outside the encoder.
        from models.embeddings import get_token_embedding

        self.embedder = get_token_embedding().to(self.device)
        self.criterion = HybridLoss(
            w_pearson=config.w_pearson,
            w_spearman=config.w_spearman,
            w_contrastive=config.w_contrastive,
            w_cosent=config.w_cosent,
            tau_spearman=config.tau_spearman,
            cosent_tau=config.cosent_tau,
            margin=config.margin,
        )
        self.optimizer = create_optimizer(
            self.model, lr=config.lr, weight_decay=config.weight_decay
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

        self.use_amp = self.device.type == "cuda" and config.fp16
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def _run_epoch(self, train: bool = True) -> Tuple[float, float]:
        loader = self.train_loader if train else self.val_loader
        self.model.train(train)
        total_loss = 0.0
        total_counted = 0
        all_preds: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []

        for batch in loader:
            labels = batch["label"].to(self.device)
            # input_ids -> embedding lookup -> token_embeddings (B, T, 768).
            inputs = self.embedder.embed_pair(batch, self.device)

            with torch.set_grad_enabled(train), torch.autocast(
                device_type=self.device.type, enabled=(self.use_amp and train)
            ):
                out = self.model(**inputs)
                score = out["score"]
                loss, _ = self.criterion(score, labels, out["z1"], out["z2"])

            if train:
                if not torch.isfinite(loss):
                    continue
                self.optimizer.zero_grad()
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    self.optimizer.step()

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
        history: Dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_pearson": []}

        best_score: float = float("-inf")
        best_state: Optional[Dict[str, Any]] = None
        best_val_pearson: float = float("-inf")

        for epoch in range(self.config.num_epochs):
            start = time.perf_counter()
            train_loss, _ = self._run_epoch(train=True)
            val_loss, val_pearson = self._run_epoch(train=False)
            elapsed = time.perf_counter() - start

            val_pearson_safe = -1.0 if math.isnan(val_pearson) else val_pearson
            # Maximize val Pearson; fall back to -val_loss while predictions are constant.
            score = val_pearson_safe if val_pearson_safe > -1.0 else -val_loss

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

            print(
                f"Epoch {epoch+1}/{self.config.num_epochs} "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_pearson={val_pearson:.4f} best={best_val_pearson:.4f} "
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
        return history


def create_dataloader(
    dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
    collate_fn=None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
