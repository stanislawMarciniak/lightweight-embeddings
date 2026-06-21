from __future__ import annotations

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau


def create_optimizer(
    model: nn.Module,
    lr: float = 2e-3,
    weight_decay: float = 5e-5,
) -> AdamW:
    """AdamW with decoupled, layer-wise weight decay.

    Biases and 1-D parameters (LayerNorm/RMSNorm scales, etc.) are excluded from
    weight decay — the standard practice that lets us apply a meaningfully strong
    L2 penalty on weight matrices without distorting normalization/bias terms.
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    param_groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return AdamW(param_groups, lr=lr)


def create_warmup_scheduler(
    optimizer: AdamW,
    warmup_epochs: int,
) -> LambdaLR:

    def lr_lambda(epoch: int) -> float:
        if warmup_epochs <= 0:
            return 1.0
        return min(1.0, (epoch + 1) / warmup_epochs)

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def create_plateau_scheduler(
    optimizer: AdamW,
    patience: int = 5,
    factor: float = 0.5,
    min_lr: float = 1e-5,
) -> ReduceLROnPlateau:

    return ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=patience,
        factor=factor,
        min_lr=min_lr,
    )