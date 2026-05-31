from __future__ import annotations

from typing import Iterable

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau


def create_optimizer(
    params: Iterable[torch.nn.Parameter],
    lr: float = 2e-3,
    weight_decay: float = 5e-5,
) -> AdamW:
    return AdamW(params, lr=lr, weight_decay=weight_decay)


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