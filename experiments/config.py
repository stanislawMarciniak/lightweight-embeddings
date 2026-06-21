from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(slots=True)
class TrainingConfig:
    model_name: str = "dan_standard"
    batch_size: int = 32
    num_epochs: int = 200
    lr: float = 2e-3
    weight_decay: float = 5e-5
    device: str = "cuda"
    seed: int = 42
    num_workers: int = 4
    max_grad_norm: float = 1.0
    margin: float = 0.5
    tau_spearman: float = 1
    early_stopping_patience_epochs: int = 20
    early_stopping_min_improvement: float = 1e-4
    early_stopping_ma_window: int = 5
    fp16: bool = False
    # HybridLoss weights (normalized in HybridLoss if they do not sum to 1).
    w_pearson: float = 0.2
    w_spearman: float = 0.2
    w_contrastive: float = 0.3
    # CoSENT ranking loss on cosine(z1, z2): training-only, strong STS signal,
    # zero inference cost. tau is the inverse temperature scale (1/20 by default).
    w_cosent: float = 0.3
    cosent_tau: float = 0.05


@dataclass(slots=True)
class QuantizationConfig:
    enabled: bool = True
    backend: Literal["fbgemm", "qnnpack"] = "fbgemm"


@dataclass(slots=True)
class ExperimentConfig:
    training: TrainingConfig = field(default_factory=TrainingConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    results_dir: str = "results"
    # Accuracy eval and latency both use this batch size on single-core CPU.
    eval_batch_size: int = 128


DEFAULT_CONFIG = ExperimentConfig()