from __future__ import annotations

import copy
import inspect
import time
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn


def _get_quantization_backend() -> str:
    """Use qnnpack on Linux/WSL to avoid 'apply_dynamic is not implemented' with fbgemm."""
    try:
        # qnnpack has broader support for dynamic quantization across platforms
        torch.backends.quantized.engine = "qnnpack"
        return "qnnpack"
    except Exception:
        pass
    try:
        torch.backends.quantized.engine = "fbgemm"
        return "fbgemm"
    except Exception:
        pass
    return "qnnpack"


def quantize_model_int8(model: nn.Module) -> nn.Module:
    """
    Post-training dynamic INT8 quantization.

    - Uses a CPU copy; quantized model runs on CPU only.
    - Backend: prefers qnnpack (avoids fbgemm 'apply_dynamic' errors on Linux/WSL).
    - Does not modify the original model.
    """
    model_cpu = copy.deepcopy(model).cpu()
    model_cpu.eval()

    backend = _get_quantization_backend()
    torch.backends.quantized.engine = backend

    qmodel = torch.ao.quantization.quantize_dynamic(
        model_cpu,
        {nn.Linear, nn.GRU},
        dtype=torch.qint8,
    )
    return qmodel


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model: nn.Module) -> float:
    total_bytes = sum(
        p.nelement() * p.element_size()
        for p in model.state_dict().values()
        if isinstance(p, torch.Tensor)
    )
    return total_bytes / (1024**2)


def _is_quantized(model: nn.Module) -> bool:
    """True if model contains dynamically quantized layers (CPU-only)."""
    for m in model.modules():
        mod = type(m).__module__
        if "quantized" in mod and "dynamic" in mod:
            return True
    return False


def _filter_model_inputs(
    model: nn.Module,
    batch: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    """
    Keep only inputs accepted by model.forward; move tensors to device,
    pass non-tensors (e.g. sentence1, sentence2) as-is.
    """
    forward_params = set(inspect.signature(model.forward).parameters.keys())
    inputs = {}
    for k, v in batch.items():
        if k not in forward_params:
            continue
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device, non_blocking=True)
        else:
            inputs[k] = v
    if not inputs:
        raise ValueError("No valid inputs found for model.forward")
    return inputs


def measure_inference_time(
    model: nn.Module,
    batch: dict,
    device: torch.device,
    repeats: int = 10,
    warmup: int = 3,
) -> Tuple[float, float]:
    """
    Returns:
        avg_time_per_batch (seconds)
        avg_time_per_sample (seconds)

    Quantized models (INT8) only support CPU; device is forced to CPU for them.
    """
    # quantized::linear_dynamic has no CUDA implementation
    if _is_quantized(model) and device.type == "cuda":
        device = torch.device("cpu")

    model.eval()
    model.to(device)

    inputs = _filter_model_inputs(model, batch, device)

    first_val = next(iter(inputs.values()))
    if isinstance(first_val, torch.Tensor):
        batch_size = first_val.size(0)
    else:
        batch_size = len(first_val)

    with torch.no_grad():

        # Warmup
        for _ in range(warmup):
            _ = model(**inputs)

        if device.type == "cuda":
            torch.cuda.synchronize()

        times = []

        for _ in range(repeats):

            start = time.perf_counter()

            _ = model(**inputs)

            if device.type == "cuda":
                torch.cuda.synchronize()

            end = time.perf_counter()

            times.append(end - start)

    avg_batch = (sum(times) / len(times)) * 1000
    avg_sample = (avg_batch / batch_size) * 1000

    return avg_batch, avg_sample