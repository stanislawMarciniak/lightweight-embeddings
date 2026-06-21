from __future__ import annotations

import copy
import inspect
import os
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Tuple

import torch
import torch.nn as nn


def _get_quantization_backend() -> str:
    """Pick the fastest available dynamic-quant backend for the current CPU.

    qnnpack is tuned for ARM and is catastrophically slow for dynamic INT8 linear
    on x86 (often 40-200x slower than FP32), which makes "INT8" look slower than
    FP32. On x86 we therefore prefer fbgemm (falling back to the unified ``x86``
    engine), and only use qnnpack on ARM. Modern PyTorch supports dynamic Linear
    *and* GRU on fbgemm, so the historical reason for forcing qnnpack is gone.
    """
    import platform

    supported = list(torch.backends.quantized.supported_engines)
    machine = platform.machine().lower()
    if machine in ("aarch64", "arm64", "armv7l", "armv8l"):
        prefs = ["qnnpack", "x86", "fbgemm"]
    else:
        prefs = ["fbgemm", "x86", "qnnpack"]

    for engine in prefs:
        if engine in supported:
            torch.backends.quantized.engine = engine
            return engine
    return supported[0] if supported else "qnnpack"


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
    """On-disk size of the (serialized) state dict.

    Using torch.save into a buffer is robust for *both* FP32 and dynamically
    quantized models: dynamic-INT8 packs Linear/GRU weights into special objects
    that are NOT plain tensors in state_dict(), so the old `sum(tensor bytes)`
    approach reported ~0 MB for Linear-only models (e.g. DAN). Serializing counts
    the packed INT8 weights, giving a realistic, comparable size.
    """
    import io

    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / (1024**2)


def _is_quantized(model: nn.Module) -> bool:
    """True if model contains dynamically quantized layers (CPU-only)."""
    for m in model.modules():
        mod = type(m).__module__
        if "quantized" in mod and "dynamic" in mod:
            return True
    return False


_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


@contextmanager
def single_core_cpu() -> Iterator[torch.device]:
    """Run CPU work on a single core/thread for fair latency comparisons.

    Pins PyTorch and common BLAS libraries to one thread. Restores prior settings
    on exit.
    """
    cpu = torch.device("cpu")
    old_threads = torch.get_num_threads()
    try:
        old_interop = torch.get_num_interop_threads()
    except RuntimeError:
        old_interop = None
    old_env = {k: os.environ.get(k) for k in _THREAD_ENV_VARS}
    try:
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        for k in _THREAD_ENV_VARS:
            os.environ[k] = "1"
        yield cpu
    finally:
        torch.set_num_threads(old_threads)
        if old_interop is not None:
            try:
                torch.set_num_interop_threads(old_interop)
            except RuntimeError:
                pass
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


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
        avg_time_per_batch (milliseconds)
        avg_time_per_sample (milliseconds)

    Expects ``device`` to be CPU when benchmarking fair single-core latency.
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

    avg_batch = (sum(times) / len(times)) * 1000  # ms per batch
    avg_sample = avg_batch / batch_size  # ms per sample

    return avg_batch, avg_sample