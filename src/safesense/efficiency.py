"""Model-only FP32 latency and CUDA-Graph helpers."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence
import numpy as np
import torch
from torch import Tensor, nn

def benchmark_model(
    model: nn.Module,
    inputs: Sequence[Tensor],
    *,
    device: str = "cuda",
    warmups: int = 200,
    iterations: int = 1000,
    checkpoint_bytes: int | None = None,
) -> dict[str, object]:
    """Measure only the forward call, synchronizing CUDA around each sample."""
    if warmups < 0 or iterations < 1:
        raise ValueError("warmups must be non-negative and iterations positive")
    selected = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    model = model.to(selected).eval()
    values = tuple(value.to(selected) for value in inputs)
    with torch.inference_mode():
        for _ in range(warmups):
            model(*values)
        if selected.type == "cuda":
            torch.cuda.synchronize(selected)
        samples: list[float] = []
        for _ in range(iterations):
            if selected.type == "cuda":
                torch.cuda.synchronize(selected)
            started = time.perf_counter()
            model(*values)
            if selected.type == "cuda":
                torch.cuda.synchronize(selected)
            samples.append((time.perf_counter() - started) * 1000)
    return {
        "device": str(selected),
        "warmups": warmups,
        "iterations": iterations,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "checkpoint_bytes_fp32": checkpoint_bytes,
        "median_ms_model_only": float(np.median(samples)),
        "p95_ms_model_only": float(np.percentile(samples, 95)),
    }


def checkpoint_bytes(path: str | Path) -> int:
    return Path(path).stat().st_size


def _cuda_graph_repair_internal_gaps(value: Tensor, mask: Tensor, *, radius: int = 3) -> Tensor:
    """Branch-free equivalent of SafeSense internal-gap repair for graph capture."""
    from .utils import zero_padded_shift

    valid = mask.float().clamp(0, 1)
    left = torch.zeros_like(valid, dtype=torch.bool)
    right = torch.zeros_like(valid, dtype=torch.bool)
    for offset in range(1, radius + 1):
        left |= zero_padded_shift(valid, offset, time_dim=1).bool()
        right |= zero_padded_shift(valid, -offset, time_dim=1).bool()
    repair = (valid <= 0) & left & right
    channels = value.transpose(1, 2)
    weights = valid.unsqueeze(1)
    kernel = 2 * radius + 1
    numerator = torch.nn.functional.avg_pool1d(channels * weights, kernel, stride=1, padding=radius) * kernel
    denominator = torch.nn.functional.avg_pool1d(weights, kernel, stride=1, padding=radius) * kernel
    filled = numerator / denominator.clamp_min(1.0)
    return torch.where(repair.unsqueeze(1), filled, channels).transpose(1, 2)


def _cuda_graph_safe_valid(valid: Tensor) -> Tensor:
    """Branch-free equivalent of FactorizedTimeModalityBlock._safe_valid."""
    safe = valid.clone()
    missing = ~safe.any(dim=1)
    safe[:, 0] = safe[:, 0] | missing
    return safe


class CudaGraphBatch1Inference(nn.Module):
    """Replay a frozen batch-1 CUDA forward with static device input buffers."""

    def __init__(self, model: nn.Module, example_inputs: Sequence[Tensor]):
        super().__init__()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA Graph inference requires CUDA")
        values = tuple(example_inputs)
        if not values or any(value.device.type != "cuda" or value.shape[0] != 1 for value in values):
            raise ValueError("CUDA Graph inference requires non-empty CUDA batch-1 inputs")
        devices = {value.device for value in values}
        if len(devices) != 1:
            raise ValueError("CUDA Graph inputs must share one device")
        self.model = model.eval()
        self._device = values[0].device
        self._static_inputs = tuple(value.detach().clone() for value in values)
        self._graph = torch.cuda.CUDAGraph()

        from . import model as safesense_module

        original_repair = safesense_module._repair_internal_gaps
        original_safe_valid = safesense_module.FactorizedTimeModalityBlock.__dict__["_safe_valid"]
        patch_safesense = isinstance(self.model, safesense_module.SafeSenseFactorized)
        if patch_safesense:
            safesense_module._repair_internal_gaps = _cuda_graph_repair_internal_gaps
            safesense_module.FactorizedTimeModalityBlock._safe_valid = staticmethod(_cuda_graph_safe_valid)
        try:
            with torch.inference_mode():
                for _ in range(20):
                    self.model(*self._static_inputs)
            torch.cuda.synchronize(self._device)
            with torch.inference_mode(), torch.cuda.graph(self._graph):
                self._static_output = self.model(*self._static_inputs)
        finally:
            if patch_safesense:
                safesense_module._repair_internal_gaps = original_repair
                safesense_module.FactorizedTimeModalityBlock._safe_valid = original_safe_valid

    def forward(self, *inputs: Tensor):
        if len(inputs) != len(self._static_inputs):
            raise ValueError("CUDA Graph input count changed after capture")
        for static, value in zip(self._static_inputs, inputs):
            if value.shape != static.shape or value.dtype != static.dtype or value.device != static.device:
                raise ValueError("CUDA Graph input shape, dtype, and device must match capture")
            static.copy_(value)
        self._graph.replay()
        return self._static_output
