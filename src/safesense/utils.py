"""Small deterministic and metric utilities used across the public release."""

from __future__ import annotations

import random
import numpy as np
import torch
from torch import Tensor, nn

from .constants import NUM_CLASSES


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def zero_padded_shift(value: Tensor, amount: int, *, time_dim: int) -> Tensor:
    """Shift along one temporal dimension using zeros, never circular wraparound."""
    if not -value.shape[time_dim] < amount < value.shape[time_dim]:
        return torch.zeros_like(value)
    shifted = torch.zeros_like(value)
    source = [slice(None)] * value.ndim
    destination = [slice(None)] * value.ndim
    if amount >= 0:
        source[time_dim] = slice(0, value.shape[time_dim] - amount if amount else None)
        destination[time_dim] = slice(amount, None)
    else:
        source[time_dim] = slice(-amount, None)
        destination[time_dim] = slice(0, amount)
    shifted[tuple(destination)] = value[tuple(source)]
    return shifted


def classification_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict[str, object]:
    predictions = np.asarray(predictions, dtype=np.int64)
    targets = np.asarray(targets, dtype=np.int64)
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    np.add.at(confusion, (targets, predictions), 1)
    tp = np.diag(confusion).astype(np.float64)
    support = confusion.sum(axis=1).astype(np.float64)
    predicted = confusion.sum(axis=0).astype(np.float64)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) > 0)
    return {
        "accuracy": float(np.mean(predictions == targets)),
        "macro_f1": float(np.mean(f1)),
        "per_class_f1": [float(v) for v in f1],
        "confusion_matrix": confusion.tolist(),
    }


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
