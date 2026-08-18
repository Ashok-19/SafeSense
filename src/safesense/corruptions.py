"""The fixed 25-scenario SafeSense corruption suite."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence
import numpy as np

from .constants import FRAME_COUNT

class RobustnessError(ValueError):
    """Raised for invalid robustness inputs or scenarios."""


@dataclass(frozen=True)
class CorruptionSpec:
    name: str
    category: str
    modality: str | None = None
    shift: int = 0
    loss_fraction: float = 0.0
    noise_std: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def scenario_specs() -> tuple[CorruptionSpec, ...]:
    specs = [CorruptionSpec("clean", "clean")]
    for modality in ("skeleton", "imu"):
        for shift in (-4, -2, 2, 4):
            specs.append(CorruptionSpec(f"{modality}_shift_{shift:+d}", "shift", modality, shift=shift))
        for fraction in (0.1, 0.3, 0.5):
            specs.append(CorruptionSpec(f"{modality}_frame_loss_{int(fraction * 100)}", "frame_loss", modality, loss_fraction=fraction))
        for noise in (0.1, 0.3, 0.5):
            specs.append(CorruptionSpec(f"{modality}_noise_{noise}", "noise", modality, noise_std=noise))
    specs.extend((
        CorruptionSpec("missing_skeleton", "missing", "skeleton"),
        CorruptionSpec("missing_imu", "missing", "imu"),
        CorruptionSpec("missing_both", "missing", "both"),
        CorruptionSpec("imu_composite_shift_+4_loss_30_noise_0.3", "composite", "imu", shift=4, loss_fraction=0.3, noise_std=0.3),
    ))
    return tuple(specs)


def base_temporal_masks(fused: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(fused)
    if values.ndim != 5 or values.shape[1:] != (1, FRAME_COUNT, 22, 3):
        raise RobustnessError(f"expected fused array [N,1,128,22,3], got {values.shape}")
    sequence = values[:, 0]
    # Official preprocessing repeats/pads skeleton frames to 128. The appended
    # IMU nodes retain the only reliable clean temporal validity signal.
    clean_mask = np.any(sequence[:, :, 20:] != 0, axis=(2, 3))
    return {
        "skeleton": clean_mask.copy(),
        "imu": clean_mask.copy(),
    }


def _shift(values: np.ndarray, amount: int) -> np.ndarray:
    shifted = np.zeros_like(values)
    if amount >= 0:
        if amount < values.shape[1]:
            shifted[:, amount:] = values[:, : values.shape[1] - amount]
    elif -amount < values.shape[1]:
        shifted[:, : amount] = values[:, -amount:]
    return shifted


def _apply_shift(values: np.ndarray, mask: np.ndarray, amount: int) -> tuple[np.ndarray, np.ndarray]:
    return _shift(values, amount), _shift(mask.astype(np.float32)[..., None], amount)[..., 0].astype(bool)


def training_noise_scales(fused: np.ndarray, masks: dict[str, np.ndarray], train_indices: Sequence[int]) -> dict[str, float]:
    values = np.asarray(fused)[:, 0]
    indices = np.asarray(train_indices, dtype=np.int64)
    scales: dict[str, float] = {}
    for name, slc in (("skeleton", slice(0, 20)), ("imu", slice(20, 22))):
        valid_values = values[indices][:, :, slc][masks[name][indices]]
        if valid_values.size == 0:
            raise RobustnessError(f"no valid training values for {name} noise scale")
        scales[name] = float(valid_values.std(dtype=np.float64)) or 1.0
    return scales


def _frame_loss(mask: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    result = mask.copy()
    for row in range(mask.shape[0]):
        valid = np.flatnonzero(mask[row])
        count = min(len(valid), int(round(len(valid) * fraction)))
        if count:
            result[row, rng.choice(valid, size=count, replace=False)] = False
    return result


def apply_corruption(
    fused: np.ndarray,
    spec: CorruptionSpec,
    *,
    seed: int,
    scales: dict[str, float],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    values = np.asarray(fused, dtype=np.float32).copy()
    masks = base_temporal_masks(values)
    rng = np.random.default_rng(seed)
    modalities = ("skeleton", "imu") if spec.modality == "both" else ((spec.modality,) if spec.modality else ())
    if spec.name == "clean":
        return values, masks
    for modality in modalities:
        slc = slice(0, 20) if modality == "skeleton" else slice(20, 22)
        if spec.name.startswith("missing"):
            values[:, 0, :, slc] = 0
            masks[modality][:] = False
            continue
        branch = values[:, 0, :, slc]
        branch_mask = masks[modality]
        if spec.shift:
            branch, branch_mask = _apply_shift(branch, branch_mask, spec.shift)
        if spec.loss_fraction:
            old_mask = branch_mask
            branch_mask = _frame_loss(old_mask, spec.loss_fraction, rng)
            branch[old_mask & ~branch_mask] = 0
        if spec.noise_std:
            noise = rng.normal(0.0, scales[modality] * spec.noise_std, size=branch.shape).astype(np.float32)
            branch = np.where(branch_mask[..., None, None], branch + noise, branch)
        values[:, 0, :, slc] = branch
        masks[modality] = branch_mask
    return values, masks
