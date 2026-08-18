"""Corruption-aware training views used by the final SafeSense objective."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from .constants import BONE_EDGES, LAG_SHIFTS
from .utils import zero_padded_shift

def _corrective_lag_target(applied_shift: int) -> int:
    corrective = -int(applied_shift)
    if corrective not in LAG_SHIFTS:
        raise ValueError(f"applied shift must map into {LAG_SHIFTS}, got {applied_shift}")
    return LAG_SHIFTS.index(corrective)


def _corrective_lag_target_for_modality(applied_shift: int, modality: str) -> int:
    """Return the corrective IMU-shift class for a shifted input modality."""
    if modality not in ("skeleton", "imu"):
        raise ValueError("modality must be skeleton or imu")
    corrective = int(applied_shift) if modality == "skeleton" else -int(applied_shift)
    if corrective not in LAG_SHIFTS:
        raise ValueError(f"applied shift must map into {LAG_SHIFTS}, got {applied_shift}")
    return LAG_SHIFTS.index(corrective)


def _pair_mask(mask: Tensor, shape: tuple[int, int]) -> Tensor:
    if mask.ndim == 2 and tuple(mask.shape) == shape:
        return torch.stack((mask, mask), dim=-1)
    if mask.ndim == 3 and tuple(mask.shape[:2]) == shape and mask.shape[-1] == 2:
        return mask.clone()
    raise ValueError("mask must have shape [B,T] or [B,T,2]")


def _drop_frames(value: Tensor, mask: Tensor, fraction: float) -> tuple[Tensor, Tensor]:
    valid = torch.nonzero(mask > 0, as_tuple=False).flatten()
    count = min(int(round(valid.numel() * fraction)), int(valid.numel()))
    if count:
        selected = valid[torch.randperm(valid.numel(), device=value.device)[:count]]
        value[selected] = 0
        mask[selected] = 0
    return value, mask


def _augment_temporal_batch(
    skeleton: Tensor,
    imu: Tensor,
    mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Create one corrupted view per sample while retaining a separate clean view."""
    skeleton = skeleton.clone()
    imu = imu.clone()
    pair_mask = _pair_mask(mask, tuple(skeleton.shape[:2])).float()
    batch_size = skeleton.shape[0]
    lag_targets = torch.full((batch_size,), _corrective_lag_target(0), dtype=torch.long, device=skeleton.device)
    categories = torch.randint(0, 5, (batch_size,), device=skeleton.device)
    fractions = (0.1, 0.3, 0.5)
    noise_levels = (0.1, 0.3, 0.5)

    for row in range(batch_size):
        category = int(categories[row])
        modality_index = int(torch.randint(0, 2, (1,), device=skeleton.device))
        modality = "skeleton" if modality_index == 0 else "imu"
        if category == 0:
            magnitude = int(torch.randint(1, 5, (1,), device=skeleton.device))
            direction = -1 if int(torch.randint(0, 2, (1,), device=skeleton.device)) == 0 else 1
            applied = magnitude * direction
            target = skeleton[row] if modality == "skeleton" else imu[row]
            target.copy_(zero_padded_shift(target, applied, time_dim=0))
            pair_mask[row, :, modality_index] = zero_padded_shift(pair_mask[row, :, modality_index], applied, time_dim=0)
            lag_targets[row] = _corrective_lag_target_for_modality(applied, modality)
        elif category == 1:
            fraction = fractions[int(torch.randint(0, len(fractions), (1,), device=skeleton.device))]
            target = skeleton[row] if modality == "skeleton" else imu[row]
            _drop_frames(target, pair_mask[row, :, modality_index], fraction)
        elif category == 2:
            noise_std = noise_levels[int(torch.randint(0, len(noise_levels), (1,), device=skeleton.device))]
            target = skeleton[row] if modality == "skeleton" else imu[row]
            valid = pair_mask[row, :, modality_index].bool().unsqueeze(-1)
            target.add_(torch.randn_like(target) * noise_std * valid)
        elif category == 3:
            if modality == "skeleton":
                skeleton[row].zero_()
            else:
                imu[row].zero_()
            pair_mask[row, :, modality_index] = 0
        else:
            magnitude = int(torch.randint(1, 5, (1,), device=skeleton.device))
            direction = -1 if int(torch.randint(0, 2, (1,), device=skeleton.device)) == 0 else 1
            applied = magnitude * direction
            imu[row] = zero_padded_shift(imu[row], applied, time_dim=0)
            pair_mask[row, :, 1] = zero_padded_shift(pair_mask[row, :, 1], applied, time_dim=0)
            _drop_frames(imu[row], pair_mask[row, :, 1], 0.3)
            valid = pair_mask[row, :, 1].bool().unsqueeze(-1)
            imu[row].add_(torch.randn_like(imu[row]) * 0.3 * valid)
            lag_targets[row] = _corrective_lag_target(applied)

    availability = torch.stack((pair_mask[..., 0].any(dim=1), pair_mask[..., 1].any(dim=1)), dim=1).float()
    lag_supervision = availability.bool().all(dim=1)
    return skeleton, imu, pair_mask, availability, lag_targets, lag_supervision


def _rebuild_engineered_features(skeleton_positions: Tensor, imu_values: Tensor) -> tuple[Tensor, Tensor]:
    """Rebuild SafeSense engineered features from raw skeleton positions and IMU channels."""
    if skeleton_positions.ndim != 3 or skeleton_positions.shape[-1] != 60:
        raise ValueError("skeleton_positions must have shape [B,T,60]")
    if imu_values.ndim != 3 or imu_values.shape[-1] != 6:
        raise ValueError("imu_values must have shape [B,T,6]")
    skeleton_difference = torch.zeros_like(skeleton_positions)
    imu_difference = torch.zeros_like(imu_values)
    if skeleton_positions.shape[1] > 1:
        skeleton_difference[:, 1:] = skeleton_positions[:, 1:] - skeleton_positions[:, :-1]
        imu_difference[:, 1:] = imu_values[:, 1:] - imu_values[:, :-1]
    joints = skeleton_positions.reshape(*skeleton_positions.shape[:2], 20, 3)
    bones = torch.stack([joints[:, :, child] - joints[:, :, parent] for child, parent in BONE_EDGES], dim=2)
    skeleton = torch.cat((skeleton_positions, skeleton_difference, bones.flatten(2)), dim=-1)
    imu = torch.cat((imu_values, imu_difference), dim=-1)
    return skeleton, imu


def _augment_subject_style_batch(
    skeleton: Tensor,
    imu: Tensor,
    mask: Tensor,
    stats: dict[str, Tensor],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Apply label-preserving subject/sensor style changes in raw physical space."""
    pair_mask = _pair_mask(mask, tuple(skeleton.shape[:2])).float()
    skeleton_positions = (skeleton * stats["skeleton_std"] + stats["skeleton_mean"])[:, :, :60].clone()
    imu_values = (imu * stats["imu_std"] + stats["imu_mean"])[:, :, :6].clone()
    joints = skeleton_positions.reshape(*skeleton_positions.shape[:2], 20, 3)
    valid = pair_mask[..., 0].float()
    root = joints[:, :, 1]
    anchor = (root * valid[..., None]).sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1.0)
    scale = 0.85 + 0.30 * torch.rand((len(joints), 1, 1, 1), device=joints.device, dtype=joints.dtype)
    yaw = (torch.rand(len(joints), device=joints.device, dtype=joints.dtype) * 2.0 - 1.0) * (15.0 * np.pi / 180.0)
    cosine, sine = torch.cos(yaw), torch.sin(yaw)
    rotation = torch.zeros((len(joints), 3, 3), device=joints.device, dtype=joints.dtype)
    rotation[:, 0, 0] = cosine; rotation[:, 0, 2] = sine
    rotation[:, 1, 1] = 1.0
    rotation[:, 2, 0] = -sine; rotation[:, 2, 2] = cosine
    centered = joints - anchor[:, None, None, :]
    transformed = torch.einsum("bij,btkj->btki", rotation, centered) * scale + anchor[:, None, None, :]
    translation_scale = stats["skeleton_noise_scale"] * 0.10
    translation = torch.randn((len(joints), 1, 1, 3), device=joints.device, dtype=joints.dtype) * translation_scale
    transformed = transformed + translation
    joints.copy_(torch.where(valid[..., None, None].bool(), transformed, joints))

    imu_triplets = imu_values.reshape(*imu_values.shape[:2], 2, 3)
    imu_rotated = torch.einsum("bij,btkj->btki", rotation, imu_triplets)
    gain = 0.90 + 0.20 * torch.rand((len(joints), 1, 2, 1), device=joints.device, dtype=joints.dtype)
    bias = torch.randn((len(joints), 1, 2, 1), device=joints.device, dtype=joints.dtype) * stats["imu_noise_scale"] * 0.03
    imu_styled = imu_rotated * gain + bias
    imu_valid = pair_mask[..., 1, None, None].bool()
    imu_triplets.copy_(torch.where(imu_valid, imu_styled, imu_triplets))

    rebuilt_skeleton, rebuilt_imu = _rebuild_engineered_features(joints.flatten(2), imu_triplets.flatten(2))
    rebuilt_skeleton = (rebuilt_skeleton - stats["skeleton_mean"]) / stats["skeleton_std"]
    rebuilt_imu = (rebuilt_imu - stats["imu_mean"]) / stats["imu_std"]
    rebuilt_skeleton *= pair_mask[..., 0, None]
    rebuilt_imu *= pair_mask[..., 1, None]
    availability = torch.stack((pair_mask[..., 0].any(dim=1), pair_mask[..., 1].any(dim=1)), dim=1).float()
    return rebuilt_skeleton, rebuilt_imu, pair_mask, availability


def _augment_structured_temporal_batch(
    skeleton: Tensor,
    imu: Tensor,
    mask: Tensor,
    stats: dict[str, Tensor],
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Corrupt raw physical channels, then rebuild/renormalize engineered features.

    This mirrors the robustness evaluator more closely than perturbing already-engineered
    standardized channels. Reconstruction supervision is emitted only for noise/frame-loss
    cases that a local denoiser can reasonably invert.
    """
    pair_mask = _pair_mask(mask, tuple(skeleton.shape[:2])).float()
    skeleton_positions = skeleton * stats["skeleton_std"] + stats["skeleton_mean"]
    imu_values = imu * stats["imu_std"] + stats["imu_mean"]
    skeleton_positions = skeleton_positions[:, :, :60].clone()
    imu_values = imu_values[:, :, :6].clone()
    batch_size = skeleton.shape[0]
    lag_targets = torch.full((batch_size,), _corrective_lag_target(0), dtype=torch.long, device=skeleton.device)
    reconstruction_supervision = torch.zeros((batch_size, 2), dtype=torch.bool, device=skeleton.device)
    correction_targets = torch.zeros((batch_size, 2), dtype=torch.float32, device=skeleton.device)
    categories = torch.randint(0, 5, (batch_size,), device=skeleton.device)
    fractions = (0.1, 0.3, 0.5)
    noise_levels = (0.1, 0.3, 0.5)

    for row in range(batch_size):
        category = int(categories[row])
        modality_index = int(torch.randint(0, 2, (1,), device=skeleton.device))
        modality = "skeleton" if modality_index == 0 else "imu"
        target = skeleton_positions[row] if modality == "skeleton" else imu_values[row]
        if category == 0:
            magnitude = int(torch.randint(1, 5, (1,), device=skeleton.device))
            direction = -1 if int(torch.randint(0, 2, (1,), device=skeleton.device)) == 0 else 1
            applied = magnitude * direction
            target.copy_(zero_padded_shift(target, applied, time_dim=0))
            pair_mask[row, :, modality_index] = zero_padded_shift(pair_mask[row, :, modality_index], applied, time_dim=0)
            lag_targets[row] = _corrective_lag_target_for_modality(applied, modality)
        elif category == 1:
            fraction = fractions[int(torch.randint(0, len(fractions), (1,), device=skeleton.device))]
            _drop_frames(target, pair_mask[row, :, modality_index], fraction)
            reconstruction_supervision[row, modality_index] = True
        elif category == 2:
            level = noise_levels[int(torch.randint(0, len(noise_levels), (1,), device=skeleton.device))]
            scale = stats[f"{modality}_noise_scale"] * level
            valid = pair_mask[row, :, modality_index].bool().unsqueeze(-1)
            target.add_(torch.randn_like(target) * scale * valid)
            reconstruction_supervision[row, modality_index] = True
            correction_targets[row, modality_index] = 1.0
        elif category == 3:
            target.zero_()
            pair_mask[row, :, modality_index] = 0
        else:
            magnitude = int(torch.randint(1, 5, (1,), device=skeleton.device))
            direction = -1 if int(torch.randint(0, 2, (1,), device=skeleton.device)) == 0 else 1
            applied = magnitude * direction
            imu_values[row] = zero_padded_shift(imu_values[row], applied, time_dim=0)
            pair_mask[row, :, 1] = zero_padded_shift(pair_mask[row, :, 1], applied, time_dim=0)
            _drop_frames(imu_values[row], pair_mask[row, :, 1], 0.3)
            valid = pair_mask[row, :, 1].bool().unsqueeze(-1)
            imu_values[row].add_(torch.randn_like(imu_values[row]) * stats["imu_noise_scale"] * 0.3 * valid)
            lag_targets[row] = _corrective_lag_target(applied)
            correction_targets[row, 1] = 1.0

    rebuilt_skeleton, rebuilt_imu = _rebuild_engineered_features(skeleton_positions, imu_values)
    rebuilt_skeleton = (rebuilt_skeleton - stats["skeleton_mean"]) / stats["skeleton_std"]
    rebuilt_imu = (rebuilt_imu - stats["imu_mean"]) / stats["imu_std"]
    rebuilt_skeleton = rebuilt_skeleton * pair_mask[..., 0, None]
    rebuilt_imu = rebuilt_imu * pair_mask[..., 1, None]
    availability = torch.stack((pair_mask[..., 0].any(dim=1), pair_mask[..., 1].any(dim=1)), dim=1).float()
    lag_supervision = availability.bool().all(dim=1)
    return rebuilt_skeleton, rebuilt_imu, pair_mask, availability, lag_targets, lag_supervision, reconstruction_supervision, correction_targets
