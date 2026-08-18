"""Final SafeSenseFactorized architecture used in the paper experiments."""

from __future__ import annotations

from typing import Sequence
import torch
from torch import Tensor, nn

from .constants import BONE_EDGES, IMU_INPUTS, NUM_CLASSES, SKELETON_INPUTS
from .utils import zero_padded_shift


def _pool(sequence: Tensor, mask: Tensor) -> Tensor:
    valid = mask.unsqueeze(1).bool()
    masked = sequence.masked_fill(~valid, 0.0)
    count = valid.sum(dim=-1).clamp_min(1)
    mean = masked.sum(dim=-1) / count
    maximum = sequence.masked_fill(~valid, -torch.inf).amax(dim=-1)
    maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    return torch.cat((mean, maximum), dim=1)


def _modality_masks(mask: Tensor, shape: tuple[int, int]) -> tuple[Tensor, Tensor]:
    if mask.ndim == 2 and tuple(mask.shape) == shape:
        return mask, mask
    if mask.ndim == 3 and tuple(mask.shape[:2]) == shape and mask.shape[-1] == 2:
        return mask[..., 0], mask[..., 1]
    raise ValueError("mask must have shape [B,T] or [B,T,2]")

def _repair_internal_gaps(value: Tensor, mask: Tensor, *, radius: int = 3) -> Tensor:
    """Fill only short internal missing runs from nearby valid frames; preserve boundaries."""
    if value.ndim != 3 or mask.ndim != 2 or tuple(value.shape[:2]) != tuple(mask.shape):
        raise ValueError("value/mask must have shapes [B,T,F] and [B,T]")
    valid = mask.float().clamp(0, 1)
    left = torch.zeros_like(valid, dtype=torch.bool)
    right = torch.zeros_like(valid, dtype=torch.bool)
    for offset in range(1, radius + 1):
        left |= zero_padded_shift(valid, offset, time_dim=1).bool()
        right |= zero_padded_shift(valid, -offset, time_dim=1).bool()
    repair = (valid <= 0) & left & right
    if not bool(repair.any()):
        return value
    channels = value.transpose(1, 2)
    weights = valid.unsqueeze(1)
    kernel = 2 * radius + 1
    numerator = nn.functional.avg_pool1d(channels * weights, kernel, stride=1, padding=radius) * kernel
    denominator = nn.functional.avg_pool1d(weights, kernel, stride=1, padding=radius) * kernel
    filled = numerator / denominator.clamp_min(1.0)
    repaired = torch.where(repair.unsqueeze(1), filled, channels)
    return repaired.transpose(1, 2)


def _sanitize_masked_motion_features(
    skeleton: Tensor,
    imu: Tensor,
    skeleton_mask: Tensor,
    imu_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    """Remove derivative artifacts across invalid temporal transitions."""
    skeleton = skeleton.clone()
    imu = imu.clone()
    skeleton_valid = skeleton_mask.float().clamp(0, 1)
    imu_valid = imu_mask.float().clamp(0, 1)
    skeleton = skeleton * skeleton_valid.unsqueeze(-1)
    imu = imu * imu_valid.unsqueeze(-1)
    skeleton_transition = skeleton_valid.clone()
    imu_transition = imu_valid.clone()
    if skeleton.shape[1] > 1:
        skeleton_transition[:, 1:] = skeleton_valid[:, 1:] * skeleton_valid[:, :-1]
        imu_transition[:, 1:] = imu_valid[:, 1:] * imu_valid[:, :-1]
    skeleton[:, :, 60:120] = skeleton[:, :, 60:120] * skeleton_transition.unsqueeze(-1)
    imu[:, :, 6:12] = imu[:, :, 6:12] * imu_transition.unsqueeze(-1)
    return skeleton, imu


class ResidualDepthwiseTCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        padding = dilation * 2
        groups = max(1, min(16, channels))
        while channels % groups:
            groups -= 1
        self.depthwise = nn.Conv1d(channels, channels, 5, padding=padding, dilation=dilation, groups=channels)
        self.pointwise = nn.Conv1d(channels, channels, 1)
        self.norm = nn.GroupNorm(groups, channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, value: Tensor) -> Tensor:
        update = self.depthwise(value)
        update = self.pointwise(update)
        return value + self.dropout(self.activation(self.norm(update)))


class SafeSenseTCNBranch(nn.Module):
    def __init__(self, input_size: int, width: int, dropout: float = 0.15, dilations: Sequence[int] = (1, 2, 4)):
        super().__init__()
        self.input = nn.Conv1d(input_size, width, 1)
        self.blocks = nn.Sequential(*(ResidualDepthwiseTCNBlock(width, dilation, dropout) for dilation in dilations))

    def forward(self, value: Tensor) -> Tensor:
        return self.blocks(self.input(value))


class TopologyGraphBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.1):
        super().__init__()
        self.self_linear = nn.Linear(channels, channels)
        self.neighbor_linear = nn.Linear(channels, channels, bias=False)
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, nodes: Tensor, adjacency: Tensor) -> Tensor:
        neighbors = torch.einsum("ij,btjc->btic", adjacency, nodes)
        update = self.self_linear(nodes) + self.neighbor_linear(neighbors)
        return nodes + self.dropout(nn.functional.gelu(self.norm(update)))


class SkeletonTopologyBranch(nn.Module):
    """Explicit fixed-topology joint encoder feeding a compact temporal branch."""

    def __init__(self):
        super().__init__()
        adjacency = torch.eye(20, dtype=torch.float32)
        for child, parent in BONE_EDGES:
            adjacency[child, parent] = 1.0
            adjacency[parent, child] = 1.0
        degree = adjacency.sum(dim=1).clamp_min(1.0)
        inv_sqrt = degree.rsqrt()
        adjacency = inv_sqrt[:, None] * adjacency * inv_sqrt[None, :]
        self.register_buffer("adjacency", adjacency)
        self.input = nn.Sequential(nn.Linear(6, 96), nn.LayerNorm(96), nn.GELU())
        self.blocks = nn.ModuleList((TopologyGraphBlock(96), TopologyGraphBlock(96), TopologyGraphBlock(96)))
        self.joint_attention = nn.Linear(96, 1)
        self.temporal = SafeSenseTCNBranch(96, 96, dilations=(1, 2, 4))

    def forward(self, skeleton: Tensor, mask: Tensor) -> Tensor:
        positions = skeleton[:, :, :60].reshape(*skeleton.shape[:2], 20, 3)
        motion = skeleton[:, :, 60:120].reshape(*skeleton.shape[:2], 20, 3)
        nodes = self.input(torch.cat((positions, motion), dim=-1))
        for block in self.blocks:
            nodes = block(nodes, self.adjacency)
        scores = self.joint_attention(nodes).squeeze(-1)
        weights = torch.softmax(scores, dim=2)
        pooled = (nodes * weights.unsqueeze(-1)).sum(dim=2)
        pooled = pooled * mask.unsqueeze(-1)
        return self.temporal(pooled.transpose(1, 2))


class EfficientTemporalReducer(nn.Module):
    """Reduce 128-frame modality streams to 32 tokens with depthwise separable convolutions."""

    def __init__(self, channels: int):
        super().__init__()
        groups = max(1, min(16, channels))
        while channels % groups:
            groups -= 1
        self.steps = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(channels, channels, 5, stride=2, padding=2, groups=channels),
                nn.Conv1d(channels, channels, 1),
                nn.GroupNorm(groups, channels),
                nn.GELU(),
            )
            for _ in range(2)
        ])

    def forward(self, sequence: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        valid = mask.float()
        for step in self.steps:
            sequence = step(sequence)
            valid = nn.functional.max_pool1d(valid.unsqueeze(1), 3, stride=2, padding=1).squeeze(1)
            valid = (valid > 0).float()
            sequence = sequence * valid.unsqueeze(1)
        return sequence, valid


class FactorizedTimeModalityBlock(nn.Module):
    """Alternate efficient attention over time and the two sensor modalities."""

    def __init__(self, width: int = 96, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.temporal_norm = nn.LayerNorm(width)
        self.temporal_attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.modality_norm = nn.LayerNorm(width)
        self.modality_attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * 2, width),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _safe_valid(valid: Tensor) -> Tensor:
        safe = valid.clone()
        missing = ~safe.any(dim=1)
        if bool(missing.any()):
            safe[missing, 0] = True
        return safe

    def forward(self, tokens: Tensor, valid: Tensor) -> Tensor:
        batch, modalities, steps, width = tokens.shape
        temporal = tokens.reshape(batch * modalities, steps, width)
        temporal_valid = valid.reshape(batch * modalities, steps)
        normalized = self.temporal_norm(temporal)
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            temporal_update, _ = self.temporal_attention(
                normalized, normalized, normalized,
                key_padding_mask=~self._safe_valid(temporal_valid),
                need_weights=False,
            )
        temporal = temporal + self.dropout(temporal_update)
        temporal = temporal * temporal_valid.unsqueeze(-1)
        tokens = temporal.reshape(batch, modalities, steps, width)

        modality = tokens.permute(0, 2, 1, 3).reshape(batch * steps, modalities, width)
        modality_valid = valid.permute(0, 2, 1).reshape(batch * steps, modalities)
        normalized = self.modality_norm(modality)
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            modality_update, _ = self.modality_attention(
                normalized, normalized, normalized,
                key_padding_mask=~self._safe_valid(modality_valid),
                need_weights=False,
            )
        modality = modality + self.dropout(modality_update)
        modality = modality * modality_valid.unsqueeze(-1)
        tokens = modality.reshape(batch, steps, modalities, width).permute(0, 2, 1, 3)
        tokens = tokens + self.ffn(self.ffn_norm(tokens))
        return tokens * valid.unsqueeze(-1)


class SafeSenseFactorized(nn.Module):
    """Graph-aware dual-modality model with factorized time/modality attention."""

    variant = "factorized"
    subject_style_training = True

    def __init__(self, width: int = 96, blocks: int = 2):
        super().__init__()
        self.skeleton_flat = SafeSenseTCNBranch(SKELETON_INPUTS, 64)
        self.skeleton_graph = SkeletonTopologyBranch()
        self.skeleton_fuse = nn.Sequential(
            nn.Conv1d(160, width, 1),
            nn.GroupNorm(16, width),
            nn.GELU(),
        )
        self.imu_branch = SafeSenseTCNBranch(IMU_INPUTS, 64)
        self.imu_project = nn.Sequential(
            nn.Conv1d(64, width, 1),
            nn.GroupNorm(16, width),
            nn.GELU(),
        )
        self.skeleton_reduce = EfficientTemporalReducer(width)
        self.imu_reduce = EfficientTemporalReducer(width)
        self.factorized_blocks = nn.ModuleList(FactorizedTimeModalityBlock(width) for _ in range(blocks))
        self.classifier = nn.Sequential(nn.LayerNorm(width * 2), nn.Linear(width * 2, NUM_CLASSES))
        self.skeleton_expert = nn.Sequential(nn.LayerNorm(width * 2), nn.Linear(width * 2, NUM_CLASSES))
        self.imu_expert = nn.Sequential(nn.LayerNorm(width * 2), nn.Linear(width * 2, NUM_CLASSES))
        self._last_expert_logits: tuple[Tensor, Tensor] | None = None
        self._last_alignment_embeddings: tuple[Tensor, Tensor] | None = None

    @staticmethod
    def _masked_mean(tokens: Tensor, valid: Tensor) -> Tensor:
        weights = valid.float().unsqueeze(-1)
        return (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def forward(
        self,
        skeleton: Tensor,
        imu: Tensor,
        mask: Tensor | None = None,
        availability: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if skeleton.ndim != 3 or skeleton.shape[-1] != SKELETON_INPUTS:
            raise ValueError(f"skeleton must have shape [B,T,{SKELETON_INPUTS}]")
        if imu.ndim != 3 or imu.shape[-1] != IMU_INPUTS:
            raise ValueError(f"imu must have shape [B,T,{IMU_INPUTS}]")
        if mask is None:
            mask = torch.ones(skeleton.shape[:2], device=skeleton.device)
        skeleton_mask, imu_mask = _modality_masks(mask, tuple(skeleton.shape[:2]))
        if availability is None:
            availability = torch.stack((skeleton_mask.any(dim=1), imu_mask.any(dim=1)), dim=1).float()
        availability = availability.float().clamp(0, 1)
        skeleton, imu = _sanitize_masked_motion_features(skeleton, imu, skeleton_mask, imu_mask)
        skeleton = _repair_internal_gaps(skeleton, skeleton_mask)
        imu = _repair_internal_gaps(imu, imu_mask)

        flat_sequence = self.skeleton_flat(skeleton.transpose(1, 2))
        graph_sequence = self.skeleton_graph(skeleton, skeleton_mask)
        skeleton_sequence = self.skeleton_fuse(torch.cat((flat_sequence, graph_sequence), dim=1))
        imu_sequence = self.imu_project(self.imu_branch(imu.transpose(1, 2)))
        skeleton_sequence, skeleton_mask = self.skeleton_reduce(skeleton_sequence, skeleton_mask)
        imu_sequence, imu_mask = self.imu_reduce(imu_sequence, imu_mask)
        skeleton_mask = skeleton_mask * availability[:, 0:1]
        imu_mask = imu_mask * availability[:, 1:2]

        tokens = torch.stack((skeleton_sequence.transpose(1, 2), imu_sequence.transpose(1, 2)), dim=1)
        valid = torch.stack((skeleton_mask.bool(), imu_mask.bool()), dim=1)
        for block in self.factorized_blocks:
            tokens = block(tokens, valid)

        skeleton_tokens, imu_tokens = tokens[:, 0], tokens[:, 1]
        skeleton_embedding = _pool(skeleton_tokens.transpose(1, 2), skeleton_mask)
        imu_embedding = _pool(imu_tokens.transpose(1, 2), imu_mask)
        skeleton_logits = self.skeleton_expert(skeleton_embedding) * availability[:, 0:1]
        imu_logits = self.imu_expert(imu_embedding) * availability[:, 1:2]
        self._last_expert_logits = (skeleton_logits, imu_logits)
        self._last_alignment_embeddings = (
            self._masked_mean(skeleton_tokens, skeleton_mask),
            self._masked_mean(imu_tokens, imu_mask),
        )

        modality_count = valid.float().sum(dim=1)
        fused_tokens = (tokens * valid.unsqueeze(-1)).sum(dim=1) / modality_count.unsqueeze(-1).clamp_min(1.0)
        fused_mask = valid.any(dim=1).float()
        fused_embedding = _pool(fused_tokens.transpose(1, 2), fused_mask)
        logits = self.classifier(fused_embedding)
        weights = availability / availability.sum(dim=1, keepdim=True).clamp_min(1.0)
        weights = torch.where(availability.sum(dim=1, keepdim=True) > 0, weights, torch.full_like(weights, 0.5))
        return logits, weights

    @property
    def expert_logits(self) -> tuple[Tensor, Tensor] | None:
        return self._last_expert_logits

    @property
    def alignment_embeddings(self) -> tuple[Tensor, Tensor] | None:
        return self._last_alignment_embeddings
