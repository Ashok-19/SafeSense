"""UTD-MHAD skeleton/IMU preparation and SafeSense feature construction."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from scipy.io import loadmat
from scipy.spatial.transform import Rotation
from torch import Tensor
from torch.utils.data import Dataset

from .constants import BONE_EDGES, FRAME_COUNT

_MAT_RE = re.compile(
    r"^a(?P<action>\d+)_s(?P<subject>\d+)_t(?P<trial>\d+)_(?P<modality>skeleton|inertial)\.mat$",
    re.IGNORECASE,
)
_MAX_SKELETON_FRAMES = 128

class DataPreparationError(ValueError):
    """Raised when UTD-MHAD preparation inputs are invalid."""


@dataclass(frozen=True)
class UTDSample:
    sample_id: str
    action: int
    subject: int
    trial: int
    skeleton: str
    inertial: str

    @property
    def key(self) -> tuple[int, int, int]:
        return self.action, self.subject, self.trial

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


def _sample_id(action: int, subject: int, trial: int) -> str:
    return f"a{action}_s{subject}_t{trial}"


def _modality_roots(data_dir: Path, modality: str) -> tuple[Path, ...]:
    names = (modality, modality.title(), modality.upper())
    roots = tuple(data_dir / name for name in names if (data_dir / name).is_dir())
    return roots or (data_dir,)


def _find_members(data_dir: Path, modality: str) -> dict[str, tuple[Path, int, int, int]]:
    found: dict[str, tuple[Path, int, int, int]] = {}
    for root in _modality_roots(data_dir, modality):
        for path in sorted(root.rglob("*.mat")):
            match = _MAT_RE.fullmatch(path.name)
            if match is None or match.group("modality").lower() != modality:
                continue
            action = int(match.group("action"))
            subject = int(match.group("subject"))
            trial = int(match.group("trial"))
            if not 1 <= action <= 27 or not 1 <= subject <= 8 or not 1 <= trial <= 4:
                raise DataPreparationError(f"out-of-range {modality} sample: {path.name}")
            sample_id = _sample_id(action, subject, trial)
            if sample_id in found:
                raise DataPreparationError(f"duplicate {modality} sample: {sample_id}")
            found[sample_id] = path, action, subject, trial
    if not found:
        raise DataPreparationError(f"no {modality} MAT files found under {data_dir}")
    return found


def pair_utd_samples(data_dir: str | Path) -> list[UTDSample]:
    """Pair only skeleton and inertial MAT files in numeric sample order."""
    root = Path(data_dir)
    if not root.is_dir():
        raise DataPreparationError(f"UTD-MHAD data directory does not exist: {root}")
    skeleton = _find_members(root, "skeleton")
    inertial = _find_members(root, "inertial")
    if set(skeleton) != set(inertial):
        missing_skeleton = sorted(set(inertial) - set(skeleton))
        missing_inertial = sorted(set(skeleton) - set(inertial))
        raise DataPreparationError(
            f"skeleton/IMU sample IDs differ (missing_skeleton={missing_skeleton}, "
            f"missing_inertial={missing_inertial})"
        )
    samples: list[UTDSample] = []
    for sample_id in sorted(skeleton, key=lambda value: tuple(int(part[1:]) for part in value.split("_"))):
        _, action, subject, trial = skeleton[sample_id]
        imu_path, imu_action, imu_subject, imu_trial = inertial[sample_id]
        if (action, subject, trial) != (imu_action, imu_subject, imu_trial):
            raise DataPreparationError(f"metadata mismatch for {sample_id}")
        samples.append(
            UTDSample(
                sample_id=sample_id,
                action=action,
                subject=subject,
                trial=trial,
                skeleton=str(skeleton[sample_id][0].resolve()),
                inertial=str(imu_path.resolve()),
            )
        )
    return samples


def _nearest_indices(source_length: int, target_length: int) -> np.ndarray:
    if source_length < 1 or target_length < 1:
        raise DataPreparationError("sequence lengths must be positive")
    if source_length == 1 or target_length == 1:
        return np.zeros(target_length, dtype=np.int64)
    return np.rint(np.arange(target_length) * (source_length - 1) / (target_length - 1)).astype(np.int64)


def _rotate_vector_pair(sequence: np.ndarray, source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Match upstream ``scipy.spatial.transform.Rotation`` ordering/casting."""
    source_norm = float(np.linalg.norm(source))
    if source_norm <= 1e-6:
        return sequence
    rotation_axis = np.cross(source, target)
    rotation_angle = float(np.arccos(np.dot(source / source_norm, target)))
    if np.abs(rotation_axis).sum() < 1e-6 or np.abs(rotation_angle) < 1e-6:
        return sequence
    rotation_axis /= np.linalg.norm(rotation_axis)
    rotation = Rotation.from_rotvec(rotation_axis * rotation_angle)
    result = sequence.copy()
    for frame in result:
        if np.sum(frame) != 0:
            frame[:] = rotation.apply(frame)
    return result


def _normalize_skeleton(sequence: np.ndarray) -> np.ndarray:
    """Match upstream UTD-MHAD skeleton normalization without importing it."""
    if sequence.shape != (_MAX_SKELETON_FRAMES, 20, 3):
        raise DataPreparationError(f"expected padded skeleton [128,20,3], got {sequence.shape}")
    sequence = np.asarray(sequence, dtype=np.float32).copy()
    valid = np.any(sequence != 0, axis=(1, 2))
    if not np.any(valid):
        return sequence
    first = int(np.flatnonzero(valid)[0])
    last = int(np.flatnonzero(valid)[-1])
    sequence[:first] = sequence[first]
    sequence[last + 1 :] = sequence[last]
    joint_mask = np.sum(sequence, axis=-1, keepdims=True) != 0
    sequence = (sequence - sequence[:, 2:3]) * joint_mask
    sequence = _rotate_vector_pair(sequence, sequence[0, 2] - sequence[0, 3], (0, 0, 1))
    sequence = _rotate_vector_pair(sequence, sequence[0, 8] - sequence[0, 4], (1, 0, 0))
    return sequence


def _official_pad_skeleton(raw: np.ndarray) -> np.ndarray:
    """Replicate the upstream loader/``pad_null_frames`` behavior."""
    if raw.shape[0] > 128:
        raise DataPreparationError(f"skeleton exceeds 128 frames: {raw.shape[0]}")
    sequence = np.zeros((128, 20, 3), dtype=np.float32)
    sequence[: len(raw)] = raw
    valid = np.sum(sequence, axis=(1, 2)) != 0
    if not np.any(valid):
        return sequence
    if not valid[0]:
        non_null = sequence[valid].copy()
        sequence.fill(0)
        sequence[: len(non_null)] = non_null
        valid = np.sum(sequence, axis=(1, 2)) != 0
    last = int(np.flatnonzero(valid)[-1]) + 1
    if last < len(sequence):
        repeats = int(np.ceil((len(sequence) - last) / last))
        sequence[last:] = np.concatenate([sequence[:last]] * repeats, axis=0)[: len(sequence) - last]
    return sequence


def _load_fused_sample(sample: UTDSample) -> np.ndarray:
    try:
        skeleton = np.asarray(loadmat(sample.skeleton)["d_skel"], dtype=np.float32).transpose(2, 0, 1)
        inertial = np.asarray(loadmat(sample.inertial)["d_iner"], dtype=np.float32)
    except (KeyError, OSError, ValueError) as error:
        raise DataPreparationError(f"cannot load {sample.sample_id}: {error}") from error
    if skeleton.ndim != 3 or skeleton.shape[1:] != (20, 3):
        raise DataPreparationError(f"{sample.sample_id}: invalid skeleton shape {skeleton.shape}")
    if inertial.ndim != 2 or inertial.shape[1] != 6:
        raise DataPreparationError(f"{sample.sample_id}: invalid inertial shape {inertial.shape}")
    if not np.isfinite(skeleton).all() or not np.isfinite(inertial).all():
        raise DataPreparationError(f"{sample.sample_id}: non-finite skeleton or inertial values")
    if skeleton.shape[0] > _MAX_SKELETON_FRAMES:
        raise DataPreparationError(f"{sample.sample_id}: skeleton exceeds 128 frames")
    target = skeleton.shape[0]
    skeleton_padded = _normalize_skeleton(_official_pad_skeleton(skeleton))
    imu_padded = np.zeros((128, 2, 3), dtype=np.float32)
    imu_padded[:target] = inertial[_nearest_indices(len(inertial), target)].reshape(target, 2, 3)
    fused = np.zeros((1, 128, 22, 3), dtype=np.float32)
    fused[0, :, :20] = skeleton_padded
    fused[0, :, 20:] = imu_padded
    return fused


def _split(subject: int) -> str:
    return "train" if subject in {1, 3, 5, 7} else "test"


def _write_csv(path: Path, samples: Sequence[UTDSample]) -> None:
    fields = ("sample_id", "action", "subject", "trial", "skeleton", "inertial", "split")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for sample in samples:
            row = sample.to_dict()
            row["split"] = _split(sample.subject)
            writer.writerow(row)
    temporary.replace(path)


class SafeSenseDataError(ValueError):
    """Raised for invalid SafeSense artifacts or configurations."""


@dataclass(frozen=True)
class SafeSenseSplits:
    development_train: np.ndarray
    development_valid: np.ndarray
    official_test: np.ndarray


def _load_manifest(prepared_dir: Path) -> list[dict[str, str]]:
    path = prepared_dir / "manifest.csv"
    if not path.is_file():
        raise SafeSenseDataError(f"missing prepared manifest: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return rows


def odd_fold_splits(prepared_dir: str | Path, validation_subject: int = 7) -> SafeSenseSplits:
    if validation_subject not in (1, 3, 5, 7):
        raise SafeSenseDataError("odd-subject validation fold must be one of 1, 3, 5, 7")
    rows = _load_manifest(Path(prepared_dir))
    subjects = np.asarray([int(row["subject"]) for row in rows], dtype=np.int64)
    official_train_rows = np.flatnonzero(np.isin(subjects, (1, 3, 5, 7)))
    official_test_rows = np.flatnonzero(np.isin(subjects, (2, 4, 6, 8)))
    # prepare-fusion writes official train arrays first and held-out arrays second;
    # translate manifest order into that array order without changing the manifest.
    train_positions = np.full(len(rows), -1, dtype=np.int64)
    test_positions = np.full(len(rows), -1, dtype=np.int64)
    train_positions[official_train_rows] = np.arange(len(official_train_rows))
    test_positions[official_test_rows] = len(official_train_rows) + np.arange(len(official_test_rows))
    array_positions = np.where(train_positions >= 0, train_positions, test_positions)
    train_subjects = tuple(subject for subject in (1, 3, 5, 7) if subject != validation_subject)
    splits = SafeSenseSplits(
        development_train=array_positions[np.isin(subjects, train_subjects)],
        development_valid=array_positions[subjects == validation_subject],
        official_test=array_positions[np.isin(subjects, (2, 4, 6, 8))],
    )
    if len(rows) == 861:
        if len(splits.development_train) + len(splits.development_valid) != 431 or len(splits.official_test) != 430:
            raise SafeSenseDataError("manifest does not match the frozen UTD-MHAD subject protocol")
    if np.intersect1d(splits.development_train, splits.development_valid).size:
        raise SafeSenseDataError("odd-subject development train and validation folds overlap")
    if np.intersect1d(np.concatenate((splits.development_train, splits.development_valid)), splits.official_test).size:
        raise SafeSenseDataError("development and official test splits overlap")
    return splits


def _feature_arrays(array: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if array.ndim != 5 or array.shape[1:] != (1, FRAME_COUNT, 22, 3):
        raise SafeSenseDataError(f"expected prepared array [N,1,128,22,3], got {array.shape}")
    sequence = np.asarray(array[:, 0], dtype=np.float32)
    skeleton = sequence[:, :, :20]
    imu = sequence[:, :, 20:]
    skeleton_positions = skeleton.reshape(len(array), FRAME_COUNT, 60)
    skeleton_difference = np.zeros_like(skeleton_positions)
    skeleton_difference[:, 1:] = skeleton_positions[:, 1:] - skeleton_positions[:, :-1]
    bones = np.stack([skeleton[:, :, child] - skeleton[:, :, parent] for child, parent in BONE_EDGES], axis=2)
    skeleton_features = np.concatenate((skeleton_positions, skeleton_difference, bones.reshape(len(array), FRAME_COUNT, 57)), axis=-1)
    imu_features = imu.reshape(len(array), FRAME_COUNT, 6)
    imu_difference = np.zeros_like(imu_features)
    imu_difference[:, 1:] = imu_features[:, 1:] - imu_features[:, :-1]
    imu_features = np.concatenate((imu_features, imu_difference), axis=-1)
    temporal_mask = (np.abs(imu_features[:, :, :6]).sum(axis=-1) > 0).astype(np.float32)
    return skeleton_features, imu_features, temporal_mask


def _standard_stats(features: np.ndarray, indices: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = features[indices]
    valid = mask[indices].astype(bool)
    flattened = values[valid]
    if flattened.size == 0:
        raise SafeSenseDataError("training split has no valid IMU frames")
    mean = flattened.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = flattened.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def _raw_noise_scale(features: np.ndarray, indices: np.ndarray, mask: np.ndarray, width: int) -> np.ndarray:
    positions = np.asarray(indices, dtype=np.int64)
    values = features[positions, :, :width]
    valid = mask[positions].astype(bool)
    flattened = values[valid]
    if flattened.size == 0:
        raise SafeSenseDataError("training split has no valid raw values for corruption scaling")
    scale = float(flattened.std(dtype=np.float64)) or 1.0
    return np.asarray(scale, dtype=np.float32)


class SafeSenseDataset(Dataset[tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]):
    def __init__(self, skeleton: np.ndarray, imu: np.ndarray, mask: np.ndarray, labels: np.ndarray, indices: Sequence[int], stats: dict[str, np.ndarray]):
        self.skeleton = torch.from_numpy(((skeleton - stats["skeleton_mean"]) / stats["skeleton_std"]).astype(np.float32, copy=False))
        self.imu = torch.from_numpy(((imu - stats["imu_mean"]) / stats["imu_std"]).astype(np.float32, copy=False))
        self.mask = torch.from_numpy(mask.astype(np.float32, copy=False))
        if self.mask.ndim == 2:
            skeleton_mask = imu_mask = self.mask
        elif self.mask.ndim == 3 and self.mask.shape[-1] == 2:
            skeleton_mask, imu_mask = self.mask[..., 0], self.mask[..., 1]
        else:
            raise SafeSenseDataError("mask must have shape [N,T] or [N,T,2]")
        self.skeleton = self.skeleton * skeleton_mask.unsqueeze(-1)
        self.imu = self.imu * imu_mask.unsqueeze(-1)
        self.availability = torch.stack((skeleton_mask.any(dim=1), imu_mask.any(dim=1)), dim=1).float()
        self.labels = torch.from_numpy(labels.astype(np.int64, copy=False))
        self.indices = np.asarray(indices, dtype=np.int64)
        self.augmentation_stats = {
            name: torch.as_tensor(stats[name], dtype=torch.float32)
            for name in (
                "skeleton_mean", "skeleton_std", "imu_mean", "imu_std",
                "skeleton_noise_scale", "imu_noise_scale",
            )
        }

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, position: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        index = int(self.indices[position])
        return self.skeleton[index], self.imu[index], self.mask[index], self.availability[index], self.labels[index]


def _load_partition(prepared_dir: Path, partition: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load one prepared partition; development never opens the held-out arrays."""
    if partition not in ("train", "val"):
        raise ValueError(partition)
    features = np.load(prepared_dir / f"skeleton_{partition}_features.npy")
    labels = np.load(prepared_dir / f"{partition}_labels.npy").astype(np.int64, copy=False)
    skeleton, imu, mask = _feature_arrays(features)
    return skeleton, imu, mask, labels

def prepare_utd_mhad(data_dir: str | Path, output_dir: str | Path, *, expected_count: int | None = 861) -> Path:
    """Prepare synchronized UTD-MHAD skeleton+IMU arrays used by SafeSense.

    Expected raw layout: extracted ``Skeleton`` and ``Inertial`` folders containing
    the original ``a*_s*_t*_{skeleton,inertial}.mat`` files. RGB and depth are not read.
    """
    samples = pair_utd_samples(data_dir)
    if expected_count is not None and len(samples) != expected_count:
        raise DataPreparationError(f"expected {expected_count} paired samples, found {len(samples)}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train = [sample for sample in samples if _split(sample.subject) == "train"]
    test = [sample for sample in samples if _split(sample.subject) == "test"]
    if not train or not test:
        raise DataPreparationError("preparation requires both odd-subject train and even-subject test samples")
    for split_name, split_samples in (("train", train), ("val", test)):
        arrays = np.stack([_load_fused_sample(sample) for sample in split_samples])
        labels = np.asarray([sample.action - 1 for sample in split_samples], dtype=np.int64)
        np.save(output / f"skeleton_{split_name}_features.npy", arrays)
        np.save(output / f"{split_name}_labels.npy", labels)
    _write_csv(output / "manifest.csv", samples)
    metadata = {
        "dataset": "UTD-MHAD",
        "modalities": ["skeleton", "inertial"],
        "sample_count": len(samples),
        "train_subjects": [1, 3, 5, 7],
        "test_subjects": [2, 4, 6, 8],
        "train_count": len(train),
        "test_count": len(test),
        "prepared_shape": [1, 128, 22, 3],
        "labels_zero_based": True,
    }
    metadata_path = output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata_path


def feature_arrays(array: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _feature_arrays(array)


def load_partition(prepared_dir: str | Path, partition: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return _load_partition(Path(prepared_dir), partition)


def fit_stats(skeleton: np.ndarray, imu: np.ndarray, mask: np.ndarray, indices: Sequence[int]) -> dict[str, np.ndarray]:
    idx = np.asarray(indices, dtype=np.int64)
    skeleton_mean, skeleton_std = _standard_stats(skeleton, idx, mask)
    imu_mean, imu_std = _standard_stats(imu, idx, mask)
    return {
        "skeleton_mean": skeleton_mean,
        "skeleton_std": skeleton_std,
        "imu_mean": imu_mean,
        "imu_std": imu_std,
        "skeleton_noise_scale": _raw_noise_scale(skeleton, idx, mask, 60),
        "imu_noise_scale": _raw_noise_scale(imu, idx, mask, 6),
    }


def load_stats(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {name: np.array(archive[name]) for name in archive.files}
