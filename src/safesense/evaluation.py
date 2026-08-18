"""Clean and corruption evaluation for a frozen SafeSense checkpoint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .corruptions import apply_corruption, base_temporal_masks, scenario_specs, training_noise_scales
from .data import feature_arrays, load_stats
from .model import SafeSenseFactorized
from .utils import classification_metrics


def load_model(checkpoint: str | Path, *, device: str = "cpu") -> tuple[SafeSenseFactorized, torch.device]:
    selected = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    model = SafeSenseFactorized().to(selected)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, selected


def _normalize(
    skeleton: np.ndarray,
    imu: np.ndarray,
    pair_mask: np.ndarray,
    stats: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    skeleton_input = ((skeleton - stats["skeleton_mean"]) / stats["skeleton_std"]).astype(np.float32)
    imu_input = ((imu - stats["imu_mean"]) / stats["imu_std"]).astype(np.float32)
    skeleton_input *= pair_mask[..., 0, None]
    imu_input *= pair_mask[..., 1, None]
    return skeleton_input, imu_input


def predict(
    model: SafeSenseFactorized,
    skeleton: np.ndarray,
    imu: np.ndarray,
    mask: np.ndarray,
    device: torch.device,
    *,
    batch_size: int = 64,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(skeleton), batch_size):
            stop = min(start + batch_size, len(skeleton))
            s = torch.from_numpy(skeleton[start:stop]).to(device)
            i = torch.from_numpy(imu[start:stop]).to(device)
            m = torch.from_numpy(mask[start:stop].astype(np.float32)).to(device)
            logits, _ = model(s, i, m)
            outputs.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(outputs)


def evaluate_clean(
    prepared_dir: str | Path,
    checkpoint: str | Path,
    stats_file: str | Path,
    *,
    device: str = "cpu",
) -> dict[str, object]:
    prepared = Path(prepared_dir)
    fused = np.load(prepared / "skeleton_val_features.npy")
    labels = np.load(prepared / "val_labels.npy").astype(np.int64)
    masks = base_temporal_masks(fused)
    pair_mask = np.stack((masks["skeleton"], masks["imu"]), axis=-1).astype(np.float32)
    skeleton, imu, _ = feature_arrays(fused)
    stats = load_stats(stats_file)
    skeleton, imu = _normalize(skeleton, imu, pair_mask, stats)
    model, selected = load_model(checkpoint, device=device)
    predictions = predict(model, skeleton, imu, pair_mask, selected)
    metrics = classification_metrics(predictions, labels)
    return {
        "samples": int(len(labels)),
        "correct": int(np.sum(predictions == labels)),
        **metrics,
    }


def evaluate_robustness(
    prepared_dir: str | Path,
    checkpoint: str | Path,
    stats_file: str | Path,
    *,
    device: str = "cpu",
    corruption_seed: int = 20260814,
) -> dict[str, object]:
    prepared = Path(prepared_dir)
    train = np.load(prepared / "skeleton_train_features.npy")
    test = np.load(prepared / "skeleton_val_features.npy")
    train_labels = np.load(prepared / "train_labels.npy").astype(np.int64)
    test_labels = np.load(prepared / "val_labels.npy").astype(np.int64)
    fused = np.concatenate((train, test), axis=0)
    masks = base_temporal_masks(fused)
    scales = training_noise_scales(fused, masks, np.arange(len(train), dtype=np.int64))
    stats = load_stats(stats_file)
    model, selected = load_model(checkpoint, device=device)
    test_slice = slice(len(train), len(fused))
    records: list[dict[str, object]] = []
    for spec in scenario_specs():
        corrupted, corrupted_masks = apply_corruption(fused, spec, seed=corruption_seed, scales=scales)
        skeleton, imu, _ = feature_arrays(corrupted)
        pair_mask = np.stack((corrupted_masks["skeleton"], corrupted_masks["imu"]), axis=-1).astype(np.float32)
        skeleton, imu = _normalize(skeleton, imu, pair_mask, stats)
        predictions = predict(model, skeleton[test_slice], imu[test_slice], pair_mask[test_slice], selected)
        metrics = classification_metrics(predictions, test_labels)
        records.append({
            "scenario": spec.name,
            "category": spec.category,
            "accuracy": float(metrics["accuracy"]),
            "macro_f1": float(metrics["macro_f1"]),
        })
    categories: dict[str, dict[str, float]] = {}
    for category in ("shift", "frame_loss", "noise", "missing", "composite"):
        rows = [r for r in records if r["category"] == category]
        categories[category] = {
            "accuracy": float(np.mean([float(r["accuracy"]) for r in rows])),
            "macro_f1": float(np.mean([float(r["macro_f1"]) for r in rows])),
        }
    equal_accuracy = float(np.mean([v["accuracy"] for v in categories.values()]))
    equal_f1 = float(np.mean([v["macro_f1"] for v in categories.values()]))
    return {
        "corruption_seed": corruption_seed,
        "test_samples": int(len(test_labels)),
        "noise_scales": scales,
        "scenario_count": len(records),
        "scenarios": records,
        "category_means": categories,
        "equal_category_corrupted_accuracy": equal_accuracy,
        "equal_category_corrupted_macro_f1": equal_f1,
    }


def write_json(path: str | Path, payload: dict[str, object]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
