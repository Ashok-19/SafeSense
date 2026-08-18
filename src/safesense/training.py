"""Training loop for the frozen SafeSenseFactorized paper configuration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .augment import _augment_structured_temporal_batch, _augment_subject_style_batch, _augment_temporal_batch
from .constants import (
    FACTORIZED_ALIGNMENT_WEIGHT,
    FACTORIZED_CLEAN_WEIGHT,
    FACTORIZED_CONSISTENCY_WEIGHT,
    FACTORIZED_EXPERT_AUXILIARY_WEIGHT,
    FACTORIZED_STRUCTURED_WEIGHT,
    FACTORIZED_STYLE_WEIGHT,
    FACTORIZED_TEMPORAL_WEIGHT,
    TEMPORAL_CONSISTENCY_TEMPERATURE,
)
from .data import SafeSenseDataset, fit_stats, load_partition, odd_fold_splits
from .model import SafeSenseFactorized
from .utils import classification_metrics, parameter_count, set_seed


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 22
    epochs: int = 60
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    swa_start_epoch: int = 45
    gradient_clip: float = 1.0

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if not 1 <= self.swa_start_epoch <= self.epochs:
            raise ValueError("swa_start_epoch must be inside the training run")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer settings")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

def _loader(dataset: SafeSenseDataset, config: TrainingConfig, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle, generator=generator, num_workers=0)


def _expert_auxiliary_loss(experts: tuple[Tensor, Tensor], labels: Tensor) -> Tensor:
    return 0.5 * sum(nn.functional.cross_entropy(expert, labels) for expert in experts)


def _temporal_consistency_loss(clean_logits: Tensor, corrupted_logits: Tensor) -> Tensor:
    temperature = TEMPORAL_CONSISTENCY_TEMPERATURE
    target = torch.softmax(clean_logits.detach() / temperature, dim=1)
    prediction = torch.log_softmax(corrupted_logits / temperature, dim=1)
    return nn.functional.kl_div(prediction, target, reduction="batchmean") * (temperature * temperature)


def _factorized_alignment_loss(model: SafeSenseFactorized, availability: Tensor) -> Tensor:
    embeddings = model.alignment_embeddings
    if embeddings is None:
        raise RuntimeError("factorized alignment embeddings were not produced")
    both = (availability[:, 0] > 0) & (availability[:, 1] > 0)
    if not bool(both.any()):
        return embeddings[0].sum() * 0.0
    skeleton = nn.functional.normalize(embeddings[0][both], dim=1)
    imu = nn.functional.normalize(embeddings[1][both], dim=1)
    return (1.0 - (skeleton * imu).sum(dim=1)).mean()


def _epoch_factorized(
    model: SafeSenseFactorized,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    loss_total = 0.0
    sample_total = 0
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    augmentation_stats = None
    if training:
        dataset = loader.dataset
        if not isinstance(dataset, SafeSenseDataset):
            raise RuntimeError("factorized training requires SafeSenseDataset augmentation statistics")
        augmentation_stats = {name: value.to(device) for name, value in dataset.augmentation_stats.items()}
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for skeleton, imu, mask, availability, labels in loader:
            skeleton, imu, mask, availability, labels = (
                value.to(device) for value in (skeleton, imu, mask, availability, labels)
            )
            clean_logits, _ = model(skeleton, imu, mask, availability)
            clean_experts = model.expert_logits
            if clean_experts is None:
                raise RuntimeError("factorized clean expert logits were not produced")
            if training:
                if augmentation_stats is None:
                    raise RuntimeError("factorized augmentation statistics are unavailable")
                temporal = _augment_temporal_batch(skeleton, imu, mask)
                temporal_logits, _ = model(temporal[0], temporal[1], temporal[2], temporal[3])
                temporal_experts = model.expert_logits
                structured = _augment_structured_temporal_batch(skeleton, imu, mask, augmentation_stats)
                structured_logits, _ = model(structured[0], structured[1], structured[2], structured[3])
                structured_experts = model.expert_logits
                styled = _augment_subject_style_batch(skeleton, imu, mask, augmentation_stats)
                styled_logits, _ = model(styled[0], styled[1], styled[2], styled[3])
                styled_experts = model.expert_logits
                if temporal_experts is None or structured_experts is None or styled_experts is None:
                    raise RuntimeError("factorized augmented expert logits were not produced")
                style_alignment = _factorized_alignment_loss(model, styled[3])
                # Recompute the clean view last so the stored alignment embeddings match it.
                clean_logits, _ = model(skeleton, imu, mask, availability)
                clean_experts = model.expert_logits
                if clean_experts is None:
                    raise RuntimeError("factorized clean expert logits were not restored")
                clean_alignment = _factorized_alignment_loss(model, availability)

                loss = (
                    FACTORIZED_CLEAN_WEIGHT * nn.functional.cross_entropy(clean_logits, labels)
                    + FACTORIZED_TEMPORAL_WEIGHT * nn.functional.cross_entropy(temporal_logits, labels)
                    + FACTORIZED_STRUCTURED_WEIGHT * nn.functional.cross_entropy(structured_logits, labels)
                    + FACTORIZED_STYLE_WEIGHT * nn.functional.cross_entropy(styled_logits, labels)
                )
                loss = loss + FACTORIZED_EXPERT_AUXILIARY_WEIGHT * (
                    FACTORIZED_CLEAN_WEIGHT * _expert_auxiliary_loss(clean_experts, labels)
                    + FACTORIZED_TEMPORAL_WEIGHT * _expert_auxiliary_loss(temporal_experts, labels)
                    + FACTORIZED_STRUCTURED_WEIGHT * _expert_auxiliary_loss(structured_experts, labels)
                    + FACTORIZED_STYLE_WEIGHT * _expert_auxiliary_loss(styled_experts, labels)
                )
                loss = loss + FACTORIZED_ALIGNMENT_WEIGHT * (0.75 * clean_alignment + 0.25 * style_alignment)
                loss = loss + FACTORIZED_CONSISTENCY_WEIGHT * (
                    _temporal_consistency_loss(clean_logits, temporal_logits)
                    + _temporal_consistency_loss(clean_logits, structured_logits)
                    + _temporal_consistency_loss(clean_logits, styled_logits)
                ) / 3.0
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            else:
                loss = nn.functional.cross_entropy(clean_logits, labels)
            batch_size = int(labels.shape[0])
            loss_total += float(loss.detach().cpu()) * batch_size
            sample_total += batch_size
            predictions.append(clean_logits.detach().argmax(1).cpu().numpy())
            targets.append(labels.cpu().numpy())
    return loss_total / max(sample_total, 1), np.concatenate(predictions), np.concatenate(targets)


def _update_swa_state(model: nn.Module, swa_state: dict[str, Tensor] | None, count: int) -> tuple[dict[str, Tensor], int]:
    state = model.state_dict()
    if swa_state is None:
        return {name: value.detach().clone() for name, value in state.items()}, 1
    next_count = count + 1
    with torch.no_grad():
        for name, value in state.items():
            detached = value.detach()
            if torch.is_floating_point(detached):
                swa_state[name].add_((detached - swa_state[name]) / float(next_count))
            else:
                swa_state[name].copy_(detached)
    return swa_state, next_count

def _train_with_swa(
    train: SafeSenseDataset,
    config: TrainingConfig,
    device: torch.device,
) -> tuple[SafeSenseFactorized, list[dict[str, float]], int]:
    set_seed(config.seed)
    model = SafeSenseFactorized().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    swa_state: dict[str, Tensor] | None = None
    swa_count = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, config.epochs + 1):
        train_loss, _, _ = _epoch_factorized(
            model,
            _loader(train, config, True, config.seed + epoch),
            device,
            optimizer,
        )
        if epoch >= config.swa_start_epoch:
            swa_state, swa_count = _update_swa_state(model, swa_state, swa_count)
        history.append({"epoch": float(epoch), "train_loss": float(train_loss)})
        if epoch == 1 or epoch == config.epochs or epoch % 5 == 0:
            print(f"epoch={epoch:03d}/{config.epochs} train_loss={train_loss:.6f} swa_states={swa_count}", flush=True)
    if swa_state is None:
        raise RuntimeError("SWA collected no states")
    model.load_state_dict(swa_state)
    return model, history, swa_count


def train_final(
    prepared_dir: str | Path,
    output_dir: str | Path,
    *,
    config: TrainingConfig | None = None,
    device: str = "cuda",
) -> dict[str, object]:
    """Train seed-22 SafeSense on odd subjects and evaluate the even-subject split."""
    config = config or TrainingConfig()
    if config.seed != 22:
        raise ValueError("the paper-selected final configuration uses seed 22")
    prepared = Path(prepared_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected_device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")

    skeleton, imu, mask, labels = load_partition(prepared, "train")
    train_indices = np.arange(len(labels), dtype=np.int64)
    stats = fit_stats(skeleton, imu, mask, train_indices)
    train_set = SafeSenseDataset(skeleton, imu, mask, labels, train_indices, stats)
    model, history, swa_count = _train_with_swa(train_set, config, selected_device)

    test_skeleton, test_imu, test_mask, test_labels = load_partition(prepared, "val")
    test_indices = np.arange(len(test_labels), dtype=np.int64)
    test_set = SafeSenseDataset(test_skeleton, test_imu, test_mask, test_labels, test_indices, stats)
    test_loss, predictions, targets = _epoch_factorized(
        model, _loader(test_set, config, False, config.seed), selected_device, None
    )
    metrics = classification_metrics(predictions, targets)

    checkpoint = output / "checkpoint.pt"
    stats_path = output / "stats.npz"
    torch.save({name: value.detach().cpu() for name, value in model.state_dict().items()}, checkpoint)
    np.savez(stats_path, **stats)
    np.save(output / "predictions.npy", predictions)
    np.save(output / "labels.npy", targets)
    (output / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    result = {
        "model": "SafeSenseFactorized",
        "status": "completed",
        "config": config.to_dict(),
        "training_subjects": [1, 3, 5, 7],
        "test_subjects": [2, 4, 6, 8],
        "parameter_count": parameter_count(model),
        "swa_state_count": swa_count,
        "test_loss": float(test_loss),
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "checkpoint": str(checkpoint),
        "stats": str(stats_path),
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def train_odd_fold(
    prepared_dir: str | Path,
    output_dir: str | Path,
    *,
    validation_subject: int,
    config: TrainingConfig | None = None,
    device: str = "cuda",
) -> dict[str, object]:
    """Train one leave-one-odd-subject-out development fold without loading even arrays."""
    config = config or TrainingConfig()
    prepared = Path(prepared_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected_device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    splits = odd_fold_splits(prepared, validation_subject)
    skeleton, imu, mask, labels = load_partition(prepared, "train")
    stats = fit_stats(skeleton, imu, mask, splits.development_train)
    train_set = SafeSenseDataset(skeleton, imu, mask, labels, splits.development_train, stats)
    valid_set = SafeSenseDataset(skeleton, imu, mask, labels, splits.development_valid, stats)
    model, history, swa_count = _train_with_swa(train_set, config, selected_device)
    valid_loss, predictions, targets = _epoch_factorized(
        model, _loader(valid_set, config, False, config.seed), selected_device, None
    )
    metrics = classification_metrics(predictions, targets)
    torch.save({name: value.detach().cpu() for name, value in model.state_dict().items()}, output / "checkpoint.pt")
    np.savez(output / "stats.npz", **stats)
    result = {
        "model": "SafeSenseFactorized",
        "validation_subject": validation_subject,
        "training_subjects": [s for s in (1, 3, 5, 7) if s != validation_subject],
        "config": config.to_dict(),
        "parameter_count": parameter_count(model),
        "swa_state_count": swa_count,
        "validation_loss": float(valid_loss),
        "validation_accuracy": float(metrics["accuracy"]),
        "validation_macro_f1": float(metrics["macro_f1"]),
    }
    (output / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
