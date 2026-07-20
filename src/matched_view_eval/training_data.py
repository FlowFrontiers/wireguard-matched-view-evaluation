from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from matched_view_eval.errors import PipelineInvariantError
from matched_view_eval.features import (
    build_flattened_splt,
    build_matched_flow_stats,
    build_sequential_splt,
)
from matched_view_eval.training_config import TrainingConfig


@dataclass(frozen=True)
class ClassicalData:
    train_values: np.ndarray
    test_values: np.ndarray
    labels: np.ndarray
    pair_ids: np.ndarray
    feature_names: tuple[str, ...]
    medians: np.ndarray | None


@dataclass(frozen=True)
class CNNData:
    train_values: np.ndarray
    train_labels: np.ndarray
    validation_values: np.ndarray
    validation_labels: np.ndarray
    test_values: np.ndarray
    test_labels: np.ndarray
    pair_ids: np.ndarray
    training_pair_ids: np.ndarray
    validation_pair_ids: np.ndarray


def pair_id_sha256(pair_ids: np.ndarray) -> str:
    """Hash an ordered pair-id sequence without delimiter ambiguity."""
    digest = hashlib.sha256()
    for value in pair_ids.astype(str):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big"))
        digest.update(encoded)
    return digest.hexdigest()


def encode_labels(frame: pd.DataFrame, classes: tuple[str, ...]) -> np.ndarray:
    if not classes:
        raise PipelineInvariantError("The class vocabulary cannot be empty")
    mapping = {label: index for index, label in enumerate(classes)}
    labels = frame["application_category"].astype(str)
    unknown = sorted(set(labels) - set(mapping))
    missing = sorted(set(mapping) - set(labels))
    if unknown or missing:
        raise PipelineInvariantError(
            f"Class vocabulary mismatch: unknown={unknown}, missing={missing}"
        )
    return labels.map(mapping).to_numpy(dtype=np.int64)


def balanced_class_weights(labels: np.ndarray, class_count: int) -> np.ndarray:
    counts = np.bincount(labels, minlength=class_count)
    if len(counts) != class_count or (counts == 0).any():
        raise PipelineInvariantError("Balanced weights require every configured class")
    return len(labels) / (class_count * counts.astype(np.float64))


def fit_stat_medians(source_values: np.ndarray) -> np.ndarray:
    """Fit NaN-aware medians on the source view only."""
    if source_values.ndim != 2:
        raise PipelineInvariantError("Statistic matrices must be two-dimensional")
    medians = np.nanmedian(source_values, axis=0)
    if not np.isfinite(medians).all():
        raise PipelineInvariantError("At least one statistic has no finite source median")
    return medians


def apply_stat_medians(values: np.ndarray, medians: np.ndarray) -> np.ndarray:
    if values.ndim != 2 or medians.shape != (values.shape[1],):
        raise PipelineInvariantError("Statistic matrix and median shapes disagree")
    transformed = np.where(np.isnan(values), medians[None, :], values).astype(
        np.float64, copy=False
    )
    if not np.isfinite(transformed).all():
        raise PipelineInvariantError("Imputed statistics contain non-finite values")
    return transformed


def prepare_classical_data(
    frame: pd.DataFrame,
    config: TrainingConfig,
    *,
    representation: str,
) -> ClassicalData:
    labels = encode_labels(frame, config.dataset.expected_classes)
    pair_ids = frame["pair_id"].astype(str).to_numpy()
    if len(frame) != config.expected_pair_count or len(np.unique(pair_ids)) != len(pair_ids):
        raise PipelineInvariantError("Canonical pair count or uniqueness is invalid")

    medians: np.ndarray | None = None
    if representation == "matched_flow_stats":
        source = build_matched_flow_stats(frame, domain=config.training_view)
        target = build_matched_flow_stats(frame, domain=config.test_view)
        medians = fit_stat_medians(source.values)
        train_values = apply_stat_medians(source.values, medians)
        test_values = apply_stat_medians(target.values, medians)
    elif representation == "flattened_splt":
        source = build_flattened_splt(
            frame,
            domain=config.training_view,
            prefix_length=config.dataset.primary_prefix_length,
        )
        target = build_flattened_splt(
            frame,
            domain=config.test_view,
            prefix_length=config.dataset.primary_prefix_length,
        )
        train_values = source.values
        test_values = target.values
    else:
        raise PipelineInvariantError(f"Unsupported classical representation: {representation}")

    if source.feature_names != target.feature_names:
        raise PipelineInvariantError("Source and target feature schemas disagree")
    if train_values.shape != test_values.shape or train_values.shape[0] != len(frame):
        raise PipelineInvariantError("Matched-view feature matrix shapes disagree")
    return ClassicalData(
        train_values=train_values,
        test_values=test_values,
        labels=labels,
        pair_ids=pair_ids,
        feature_names=source.feature_names,
        medians=medians,
    )


def stratified_validation_indices(
    labels: np.ndarray,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.model_selection import train_test_split

    all_indices = np.arange(len(labels), dtype=np.int64)
    training, validation = train_test_split(
        all_indices,
        test_size=validation_fraction,
        stratify=labels,
        random_state=seed,
    )
    training = np.sort(training)
    validation = np.sort(validation)
    if np.intersect1d(training, validation).size or len(training) + len(validation) != len(labels):
        raise PipelineInvariantError("CNN training and validation partitions overlap or lose rows")
    if set(np.unique(labels[training])) != set(np.unique(labels[validation])):
        raise PipelineInvariantError("CNN validation split does not retain every class")
    return training, validation


def augment_sequential_splt(
    values: np.ndarray,
    mask: np.ndarray,
    *,
    mask_probability: float,
    noise_stddev: float,
    seed: int,
) -> np.ndarray:
    """Mask complete valid timesteps and perturb magnitude channels only."""
    if not 0 <= mask_probability <= 1 or noise_stddev < 0:
        raise PipelineInvariantError("Invalid augmentation probability or noise scale")
    if values.ndim != 3 or values.shape[-1] != 3 or mask.shape != values.shape[:2]:
        raise PipelineInvariantError("SPLT values and mask have incompatible shapes")
    if not np.all(values[~mask] == 0):
        raise PipelineInvariantError("Input SPLT padding must be zero")
    rng = np.random.default_rng(seed)
    augmented = values.copy()
    dropped = (rng.random(mask.shape) < mask_probability) & mask
    augmented[dropped] = 0.0
    retained = mask & ~dropped
    noise = rng.normal(0.0, noise_stddev, size=values[:, :, 1:].shape).astype(np.float32)
    augmented[:, :, 1:] += noise * retained[:, :, None]
    augmented[~mask] = 0.0
    if not np.array_equal(augmented[:, :, 0][retained], values[:, :, 0][retained]):
        raise PipelineInvariantError("Augmentation changed direction values")
    if not np.all(augmented[~mask] == 0) or not np.isfinite(augmented).all():
        raise PipelineInvariantError("Augmentation corrupted padding or produced non-finite values")
    return augmented


def prepare_cnn_data(frame: pd.DataFrame, config: TrainingConfig) -> CNNData:
    labels = encode_labels(frame, config.dataset.expected_classes)
    pair_ids = frame["pair_id"].astype(str).to_numpy()
    source = build_sequential_splt(
        frame,
        domain=config.training_view,
        prefix_length=config.dataset.primary_prefix_length,
    )
    target = build_sequential_splt(
        frame,
        domain=config.test_view,
        prefix_length=config.dataset.primary_prefix_length,
    )
    if source.channels != ("direction", "size", "iat_ms") or source.channels != target.channels:
        raise PipelineInvariantError("CNN1D requires the frozen three-channel SPLT tensor")
    training_indices, validation_indices = stratified_validation_indices(
        labels,
        validation_fraction=config.cnn1d.validation_fraction,
        seed=config.seed,
    )
    source_training = source.values[training_indices]
    augmented = augment_sequential_splt(
        source_training,
        source.mask[training_indices],
        mask_probability=config.cnn1d.augmentation_mask_probability,
        noise_stddev=config.cnn1d.augmentation_noise_stddev,
        seed=config.cnn1d.augmentation_seed,
    )
    train_values = np.concatenate((source_training, augmented), axis=0)
    train_labels = np.concatenate((labels[training_indices], labels[training_indices]), axis=0)
    return CNNData(
        train_values=train_values,
        train_labels=train_labels,
        validation_values=source.values[validation_indices],
        validation_labels=labels[validation_indices],
        test_values=target.values,
        test_labels=labels,
        pair_ids=pair_ids,
        training_pair_ids=pair_ids[training_indices],
        validation_pair_ids=pair_ids[validation_indices],
    )
