from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from matched_view_eval.analysis_config import INTERVAL_METRICS
from matched_view_eval.errors import PipelineInvariantError


@dataclass(frozen=True)
class BootstrapResult:
    point: dict[str, float]
    lower: dict[str, float]
    upper: dict[str, float]
    mean: dict[str, float]
    standard_deviation: dict[str, float]
    distributions: np.ndarray


def confusion_metrics(confusion: np.ndarray) -> dict[str, float]:
    """Compute the four table metrics from a square confusion matrix."""
    matrix = np.asarray(confusion, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or (matrix < 0).any():
        raise PipelineInvariantError("Confusion matrix must be square and nonnegative")
    total = matrix.sum()
    row_support = matrix.sum(axis=1)
    column_support = matrix.sum(axis=0)
    if total <= 0 or (row_support == 0).any():
        raise PipelineInvariantError("Every class must be represented in the confusion matrix")
    true_positive = np.diag(matrix)
    recall = np.divide(
        true_positive,
        row_support,
        out=np.zeros_like(true_positive),
        where=row_support > 0,
    )
    f1_denominator = row_support + column_support
    per_class_f1 = np.divide(
        2.0 * true_positive,
        f1_denominator,
        out=np.zeros_like(true_positive),
        where=f1_denominator > 0,
    )
    return {
        "accuracy": float(true_positive.sum() / total),
        "balanced_accuracy": float(recall.mean()),
        "macro_f1": float(per_class_f1.mean()),
        "weighted_f1": float(np.dot(per_class_f1, row_support) / total),
    }


def confusion_from_labels(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    *,
    class_count: int,
) -> np.ndarray:
    true_values = np.asarray(true_labels, dtype=np.int64)
    predicted_values = np.asarray(predicted_labels, dtype=np.int64)
    if true_values.shape != predicted_values.shape or true_values.ndim != 1:
        raise PipelineInvariantError("True and predicted label vectors must align")
    if (
        (true_values < 0).any()
        or (predicted_values < 0).any()
        or (true_values >= class_count).any()
        or (predicted_values >= class_count).any()
    ):
        raise PipelineInvariantError("Labels are outside the configured class vocabulary")
    states = true_values * class_count + predicted_values
    return np.bincount(states, minlength=class_count * class_count).reshape(
        class_count, class_count
    )


def classification_metrics(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    class_count: int,
) -> tuple[dict[str, float], np.ndarray]:
    from sklearn.metrics import average_precision_score
    from sklearn.preprocessing import label_binarize

    confusion = confusion_from_labels(
        true_labels, predicted_labels, class_count=class_count
    )
    metrics = confusion_metrics(confusion)
    probability_values = np.asarray(probabilities, dtype=np.float64)
    if probability_values.shape != (len(true_labels), class_count):
        raise PipelineInvariantError("Probability matrix shape is invalid")
    if not np.isfinite(probability_values).all() or not np.allclose(
        probability_values.sum(axis=1), 1.0, rtol=1e-6, atol=1e-6
    ):
        raise PipelineInvariantError("Probability matrix is invalid")
    binarized = label_binarize(true_labels, classes=np.arange(class_count))
    metrics["macro_average_precision"] = float(
        average_precision_score(binarized, probability_values, average="macro")
    )
    return metrics, confusion


def paired_bootstrap_confusion_metrics(
    confusion: np.ndarray,
    *,
    replicates: int,
    confidence_level: float,
    seed: int,
) -> BootstrapResult:
    """Exact multinomial implementation of pair-level nonparametric bootstrap."""
    matrix = np.asarray(confusion, dtype=np.int64)
    point = confusion_metrics(matrix)
    total = int(matrix.sum())
    if replicates < 1 or not 0 < confidence_level < 1:
        raise PipelineInvariantError("Invalid bootstrap configuration")
    probabilities = matrix.reshape(-1).astype(np.float64) / total
    rng = np.random.default_rng(seed)
    sampled = rng.multinomial(total, probabilities, size=replicates).reshape(
        replicates, *matrix.shape
    )
    row_support = sampled.sum(axis=2)
    column_support = sampled.sum(axis=1)
    true_positive = np.diagonal(sampled, axis1=1, axis2=2)
    recall = np.divide(
        true_positive,
        row_support,
        out=np.zeros_like(true_positive, dtype=np.float64),
        where=row_support > 0,
    )
    f1_denominator = row_support + column_support
    per_class_f1 = np.divide(
        2.0 * true_positive,
        f1_denominator,
        out=np.zeros_like(true_positive, dtype=np.float64),
        where=f1_denominator > 0,
    )
    distributions = np.column_stack(
        (
            true_positive.sum(axis=1) / total,
            recall.mean(axis=1),
            per_class_f1.mean(axis=1),
            (per_class_f1 * row_support).sum(axis=1) / total,
        )
    )
    alpha = (1.0 - confidence_level) / 2.0
    lower_values = np.quantile(distributions, alpha, axis=0)
    upper_values = np.quantile(distributions, 1.0 - alpha, axis=0)
    mean_values = distributions.mean(axis=0)
    std_values = distributions.std(axis=0)
    return BootstrapResult(
        point=point,
        lower=dict(zip(INTERVAL_METRICS, lower_values, strict=True)),
        upper=dict(zip(INTERVAL_METRICS, upper_values, strict=True)),
        mean=dict(zip(INTERVAL_METRICS, mean_values, strict=True)),
        standard_deviation=dict(zip(INTERVAL_METRICS, std_values, strict=True)),
        distributions=distributions,
    )
