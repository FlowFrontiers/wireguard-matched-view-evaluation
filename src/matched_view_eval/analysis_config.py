from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from matched_view_eval.errors import PipelineInvariantError
from matched_view_eval.training_config import TrainingConfig, load_training_config

METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "macro_average_precision",
)
INTERVAL_METRICS = METRICS[:4]


@dataclass(frozen=True)
class AnalysisConfig:
    training: TrainingConfig
    output_dir: Path
    bootstrap_replicates: int
    confidence_level: float
    bootstrap_seed: int
    precision_recall_grid_points: int
    primary_metrics: tuple[str, ...]
    interval_metrics: tuple[str, ...]
    key_per_class_models: tuple[str, ...]


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_analysis_config(
    config_path: Path,
    *,
    input_root: Path | None = None,
    artifact_dir: Path | None = None,
    run_root: Path | None = None,
    analysis_output_dir: Path | None = None,
) -> AnalysisConfig:
    training = load_training_config(
        config_path,
        input_root=input_root,
        artifact_dir=artifact_dir,
        output_root=run_root,
    )
    with config_path.expanduser().resolve().open(encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)
    analysis = raw.get("analysis", {})
    configured_output = analysis_output_dir or analysis.get("output_dir")
    if configured_output is None:
        raise PipelineInvariantError("analysis.output_dir is required")
    config = AnalysisConfig(
        training=training,
        output_dir=_resolve(training.dataset.project_root, configured_output),
        bootstrap_replicates=int(analysis.get("bootstrap_replicates", 1_000)),
        confidence_level=float(analysis.get("confidence_level", 0.95)),
        bootstrap_seed=int(analysis.get("bootstrap_seed", 42)),
        precision_recall_grid_points=int(analysis.get("precision_recall_grid_points", 500)),
        primary_metrics=tuple(str(value) for value in analysis.get("primary_metrics", [])),
        interval_metrics=tuple(str(value) for value in analysis.get("interval_metrics", [])),
        key_per_class_models=tuple(
            str(value) for value in analysis.get("key_per_class_models", [])
        ),
    )
    _validate_analysis_config(config)
    return config


def _validate_analysis_config(config: AnalysisConfig) -> None:
    if config.bootstrap_replicates != 1_000:
        raise PipelineInvariantError("The analysis requires exactly 1,000 bootstrap replicates")
    if config.confidence_level != 0.95 or config.bootstrap_seed != 42:
        raise PipelineInvariantError("The confidence level and bootstrap seed are frozen")
    if config.precision_recall_grid_points != 500:
        raise PipelineInvariantError("The precision-recall grid requires exactly 500 points")
    if config.primary_metrics != ("balanced_accuracy", "macro_f1"):
        raise PipelineInvariantError("Primary metrics differ from the frozen definition")
    if config.interval_metrics != INTERVAL_METRICS:
        raise PipelineInvariantError(
            "Only the four confusion-derived metrics receive confidence intervals"
        )
    expected_key_models = (
        "cnn1d_sequential_splt",
        "xgboost_matched_flow_stats",
        "xgboost_flattened_splt",
    )
    if config.key_per_class_models != expected_key_models:
        raise PipelineInvariantError(
            "Per-class figure model order differs from the frozen definition"
        )
