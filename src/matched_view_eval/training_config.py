from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from matched_view_eval.config import DatasetConfig, load_config
from matched_view_eval.errors import PipelineInvariantError

MODEL_IDS = (
    "rf_matched_flow_stats",
    "xgboost_matched_flow_stats",
    "rf_flattened_splt",
    "xgboost_flattened_splt",
    "cnn1d_sequential_splt",
)


@dataclass(frozen=True)
class RandomForestConfig:
    n_estimators: int
    max_depth: int | None
    class_weight: str


@dataclass(frozen=True)
class XGBoostConfig:
    n_estimators: int
    max_depth: int
    subsample: float
    colsample_bytree: float
    matched_flow_stats_learning_rate: float
    flattened_splt_learning_rate: float


@dataclass(frozen=True)
class CNN1DConfig:
    batch_size: int
    maximum_epochs: int
    validation_fraction: float
    learning_rate: float
    dropout: float
    focal_gamma: float
    focal_alpha: float
    early_stopping_patience: int
    scheduler_patience: int
    augmentation_mask_probability: float
    augmentation_noise_stddev: float
    augmentation_seed: int
    expected_trainable_parameters: int
    expected_total_parameters: int


@dataclass(frozen=True)
class TrainingConfig:
    dataset: DatasetConfig
    output_root: Path
    seed: int
    training_view: str
    test_view: str
    expected_pair_count: int
    model_ids: tuple[str, ...]
    random_forest: RandomForestConfig
    xgboost: XGBoostConfig
    cnn1d: CNN1DConfig


def _required(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise PipelineInvariantError(f"Missing evaluation configuration: {key}")
    return mapping[key]


def _resolve(project_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def load_training_config(
    config_path: Path,
    *,
    input_root: Path | None = None,
    artifact_dir: Path | None = None,
    output_root: Path | None = None,
) -> TrainingConfig:
    """Load and validate the frozen five-model evaluation contract."""
    dataset = load_config(config_path, input_root=input_root, output_dir=artifact_dir)
    with config_path.expanduser().resolve().open(encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)
    evaluation = raw.get("evaluation", {})
    rf = evaluation.get("random_forest", {})
    xgb = evaluation.get("xgboost", {})
    cnn = evaluation.get("cnn1d", {})
    configured_output = output_root or _required(evaluation, "output_root")

    config = TrainingConfig(
        dataset=dataset,
        output_root=_resolve(dataset.project_root, configured_output),
        seed=int(_required(evaluation, "seed")),
        training_view=str(_required(evaluation, "training_view")),
        test_view=str(_required(evaluation, "test_view")),
        expected_pair_count=int(_required(evaluation, "expected_pair_count")),
        model_ids=tuple(str(value) for value in _required(evaluation, "model_ids")),
        random_forest=RandomForestConfig(
            n_estimators=int(_required(rf, "n_estimators")),
            max_depth=(int(rf["max_depth"]) if rf.get("max_depth") is not None else None),
            class_weight=str(_required(rf, "class_weight")),
        ),
        xgboost=XGBoostConfig(
            n_estimators=int(_required(xgb, "n_estimators")),
            max_depth=int(_required(xgb, "max_depth")),
            subsample=float(_required(xgb, "subsample")),
            colsample_bytree=float(_required(xgb, "colsample_bytree")),
            matched_flow_stats_learning_rate=float(
                _required(xgb, "matched_flow_stats_learning_rate")
            ),
            flattened_splt_learning_rate=float(
                _required(xgb, "flattened_splt_learning_rate")
            ),
        ),
        cnn1d=CNN1DConfig(
            batch_size=int(_required(cnn, "batch_size")),
            maximum_epochs=int(_required(cnn, "maximum_epochs")),
            validation_fraction=float(_required(cnn, "validation_fraction")),
            learning_rate=float(_required(cnn, "learning_rate")),
            dropout=float(_required(cnn, "dropout")),
            focal_gamma=float(_required(cnn, "focal_gamma")),
            focal_alpha=float(_required(cnn, "focal_alpha")),
            early_stopping_patience=int(_required(cnn, "early_stopping_patience")),
            scheduler_patience=int(_required(cnn, "scheduler_patience")),
            augmentation_mask_probability=float(
                _required(cnn, "augmentation_mask_probability")
            ),
            augmentation_noise_stddev=float(_required(cnn, "augmentation_noise_stddev")),
            augmentation_seed=int(_required(cnn, "augmentation_seed")),
            expected_trainable_parameters=int(
                _required(cnn, "expected_trainable_parameters")
            ),
            expected_total_parameters=int(_required(cnn, "expected_total_parameters")),
        ),
    )
    _validate_training_config(config)
    return config


def _validate_training_config(config: TrainingConfig) -> None:
    if config.training_view != "inner" or config.test_view != "outer":
        raise PipelineInvariantError("The evaluation contract is fixed to inner-to-outer transfer")
    if config.model_ids != MODEL_IDS:
        raise PipelineInvariantError(f"model_ids must equal the frozen order: {MODEL_IDS}")
    if config.expected_pair_count != config.dataset.expected_retained_rows:
        raise PipelineInvariantError("Evaluation and canonical pair counts disagree")
    if config.random_forest.n_estimators != 500:
        raise PipelineInvariantError("Random Forest must use exactly 500 estimators")
    if config.random_forest.class_weight != "balanced":
        raise PipelineInvariantError("Random Forest must use balanced class weights")
    if config.xgboost.n_estimators != 500 or config.xgboost.max_depth != 6:
        raise PipelineInvariantError("XGBoost topology differs from the frozen definition")
    for value in (config.xgboost.subsample, config.xgboost.colsample_bytree):
        if not 0 < value <= 1:
            raise PipelineInvariantError("XGBoost sampling fractions must lie in (0, 1]")
    cnn = config.cnn1d
    if not 0 < cnn.validation_fraction < 1:
        raise PipelineInvariantError("CNN validation_fraction must lie in (0, 1)")
    if not 0 <= cnn.augmentation_mask_probability < 1:
        raise PipelineInvariantError("CNN mask probability must lie in [0, 1)")
    if cnn.augmentation_noise_stddev < 0:
        raise PipelineInvariantError("CNN noise standard deviation must be nonnegative")
    integer_values = (
        config.seed,
        cnn.batch_size,
        cnn.maximum_epochs,
        cnn.early_stopping_patience,
        cnn.scheduler_patience,
        cnn.expected_trainable_parameters,
        cnn.expected_total_parameters,
    )
    if any(value < 1 for value in integer_values):
        raise PipelineInvariantError("Seeds, counts, and patience values must be positive")
