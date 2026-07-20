from __future__ import annotations

import json
import platform
import shutil
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from matched_view_eval import __version__
from matched_view_eval.data import CANONICAL_FILENAME, MANIFEST_FILENAME
from matched_view_eval.errors import PipelineInvariantError
from matched_view_eval.hashing import sha256_file
from matched_view_eval.models import (
    build_cnn1d,
    build_random_forest,
    build_xgboost,
    configure_tensorflow,
)
from matched_view_eval.provenance import git_provenance
from matched_view_eval.training_config import MODEL_IDS, TrainingConfig
from matched_view_eval.training_data import (
    balanced_class_weights,
    pair_id_sha256,
    prepare_classical_data,
    prepare_cnn_data,
)
from matched_view_eval.validation import validate_dataset

FEATURE_AUDIT_FILENAME = "feature_audit.json"
PREDICTIONS_FILENAME = "predictions.parquet"
RUN_FILENAME = "run.json"

MODEL_DEFINITIONS = {
    "rf_matched_flow_stats": ("random_forest", "matched_flow_stats"),
    "xgboost_matched_flow_stats": ("xgboost", "matched_flow_stats"),
    "rf_flattened_splt": ("random_forest", "flattened_splt"),
    "xgboost_flattened_splt": ("xgboost", "flattened_splt"),
    "cnn1d_sequential_splt": ("cnn1d", "sequential_splt"),
}


def _json_write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _package_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    versions: dict[str, str] = {}
    for package in ("numpy", "pandas", "pyarrow", "scikit-learn", "xgboost", "tensorflow"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            continue
    return versions


def _input_hashes(config: TrainingConfig) -> dict[str, str]:
    paths = {
        "canonical": config.dataset.output_dir / CANONICAL_FILENAME,
        "dataset_manifest": config.dataset.output_dir / MANIFEST_FILENAME,
        "feature_audit": config.dataset.output_dir / FEATURE_AUDIT_FILENAME,
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    return {name: sha256_file(path) for name, path in paths.items()}


def _probability_columns(class_count: int) -> list[str]:
    return [f"probability_{index:02d}" for index in range(class_count)]


def _write_predictions(
    path: Path,
    *,
    pair_ids: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    classes: tuple[str, ...],
) -> None:
    if probabilities.shape != (len(pair_ids), len(classes)):
        raise PipelineInvariantError("Prediction probability shape is invalid")
    if not np.isfinite(probabilities).all() or not np.allclose(
        probabilities.sum(axis=1), 1.0, rtol=1e-6, atol=1e-6
    ):
        raise PipelineInvariantError("Predicted probabilities are non-finite or do not sum to one")
    predicted = probabilities.argmax(axis=1).astype(np.int64)
    payload: dict[str, Any] = {
        "pair_id": pair_ids.astype(str),
        "true_index": labels.astype(np.int16),
        "predicted_index": predicted.astype(np.int16),
        "true_label": np.asarray(classes, dtype=object)[labels],
        "predicted_label": np.asarray(classes, dtype=object)[predicted],
    }
    for index, name in enumerate(_probability_columns(len(classes))):
        payload[name] = probabilities[:, index].astype(np.float32)
    pd.DataFrame(payload).to_parquet(path, index=False)


def _fit_classical(
    model_id: str,
    frame: pd.DataFrame,
    config: TrainingConfig,
    temporary: Path,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    import joblib

    family, representation = MODEL_DEFINITIONS[model_id]
    data = prepare_classical_data(frame, config, representation=representation)
    class_count = len(config.dataset.expected_classes)
    if family == "random_forest":
        model = build_random_forest(config)
        model.fit(data.train_values, data.labels)
        model_path = temporary / "model.joblib"
        joblib.dump(model, model_path)
        balance = {"mechanism": "class_weight", "value": "balanced"}
        hyperparameters = {
            "n_estimators": config.random_forest.n_estimators,
            "max_depth": config.random_forest.max_depth,
            "n_jobs": -1,
            "random_state": config.seed,
        }
    else:
        model = build_xgboost(config, representation=representation)
        weights = balanced_class_weights(data.labels, class_count)[data.labels]
        model.fit(data.train_values, data.labels, sample_weight=weights)
        model_path = temporary / "model.json"
        model.save_model(model_path)
        balance = {"mechanism": "sample_weight", "value": "balanced"}
        learning_rate = (
            config.xgboost.matched_flow_stats_learning_rate
            if representation == "matched_flow_stats"
            else config.xgboost.flattened_splt_learning_rate
        )
        hyperparameters = {
            "n_estimators": config.xgboost.n_estimators,
            "max_depth": config.xgboost.max_depth,
            "learning_rate": learning_rate,
            "subsample": config.xgboost.subsample,
            "colsample_bytree": config.xgboost.colsample_bytree,
            "eval_metric": "mlogloss",
            "n_jobs": -1,
            "random_state": config.seed,
        }
    probabilities = np.asarray(model.predict_proba(data.test_values), dtype=np.float64)
    details: dict[str, Any] = {
        "family": family,
        "representation": representation,
        "feature_count": data.train_values.shape[1],
        "feature_names": list(data.feature_names),
        "class_balance": balance,
        "hyperparameters": hyperparameters,
        "source_medians": data.medians.tolist() if data.medians is not None else None,
        "model_artifact": model_path.name,
    }
    return details, data.pair_ids, data.labels, probabilities


def _fit_cnn(
    frame: pd.DataFrame,
    config: TrainingConfig,
    temporary: Path,
    *,
    device: str,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    tf = configure_tensorflow(config.seed, device=device)
    tf.keras.backend.clear_session()
    data = prepare_cnn_data(frame, config)
    model = build_cnn1d(config, tf)
    class_weights = balanced_class_weights(
        data.train_labels, len(config.dataset.expected_classes)
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=config.cnn1d.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=config.cnn1d.scheduler_patience,
            min_lr=1e-6,
            verbose=1,
        ),
    ]
    history = model.fit(
        data.train_values,
        data.train_labels,
        batch_size=config.cnn1d.batch_size,
        epochs=config.cnn1d.maximum_epochs,
        validation_data=(data.validation_values, data.validation_labels),
        class_weight={index: float(weight) for index, weight in enumerate(class_weights)},
        callbacks=callbacks,
        shuffle=True,
        verbose=1,
    )
    probabilities = np.asarray(
        model.predict(data.test_values, batch_size=config.cnn1d.batch_size, verbose=1),
        dtype=np.float64,
    )
    (temporary / "model_architecture.json").write_text(model.to_json(), encoding="utf-8")
    model.save_weights(temporary / "model.weights.h5")
    pd.DataFrame(history.history).to_csv(temporary / "training_history.csv", index=False)
    details = {
        "family": "cnn1d",
        "representation": "sequential_splt",
        "feature_shape": [config.dataset.primary_prefix_length, 3],
        "class_balance": {"mechanism": "class_weight", "value": class_weights.tolist()},
        "training_pair_count_before_augmentation": len(data.training_pair_ids),
        "training_pair_count_after_augmentation": len(data.train_labels),
        "validation_pair_count": len(data.validation_pair_ids),
        "training_pair_ids_sha256": pair_id_sha256(data.training_pair_ids),
        "validation_pair_ids_sha256": pair_id_sha256(data.validation_pair_ids),
        "epochs_completed": len(history.history["loss"]),
        "model_artifact": "model.weights.h5",
        "architecture_artifact": "model_architecture.json",
        "history_artifact": "training_history.csv",
        "device": device,
        "topology": {
            "parallel_kernels": [3, 7, 11],
            "branch_filters": 64,
            "residual_filters": 192,
            "attention_heads": 4,
            "attention_key_dim": 48,
            "final_filters": 256,
            "dense_units": [256, 128],
            "trainable_parameters": config.cnn1d.expected_trainable_parameters,
            "total_parameters": config.cnn1d.expected_total_parameters,
        },
        "training_policy": {
            "batch_size": config.cnn1d.batch_size,
            "maximum_epochs": config.cnn1d.maximum_epochs,
            "validation_fraction": config.cnn1d.validation_fraction,
            "learning_rate": config.cnn1d.learning_rate,
            "dropout": config.cnn1d.dropout,
            "focal_gamma": config.cnn1d.focal_gamma,
            "focal_alpha": config.cnn1d.focal_alpha,
            "early_stopping_metric": "val_accuracy",
            "early_stopping_patience": config.cnn1d.early_stopping_patience,
            "scheduler_metric": "val_loss",
            "scheduler_factor": 0.5,
            "scheduler_patience": config.cnn1d.scheduler_patience,
            "minimum_learning_rate": 1e-6,
            "shuffle": True,
            "deterministic_operations": True,
        },
        "augmentation": {
            "mask_probability": config.cnn1d.augmentation_mask_probability,
            "noise_stddev": config.cnn1d.augmentation_noise_stddev,
            "seed": config.cnn1d.augmentation_seed,
            "direction_perturbed": False,
            "padding_perturbed": False,
        },
    }
    return details, data.pair_ids, data.test_labels, probabilities


def train_model(
    config: TrainingConfig,
    *,
    model_id: str,
    device: str = "auto",
    force: bool = False,
    require_clean_git: bool = True,
) -> dict[str, Any]:
    """Train one frozen configuration and atomically publish outer-view predictions."""
    if model_id not in MODEL_IDS or model_id not in config.model_ids:
        raise PipelineInvariantError(f"Unknown model_id: {model_id}")
    provenance = git_provenance(config.dataset.project_root)
    if require_clean_git and (
        not provenance.get("status_available") or provenance.get("dirty") is not False
    ):
        raise PipelineInvariantError("Training requires a clean committed Git revision")
    validate_dataset(config.dataset)
    input_hashes = _input_hashes(config)
    canonical_path = config.dataset.output_dir / CANONICAL_FILENAME
    frame = pd.read_parquet(canonical_path)
    if len(frame) != config.expected_pair_count:
        raise PipelineInvariantError("Canonical row count changed before training")

    destination = config.output_root / model_id
    if destination.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing run: {destination}")
    config.output_root.mkdir(parents=True, exist_ok=True)
    temporary = config.output_root / f".{model_id}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir()
    started = time.monotonic()
    try:
        family = MODEL_DEFINITIONS[model_id][0]
        if family == "cnn1d":
            details, pair_ids, labels, probabilities = _fit_cnn(
                frame, config, temporary, device=device
            )
        else:
            details, pair_ids, labels, probabilities = _fit_classical(
                model_id, frame, config, temporary
            )
        _write_predictions(
            temporary / PREDICTIONS_FILENAME,
            pair_ids=pair_ids,
            labels=labels,
            probabilities=probabilities,
            classes=config.dataset.expected_classes,
        )
        artifact_paths = sorted(
            path for path in temporary.iterdir() if path.is_file() and path.name != RUN_FILENAME
        )
        run = {
            "schema_version": 1,
            "created_utc": datetime.now(UTC).isoformat(),
            "package_version": __version__,
            "model_id": model_id,
            "protocol": "matched_view_same_flow",
            "training_view": config.training_view,
            "test_view": config.test_view,
            "seed": config.seed,
            "pair_count": len(pair_ids),
            "pair_ids_sha256": pair_id_sha256(pair_ids),
            "class_order": list(config.dataset.expected_classes),
            "input_hashes": input_hashes,
            "git": provenance,
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "packages": _package_versions(),
            },
            "training_seconds": time.monotonic() - started,
            "definition": details,
            "artifacts": {
                path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
                for path in artifact_paths
            },
        }
        _json_write(temporary / RUN_FILENAME, run)
        validate_run_directory(temporary)
        if destination.exists():
            shutil.rmtree(destination)
        temporary.replace(destination)
        return run
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def validate_run_directory(path: Path) -> dict[str, Any]:
    """Validate artifact hashes and prediction/probability consistency."""
    run_path = path / RUN_FILENAME
    predictions_path = path / PREDICTIONS_FILENAME
    if not run_path.is_file() or not predictions_path.is_file():
        raise FileNotFoundError(f"Incomplete run directory: {path}")
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if run.get("model_id") not in MODEL_IDS or run.get("protocol") != "matched_view_same_flow":
        raise PipelineInvariantError("Run identity is invalid")
    required_artifacts = {
        "rf_matched_flow_stats": {PREDICTIONS_FILENAME, "model.joblib"},
        "rf_flattened_splt": {PREDICTIONS_FILENAME, "model.joblib"},
        "xgboost_matched_flow_stats": {PREDICTIONS_FILENAME, "model.json"},
        "xgboost_flattened_splt": {PREDICTIONS_FILENAME, "model.json"},
        "cnn1d_sequential_splt": {
            PREDICTIONS_FILENAME,
            "model_architecture.json",
            "model.weights.h5",
            "training_history.csv",
        },
    }[run["model_id"]]
    if set(run.get("artifacts", {})) != required_artifacts:
        raise PipelineInvariantError("Run artifact inventory is incomplete or unexpected")
    for name, metadata in run.get("artifacts", {}).items():
        artifact = path / name
        if not artifact.is_file() or sha256_file(artifact) != metadata.get("sha256"):
            raise PipelineInvariantError(f"Run artifact hash mismatch: {name}")
    predictions = pd.read_parquet(predictions_path)
    classes = tuple(str(value) for value in run["class_order"])
    probability_columns = _probability_columns(len(classes))
    required = {
        "pair_id",
        "true_index",
        "predicted_index",
        "true_label",
        "predicted_label",
        *probability_columns,
    }
    if set(predictions.columns) != required:
        raise PipelineInvariantError("Prediction artifact schema is invalid")
    if len(predictions) != run.get("pair_count") or predictions["pair_id"].duplicated().any():
        raise PipelineInvariantError("Prediction pair coverage is invalid")
    if pair_id_sha256(predictions["pair_id"].astype(str).to_numpy()) != run.get(
        "pair_ids_sha256"
    ):
        raise PipelineInvariantError("Prediction pair order/hash is invalid")
    probabilities = predictions.loc[:, probability_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(probabilities).all() or not np.allclose(
        probabilities.sum(axis=1), 1.0, rtol=1e-6, atol=1e-6
    ):
        raise PipelineInvariantError("Stored probabilities are invalid")
    predicted = probabilities.argmax(axis=1)
    if not np.array_equal(predicted, predictions["predicted_index"].to_numpy(dtype=int)):
        raise PipelineInvariantError("Stored predictions disagree with probability argmax")
    class_array = np.asarray(classes, dtype=object)
    for index_column, label_column in (
        ("true_index", "true_label"),
        ("predicted_index", "predicted_label"),
    ):
        indices = predictions[index_column].to_numpy(dtype=int)
        if (indices < 0).any() or (indices >= len(classes)).any():
            raise PipelineInvariantError(f"{index_column} is outside the class vocabulary")
        if not np.array_equal(class_array[indices], predictions[label_column].to_numpy()):
            raise PipelineInvariantError(f"{label_column} disagrees with class indices")
    return {
        "valid": True,
        "model_id": run["model_id"],
        "pair_count": len(predictions),
        "predictions_sha256": sha256_file(predictions_path),
    }
