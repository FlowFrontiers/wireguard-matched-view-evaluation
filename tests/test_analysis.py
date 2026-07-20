from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

import matched_view_eval.analysis as analysis_module
import matched_view_eval.orchestration as orchestration_module
from matched_view_eval.analysis import (
    ANALYSIS_MANIFEST_FILENAME,
    analyze_runs,
    validate_analysis_directory,
)
from matched_view_eval.analysis_config import load_analysis_config
from matched_view_eval.errors import PipelineInvariantError
from matched_view_eval.hashing import sha256_file
from matched_view_eval.metrics import (
    classification_metrics,
    confusion_from_labels,
    confusion_metrics,
    paired_bootstrap_confusion_metrics,
)
from matched_view_eval.orchestration import _training_command, preflight
from matched_view_eval.training import PREDICTIONS_FILENAME, RUN_FILENAME, _write_predictions
from matched_view_eval.training_data import pair_id_sha256

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "experiment.yaml"


def test_confusion_metrics_match_sklearn() -> None:
    true = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
    predicted = np.array([0, 1, 0, 1, 2, 2, 0, 2, 1])
    confusion = confusion_from_labels(true, predicted, class_count=3)
    observed = confusion_metrics(confusion)
    assert observed["accuracy"] == pytest.approx(accuracy_score(true, predicted))
    assert observed["balanced_accuracy"] == pytest.approx(
        balanced_accuracy_score(true, predicted)
    )
    assert observed["macro_f1"] == pytest.approx(
        f1_score(true, predicted, average="macro", zero_division=0)
    )
    assert observed["weighted_f1"] == pytest.approx(
        f1_score(true, predicted, average="weighted", zero_division=0)
    )


def test_bootstrap_is_deterministic_and_preserves_point_estimate() -> None:
    confusion = np.array([[80, 10, 10], [5, 40, 5], [2, 3, 25]])
    first = paired_bootstrap_confusion_metrics(
        confusion, replicates=1_000, confidence_level=0.95, seed=42
    )
    second = paired_bootstrap_confusion_metrics(
        confusion, replicates=1_000, confidence_level=0.95, seed=42
    )
    assert np.array_equal(first.distributions, second.distributions)
    assert first.point == confusion_metrics(confusion)
    for metric, point in first.point.items():
        assert first.lower[metric] < point < first.upper[metric]


def test_macro_average_precision_uses_probabilities() -> None:
    true = np.repeat(np.arange(3), 4)
    predicted = true.copy()
    probabilities = np.full((len(true), 3), 0.05)
    probabilities[np.arange(len(true)), true] = 0.90
    metrics, _ = classification_metrics(
        true, predicted, probabilities, class_count=3
    )
    assert metrics["macro_average_precision"] == pytest.approx(1.0)
    probabilities[[0, 4, 8]] = probabilities[[0, 4, 8]][:, ::-1]
    degraded, _ = classification_metrics(
        true, probabilities.argmax(axis=1), probabilities, class_count=3
    )
    assert degraded["macro_average_precision"] < metrics["macro_average_precision"]


def _probabilities(true: np.ndarray, predicted: np.ndarray, class_count: int) -> np.ndarray:
    values = np.empty((len(true), class_count), dtype=np.float64)
    for index, (true_label, predicted_label) in enumerate(zip(true, predicted, strict=True)):
        if true_label == predicted_label:
            values[index] = 0.1 / (class_count - 1)
            values[index, true_label] = 0.9
        else:
            values[index] = 0.1 / (class_count - 2)
            values[index, true_label] = 0.2
            values[index, predicted_label] = 0.7
    return values


def _write_synthetic_run(
    run_root: Path,
    *,
    model_id: str,
    classes: tuple[str, ...],
    pair_ids: np.ndarray,
    true: np.ndarray,
    predicted: np.ndarray,
) -> None:
    run_dir = run_root / model_id
    run_dir.mkdir(parents=True)
    probabilities = _probabilities(true, predicted, len(classes))
    predictions_path = run_dir / PREDICTIONS_FILENAME
    _write_predictions(
        predictions_path,
        pair_ids=pair_ids,
        labels=true,
        probabilities=probabilities,
        classes=classes,
    )
    if model_id.startswith("rf_"):
        model_files = {"model.joblib": b"rf"}
    elif model_id.startswith("xgboost_"):
        model_files = {"model.json": b"{}"}
    else:
        model_files = {
            "model_architecture.json": b"{}",
            "model.weights.h5": b"weights",
            "training_history.csv": b"loss,accuracy\n1.0,0.5\n",
        }
    for name, content in model_files.items():
        (run_dir / name).write_bytes(content)
    artifact_names = {PREDICTIONS_FILENAME, *model_files}
    run = {
        "model_id": model_id,
        "protocol": "matched_view_same_flow",
        "pair_count": len(pair_ids),
        "pair_ids_sha256": pair_id_sha256(pair_ids),
        "class_order": list(classes),
        "input_hashes": {"canonical": "a" * 64, "feature_audit": "b" * 64},
        "artifacts": {
            name: {"sha256": sha256_file(run_dir / name)} for name in artifact_names
        },
    }
    (run_dir / RUN_FILENAME).write_text(json.dumps(run), encoding="utf-8")


def _synthetic_analysis_config(tmp_path: Path):
    config = load_analysis_config(CONFIG_PATH)
    run_root = tmp_path / "runs"
    classes = config.training.dataset.expected_classes
    true = np.repeat(np.arange(len(classes)), 10)
    pair_ids = np.array([f"pair-{index:04d}" for index in range(len(true))])
    for model_index, model_id in enumerate(config.training.model_ids):
        predicted = true.copy()
        error_indices = np.arange(model_index, len(true), 17 + model_index)
        predicted[error_indices] = (predicted[error_indices] + 1) % len(classes)
        _write_synthetic_run(
            run_root,
            model_id=model_id,
            classes=classes,
            pair_ids=pair_ids,
            true=true,
            predicted=predicted,
        )
    dataset = replace(
        config.training.dataset,
        expected_retained_rows=len(true),
    )
    training = replace(
        config.training,
        dataset=dataset,
        output_root=run_root,
        expected_pair_count=len(true),
    )
    return replace(config, training=training, output_dir=tmp_path / "analysis")


def test_analysis_pipeline_and_content_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _synthetic_analysis_config(tmp_path)
    monkeypatch.setattr(
        analysis_module,
        "git_provenance",
        lambda _: {"revision": "test", "dirty": False, "status_available": True},
    )
    manifest = analyze_runs(config)
    assert manifest["pair_count"] == 140
    result = validate_analysis_directory(
        config.output_dir,
        expected_models=config.training.model_ids,
        run_root=config.training.output_root,
    )
    assert result == {"valid": True, "pair_count": 140, "model_count": 5, "artifact_count": 14}
    macros = (config.output_dir / "latex" / "results_macros.tex").read_text()
    assert r"\newcommand{\SampleCount}{140}" in macros
    assert (config.output_dir / "figures" / "macro_precision_recall.pdf").stat().st_size > 0

    first_hashes = manifest["artifacts"]
    second_manifest = analyze_runs(config, force=True)
    assert second_manifest["artifacts"] == first_hashes

    metrics_path = config.output_dir / "metrics.csv"
    metrics = pd.read_csv(metrics_path)
    metrics.loc[0, "macro_average_precision"] = 0.0
    metrics.to_csv(metrics_path, index=False)
    manifest_path = config.output_dir / ANALYSIS_MANIFEST_FILENAME
    tampered = json.loads(manifest_path.read_text(encoding="utf-8"))
    tampered["artifacts"]["metrics.csv"]["sha256"] = sha256_file(metrics_path)
    manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(PipelineInvariantError, match="disagrees with predictions"):
        validate_analysis_directory(
            config.output_dir,
            expected_models=config.training.model_ids,
            run_root=config.training.output_root,
        )


def test_training_commands_use_isolated_python_processes(tmp_path: Path) -> None:
    config = _synthetic_analysis_config(tmp_path)
    command = _training_command(
        CONFIG_PATH,
        config,
        model_id="cnn1d_sequential_splt",
        device="gpu",
        force=False,
    )
    assert command[:3] == [
        __import__("sys").executable,
        "-m",
        "matched_view_eval.cli",
    ]
    assert command[command.index("--model") + 1] == "cnn1d_sequential_splt"
    assert command[command.index("--device") + 1] == "gpu"


def test_preflight_does_not_import_tensorflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _synthetic_analysis_config(tmp_path)
    monkeypatch.setattr(
        orchestration_module,
        "git_provenance",
        lambda _: {"revision": "test", "dirty": False, "status_available": True},
    )
    result = preflight(config)
    assert result["completed_runs"] == 5
    assert result["pending_runs"] == 0
    assert result["analysis_status"] == "pending"
    assert "tensorflow" not in sys.modules
