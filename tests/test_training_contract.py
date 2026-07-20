from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from matched_view_eval.errors import PipelineInvariantError
from matched_view_eval.hashing import sha256_file
from matched_view_eval.models import (
    build_random_forest,
    build_xgboost,
)
from matched_view_eval.schema import CANONICAL_STATS
from matched_view_eval.training import (
    PREDICTIONS_FILENAME,
    RUN_FILENAME,
    _write_predictions,
    train_model,
    validate_run_directory,
)
from matched_view_eval.training_config import MODEL_IDS, load_training_config
from matched_view_eval.training_data import (
    apply_stat_medians,
    augment_sequential_splt,
    balanced_class_weights,
    fit_stat_medians,
    pair_id_sha256,
    stratified_validation_indices,
)

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "experiment.yaml"


def _run_tensorflow_check(source: str) -> None:
    if find_spec("tensorflow") is None:
        pytest.skip("TensorFlow is not installed")
    environment = os.environ.copy()
    source_root = str(Path(__file__).parents[1] / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (source_root, environment.get("PYTHONPATH", "")) if value
    )
    result = subprocess.run(
        [sys.executable, "-c", source],
        cwd=Path(__file__).parents[1],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_frozen_training_configuration() -> None:
    config = load_training_config(CONFIG_PATH)
    assert config.model_ids == MODEL_IDS
    assert config.training_view == "inner"
    assert config.test_view == "outer"
    assert config.expected_pair_count == 226_281
    assert config.random_forest.n_estimators == 500
    assert config.xgboost.n_estimators == 500
    assert config.cnn1d.validation_fraction == 0.1


def test_source_medians_do_not_depend_on_target_values() -> None:
    source = np.array([[1.0, np.nan], [3.0, 8.0], [5.0, 4.0]])
    target = np.array([[np.nan, 1e12], [7.0, np.nan]])
    medians = fit_stat_medians(source)
    transformed = apply_stat_medians(target, medians)
    assert np.array_equal(medians, np.array([3.0, 6.0]))
    assert transformed[0, 0] == 3.0
    assert transformed[1, 1] == 6.0
    poisoned_target = np.full_like(target, -1e15)
    assert np.array_equal(fit_stat_medians(source), medians)
    assert not np.array_equal(apply_stat_medians(poisoned_target, medians), transformed)


def test_balanced_weights_match_closed_form() -> None:
    labels = np.array([0, 0, 0, 1, 2, 2])
    observed = balanced_class_weights(labels, 3)
    expected = np.array([6 / (3 * 3), 6 / (3 * 1), 6 / (3 * 2)])
    assert np.allclose(observed, expected)


def test_validation_split_is_deterministic_stratified_and_disjoint() -> None:
    labels = np.repeat(np.arange(4), [50, 40, 30, 20])
    train_a, validation_a = stratified_validation_indices(
        labels, validation_fraction=0.1, seed=42
    )
    train_b, validation_b = stratified_validation_indices(
        labels, validation_fraction=0.1, seed=42
    )
    assert np.array_equal(train_a, train_b)
    assert np.array_equal(validation_a, validation_b)
    assert not np.intersect1d(train_a, validation_a).size
    assert len(validation_a) == 14
    assert set(labels[validation_a]) == {0, 1, 2, 3}


def test_augmentation_preserves_direction_and_padding() -> None:
    values = np.array(
        [
            [[-1.0, 4.0, 0.0], [1.0, 5.0, 2.0], [0.0, 0.0, 0.0]],
            [[1.0, 3.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    mask = np.array([[True, True, False], [True, False, False]])
    augmented = augment_sequential_splt(
        values,
        mask,
        mask_probability=0.0,
        noise_stddev=0.2,
        seed=99,
    )
    assert np.array_equal(augmented[:, :, 0][mask], values[:, :, 0][mask])
    assert np.all(augmented[~mask] == 0)
    assert not np.array_equal(augmented[:, :, 1:][mask], values[:, :, 1:][mask])
    dropped = augment_sequential_splt(
        values,
        mask,
        mask_probability=1.0,
        noise_stddev=0.2,
        seed=99,
    )
    assert np.all(dropped == 0)


def test_classical_model_parameters_are_frozen() -> None:
    pytest.importorskip("sklearn")
    pytest.importorskip("xgboost")
    config = load_training_config(CONFIG_PATH)
    forest = build_random_forest(config)
    assert forest.get_params()["n_estimators"] == 500
    assert forest.get_params()["class_weight"] == "balanced"
    stats_xgb = build_xgboost(config, representation="matched_flow_stats")
    splt_xgb = build_xgboost(config, representation="flattened_splt")
    assert stats_xgb.get_params()["learning_rate"] == 0.05
    assert splt_xgb.get_params()["learning_rate"] == 0.1
    assert stats_xgb.get_params()["max_depth"] == 6


def test_focal_loss_retains_sample_axis_for_class_weighting() -> None:
    _run_tensorflow_check(
        """
import numpy as np
from matched_view_eval.models import configure_tensorflow, make_sparse_focal_loss
tf = configure_tensorflow(42, device="cpu")
loss = make_sparse_focal_loss(tf, gamma=2.0, alpha=0.25)
labels = tf.constant([0, 1], dtype=tf.int32)
predictions = tf.constant([[0.8, 0.2], [0.3, 0.7]], dtype=tf.float32)
per_sample = loss(labels, predictions).numpy()
assert per_sample.shape == (2,)
baseline = per_sample * np.array([1.0, 1.0])[labels.numpy()]
changed = per_sample * np.array([5.0, 1.0])[labels.numpy()]
assert np.isclose(changed[0], 5.0 * baseline[0])
assert np.isclose(changed[1], baseline[1])
"""
    )


def test_cnn_topology_parameter_counts() -> None:
    _run_tensorflow_check(
        """
from pathlib import Path
import numpy as np
from matched_view_eval.models import build_cnn1d, configure_tensorflow
from matched_view_eval.training_config import load_training_config
config = load_training_config(Path("configs/experiment.yaml"))
tf = configure_tensorflow(42, device="cpu")
tf.keras.backend.clear_session()
model = build_cnn1d(config, tf)
trainable = int(sum(np.prod(variable.shape) for variable in model.trainable_weights))
assert trainable == 550_158
assert model.count_params() == 551_438
rng = np.random.default_rng(42)
values = rng.normal(size=(28, 50, 3)).astype(np.float32)
labels = np.repeat(np.arange(14), 2)
history = model.fit(
    values,
    labels,
    batch_size=28,
    epochs=1,
    class_weight={index: 1.0 + index / 10.0 for index in range(14)},
    verbose=0,
)
probabilities = model.predict(values[:3], verbose=0)
assert np.isfinite(history.history["loss"]).all()
assert probabilities.shape == (3, 14)
assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
"""
    )


def _make_run(path: Path) -> None:
    classes = ("a", "b")
    pair_ids = np.array(["p1", "p2", "p3"])
    labels = np.array([0, 1, 1])
    probabilities = np.array([[0.8, 0.2], [0.1, 0.9], [0.6, 0.4]])
    predictions_path = path / PREDICTIONS_FILENAME
    _write_predictions(
        predictions_path,
        pair_ids=pair_ids,
        labels=labels,
        probabilities=probabilities,
        classes=classes,
    )
    payload = {
        "model_id": "rf_matched_flow_stats",
        "protocol": "matched_view_same_flow",
        "pair_count": 3,
        "pair_ids_sha256": pair_id_sha256(pair_ids),
        "class_order": list(classes),
        "artifacts": {
            PREDICTIONS_FILENAME: {"sha256": sha256_file(predictions_path)},
            "model.joblib": {"sha256": ""},
        },
    }
    model_path = path / "model.joblib"
    model_path.write_bytes(b"synthetic-model")
    payload["artifacts"]["model.joblib"]["sha256"] = sha256_file(model_path)
    (path / RUN_FILENAME).write_text(json.dumps(payload), encoding="utf-8")


def test_prediction_artifact_validation_recomputes_content(tmp_path: Path) -> None:
    _make_run(tmp_path)
    assert validate_run_directory(tmp_path)["valid"] is True
    predictions_path = tmp_path / PREDICTIONS_FILENAME
    predictions = pd.read_parquet(predictions_path)
    predictions.loc[0, "predicted_index"] = 1
    predictions.loc[0, "predicted_label"] = "b"
    predictions.to_parquet(predictions_path, index=False)
    run_path = tmp_path / RUN_FILENAME
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["artifacts"][PREDICTIONS_FILENAME]["sha256"] = sha256_file(predictions_path)
    run_path.write_text(json.dumps(run), encoding="utf-8")
    with pytest.raises(PipelineInvariantError, match="probability argmax"):
        validate_run_directory(tmp_path)


def test_classical_run_is_published_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_training_config(CONFIG_PATH)
    artifact_dir = tmp_path / "processed"
    artifact_dir.mkdir()
    canonical_path = artifact_dir / "canonical_pairs.parquet"
    canonical_path.write_bytes(b"synthetic-canonical-anchor")
    (artifact_dir / "dataset_manifest.json").write_text("{}", encoding="utf-8")
    (artifact_dir / "feature_audit.json").write_text("{}", encoding="utf-8")

    classes = config.dataset.expected_classes
    labels = np.repeat(np.asarray(classes), 2)
    row_count = len(labels)
    frame: dict[str, object] = {
        "pair_id": [f"pair-{index:03d}" for index in range(row_count)],
        "application_category": labels,
    }
    for domain in ("inner", "outer"):
        for feature_index, name in enumerate(CANONICAL_STATS):
            frame[f"{domain}_{name}"] = np.arange(row_count, dtype=float) + feature_index + 1
    canonical = pd.DataFrame(frame)
    dataset = replace(
        config.dataset,
        output_dir=artifact_dir,
        expected_retained_rows=row_count,
    )
    test_config = replace(
        config,
        dataset=dataset,
        output_root=tmp_path / "runs",
        expected_pair_count=row_count,
    )
    import matched_view_eval.training as training_module

    original_read_parquet = training_module.pd.read_parquet

    def read_parquet(path: Path, *args: object, **kwargs: object) -> pd.DataFrame:
        if Path(path) == canonical_path:
            return canonical.copy()
        return original_read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(training_module, "validate_dataset", lambda _: {"valid": True})
    monkeypatch.setattr(training_module.pd, "read_parquet", read_parquet)
    run = train_model(
        test_config,
        model_id="rf_matched_flow_stats",
        require_clean_git=False,
    )
    destination = test_config.output_root / "rf_matched_flow_stats"
    assert run["pair_count"] == row_count
    assert destination.is_dir()
    assert not list(test_config.output_root.glob(".*.tmp"))
    assert validate_run_directory(destination)["pair_count"] == row_count
