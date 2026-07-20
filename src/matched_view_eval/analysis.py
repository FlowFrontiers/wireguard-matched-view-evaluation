from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from matched_view_eval import __version__
from matched_view_eval.analysis_config import INTERVAL_METRICS, METRICS, AnalysisConfig
from matched_view_eval.errors import PipelineInvariantError
from matched_view_eval.hashing import sha256_file
from matched_view_eval.metrics import (
    classification_metrics,
    confusion_metrics,
    paired_bootstrap_confusion_metrics,
)
from matched_view_eval.provenance import git_provenance
from matched_view_eval.training import (
    PREDICTIONS_FILENAME,
    RUN_FILENAME,
    validate_run_directory,
)

ANALYSIS_MANIFEST_FILENAME = "analysis_manifest.json"

MODEL_LABELS = {
    "rf_matched_flow_stats": ("RF", "FlowFeatures"),
    "xgboost_matched_flow_stats": ("XGBoost", "FlowFeatures"),
    "rf_flattened_splt": ("RF", "SPLT"),
    "xgboost_flattened_splt": ("XGBoost", "SPLT"),
    "cnn1d_sequential_splt": ("CNN1D", "SPLT"),
}

PLOT_LABELS = {
    "rf_matched_flow_stats": "RF + FlowFeatures",
    "xgboost_matched_flow_stats": "XGBoost + FlowFeatures",
    "rf_flattened_splt": "RF + SPLT",
    "xgboost_flattened_splt": "XGBoost + SPLT",
    "cnn1d_sequential_splt": "CNN1D + SPLT",
}

PLOT_STYLES = {
    "cnn1d_sequential_splt": ("#2171B5", "-"),
    "xgboost_matched_flow_stats": ("#D95F0E", "-"),
    "xgboost_flattened_splt": ("#238B45", "-"),
    "rf_matched_flow_stats": ("#756BB1", "--"),
    "rf_flattened_splt": ("#CB181D", "--"),
}


def _probability_columns(class_count: int) -> list[str]:
    return [f"probability_{index:02d}" for index in range(class_count)]


def _load_runs(
    config: AnalysisConfig,
) -> tuple[dict[str, dict[str, Any]], dict[str, pd.DataFrame], tuple[str, ...]]:
    runs: dict[str, dict[str, Any]] = {}
    predictions: dict[str, pd.DataFrame] = {}
    reference_pair_hash: str | None = None
    reference_classes: tuple[str, ...] | None = None
    reference_inputs: dict[str, str] | None = None
    for model_id in config.training.model_ids:
        run_dir = config.training.output_root / model_id
        validate_run_directory(run_dir)
        run = json.loads((run_dir / RUN_FILENAME).read_text(encoding="utf-8"))
        frame = pd.read_parquet(run_dir / PREDICTIONS_FILENAME)
        classes = tuple(str(value) for value in run["class_order"])
        if reference_pair_hash is None:
            reference_pair_hash = str(run["pair_ids_sha256"])
            reference_classes = classes
            reference_inputs = dict(run["input_hashes"])
        if run["pair_ids_sha256"] != reference_pair_hash:
            raise PipelineInvariantError("Model runs use different ordered pair sets")
        if classes != reference_classes:
            raise PipelineInvariantError("Model runs use different class vocabularies")
        if run["input_hashes"] != reference_inputs:
            raise PipelineInvariantError("Model runs were produced from different inputs")
        if not frame["pair_id"].equals(
            next(iter(predictions.values()))["pair_id"] if predictions else frame["pair_id"]
        ):
            raise PipelineInvariantError("Prediction rows differ across model runs")
        runs[model_id] = run
        predictions[model_id] = frame
    if reference_classes is None:
        raise PipelineInvariantError("No completed model runs were found")
    return runs, predictions, reference_classes


def _per_class_rows(
    model_id: str,
    confusion: np.ndarray,
    classes: tuple[str, ...],
) -> list[dict[str, Any]]:
    matrix = confusion.astype(np.float64)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    true_positive = np.diag(matrix)
    precision = np.divide(
        true_positive, predicted, out=np.zeros_like(true_positive), where=predicted > 0
    )
    recall = np.divide(
        true_positive, support, out=np.zeros_like(true_positive), where=support > 0
    )
    denominator = precision + recall
    f1 = np.divide(
        2.0 * precision * recall,
        denominator,
        out=np.zeros_like(true_positive),
        where=denominator > 0,
    )
    return [
        {
            "model_id": model_id,
            "model": MODEL_LABELS[model_id][0],
            "features": MODEL_LABELS[model_id][1],
            "class_index": index,
            "class_name": label,
            "support": int(support[index]),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
        }
        for index, label in enumerate(classes)
    ]


def _macro_precision_recall(
    true_labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    class_count: int,
    grid_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.metrics import precision_recall_curve
    from sklearn.preprocessing import label_binarize

    binarized = label_binarize(true_labels, classes=np.arange(class_count))
    recall_grid = np.linspace(0.0, 1.0, grid_points)
    mean_precision = np.zeros_like(recall_grid)
    for class_index in range(class_count):
        precision, recall, _ = precision_recall_curve(
            binarized[:, class_index], probabilities[:, class_index]
        )
        mean_precision += np.interp(recall_grid, recall[::-1], precision[::-1])
    return recall_grid, mean_precision / class_count


def _top_confused_pairs(
    confusion: np.ndarray,
    classes: tuple[str, ...],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for true_index, true_label in enumerate(classes):
        support = int(confusion[true_index].sum())
        for predicted_index, predicted_label in enumerate(classes):
            count = int(confusion[true_index, predicted_index])
            if true_index != predicted_index and count:
                rows.append(
                    {
                        "true_category": true_label,
                        "predicted_category": predicted_label,
                        "count": count,
                        "true_class_support": support,
                        "error_rate": count / support,
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["count", "true_category", "predicted_category"],
        ascending=[False, True, True],
    )


def _plot_outputs(
    output_dir: Path,
    *,
    metrics: pd.DataFrame,
    per_class: pd.DataFrame,
    curves: pd.DataFrame,
    cnn_confusion: np.ndarray,
    classes: tuple[str, ...],
    key_models: tuple[str, ...],
) -> None:
    cache_dir = Path(tempfile.gettempdir()) / "wgme-matplotlib-cache"
    cache_dir.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = output_dir / "figures"
    figures.mkdir()

    x = np.arange(len(classes), dtype=float) * 1.8
    width = 0.50
    fig, axis = plt.subplots(figsize=(9.0, 3.0))
    offsets = (-width, 0.0, width)
    colors = ("#2171B5", "#D95F0E", "#238B45")
    for model_id, offset, color in zip(key_models, offsets, colors, strict=True):
        selected = per_class[per_class["model_id"] == model_id].set_index("class_name")
        axis.bar(
            x + offset,
            selected.loc[list(classes), "f1"],
            width,
            label=PLOT_LABELS[model_id],
            color=color,
            edgecolor="white",
            linewidth=0.6,
        )
    axis.set_xticks(x)
    axis.set_xticklabels(classes, rotation=35, ha="right", fontsize=8)
    axis.set_ylabel("F1 score", fontsize=9)
    axis.set_ylim(0.0, 1.05)
    axis.legend(fontsize=8, loc="lower right", frameon=True)
    axis.yaxis.grid(True, alpha=0.3, linestyle="--")
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.5)
    fig.savefig(figures / "per_class_f1.png", dpi=300, bbox_inches="tight")
    fig.savefig(
        figures / "per_class_f1.pdf",
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(3.5, 2.5))
    ap_lookup = metrics.set_index("model_id")["macro_average_precision"]
    for model_id in metrics["model_id"]:
        selected = curves[curves["model_id"] == model_id]
        color, linestyle = PLOT_STYLES[model_id]
        axis.plot(
            selected["recall"],
            selected["precision"],
            color=color,
            linestyle=linestyle,
            linewidth=1.8,
            label=f"{PLOT_LABELS[model_id]} (AP={ap_lookup[model_id]:.3f})",
        )
    axis.set_xlabel("Recall", fontsize=9)
    axis.set_ylabel("Precision", fontsize=9)
    axis.set_xlim(0.0, 1.02)
    axis.set_ylim(0.0, 1.02)
    axis.tick_params(labelsize=8)
    axis.legend(fontsize=6.5, loc="lower left", frameon=True)
    axis.yaxis.grid(True, alpha=0.3, linestyle="--")
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.4)
    fig.savefig(figures / "macro_precision_recall.png", dpi=300, bbox_inches="tight")
    fig.savefig(
        figures / "macro_precision_recall.pdf",
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)

    row_support = cnn_confusion.sum(axis=1, keepdims=True)
    normalized = np.divide(
        cnn_confusion,
        row_support,
        out=np.zeros_like(cnn_confusion, dtype=float),
        where=row_support > 0,
    )
    fig, axis = plt.subplots(figsize=(7.2, 6.2))
    image = axis.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
    axis.set_xticks(np.arange(len(classes)), labels=classes, rotation=45, ha="right", fontsize=7)
    axis.set_yticks(np.arange(len(classes)), labels=classes, fontsize=7)
    axis.set_xlabel("Predicted category", fontsize=9)
    axis.set_ylabel("True category", fontsize=9)
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Row-normalized rate")
    fig.tight_layout()
    fig.savefig(figures / "cnn1d_confusion_matrix.png", dpi=300, bbox_inches="tight")
    fig.savefig(
        figures / "cnn1d_confusion_matrix.pdf",
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)


def _latex_number(value: float) -> str:
    return f"{value:.4f}"


def _write_latex_outputs(
    output_dir: Path,
    *,
    metrics: pd.DataFrame,
    intervals: pd.DataFrame,
    pair_count: int,
) -> None:
    latex_dir = output_dir / "latex"
    latex_dir.mkdir()
    interval_lookup = intervals.set_index(["model_id", "metric"])
    best = {metric: metrics[metric].max() for metric in INTERVAL_METRICS}
    rows: list[str] = []
    for _, result in metrics.iterrows():
        cells = [str(result["model"]), str(result["features"])]
        for metric in INTERVAL_METRICS:
            point = float(result[metric])
            interval = interval_lookup.loc[(result["model_id"], metric)]
            formatted = _latex_number(point)
            if np.isclose(point, best[metric], rtol=0.0, atol=1e-12):
                formatted = f"\\textbf{{{formatted}}}"
            cells.append(
                f"{formatted} {{\\scriptsize "
                f"[{float(interval['lower']):.4f}, {float(interval['upper']):.4f}]}}"
            )
        rows.append(" & ".join(cells) + r" \\")
    table = "\n".join(
        (
            r"\begin{tabular}{@{}llcccc@{}}",
            r"\toprule",
            r"\textbf{Model} & \textbf{Features} & \textbf{Accuracy} & "
            r"\textbf{Bal. Accuracy} & \textbf{Macro F1} & \textbf{Weighted F1} \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            "",
        )
    )
    (latex_dir / "cross_domain_table.tex").write_text(table, encoding="utf-8")

    prefixes = {
        "rf_matched_flow_stats": "RFFlowFeatures",
        "xgboost_matched_flow_stats": "XGBFlowFeatures",
        "rf_flattened_splt": "RFSPLT",
        "xgboost_flattened_splt": "XGBSPLT",
        "cnn1d_sequential_splt": "CNNOneDSPLT",
    }
    metric_suffixes = {
        "accuracy": "Accuracy",
        "balanced_accuracy": "BalancedAccuracy",
        "macro_f1": "MacroFOne",
        "weighted_f1": "WeightedFOne",
        "macro_average_precision": "MacroAP",
    }
    commands = [f"\\newcommand{{\\SampleCount}}{{{pair_count:,}}}"]
    for _, result in metrics.iterrows():
        for metric, suffix in metric_suffixes.items():
            commands.append(
                f"\\newcommand{{\\{prefixes[result['model_id']]}{suffix}}}"
                f"{{{float(result[metric]):.4f}}}"
            )
    (latex_dir / "results_macros.tex").write_text("\n".join(commands) + "\n", encoding="utf-8")


def analyze_runs(config: AnalysisConfig, *, force: bool = False) -> dict[str, Any]:
    """Aggregate five validated runs into tables, figures, and LaTeX artifacts."""
    provenance = git_provenance(config.training.dataset.project_root)
    if not provenance.get("status_available") or provenance.get("dirty") is not False:
        raise PipelineInvariantError("Analysis requires a clean committed Git revision")
    if config.output_dir.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite analysis output: {config.output_dir}")
    runs, prediction_frames, classes = _load_runs(config)
    temporary = config.output_dir.parent / f".{config.output_dir.name}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir(parents=True)
    try:
        metric_rows: list[dict[str, Any]] = []
        interval_rows: list[dict[str, Any]] = []
        per_class_rows: list[dict[str, Any]] = []
        curve_rows: list[dict[str, Any]] = []
        confusion_matrices: dict[str, np.ndarray] = {}
        for model_id in config.training.model_ids:
            predictions = prediction_frames[model_id]
            true_labels = predictions["true_index"].to_numpy(dtype=np.int64)
            predicted_labels = predictions["predicted_index"].to_numpy(dtype=np.int64)
            probabilities = predictions.loc[
                :, _probability_columns(len(classes))
            ].to_numpy(dtype=np.float64)
            metric_values, confusion = classification_metrics(
                true_labels,
                predicted_labels,
                probabilities,
                class_count=len(classes),
            )
            bootstrap = paired_bootstrap_confusion_metrics(
                confusion,
                replicates=config.bootstrap_replicates,
                confidence_level=config.confidence_level,
                seed=config.bootstrap_seed,
            )
            model_label, features = MODEL_LABELS[model_id]
            metric_rows.append(
                {"model_id": model_id, "model": model_label, "features": features, **metric_values}
            )
            for metric in INTERVAL_METRICS:
                interval_rows.append(
                    {
                        "model_id": model_id,
                        "model": model_label,
                        "features": features,
                        "metric": metric,
                        "point": metric_values[metric],
                        "bootstrap_mean": bootstrap.mean[metric],
                        "lower": bootstrap.lower[metric],
                        "upper": bootstrap.upper[metric],
                        "standard_deviation": bootstrap.standard_deviation[metric],
                    }
                )
            per_class_rows.extend(_per_class_rows(model_id, confusion, classes))
            recall, precision = _macro_precision_recall(
                true_labels,
                probabilities,
                class_count=len(classes),
                grid_points=config.precision_recall_grid_points,
            )
            curve_rows.extend(
                {
                    "model_id": model_id,
                    "recall": float(recall[index]),
                    "precision": float(precision[index]),
                }
                for index in range(len(recall))
            )
            confusion_matrices[model_id] = confusion

        metrics = pd.DataFrame(metric_rows)
        intervals = pd.DataFrame(interval_rows)
        per_class = pd.DataFrame(per_class_rows)
        curves = pd.DataFrame(curve_rows)
        metrics.to_csv(temporary / "metrics.csv", index=False)
        intervals.to_csv(temporary / "bootstrap_intervals.csv", index=False)
        per_class.to_csv(temporary / "per_class_metrics.csv", index=False)
        curves.to_csv(temporary / "precision_recall_curves.csv", index=False)
        np.savez_compressed(temporary / "confusion_matrices.npz", **confusion_matrices)
        cnn_confusion = confusion_matrices["cnn1d_sequential_splt"]
        _top_confused_pairs(cnn_confusion, classes).to_csv(
            temporary / "cnn1d_confused_pairs.csv", index=False
        )
        _plot_outputs(
            temporary,
            metrics=metrics,
            per_class=per_class,
            curves=curves,
            cnn_confusion=cnn_confusion,
            classes=classes,
            key_models=config.key_per_class_models,
        )
        pair_count = int(next(iter(runs.values()))["pair_count"])
        _write_latex_outputs(
            temporary,
            metrics=metrics,
            intervals=intervals,
            pair_count=pair_count,
        )
        artifact_paths = sorted(
            path
            for path in temporary.rglob("*")
            if path.is_file() and path.name != ANALYSIS_MANIFEST_FILENAME
        )
        manifest = {
            "schema_version": 1,
            "created_utc": datetime.now(UTC).isoformat(),
            "package_version": __version__,
            "git": provenance,
            "pair_count": pair_count,
            "class_order": list(classes),
            "model_order": list(config.training.model_ids),
            "metrics": list(METRICS),
            "bootstrap": {
                "method": "paired pair-level nonparametric bootstrap via multinomial state counts",
                "replicates": config.bootstrap_replicates,
                "confidence_level": config.confidence_level,
                "seed": config.bootstrap_seed,
                "interval_metrics": list(config.interval_metrics),
                "average_precision_interval": False,
            },
            "run_inputs": {
                model_id: {
                    "run_json_sha256": sha256_file(
                        config.training.output_root / model_id / RUN_FILENAME
                    ),
                    "predictions_sha256": sha256_file(
                        config.training.output_root / model_id / PREDICTIONS_FILENAME
                    ),
                }
                for model_id in config.training.model_ids
            },
            "artifacts": {
                str(path.relative_to(temporary)): {
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
                for path in artifact_paths
            },
        }
        (temporary / ANALYSIS_MANIFEST_FILENAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        validate_analysis_directory(
            temporary,
            expected_models=config.training.model_ids,
            run_root=config.training.output_root,
            bootstrap_replicates=config.bootstrap_replicates,
            confidence_level=config.confidence_level,
            bootstrap_seed=config.bootstrap_seed,
        )
        if config.output_dir.exists():
            shutil.rmtree(config.output_dir)
        temporary.replace(config.output_dir)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def validate_analysis_directory(
    path: Path,
    *,
    expected_models: tuple[str, ...],
    run_root: Path | None = None,
    bootstrap_replicates: int = 1_000,
    confidence_level: float = 0.95,
    bootstrap_seed: int = 42,
) -> dict[str, Any]:
    manifest_path = path / ANALYSIS_MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if tuple(manifest.get("model_order", [])) != expected_models:
        raise PipelineInvariantError("Analysis model order is invalid")
    for relative, metadata in manifest.get("artifacts", {}).items():
        artifact = path / relative
        if not artifact.is_file() or sha256_file(artifact) != metadata.get("sha256"):
            raise PipelineInvariantError(f"Analysis artifact hash mismatch: {relative}")
    required = {
        "metrics.csv",
        "bootstrap_intervals.csv",
        "per_class_metrics.csv",
        "precision_recall_curves.csv",
        "confusion_matrices.npz",
        "cnn1d_confused_pairs.csv",
        "figures/per_class_f1.png",
        "figures/per_class_f1.pdf",
        "figures/macro_precision_recall.png",
        "figures/macro_precision_recall.pdf",
        "figures/cnn1d_confusion_matrix.png",
        "figures/cnn1d_confusion_matrix.pdf",
        "latex/cross_domain_table.tex",
        "latex/results_macros.tex",
    }
    if set(manifest.get("artifacts", {})) != required:
        raise PipelineInvariantError("Analysis artifact inventory is incomplete or unexpected")
    metrics = pd.read_csv(path / "metrics.csv")
    intervals = pd.read_csv(path / "bootstrap_intervals.csv")
    per_class = pd.read_csv(path / "per_class_metrics.csv")
    curves = pd.read_csv(path / "precision_recall_curves.csv")
    if tuple(metrics["model_id"]) != expected_models or len(metrics) != 5:
        raise PipelineInvariantError("Metric table model coverage is invalid")
    if len(intervals) != len(expected_models) * len(INTERVAL_METRICS):
        raise PipelineInvariantError("Bootstrap interval coverage is invalid")
    class_count = len(manifest["class_order"])
    if len(per_class) != len(expected_models) * class_count:
        raise PipelineInvariantError("Per-class metric coverage is invalid")
    if len(curves) != len(expected_models) * 500:
        raise PipelineInvariantError("Precision-recall curve coverage is invalid")
    matrices = np.load(path / "confusion_matrices.npz")
    if set(matrices.files) != set(expected_models):
        raise PipelineInvariantError("Confusion-matrix model coverage is invalid")
    metric_lookup = metrics.set_index("model_id")
    interval_lookup = intervals.set_index(["model_id", "metric"])
    for model_id in expected_models:
        derived = confusion_metrics(matrices[model_id])
        for metric in INTERVAL_METRICS:
            observed = float(metric_lookup.loc[model_id, metric])
            if not np.isclose(observed, derived[metric], rtol=1e-12, atol=1e-12):
                raise PipelineInvariantError(
                    f"Metric table disagrees with confusion matrix: {model_id}/{metric}"
                )
            interval_point = float(interval_lookup.loc[(model_id, metric), "point"])
            if not np.isclose(observed, interval_point, rtol=1e-12, atol=1e-12):
                raise PipelineInvariantError(
                    f"Bootstrap point estimate disagrees with metrics: {model_id}/{metric}"
                )
    if run_root is not None:
        for model_id in expected_models:
            run_dir = run_root / model_id
            validate_run_directory(run_dir)
            recorded_input = manifest.get("run_inputs", {}).get(model_id, {})
            if recorded_input != {
                "run_json_sha256": sha256_file(run_dir / RUN_FILENAME),
                "predictions_sha256": sha256_file(run_dir / PREDICTIONS_FILENAME),
            }:
                raise PipelineInvariantError(f"Analysis run binding is stale: {model_id}")
            predictions = pd.read_parquet(run_dir / PREDICTIONS_FILENAME)
            class_count = len(manifest["class_order"])
            probability_columns = _probability_columns(class_count)
            metric_values, confusion = classification_metrics(
                predictions["true_index"].to_numpy(dtype=np.int64),
                predictions["predicted_index"].to_numpy(dtype=np.int64),
                predictions.loc[:, probability_columns].to_numpy(dtype=np.float64),
                class_count=class_count,
            )
            if not np.array_equal(confusion, matrices[model_id]):
                raise PipelineInvariantError(
                    f"Stored confusion matrix disagrees with predictions: {model_id}"
                )
            for metric in METRICS:
                if not np.isclose(
                    float(metric_lookup.loc[model_id, metric]),
                    metric_values[metric],
                    rtol=1e-12,
                    atol=1e-12,
                ):
                    raise PipelineInvariantError(
                        f"Metric table disagrees with predictions: {model_id}/{metric}"
                    )
            bootstrap = paired_bootstrap_confusion_metrics(
                confusion,
                replicates=bootstrap_replicates,
                confidence_level=confidence_level,
                seed=bootstrap_seed,
            )
            for metric in INTERVAL_METRICS:
                interval = interval_lookup.loc[(model_id, metric)]
                for column, expected in (
                    ("bootstrap_mean", bootstrap.mean[metric]),
                    ("lower", bootstrap.lower[metric]),
                    ("upper", bootstrap.upper[metric]),
                    ("standard_deviation", bootstrap.standard_deviation[metric]),
                ):
                    if not np.isclose(
                        float(interval[column]), expected, rtol=1e-12, atol=1e-12
                    ):
                        raise PipelineInvariantError(
                            f"Bootstrap artifact disagrees with predictions: "
                            f"{model_id}/{metric}/{column}"
                        )
            expected_per_class = pd.DataFrame(
                _per_class_rows(model_id, confusion, tuple(manifest["class_order"]))
            ).reset_index(drop=True)
            observed_per_class = per_class[per_class["model_id"] == model_id].reset_index(
                drop=True
            )
            observed_identity = list(
                observed_per_class[["class_index", "class_name"]].itertuples(index=False)
            )
            expected_identity = list(
                expected_per_class[["class_index", "class_name"]].itertuples(index=False)
            )
            if observed_identity != expected_identity:
                raise PipelineInvariantError(f"Per-class row identity is invalid: {model_id}")
            for column in ("support", "precision", "recall", "f1"):
                if not np.allclose(
                    observed_per_class[column].to_numpy(dtype=float),
                    expected_per_class[column].to_numpy(dtype=float),
                    rtol=1e-12,
                    atol=1e-12,
                ):
                    raise PipelineInvariantError(
                        f"Per-class metrics disagree with predictions: {model_id}/{column}"
                    )
            expected_recall, expected_precision = _macro_precision_recall(
                predictions["true_index"].to_numpy(dtype=np.int64),
                predictions.loc[:, probability_columns].to_numpy(dtype=np.float64),
                class_count=class_count,
                grid_points=500,
            )
            observed_curve = curves[curves["model_id"] == model_id]
            if not np.allclose(
                observed_curve["recall"].to_numpy(dtype=float),
                expected_recall,
                rtol=1e-12,
                atol=1e-12,
            ) or not np.allclose(
                observed_curve["precision"].to_numpy(dtype=float),
                expected_precision,
                rtol=1e-12,
                atol=1e-12,
            ):
                raise PipelineInvariantError(
                    f"Precision-recall curve disagrees with predictions: {model_id}"
                )
        expected_pairs = _top_confused_pairs(
            matrices["cnn1d_sequential_splt"], tuple(manifest["class_order"])
        ).reset_index(drop=True)
        observed_pairs = pd.read_csv(path / "cnn1d_confused_pairs.csv")
        if not observed_pairs.equals(expected_pairs):
            numeric = ("count", "true_class_support", "error_rate")
            identity = ("true_category", "predicted_category")
            if list(observed_pairs.loc[:, list(identity)].itertuples(index=False)) != list(
                expected_pairs.loc[:, list(identity)].itertuples(index=False)
            ) or not np.allclose(
                observed_pairs.loc[:, list(numeric)].to_numpy(dtype=float),
                expected_pairs.loc[:, list(numeric)].to_numpy(dtype=float),
                rtol=1e-12,
                atol=1e-12,
            ):
                raise PipelineInvariantError("CNN1D confused-pair table is invalid")
    return {
        "valid": True,
        "pair_count": int(manifest["pair_count"]),
        "model_count": len(expected_models),
        "artifact_count": len(required),
    }
