from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from matched_view_eval.analysis import validate_analysis_directory
from matched_view_eval.errors import PipelineInvariantError
from matched_view_eval.hashing import sha256_file
from matched_view_eval.training import RUN_FILENAME, required_run_artifacts

RELEASE_MANIFEST_FILENAME = "release_manifest.json"
PUBLISHABLE_MODEL_OMISSIONS = {
    "rf_matched_flow_stats": frozenset({"model.joblib"}),
    "rf_flattened_splt": frozenset({"model.joblib"}),
    "xgboost_matched_flow_stats": frozenset({"model.json"}),
    "xgboost_flattened_splt": frozenset({"model.json"}),
    "cnn1d_sequential_splt": frozenset(),
}


def _validate_file(path: Path, metadata: dict[str, Any], *, label: str) -> None:
    if not path.is_file() or sha256_file(path) != metadata.get("sha256"):
        raise PipelineInvariantError(f"Release artifact hash mismatch: {label}")
    if path.stat().st_size != metadata.get("bytes"):
        raise PipelineInvariantError(f"Release artifact size mismatch: {label}")


def validate_release_directory(
    path: Path,
    *,
    expected_models: tuple[str, ...],
    bootstrap_replicates: int = 1_000,
    confidence_level: float = 0.95,
    bootstrap_seed: int = 42,
) -> dict[str, Any]:
    """Validate the prediction-focused evidence shipped in the Git repository."""
    manifest_path = path / RELEASE_MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise PipelineInvariantError("Release manifest schema is invalid")
    campaign_revision = str(manifest.get("campaign_revision", ""))
    if manifest.get("release_id") != campaign_revision[:7]:
        raise PipelineInvariantError("Release id does not match the campaign revision")
    if tuple(manifest.get("model_order", [])) != expected_models:
        raise PipelineInvariantError("Release model order is invalid")

    root_artifacts = manifest.get("root_artifacts", {})
    required_root = {"campaign.log", "campaign.exit"}
    if set(root_artifacts) != required_root:
        raise PipelineInvariantError("Release root artifact inventory is invalid")
    for relative, metadata in root_artifacts.items():
        _validate_file(path / relative, metadata, label=relative)
    exit_status = int((path / "campaign.exit").read_text().strip())
    if exit_status != 0 or manifest.get("campaign_exit_status") != 0:
        raise PipelineInvariantError("Released campaign did not complete successfully")

    run_root = path / "runs"
    omissions = manifest.get("omitted_run_artifacts", {})
    if set(omissions) != set(expected_models):
        raise PipelineInvariantError("Release omission inventory has invalid model coverage")
    allowed_missing: dict[str, frozenset[str]] = {}
    canonical_hash = str(manifest.get("canonical_sha256", ""))
    for model_id in expected_models:
        run_path = run_root / model_id / RUN_FILENAME
        if not run_path.is_file():
            raise FileNotFoundError(run_path)
        run = json.loads(run_path.read_text(encoding="utf-8"))
        declared = omissions[model_id]
        if not isinstance(declared, dict):
            raise PipelineInvariantError(f"Invalid release omissions for {model_id}")
        if set(declared) != PUBLISHABLE_MODEL_OMISSIONS[model_id]:
            raise PipelineInvariantError(f"Invalid release omissions for {model_id}")
        allowed_missing[model_id] = frozenset(declared)
        if set(run.get("artifacts", {})) != required_run_artifacts(model_id):
            raise PipelineInvariantError(f"Run artifact inventory is invalid: {model_id}")
        for name, metadata in declared.items():
            if (run_root / model_id / name).exists():
                raise PipelineInvariantError(
                    f"Declared release omission is present: {model_id}/{name}"
                )
            if metadata != run["artifacts"][name]:
                raise PipelineInvariantError(f"Omission metadata is stale: {model_id}/{name}")
        if run.get("git", {}).get("revision") != campaign_revision:
            raise PipelineInvariantError(f"Run revision mismatch: {model_id}")
        if run.get("git", {}).get("dirty") is not False:
            raise PipelineInvariantError(f"Run provenance is dirty: {model_id}")
        if run.get("input_hashes", {}).get("canonical") != canonical_hash:
            raise PipelineInvariantError(f"Canonical hash mismatch: {model_id}")

    analysis_path = path / "analysis"
    analysis_manifest = json.loads(
        (analysis_path / "analysis_manifest.json").read_text(encoding="utf-8")
    )
    if analysis_manifest.get("git", {}).get("revision") != campaign_revision:
        raise PipelineInvariantError("Analysis revision differs from the campaign revision")
    result = validate_analysis_directory(
        analysis_path,
        expected_models=expected_models,
        run_root=run_root,
        bootstrap_replicates=bootstrap_replicates,
        confidence_level=confidence_level,
        bootstrap_seed=bootstrap_seed,
        allowed_missing_run_artifacts=allowed_missing,
    )
    return {
        "valid": True,
        "release_id": manifest["release_id"],
        "campaign_revision": campaign_revision,
        "canonical_sha256": canonical_hash,
        "pair_count": result["pair_count"],
        "model_count": result["model_count"],
        "analysis_artifact_count": result["artifact_count"],
        "omitted_model_artifact_count": sum(len(values) for values in allowed_missing.values()),
    }
