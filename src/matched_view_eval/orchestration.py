from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from matched_view_eval.analysis import analyze_runs, validate_analysis_directory
from matched_view_eval.analysis_config import AnalysisConfig
from matched_view_eval.errors import PipelineInvariantError
from matched_view_eval.provenance import git_provenance
from matched_view_eval.training import validate_run_directory


def preflight(config: AnalysisConfig) -> dict[str, Any]:
    """Report campaign state without importing any model framework."""
    provenance = git_provenance(config.training.dataset.project_root)
    if not provenance.get("status_available") or provenance.get("dirty") is not False:
        raise PipelineInvariantError("Campaign execution requires a clean committed Git revision")
    runs: list[dict[str, str]] = []
    for model_id in config.training.model_ids:
        path = config.training.output_root / model_id
        if not path.exists():
            status = "pending"
        else:
            validate_run_directory(path)
            status = "complete"
        runs.append({"model_id": model_id, "status": status, "path": str(path)})
    if not config.output_dir.exists():
        analysis_status = "pending"
    else:
        validate_analysis_directory(
            config.output_dir,
            expected_models=config.training.model_ids,
            run_root=config.training.output_root,
        )
        analysis_status = "complete"
    return {
        "git": provenance,
        "runs": runs,
        "completed_runs": sum(run["status"] == "complete" for run in runs),
        "pending_runs": sum(run["status"] == "pending" for run in runs),
        "analysis_status": analysis_status,
    }


def _training_command(
    config_path: Path,
    config: AnalysisConfig,
    *,
    model_id: str,
    device: str,
    force: bool,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "matched_view_eval.cli",
        "train-model",
        "--config",
        str(config_path),
        "--input-root",
        str(config.training.dataset.input_root),
        "--artifact-dir",
        str(config.training.dataset.output_dir),
        "--output-root",
        str(config.training.output_root),
        "--model",
        model_id,
        "--device",
        device,
    ]
    if force:
        command.append("--force")
    return command


def run_all(
    config_path: Path,
    config: AnalysisConfig,
    *,
    device: str,
    force: bool = False,
) -> dict[str, Any]:
    """Run five models in isolated processes, then publish the analysis."""
    initial = preflight(config)
    executed: list[str] = []
    reused: list[str] = []
    status_by_model = {entry["model_id"]: entry["status"] for entry in initial["runs"]}
    for model_id in config.training.model_ids:
        if status_by_model[model_id] == "complete" and not force:
            reused.append(model_id)
            continue
        command = _training_command(
            config_path,
            config,
            model_id=model_id,
            device=device,
            force=force,
        )
        result = subprocess.run(command, check=False, text=True, capture_output=True)
        if result.returncode != 0:
            raise PipelineInvariantError(
                f"Training subprocess failed for {model_id} (exit {result.returncode})\n"
                f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )
        validate_run_directory(config.training.output_root / model_id)
        executed.append(model_id)
    if config.output_dir.exists() and not force:
        analysis = validate_analysis_directory(
            config.output_dir,
            expected_models=config.training.model_ids,
            run_root=config.training.output_root,
        )
        analysis_action = "reused"
    else:
        analyze_runs(config, force=force)
        analysis = validate_analysis_directory(
            config.output_dir,
            expected_models=config.training.model_ids,
            run_root=config.training.output_root,
        )
        analysis_action = "executed"
    return {
        "executed_models": executed,
        "reused_models": reused,
        "analysis_action": analysis_action,
        "analysis": analysis,
        "final": preflight(config),
    }
