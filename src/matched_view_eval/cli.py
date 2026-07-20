from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from matched_view_eval.config import load_config
from matched_view_eval.data import build_canonical_dataset
from matched_view_eval.feature_audit import audit_features
from matched_view_eval.training_config import MODEL_IDS, load_training_config
from matched_view_eval.validation import validate_dataset

DEFAULT_CONFIG = Path("configs/experiment.yaml")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wgme",
        description="WireGuard matched-view evaluation",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("build-data", "validate-data", "audit-features"):
        child = subparsers.add_parser(command)
        child.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        child.add_argument("--input-root", type=Path)
        child.add_argument("--output-dir", type=Path)
        if command != "validate-data":
            child.add_argument("--force", action="store_true")
    train = subparsers.add_parser("train-model")
    train.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    train.add_argument("--input-root", type=Path)
    train.add_argument("--artifact-dir", type=Path)
    train.add_argument("--output-root", type=Path)
    train.add_argument("--model", required=True, choices=MODEL_IDS)
    train.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto")
    train.add_argument("--force", action="store_true")
    validate_run = subparsers.add_parser("validate-run")
    validate_run.add_argument("path", type=Path)
    for command in ("analyze", "validate-analysis"):
        analyze = subparsers.add_parser(command)
        analyze.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        analyze.add_argument("--input-root", type=Path)
        analyze.add_argument("--artifact-dir", type=Path)
        analyze.add_argument("--run-root", type=Path)
        analyze.add_argument("--analysis-output-dir", type=Path)
        if command == "analyze":
            analyze.add_argument("--force", action="store_true")
    run_all = subparsers.add_parser("run-all")
    run_all.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run_all.add_argument("--input-root", type=Path)
    run_all.add_argument("--artifact-dir", type=Path)
    run_all.add_argument("--run-root", type=Path)
    run_all.add_argument("--analysis-output-dir", type=Path)
    run_all.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto")
    run_all.add_argument("--preflight", action="store_true")
    run_all.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "train-model":
        from matched_view_eval.training import train_model

        config = load_training_config(
            args.config,
            input_root=args.input_root,
            artifact_dir=args.artifact_dir,
            output_root=args.output_root,
        )
        result = train_model(
            config,
            model_id=args.model,
            device=args.device,
            force=args.force,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.command == "validate-run":
        from matched_view_eval.training import validate_run_directory

        result = validate_run_directory(args.path.expanduser().resolve())
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.command in {"analyze", "validate-analysis", "run-all"}:
        from matched_view_eval.analysis_config import load_analysis_config

        analysis_config = load_analysis_config(
            args.config,
            input_root=args.input_root,
            artifact_dir=args.artifact_dir,
            run_root=args.run_root,
            analysis_output_dir=args.analysis_output_dir,
        )
        if args.command in {"analyze", "validate-analysis"}:
            from matched_view_eval.analysis import (
                analyze_runs,
                validate_analysis_directory,
            )

            result = (
                analyze_runs(analysis_config, force=args.force)
                if args.command == "analyze"
                else validate_analysis_directory(
                    analysis_config.output_dir,
                    expected_models=analysis_config.training.model_ids,
                    run_root=analysis_config.training.output_root,
                    bootstrap_replicates=analysis_config.bootstrap_replicates,
                    confidence_level=analysis_config.confidence_level,
                    bootstrap_seed=analysis_config.bootstrap_seed,
                )
            )
        else:
            from matched_view_eval.orchestration import preflight, run_all

            result = (
                preflight(analysis_config)
                if args.preflight
                else run_all(
                    args.config.expanduser().resolve(),
                    analysis_config,
                    device=args.device,
                    force=args.force,
                )
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    config = load_config(
        args.config,
        input_root=args.input_root,
        output_dir=args.output_dir,
    )
    if args.command == "build-data":
        result = build_canonical_dataset(config, force=args.force)
    elif args.command == "validate-data":
        result = validate_dataset(config)
    else:
        result = audit_features(config, force=args.force)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
