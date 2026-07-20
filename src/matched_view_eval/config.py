from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from matched_view_eval.errors import PipelineInvariantError


@dataclass(frozen=True)
class DatasetConfig:
    project_root: Path
    input_root: Path
    flow_files: dict[int, Path]
    packet_match_files: dict[int, Path]
    expected_input_hashes: dict[str, str]
    output_dir: Path
    minimum_class_support: int
    maximum_sequence_length: int
    primary_prefix_length: int
    packet_batch_size: int
    aggregation_partitions: int
    assignment_padding_ms: float
    expected_retained_rows: int | None
    expected_classes: tuple[str, ...]


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_config(
    config_path: Path,
    *,
    input_root: Path | None = None,
    output_dir: Path | None = None,
) -> DatasetConfig:
    """Load the data and feature configuration."""
    config_path = config_path.expanduser().resolve()
    with config_path.open(encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)

    project_root = config_path.parent.parent
    dataset = raw.get("dataset", {})
    features = raw.get("features", {})
    configured_input = input_root if input_root is not None else dataset.get("input_root")
    configured_output = output_dir if output_dir is not None else dataset.get("output_dir")
    if configured_input is None or configured_output is None:
        raise PipelineInvariantError("dataset.input_root and dataset.output_dir are required")

    root = _resolve(project_root, configured_input)
    sessions = dataset.get("sessions", {})
    flow_files = {int(key): root / str(value["flows"]) for key, value in sessions.items()}
    packet_match_files = {
        int(key): root / str(value["packet_matches"]) for key, value in sessions.items()
    }
    expected_input_hashes = {
        f"session_{int(key)}_{kind}": str(value[f"{kind}_sha256"])
        for key, value in sessions.items()
        for kind in ("flows", "packet_matches")
        if value.get(f"{kind}_sha256")
    }
    if set(flow_files) != {1, 2} or flow_files.keys() != packet_match_files.keys():
        raise PipelineInvariantError("Exactly sessions 1 and 2 must be configured")

    expected_rows = dataset.get("expected_retained_rows")
    config = DatasetConfig(
        project_root=project_root,
        input_root=root,
        flow_files=flow_files,
        packet_match_files=packet_match_files,
        expected_input_hashes=expected_input_hashes,
        output_dir=_resolve(project_root, configured_output),
        minimum_class_support=int(dataset.get("minimum_class_support", 200)),
        maximum_sequence_length=int(dataset.get("maximum_sequence_length", 80)),
        primary_prefix_length=int(features.get("prefix_length", 50)),
        packet_batch_size=int(dataset.get("packet_batch_size", 500_000)),
        aggregation_partitions=int(dataset.get("aggregation_partitions", 64)),
        assignment_padding_ms=float(dataset.get("assignment_padding_ms", 2_000.0)),
        expected_retained_rows=int(expected_rows) if expected_rows is not None else None,
        expected_classes=tuple(str(value) for value in dataset.get("expected_classes", [])),
    )
    _validate_config(config)
    return config


def _validate_config(config: DatasetConfig) -> None:
    if config.minimum_class_support < 1:
        raise PipelineInvariantError("minimum_class_support must be positive")
    if config.maximum_sequence_length < 1:
        raise PipelineInvariantError("maximum_sequence_length must be positive")
    if not 1 <= config.primary_prefix_length <= config.maximum_sequence_length:
        raise PipelineInvariantError(
            "features.prefix_length must be within maximum_sequence_length"
        )
    if config.packet_batch_size < 1 or config.aggregation_partitions < 1:
        raise PipelineInvariantError("batch size and partition count must be positive")
    if config.assignment_padding_ms < 0:
        raise PipelineInvariantError("assignment_padding_ms must be nonnegative")
    if config.expected_retained_rows is not None and config.expected_retained_rows < 1:
        raise PipelineInvariantError("expected_retained_rows must be positive")
    if config.expected_classes and len(config.expected_classes) != len(
        set(config.expected_classes)
    ):
        raise PipelineInvariantError("expected_classes must be unique")
    for name, digest in config.expected_input_hashes.items():
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise PipelineInvariantError(f"Invalid SHA-256 configured for {name}")
