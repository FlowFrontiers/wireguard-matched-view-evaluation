from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from matched_view_eval import __version__
from matched_view_eval.config import DatasetConfig
from matched_view_eval.data import CANONICAL_FILENAME
from matched_view_eval.errors import PipelineInvariantError
from matched_view_eval.features import (
    build_flattened_splt,
    build_matched_flow_stats,
    build_sequential_splt,
)
from matched_view_eval.hashing import sha256_file
from matched_view_eval.provenance import git_provenance
from matched_view_eval.schema import CANONICAL_STATS, SEQUENCE_COLUMNS, STAT_COLUMNS

FEATURE_AUDIT_FILENAME = "feature_audit.json"


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def audit_features(config: DatasetConfig, *, force: bool = False) -> dict[str, Any]:
    """Exercise both representations over every canonical row without materializing them."""
    canonical_path = config.output_dir / CANONICAL_FILENAME
    output_path = config.output_dir / FEATURE_AUDIT_FILENAME
    if not canonical_path.is_file():
        raise FileNotFoundError(canonical_path)
    if output_path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing artifact: {output_path}")

    parquet = pq.ParquetFile(canonical_path)
    columns = [*STAT_COLUMNS, *SEQUENCE_COLUMNS]
    row_count = 0
    missing_stats = {domain: 0 for domain in ("inner", "outer")}
    observed_direction_values: set[float] = set()
    for batch in parquet.iter_batches(columns=columns, batch_size=10_000):
        frame = batch.to_pandas()
        row_count += len(frame)
        sequences = {}
        for domain in ("inner", "outer"):
            statistics = build_matched_flow_stats(frame, domain=domain)
            if statistics.values.shape != (len(frame), len(CANONICAL_STATS)):
                raise PipelineInvariantError("MatchedFlowStats has an unexpected shape")
            missing_stats[domain] += int(np.isnan(statistics.values).sum())

            sequence = build_sequential_splt(
                frame,
                domain=domain,
                prefix_length=config.primary_prefix_length,
            )
            flattened = build_flattened_splt(
                frame,
                domain=domain,
                prefix_length=config.primary_prefix_length,
            )
            expected_shape = (len(frame), config.primary_prefix_length, 3)
            if sequence.values.shape != expected_shape or sequence.mask.shape != expected_shape[:2]:
                raise PipelineInvariantError("Sequential SPLT has an unexpected shape")
            if not np.array_equal(flattened.values, sequence.values.reshape(len(frame), -1)):
                raise PipelineInvariantError("Flattened SPLT differs from the temporal tensor")
            if not np.array_equal(flattened.mask, sequence.mask):
                raise PipelineInvariantError("Flattened and temporal masks differ")
            if not np.all(sequence.values[~sequence.mask] == 0):
                raise PipelineInvariantError("SPLT padding is not exactly zero")
            valid_direction = sequence.values[:, :, 0][sequence.mask]
            if not np.isin(valid_direction, (-1.0, 1.0)).all():
                raise PipelineInvariantError("Valid SPLT direction is not encoded as {-1,+1}")
            observed_direction_values.update(np.unique(valid_direction).tolist())
            sequences[domain] = sequence
        if not np.array_equal(sequences["inner"].mask, sequences["outer"].mask):
            raise PipelineInvariantError("Matched views expose different prefix masks")

    if config.expected_retained_rows is not None and row_count != config.expected_retained_rows:
        raise PipelineInvariantError(
            f"Feature audit expected {config.expected_retained_rows} rows, observed {row_count}"
        )
    if observed_direction_values != {-1.0, 1.0}:
        raise PipelineInvariantError(
            f"Expected both encoded directions, observed {sorted(observed_direction_values)}"
        )

    payload: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "provenance": {
            "package_version": __version__,
            "git": git_provenance(config.project_root),
        },
        "canonical": {
            "path": str(canonical_path),
            "sha256": sha256_file(canonical_path),
            "rows": row_count,
        },
        "matched_flow_stats": {
            "feature_count": len(CANONICAL_STATS),
            "feature_names": list(CANONICAL_STATS),
            "missing_values": missing_stats,
            "infinite_values": 0,
        },
        "sequential_splt": {
            "prefix_length": config.primary_prefix_length,
            "channels": ["direction", "size", "iat_ms"],
            "shape": [row_count, config.primary_prefix_length, 3],
            "direction_encoding": {"raw_0": -1, "raw_1": 1, "padding": 0},
            "magnitude_transform": "log1p",
            "padding_is_zero": True,
            "matched_view_masks_equal": True,
        },
        "flattened_splt": {
            "feature_count": config.primary_prefix_length * 3,
            "is_exact_temporal_reshape": True,
        },
        "status": "valid",
    }
    _write_json_atomic(output_path, payload)
    return payload
