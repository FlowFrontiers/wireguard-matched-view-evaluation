from __future__ import annotations

import json
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from matched_view_eval.config import DatasetConfig
from matched_view_eval.data import (
    ASSIGNMENT_AUDIT_FILENAME,
    CANONICAL_FILENAME,
    MANIFEST_FILENAME,
)
from matched_view_eval.errors import PipelineInvariantError
from matched_view_eval.features import build_prefix_stats
from matched_view_eval.hashing import sha256_file
from matched_view_eval.schema import (
    ABSOLUTE_TIMESTAMP_NAMES,
    CANONICAL_COLUMNS,
    CANONICAL_STATS,
    STAT_COLUMNS,
)


def _validate_manifest(
    config: DatasetConfig,
    *,
    canonical_path: Any,
    audit_path: Any,
    manifest_path: Any,
    audit: pd.DataFrame,
) -> dict[str, Any]:
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    for name, path in (("canonical_pairs", canonical_path), ("assignment_audit", audit_path)):
        expected = manifest.get("artifacts", {}).get(name, {}).get("sha256")
        observed = sha256_file(path)
        if expected != observed:
            raise PipelineInvariantError(
                f"SHA-256 mismatch for {name}: expected {expected}, observed {observed}"
            )
    for session in sorted(config.flow_files):
        for kind, path in (
            ("flows", config.flow_files[session]),
            ("packet_matches", config.packet_match_files[session]),
        ):
            key = f"session_{session}_{kind}"
            expected = manifest.get("input_files", {}).get(key, {}).get("sha256")
            observed = sha256_file(path)
            if expected != observed:
                raise PipelineInvariantError(
                    f"Input SHA-256 mismatch for {key}: expected {expected}, observed {observed}"
                )
            frozen = config.expected_input_hashes.get(key)
            if frozen is not None and frozen != observed:
                raise PipelineInvariantError(
                    f"Configured public SHA-256 mismatch for {key}: "
                    f"expected {frozen}, observed {observed}"
                )

    invariants = manifest.get("invariants", {})
    assigned = int(audit["reproduced_matched_packets"].sum())
    checked = audit["directional_bytes_fidelity_checked"].astype(bool)
    equal = audit["directional_bytes_fidelity_equal"].astype(bool)
    expected_counts = {
        "assignment_count_equal_flows": len(audit),
        "assigned_packet_rows_conserved": assigned,
        "prefix_convention_equal_flows": int(audit["released_matched_packets"].gt(0).sum()),
        "endpoint_orientation_equal_packets": assigned,
        "directional_bytes_fidelity_equal_flows": int((checked & equal).sum()),
    }
    for name, expected in expected_counts.items():
        if invariants.get(name) != expected:
            raise PipelineInvariantError(
                f"Manifest invariant {name} is {invariants.get(name)}, expected {expected}"
            )
    return manifest


def validate_dataset(config: DatasetConfig) -> dict[str, Any]:
    """Validate canonical data, source hashes, and assignment evidence in bounded memory."""
    canonical_path = config.output_dir / CANONICAL_FILENAME
    audit_path = config.output_dir / ASSIGNMENT_AUDIT_FILENAME
    manifest_path = config.output_dir / MANIFEST_FILENAME
    for path in (canonical_path, audit_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    parquet = pq.ParquetFile(canonical_path)
    columns = parquet.schema_arrow.names
    missing = set(CANONICAL_COLUMNS) - set(columns)
    if missing:
        raise PipelineInvariantError(f"Canonical dataset is missing columns: {sorted(missing)}")
    banned = [name for name in columns if any(token in name for token in ABSOLUTE_TIMESTAMP_NAMES)]
    if banned:
        raise PipelineInvariantError(f"Absolute timestamp columns are prohibited: {banned}")

    pair_ids: set[str] = set()
    class_support: Counter[str] = Counter()
    session_support: Counter[int] = Counter()
    row_count = 0
    scan_columns = [
        "pair_id",
        "session",
        "application_category",
        "inner_direction",
        "inner_size",
        "inner_iat_ms",
        "outer_direction",
        "outer_size",
        "outer_iat_ms",
        "inner_length",
        "outer_length",
        *STAT_COLUMNS,
    ]
    for batch in parquet.iter_batches(columns=scan_columns, batch_size=10_000):
        frame = batch.to_pandas()
        if frame["pair_id"].duplicated().any() or pair_ids.intersection(frame["pair_id"]):
            raise PipelineInvariantError("Canonical dataset contains duplicate pair_id values")
        pair_ids.update(frame["pair_id"])
        class_support.update(frame["application_category"].astype(str))
        session_support.update(int(value) for value in frame["session"])
        row_count += len(frame)

        statistics = frame.loc[:, STAT_COLUMNS].apply(pd.to_numeric, errors="coerce")
        statistic_values = statistics.to_numpy(dtype=np.float64)
        if np.isinf(statistic_values).any():
            raise PipelineInvariantError("Canonical statistics contain infinite values")
        finite_values = statistic_values[np.isfinite(statistic_values)]
        if (finite_values < 0).any():
            raise PipelineInvariantError("Canonical statistics contain negative values")
        for sentinel in (1e9, 2e9):
            if np.isclose(finite_values, sentinel, rtol=0.0, atol=1e-6).any():
                raise PipelineInvariantError(
                    f"Canonical statistics contain suspect sentinel value {sentinel:g}"
                )

        for domain in ("inner", "outer"):
            duration_seconds = statistics[f"{domain}_duration_ms"] / 1000.0
            valid_duration = duration_seconds.where(duration_seconds > 0)
            for rate_name, numerator in (
                ("packet_rate", "packets"),
                ("byte_rate", "bytes"),
            ):
                expected = statistics[f"{domain}_{numerator}"] / valid_duration
                observed = statistics[f"{domain}_{rate_name}"]
                if not np.isclose(
                    observed.to_numpy(dtype=float),
                    expected.to_numpy(dtype=float),
                    rtol=1e-10,
                    atol=1e-10,
                    equal_nan=True,
                ).all():
                    raise PipelineInvariantError(
                        f"{domain}_{rate_name} is inconsistent with count/duration"
                    )

            lengths = frame[f"{domain}_length"].astype(int)
            if not lengths.between(1, config.maximum_sequence_length).all():
                raise PipelineInvariantError(f"{domain} sequence lengths are outside bounds")
            for channel in ("direction", "size", "iat_ms"):
                if not frame[f"{domain}_{channel}"].map(len).equals(lengths):
                    raise PipelineInvariantError(
                        f"{domain}_{channel} lengths disagree with {domain}_length"
                    )
                for sequence in frame[f"{domain}_{channel}"]:
                    values = np.asarray(sequence, dtype=float)
                    if not np.isfinite(values).all():
                        raise PipelineInvariantError(f"{domain}_{channel} is non-finite")
                    if channel == "direction" and not np.isin(values, (0, 1)).all():
                        raise PipelineInvariantError(f"{domain}_direction is outside {{0,1}}")
                    if channel != "direction" and (values < 0).any():
                        raise PipelineInvariantError(f"{domain}_{channel} is negative")

        if not frame["inner_length"].astype(int).equals(frame["outer_length"].astype(int)):
            raise PipelineInvariantError("Matched-view sequence lengths differ")
        if not np.array_equal(statistics["inner_packets"], statistics["outer_packets"]):
            raise PipelineInvariantError("Matched-view full packet counts differ")
        expected_lengths = np.minimum(
            statistics["inner_packets"].to_numpy(dtype=int), config.maximum_sequence_length
        )
        if not np.array_equal(expected_lengths, frame["inner_length"].to_numpy(dtype=int)):
            raise PipelineInvariantError("Sequence lengths disagree with matched packet counts")

        complete = statistics["inner_packets"].le(config.maximum_sequence_length)
        if complete.any():
            complete_frame = frame.loc[complete]
            for domain in ("inner", "outer"):
                rebuilt = build_prefix_stats(
                    complete_frame,
                    domain=domain,
                    prefix_length=config.maximum_sequence_length,
                ).values
                observed = statistics.loc[
                    complete, [f"{domain}_{name}" for name in CANONICAL_STATS]
                ].to_numpy(dtype=float)
                if not np.isclose(
                    observed, rebuilt, rtol=1e-10, atol=1e-8, equal_nan=True
                ).all():
                    raise PipelineInvariantError(
                        f"{domain} full-flow statistics disagree with sequence conventions"
                    )

    if any(count < config.minimum_class_support for count in class_support.values()):
        raise PipelineInvariantError("A retained class violates minimum support")
    if config.expected_retained_rows is not None and row_count != config.expected_retained_rows:
        raise PipelineInvariantError(
            f"Expected {config.expected_retained_rows} rows, observed {row_count}"
        )
    observed_classes = tuple(sorted(class_support))
    if config.expected_classes and observed_classes != tuple(sorted(config.expected_classes)):
        raise PipelineInvariantError(f"Unexpected retained classes: {observed_classes}")

    audit = pd.read_parquet(audit_path)
    required_audit = {
        "session",
        "source_id",
        "flow_id",
        "released_matched_packets",
        "reproduced_matched_packets",
        "counts_equal",
        "directional_bytes_fidelity_checked",
        "directional_bytes_fidelity_equal",
    }
    if required_audit - set(audit.columns):
        raise PipelineInvariantError("Assignment audit schema is incomplete")
    equal_counts = (
        audit["released_matched_packets"].astype(int)
        == audit["reproduced_matched_packets"].astype(int)
    )
    if not equal_counts.all() or not audit["counts_equal"].astype(bool).all():
        raise PipelineInvariantError("Assignment audit contains count mismatches")
    checked = audit["directional_bytes_fidelity_checked"].astype(bool)
    equal = audit["directional_bytes_fidelity_equal"].astype(bool)
    if not equal.loc[checked].all():
        raise PipelineInvariantError("Assignment audit contains directional-byte failures")
    manifest = _validate_manifest(
        config,
        canonical_path=canonical_path,
        audit_path=audit_path,
        manifest_path=manifest_path,
        audit=audit,
    )
    return {
        "status": "valid",
        "rows": row_count,
        "classes": len(class_support),
        "sessions": dict(sorted(session_support.items())),
        "assignment_rows": len(audit),
        "assigned_packets": int(audit["reproduced_matched_packets"].sum()),
        "directional_bytes_fidelity_checked": int(checked.sum()),
        "canonical_sha256": manifest["artifacts"]["canonical_pairs"]["sha256"],
    }
