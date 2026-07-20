from __future__ import annotations

import json
import shutil
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from matched_view_eval import __version__
from matched_view_eval.config import DatasetConfig
from matched_view_eval.errors import PipelineInvariantError
from matched_view_eval.hashing import sha256_file
from matched_view_eval.matched_packets import (
    SessionReconstruction,
    load_flow_table,
    reconstruct_session,
)
from matched_view_eval.provenance import git_provenance
from matched_view_eval.schema import CANONICAL_COLUMNS

CANONICAL_FILENAME = "canonical_pairs.parquet"
ASSIGNMENT_AUDIT_FILENAME = "assignment_audit.parquet"
MANIFEST_FILENAME = "dataset_manifest.json"
SCHEMA_VERSION = 1


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _canonical_session(
    flows: pd.DataFrame,
    reconstruction: SessionReconstruction,
    *,
    session: int,
    retained_classes: set[str],
) -> pd.DataFrame:
    selected = flows.loc[
        flows["matched_packets"].gt(0)
        & flows["application_category_name"].astype(str).isin(retained_classes)
    ].copy()
    selected = selected.merge(
        reconstruction.features, on="flow_id", how="left", validate="one_to_one"
    ).reset_index(drop=True)
    if selected["inner_packets"].isna().any():
        raise PipelineInvariantError("Eligible flows are missing reconstructed packet features")

    result = pd.DataFrame(
        {
            "pair_id": "s" + str(session) + ":" + selected["id"].astype(str),
            "session": np.full(len(selected), session, dtype=np.int8),
            "source_id": selected["id"].astype(np.int64),
            "source_flow_id": selected["flow_id"].astype(np.int64),
            "application_name": selected["application_name"].astype(str),
            "application_category": selected["application_category_name"].astype(str),
        }
    )
    for column in (column for column in CANONICAL_COLUMNS if column not in result):
        result[column] = selected[column].to_numpy()
    if result["pair_id"].duplicated().any():
        raise PipelineInvariantError("Canonical pair IDs are not unique within a session")
    return result.loc[:, CANONICAL_COLUMNS]


def _input_hashes(config: DatasetConfig) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for session in sorted(config.flow_files):
        for kind, path in (
            ("flows", config.flow_files[session]),
            ("packet_matches", config.packet_match_files[session]),
        ):
            result[f"session_{session}_{kind}"] = {
                "path": str(path),
                "sha256": sha256_file(path),
            }
    return result


def _validate_expected_input_hashes(
    config: DatasetConfig, observed: dict[str, dict[str, str]]
) -> None:
    for key, expected in config.expected_input_hashes.items():
        actual = observed.get(key, {}).get("sha256")
        if actual != expected:
            raise PipelineInvariantError(
                f"Public input SHA-256 mismatch for {key}: expected {expected}, observed {actual}"
            )


def build_canonical_dataset(config: DatasetConfig, *, force: bool = False) -> dict[str, Any]:
    """Rebuild both views from public matched packet pairs under one convention."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    canonical_path = config.output_dir / CANONICAL_FILENAME
    audit_path = config.output_dir / ASSIGNMENT_AUDIT_FILENAME
    manifest_path = config.output_dir / MANIFEST_FILENAME
    outputs = (canonical_path, audit_path, manifest_path)
    existing = [path for path in outputs if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "Refusing to overwrite existing artifacts: " + ", ".join(map(str, existing))
        )
    input_hashes = _input_hashes(config)
    _validate_expected_input_hashes(config, input_hashes)

    work_root = config.output_dir / ".build_work"
    if work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir()

    flows_by_session: dict[int, pd.DataFrame] = {}
    reconstructions: dict[int, SessionReconstruction] = {}
    class_support: Counter[str] = Counter()
    excluded_by_session: dict[int, int] = {}
    input_rows: dict[int, int] = {}
    try:
        for session in sorted(config.flow_files):
            flows = load_flow_table(config.flow_files[session])
            flows_by_session[session] = flows
            eligible = flows["matched_packets"].gt(0)
            class_support.update(flows.loc[eligible, "application_category_name"].astype(str))
            input_rows[session] = len(flows)
            excluded_by_session[session] = int((~eligible).sum())
            reconstructions[session] = reconstruct_session(
                config.flow_files[session],
                config.packet_match_files[session],
                work_root / f"session_{session}",
                maximum_length=config.maximum_sequence_length,
                batch_size=config.packet_batch_size,
                partitions=config.aggregation_partitions,
                padding_ms=config.assignment_padding_ms,
            )

        retained_classes = {
            label for label, support in class_support.items()
            if support >= config.minimum_class_support
        }
        if not retained_classes:
            raise PipelineInvariantError("Class filtering removed every category")

        canonical_parts: list[pd.DataFrame] = []
        audit_parts: list[pd.DataFrame] = []
        for session in sorted(flows_by_session):
            canonical_parts.append(
                _canonical_session(
                    flows_by_session[session],
                    reconstructions[session],
                    session=session,
                    retained_classes=retained_classes,
                )
            )
            audit = reconstructions[session].assignment_audit.copy()
            audit.insert(0, "session", session)
            audit_parts.append(audit)

        canonical = pd.concat(canonical_parts, ignore_index=True).sort_values(
            ["session", "source_id"], ignore_index=True
        )
        assignment_audit = pd.concat(audit_parts, ignore_index=True).sort_values(
            ["session", "source_id"], ignore_index=True
        )
        observed_classes = tuple(sorted(canonical["application_category"].unique()))
        if (
            config.expected_retained_rows is not None
            and len(canonical) != config.expected_retained_rows
        ):
            raise PipelineInvariantError(
                f"Expected {config.expected_retained_rows} retained rows, observed {len(canonical)}"
            )
        if config.expected_classes and observed_classes != tuple(sorted(config.expected_classes)):
            raise PipelineInvariantError(
                f"Retained classes differ from the frozen set: {observed_classes}"
            )

        temporary_canonical = canonical_path.with_suffix(".parquet.tmp")
        temporary_audit = audit_path.with_suffix(".parquet.tmp")
        pq.write_table(
            pa.Table.from_pandas(canonical, preserve_index=False),
            temporary_canonical,
            compression="zstd",
        )
        assignment_audit.to_parquet(temporary_audit, index=False, compression="zstd")
        temporary_canonical.replace(canonical_path)
        temporary_audit.replace(audit_path)
    finally:
        shutil.rmtree(work_root, ignore_errors=True)

    packet_rows_by_session = {
        session: reconstruction.packet_rows
        for session, reconstruction in reconstructions.items()
    }
    reorder_pairs_by_session = {
        session: reconstruction.outer_reordered_adjacent_pairs
        for session, reconstruction in reconstructions.items()
    }
    reorder_flows_by_session = {
        session: reconstruction.outer_reordered_flows
        for session, reconstruction in reconstructions.items()
    }
    checked_by_session = {
        session: int(
            reconstruction.assignment_audit["directional_bytes_fidelity_checked"].sum()
        )
        for session, reconstruction in reconstructions.items()
    }
    equal_by_session = {
        session: int(
            (
                reconstruction.assignment_audit["directional_bytes_fidelity_checked"]
                & reconstruction.assignment_audit["directional_bytes_fidelity_equal"]
            ).sum()
        )
        for session, reconstruction in reconstructions.items()
    }
    retained_rows = len(canonical)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "provenance": {
            "package_version": __version__,
            "git": git_provenance(config.project_root),
        },
        "input_files": input_hashes,
        "configuration": {
            "minimum_class_support": config.minimum_class_support,
            "maximum_sequence_length": config.maximum_sequence_length,
            "primary_prefix_length": config.primary_prefix_length,
            "packet_batch_size": config.packet_batch_size,
            "aggregation_partitions": config.aggregation_partitions,
            "assignment_padding_ms": config.assignment_padding_ms,
            "full_flow_order": "view-specific timestamp and packet index",
            "prefix_membership": "first matched pairs by inner timestamp and packet index",
            "prefix_order": "selected pairs ordered separately in each observed view",
            "statistics": "matched physical packet pairs; sample standard deviation",
        },
        "counts": {
            "input_rows": sum(input_rows.values()),
            "input_rows_by_session": input_rows,
            "packet_match_rows": sum(packet_rows_by_session.values()),
            "packet_match_rows_by_session": packet_rows_by_session,
            "outer_reordered_adjacent_pairs": sum(reorder_pairs_by_session.values()),
            "outer_reordered_adjacent_pairs_by_session": reorder_pairs_by_session,
            "outer_reordered_flows": sum(reorder_flows_by_session.values()),
            "outer_reordered_flows_by_session": reorder_flows_by_session,
            "endpoint_orientation_checked_packets": sum(packet_rows_by_session.values()),
            "directional_bytes_fidelity_checked_flows": sum(checked_by_session.values()),
            "directional_bytes_fidelity_equal_flows": sum(equal_by_session.values()),
            "eligible_rows": sum(input_rows.values()) - sum(excluded_by_session.values()),
            "excluded_no_matched_outer_packets": sum(excluded_by_session.values()),
            "excluded_no_matched_outer_packets_by_session": excluded_by_session,
            "excluded_below_class_support": (
                sum(input_rows.values()) - sum(excluded_by_session.values()) - retained_rows
            ),
            "retained_rows": retained_rows,
            "retained_classes": sorted(retained_classes),
            "class_support_after_eligibility_before_class_filtering": dict(
                sorted(class_support.items())
            ),
        },
        "invariants": {
            "assignment_count_equal_flows": sum(input_rows.values()),
            "assigned_packet_rows_conserved": sum(packet_rows_by_session.values()),
            "prefix_convention_equal_flows": sum(input_rows.values())
            - sum(excluded_by_session.values()),
            "endpoint_orientation_equal_packets": sum(packet_rows_by_session.values()),
            "directional_bytes_fidelity_equal_flows": sum(equal_by_session.values()),
        },
        "artifacts": {
            "canonical_pairs": {
                "path": canonical_path.name,
                "sha256": sha256_file(canonical_path),
            },
            "assignment_audit": {
                "path": audit_path.name,
                "sha256": sha256_file(audit_path),
            },
        },
    }
    _write_json_atomic(manifest_path, payload)
    return payload
