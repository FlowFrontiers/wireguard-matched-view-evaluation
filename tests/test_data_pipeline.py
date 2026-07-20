from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from matched_view_eval.config import DatasetConfig
from matched_view_eval.data import (
    ASSIGNMENT_AUDIT_FILENAME,
    CANONICAL_FILENAME,
    build_canonical_dataset,
)
from matched_view_eval.errors import PipelineInvariantError
from matched_view_eval.feature_audit import FEATURE_AUDIT_FILENAME, audit_features
from matched_view_eval.validation import validate_dataset


def _key(src: str, dst: str, protocol: int, sport: int, dport: int) -> str:
    return f"{src}|{dst}|{protocol}|{sport}|{dport}"


def _write_session(root: Path, session: int, rows_per_class: int = 20) -> tuple[Path, Path]:
    session_root = root / f"session{session}"
    session_root.mkdir()
    flow_rows = []
    packet_rows = []
    row_id = 0
    packet_id = 0
    for category in ("Web", "Network"):
        for _ in range(rows_per_class):
            inside = f"10.{session}.0.{row_id + 1}"
            outside = "192.0.2.1"
            externally_initiated = row_id == 0
            src, dst = (
                (outside, inside) if externally_initiated else (inside, outside)
            )
            sport = 10_000 + row_id
            dport = 443
            base_ms = row_id * 100.0
            flow_rows.append(
                {
                    "id": row_id,
                    "flow_id": row_id,
                    "application_name": f"App-{category}",
                    "application_category_name": category,
                    "src_ip": src,
                    "dst_ip": dst,
                    "flow_start_ms": base_ms,
                    "flow_end_ms": base_ms + 20.0,
                    "k5_fwd": _key(src, dst, 17, sport, dport),
                    "k5_rev": _key(dst, src, 17, dport, sport),
                    "matched_packets": 3,
                    "bidirectional_packets": 3,
                    "src2dst_bytes": 400,
                    "dst2src_bytes": 200,
                }
            )
            for offset_ms, flow_direction, inner_size, outer_size in (
                (0.0, 0, 100, 112),
                (5.0, 1, 200, 208),
                (12.0, 0, 300, 304),
            ):
                if flow_direction == 0:
                    psrc, pdst, psport, pdport = src, dst, sport, dport
                else:
                    psrc, pdst, psport, pdport = dst, src, dport, sport
                capture_direction = "OUTBOUND" if psrc == inside else "INBOUND"
                packet_rows.append(
                    {
                        "direction": capture_direction,
                        "inner_idx": packet_id,
                        "inner_time": (base_ms + offset_ms) / 1000.0,
                        "inner_src": psrc,
                        "inner_dst": pdst,
                        "inner_proto": 17,
                        "inner_sport": psport,
                        "inner_dport": pdport,
                        "inner_length": inner_size,
                        "outer_idx": packet_id,
                        "outer_time": (base_ms + offset_ms + 0.25) / 1000.0,
                        "outer_padded_length": outer_size,
                    }
                )
                packet_id += 1
            row_id += 1

    flow_rows.append(
        {
            "id": row_id,
            "flow_id": row_id,
            "application_name": "App-Web",
            "application_category_name": "Web",
            "src_ip": "198.51.100.1",
            "dst_ip": "192.0.2.1",
            "flow_start_ms": 10_000.0,
            "flow_end_ms": 10_020.0,
            "k5_fwd": _key("198.51.100.1", "192.0.2.1", 17, 50_000 + session, 443),
            "k5_rev": _key("192.0.2.1", "198.51.100.1", 17, 443, 50_000 + session),
            "matched_packets": 0,
            "bidirectional_packets": 1,
            "src2dst_bytes": 1,
            "dst2src_bytes": 0,
        }
    )
    flow_path = session_root / f"session{session}_flows.parquet"
    packet_path = session_root / f"session{session}_packet_matches.parquet"
    pd.DataFrame(flow_rows).to_parquet(flow_path, index=False)
    pd.DataFrame(packet_rows).to_parquet(packet_path, index=False)
    return flow_path, packet_path


def _config(tmp_path: Path) -> DatasetConfig:
    input_root = tmp_path / "raw"
    output_dir = tmp_path / "processed"
    input_root.mkdir()
    flow_files = {}
    packet_files = {}
    for session in (1, 2):
        flow_files[session], packet_files[session] = _write_session(input_root, session)
    return DatasetConfig(
        project_root=tmp_path,
        input_root=input_root,
        flow_files=flow_files,
        packet_match_files=packet_files,
        expected_input_hashes={},
        output_dir=output_dir,
        minimum_class_support=10,
        maximum_sequence_length=5,
        primary_prefix_length=3,
        packet_batch_size=17,
        aggregation_partitions=4,
        assignment_padding_ms=2_000.0,
        expected_retained_rows=80,
        expected_classes=("Network", "Web"),
    )


def test_build_and_validate_matched_pair_dataset(tmp_path: Path) -> None:
    config = _config(tmp_path)
    manifest = build_canonical_dataset(config)
    report = validate_dataset(config)
    feature_report = audit_features(config)

    canonical = pd.read_parquet(config.output_dir / CANONICAL_FILENAME)
    audit = pd.read_parquet(config.output_dir / ASSIGNMENT_AUDIT_FILENAME)
    assert report["status"] == "valid"
    assert report["rows"] == 80
    assert report["assignment_rows"] == 82
    assert report["assigned_packets"] == 240
    assert manifest["schema_version"] == 1
    assert manifest["counts"]["retained_rows"] == 80
    assert manifest["counts"]["input_rows"] == 82
    assert manifest["counts"]["packet_match_rows"] == 240
    assert manifest["counts"]["excluded_no_matched_outer_packets"] == 2
    assert canonical["pair_id"].is_unique
    assert canonical["inner_direction"].map(len).eq(3).all()
    assert canonical["inner_direction"].equals(canonical["outer_direction"])
    assert (canonical["inner_packets"] == canonical["outer_packets"]).all()
    assert audit["counts_equal"].all()
    assert audit.loc[
        audit["directional_bytes_fidelity_checked"],
        "directional_bytes_fidelity_equal",
    ].all()

    first = canonical.iloc[0]
    assert first["inner_duration_ms"] == pytest.approx(12.0)
    assert first["outer_duration_ms"] == pytest.approx(12.0)
    assert first["inner_packet_rate"] == pytest.approx(250.0)
    assert first["inner_byte_rate"] == pytest.approx(50_000.0)
    assert first["inner_iat_ms"].tolist() == pytest.approx([0.0, 5.0, 7.0])
    assert first["inner_direction"].tolist() == [0, 1, 0]
    assert first["inner_src2dst_bytes"] == 400
    assert first["inner_dst2src_bytes"] == 200
    first_capture = pd.read_parquet(config.packet_match_files[1]).iloc[0]
    assert first_capture["direction"] == "INBOUND"
    assert feature_report["status"] == "valid"
    assert feature_report["canonical"]["rows"] == 80
    assert feature_report["sequential_splt"]["direction_encoding"] == {
        "raw_0": -1,
        "raw_1": 1,
        "padding": 0,
    }
    assert (config.output_dir / FEATURE_AUDIT_FILENAME).is_file()


def test_canonical_artifact_is_byte_deterministic(tmp_path: Path) -> None:
    config = _config(tmp_path)
    first = build_canonical_dataset(config)
    second_config = replace(config, output_dir=tmp_path / "second")
    second = build_canonical_dataset(second_config)
    assert first["artifacts"]["canonical_pairs"]["sha256"] == second["artifacts"][
        "canonical_pairs"
    ]["sha256"]
    assert first["artifacts"]["assignment_audit"]["sha256"] == second["artifacts"][
        "assignment_audit"
    ]["sha256"]


def test_existing_artifacts_are_not_overwritten(tmp_path: Path) -> None:
    config = _config(tmp_path)
    build_canonical_dataset(config)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        build_canonical_dataset(config)


def test_wrong_frozen_public_hash_is_rejected_before_build(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config = replace(
        config,
        expected_input_hashes={"session_1_flows": "0" * 64},
    )
    with pytest.raises(PipelineInvariantError, match="Public input SHA-256 mismatch"):
        build_canonical_dataset(config)


def test_canonical_sentinel_is_rejected_before_hash_validation(tmp_path: Path) -> None:
    config = _config(tmp_path)
    build_canonical_dataset(config)
    path = config.output_dir / CANONICAL_FILENAME
    canonical = pd.read_parquet(path)
    canonical.loc[0, "outer_packet_rate"] = 1e9
    canonical.to_parquet(path, index=False)
    with pytest.raises(PipelineInvariantError, match="sentinel"):
        validate_dataset(config)


def test_source_tampering_invalidates_manifest(tmp_path: Path) -> None:
    config = _config(tmp_path)
    build_canonical_dataset(config)
    source = pd.read_parquet(config.flow_files[1])
    source.loc[0, "application_name"] = "Tampered"
    source.to_parquet(config.flow_files[1], index=False)
    with pytest.raises(PipelineInvariantError, match="Input SHA-256 mismatch"):
        validate_dataset(config)


def test_assignment_count_mismatch_is_rejected(tmp_path: Path) -> None:
    config = _config(tmp_path)
    flows = pd.read_parquet(config.flow_files[1])
    flows.loc[0, "matched_packets"] = 2
    flows.to_parquet(config.flow_files[1], index=False)
    with pytest.raises(PipelineInvariantError, match="assignment counts differ"):
        build_canonical_dataset(config)
