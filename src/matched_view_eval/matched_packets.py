from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from matched_view_eval.errors import PipelineInvariantError
from matched_view_eval.features import build_prefix_stats
from matched_view_eval.schema import (
    CANONICAL_STATS,
    FLOW_REQUIRED_COLUMNS,
    PACKET_REQUIRED_COLUMNS,
)

ASSIGNED_SCHEMA = pa.schema(
    [
        ("flow_id", pa.int64()),
        ("inner_idx", pa.int64()),
        ("outer_idx", pa.int64()),
        ("direction", pa.int8()),
        ("inner_time_ms", pa.float64()),
        ("outer_time_ms", pa.float64()),
        ("inner_size", pa.float64()),
        ("outer_size", pa.float64()),
    ]
)


@dataclass(frozen=True)
class SessionReconstruction:
    features: pd.DataFrame
    assignment_audit: pd.DataFrame
    packet_rows: int
    outer_reordered_adjacent_pairs: int
    outer_reordered_flows: int


def _check_parquet_columns(path: Path, required: tuple[str, ...]) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    columns = set(pq.ParquetFile(path).schema_arrow.names)
    missing = set(required) - columns
    if missing:
        raise PipelineInvariantError(f"{path} is missing required columns: {sorted(missing)}")


def load_flow_table(path: Path) -> pd.DataFrame:
    _check_parquet_columns(path, FLOW_REQUIRED_COLUMNS)
    frame = pq.read_table(path, columns=list(FLOW_REQUIRED_COLUMNS)).to_pandas()
    numeric = (
        "id",
        "flow_id",
        "flow_start_ms",
        "flow_end_ms",
        "matched_packets",
        "bidirectional_packets",
        "src2dst_bytes",
        "dst2src_bytes",
    )
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame["id"] = frame["id"].astype(np.int64)
    frame["flow_id"] = frame["flow_id"].astype(np.int64)
    frame["matched_packets"] = frame["matched_packets"].astype(np.int64)
    frame["bidirectional_packets"] = frame["bidirectional_packets"].astype(np.int64)
    if frame["id"].duplicated().any() or frame["flow_id"].duplicated().any():
        raise PipelineInvariantError("Released flow IDs must be unique within a session")
    if (frame["matched_packets"] < 0).any():
        raise PipelineInvariantError("Released matched packet counts must be nonnegative")
    return frame


def _packet_keys(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["inner_src"].astype(str)
        + "|"
        + frame["inner_dst"].astype(str)
        + "|"
        + frame["inner_proto"].astype(str)
        + "|"
        + frame["inner_sport"].astype(str)
        + "|"
        + frame["inner_dport"].astype(str)
    )


def _key_to_flow(flows: pd.DataFrame) -> dict[str, int]:
    keys = pd.concat(
        [
            flows.loc[:, ["k5_fwd", "flow_id"]].rename(columns={"k5_fwd": "key"}),
            flows.loc[:, ["k5_rev", "flow_id"]].rename(columns={"k5_rev": "key"}),
        ],
        ignore_index=True,
    )
    duplicated = keys["key"].duplicated(keep=False)
    if duplicated.any():
        examples = keys.loc[duplicated, "key"].head().tolist()
        raise PipelineInvariantError(
            "Flow assignment keys are ambiguous; exact fast-path assignment is unsafe: "
            f"{examples}"
        )
    return dict(zip(keys["key"].astype(str), keys["flow_id"].astype(int), strict=True))


def _assigned_table(
    frame: pd.DataFrame, flow_ids: np.ndarray, direction: np.ndarray
) -> pa.Table:
    assigned = pd.DataFrame(
        {
            "flow_id": flow_ids.astype(np.int64, copy=False),
            "inner_idx": pd.to_numeric(frame["inner_idx"], errors="raise").astype(np.int64),
            "outer_idx": pd.to_numeric(frame["outer_idx"], errors="raise").astype(np.int64),
            "direction": direction.astype(np.int8, copy=False),
            "inner_time_ms": pd.to_numeric(frame["inner_time"], errors="raise") * 1000.0,
            "outer_time_ms": pd.to_numeric(frame["outer_time"], errors="raise") * 1000.0,
            "inner_size": pd.to_numeric(frame["inner_length"], errors="raise").astype(float),
            "outer_size": pd.to_numeric(
                frame["outer_padded_length"], errors="raise"
            ).astype(float),
        }
    )
    values = assigned.loc[:, ["inner_time_ms", "outer_time_ms", "inner_size", "outer_size"]]
    invalid_magnitude = (assigned[["inner_size", "outer_size"]] < 0).any().any()
    if not np.isfinite(values.to_numpy()).all() or invalid_magnitude:
        raise PipelineInvariantError("Assigned packet measurements are invalid")
    return pa.Table.from_pandas(assigned, schema=ASSIGNED_SCHEMA, preserve_index=False)


def assign_packets(
    flows: pd.DataFrame,
    packet_path: Path,
    work_dir: Path,
    *,
    batch_size: int,
    partitions: int,
    padding_ms: float,
) -> tuple[list[Path], pd.DataFrame, int]:
    """Reproduce released assignments and hash-partition matched packet pairs."""
    _check_parquet_columns(packet_path, PACKET_REQUIRED_COLUMNS)
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    key_map = _key_to_flow(flows)
    max_flow_id = int(flows["flow_id"].max())
    starts = np.full(max_flow_id + 1, np.nan)
    ends = np.full(max_flow_id + 1, np.nan)
    released = np.zeros(max_flow_id + 1, dtype=np.int64)
    flow_src = np.full(max_flow_id + 1, None, dtype=object)
    flow_dst = np.full(max_flow_id + 1, None, dtype=object)
    flow_ids = flows["flow_id"].to_numpy(dtype=np.int64)
    starts[flow_ids] = flows["flow_start_ms"].to_numpy(dtype=float)
    ends[flow_ids] = flows["flow_end_ms"].to_numpy(dtype=float)
    released[flow_ids] = flows["matched_packets"].to_numpy(dtype=np.int64)
    flow_src[flow_ids] = flows["src_ip"].astype(str).to_numpy()
    flow_dst[flow_ids] = flows["dst_ip"].astype(str).to_numpy()
    reproduced = np.zeros_like(released)

    paths = [work_dir / f"partition_{index:03d}.parquet" for index in range(partitions)]
    writers: dict[int, pq.ParquetWriter] = {}
    packet_rows = 0
    parquet = pq.ParquetFile(packet_path)
    try:
        for batch in parquet.iter_batches(
            columns=list(PACKET_REQUIRED_COLUMNS), batch_size=batch_size
        ):
            frame = batch.to_pandas()
            packet_rows += len(frame)
            assigned = _packet_keys(frame).map(key_map)
            if assigned.isna().any():
                raise PipelineInvariantError(
                    f"{int(assigned.isna().sum())} packet matches have no flow key"
                )
            assigned_ids = assigned.to_numpy(dtype=np.int64)
            packet_src = frame["inner_src"].astype(str).to_numpy()
            packet_dst = frame["inner_dst"].astype(str).to_numpy()
            forward = (packet_src == flow_src[assigned_ids]) & (
                packet_dst == flow_dst[assigned_ids]
            )
            reverse = (packet_src == flow_dst[assigned_ids]) & (
                packet_dst == flow_src[assigned_ids]
            )
            if not np.logical_xor(forward, reverse).all():
                invalid = int((~np.logical_xor(forward, reverse)).sum())
                raise PipelineInvariantError(
                    f"{invalid} assigned packets disagree with flow endpoint orientation"
                )
            direction = np.where(forward, 0, 1).astype(np.int8)
            inner_time_ms = pd.to_numeric(frame["inner_time"], errors="raise").to_numpy() * 1000.0
            in_window = (inner_time_ms >= starts[assigned_ids] - padding_ms) & (
                inner_time_ms <= ends[assigned_ids] + padding_ms
            )
            if not in_window.all():
                raise PipelineInvariantError(
                    f"{int((~in_window).sum())} packet assignments violate the released time window"
                )
            np.add.at(reproduced, assigned_ids, 1)

            table = _assigned_table(frame, assigned_ids, direction)
            partition_ids = np.mod(assigned_ids, partitions)
            for partition in np.unique(partition_ids):
                partition = int(partition)
                selected = np.flatnonzero(partition_ids == partition)
                part = table.take(pa.array(selected, type=pa.int64()))
                writer = writers.get(partition)
                if writer is None:
                    writer = pq.ParquetWriter(paths[partition], ASSIGNED_SCHEMA, compression="zstd")
                    writers[partition] = writer
                writer.write_table(part)
    finally:
        for writer in writers.values():
            writer.close()

    audit = flows.loc[:, ["id", "flow_id", "matched_packets"]].copy()
    audit = audit.rename(
        columns={"id": "source_id", "matched_packets": "released_matched_packets"}
    )
    audit["reproduced_matched_packets"] = reproduced[
        audit["flow_id"].to_numpy(dtype=np.int64)
    ]
    audit["counts_equal"] = (
        audit["released_matched_packets"] == audit["reproduced_matched_packets"]
    )
    if not audit["counts_equal"].all():
        examples = audit.loc[~audit["counts_equal"]].head().to_dict("records")
        raise PipelineInvariantError(f"Reproduced assignment counts differ: {examples}")
    if int(reproduced.sum()) != packet_rows:
        raise PipelineInvariantError("Assigned packet count does not conserve packet-match rows")
    return [path for path in paths if path.exists()], audit, packet_rows


def _gap_aggregates(
    frame: pd.DataFrame, gap_column: str, prefix: str
) -> pd.DataFrame:
    grouped = frame.groupby("flow_id", sort=False)[gap_column]
    result = grouped.agg(["min", "mean", "std", "max"]).fillna(0.0)
    result.columns = [
        f"{prefix}_min_iat_ms",
        f"{prefix}_mean_iat_ms",
        f"{prefix}_std_iat_ms",
        f"{prefix}_max_iat_ms",
    ]
    return result


def _domain_statistics(frame: pd.DataFrame, domain: str) -> pd.DataFrame:
    size = f"{domain}_size"
    time = f"{domain}_time_ms"
    gap = f"{domain}_gap"
    direction_gap = f"{domain}_direction_gap"
    grouped = frame.groupby("flow_id", sort=False)
    statistics = grouped.agg(
        packets=(size, "size"),
        bytes=(size, "sum"),
        mean_packet_size=(size, "mean"),
        std_packet_size=(size, "std"),
        first_time=(time, "first"),
        last_time=(time, "last"),
    )
    statistics["std_packet_size"] = statistics["std_packet_size"].fillna(0.0)
    statistics["duration_ms"] = statistics.pop("last_time") - statistics.pop("first_time")
    statistics = statistics.join(_gap_aggregates(frame, gap, ""))
    statistics = statistics.rename(
        columns={name: name.removeprefix("_") for name in statistics.columns}
    )

    directional_bytes = (
        frame.groupby(["flow_id", "direction"], sort=False)[size]
        .sum()
        .unstack(fill_value=0.0)
    )
    statistics["src2dst_bytes"] = directional_bytes.get(0, 0.0)
    statistics["dst2src_bytes"] = directional_bytes.get(1, 0.0)
    for raw_direction, name in ((0, "src2dst"), (1, "dst2src")):
        selected = frame.loc[frame["direction"] == raw_direction]
        statistics = statistics.join(
            _gap_aggregates(selected, direction_gap, name), how="left"
        )
    statistics = statistics.fillna(0.0)
    duration_seconds = statistics["duration_ms"] / 1000.0
    valid = duration_seconds.where(duration_seconds > 0)
    statistics["packet_rate"] = statistics["packets"] / valid
    statistics["byte_rate"] = statistics["bytes"] / valid
    return statistics.loc[:, CANONICAL_STATS].add_prefix(f"{domain}_")


def _sequence_view(
    prefix_pairs: pd.DataFrame, domain: str, maximum_length: int
) -> pd.DataFrame:
    ordered = prefix_pairs.sort_values(
        ["flow_id", f"{domain}_time_ms", f"{domain}_idx"],
        kind="mergesort",
    ).copy()
    grouped = ordered.groupby("flow_id", sort=False)
    ordered[f"{domain}_prefix_iat"] = grouped[f"{domain}_time_ms"].diff().fillna(0.0)
    if (ordered[f"{domain}_prefix_iat"] < 0).any():
        raise PipelineInvariantError(f"{domain} prefix PIAT contains negative values")
    result = pd.DataFrame(index=grouped.size().index)
    result[f"{domain}_direction"] = grouped["direction"].agg(list)
    result[f"{domain}_size"] = grouped[f"{domain}_size"].agg(list)
    result[f"{domain}_iat_ms"] = grouped[f"{domain}_prefix_iat"].agg(list)
    result[f"{domain}_length"] = grouped.size().clip(upper=maximum_length).astype(np.int16)
    return result


def _sequences(inner_ordered: pd.DataFrame, maximum_length: int) -> pd.DataFrame:
    # Prefix membership is source-defined; each view then preserves its observed order.
    prefix_pairs = inner_ordered.groupby("flow_id", sort=False).head(maximum_length)
    inner = _sequence_view(prefix_pairs, "inner", maximum_length)
    outer = _sequence_view(prefix_pairs, "outer", maximum_length)
    result = inner.join(outer)
    return result


def _prefix_statistics(
    inner_ordered: pd.DataFrame, maximum_length: int
) -> dict[str, pd.DataFrame]:
    prefix_pairs = inner_ordered.groupby("flow_id", sort=False).head(maximum_length)
    result = {}
    for domain in ("inner", "outer"):
        ordered = prefix_pairs.sort_values(
            ["flow_id", f"{domain}_time_ms", f"{domain}_idx"],
            kind="mergesort",
        ).copy()
        ordered[f"{domain}_gap"] = ordered.groupby("flow_id", sort=False)[
            f"{domain}_time_ms"
        ].diff()
        ordered[f"{domain}_direction_gap"] = ordered.groupby(
            ["flow_id", "direction"], sort=False
        )[f"{domain}_time_ms"].diff()
        result[domain] = _domain_statistics(ordered, domain)
    return result


def aggregate_partition(path: Path, maximum_length: int) -> tuple[pd.DataFrame, int, int]:
    frame = pq.read_table(path).to_pandas()
    inner_ordered = frame.sort_values(
        ["flow_id", "inner_time_ms", "inner_idx", "outer_idx"],
        kind="mergesort",
        ignore_index=True,
    )
    observed_outer_gap = inner_ordered.groupby("flow_id", sort=False)["outer_time_ms"].diff()
    reordered = observed_outer_gap.lt(0)
    reordered_pairs = int(reordered.sum())
    reordered_flows = int(inner_ordered.loc[reordered, "flow_id"].nunique())

    ordered_views = {"inner": inner_ordered}
    ordered_views["outer"] = frame.sort_values(
        ["flow_id", "outer_time_ms", "outer_idx", "inner_idx"],
        kind="mergesort",
        ignore_index=True,
    )
    for domain in ("inner", "outer"):
        ordered = ordered_views[domain]
        ordered[f"{domain}_gap"] = ordered.groupby("flow_id", sort=False)[
            f"{domain}_time_ms"
        ].diff()
        ordered[f"{domain}_direction_gap"] = ordered.groupby(
            ["flow_id", "direction"], sort=False
        )[f"{domain}_time_ms"].diff()
        finite_gaps = ordered[f"{domain}_gap"].dropna().to_numpy()
        if (finite_gaps < 0).any():
            raise PipelineInvariantError(
                f"{domain} packet timestamps are not monotonic in view-specific order"
            )
    inner = _domain_statistics(ordered_views["inner"], "inner")
    outer = _domain_statistics(ordered_views["outer"], "outer")
    sequences = _sequences(inner_ordered, maximum_length)
    prefix_statistics = _prefix_statistics(inner_ordered, maximum_length)
    result = inner.join(outer).join(sequences)
    if not (result["inner_packets"] == result["outer_packets"]).all():
        raise PipelineInvariantError("Matched-view packet counts differ after aggregation")
    audit_frame = result.reset_index()
    for domain in ("inner", "outer"):
        reconstructed = build_prefix_stats(
            audit_frame, domain=domain, prefix_length=maximum_length
        ).values
        expected = prefix_statistics[domain].loc[result.index].to_numpy(dtype=float)
        if not np.isclose(
            expected,
            reconstructed,
            rtol=1e-10,
            atol=1e-8,
            equal_nan=True,
        ).all():
            raise PipelineInvariantError(
                f"{domain} first-{maximum_length} aggregation disagrees with PrefixStats"
            )
    return result.reset_index(), reordered_pairs, reordered_flows


def _add_directional_bytes_fidelity(
    audit: pd.DataFrame,
    flows: pd.DataFrame,
    features: pd.DataFrame,
) -> pd.DataFrame:
    source = flows.set_index("flow_id")
    rebuilt = features.set_index("flow_id")
    fully_matched = source["matched_packets"].gt(0) & source["matched_packets"].eq(
        source["bidirectional_packets"]
    )
    fully_matched_ids = source.index[fully_matched]
    src_bytes_equal = np.isclose(
        rebuilt.loc[fully_matched_ids, "inner_src2dst_bytes"].to_numpy(dtype=float),
        source.loc[fully_matched_ids, "src2dst_bytes"].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-8,
    )
    dst_bytes_equal = np.isclose(
        rebuilt.loc[fully_matched_ids, "inner_dst2src_bytes"].to_numpy(dtype=float),
        source.loc[fully_matched_ids, "dst2src_bytes"].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-8,
    )
    byte_fidelity = src_bytes_equal & dst_bytes_equal
    if not byte_fidelity.all():
        failed_ids = fully_matched_ids[~byte_fidelity].tolist()[:5]
        raise PipelineInvariantError(
            f"Rebuilt directional bytes disagree with released semantics: {failed_ids}"
        )

    result = audit.copy()
    result["directional_bytes_fidelity_checked"] = result["flow_id"].isin(
        fully_matched_ids
    )
    result["directional_bytes_fidelity_equal"] = True
    return result


def reconstruct_session(
    flow_path: Path,
    packet_path: Path,
    work_dir: Path,
    *,
    maximum_length: int,
    batch_size: int,
    partitions: int,
    padding_ms: float,
) -> SessionReconstruction:
    flows = load_flow_table(flow_path)
    partition_paths, audit, packet_rows = assign_packets(
        flows,
        packet_path,
        work_dir,
        batch_size=batch_size,
        partitions=partitions,
        padding_ms=padding_ms,
    )
    aggregated = [aggregate_partition(path, maximum_length) for path in partition_paths]
    pieces = [item[0] for item in aggregated]
    features = pd.concat(pieces, ignore_index=True).sort_values("flow_id").reset_index(drop=True)
    expected_ids = set(flows.loc[flows["matched_packets"] > 0, "flow_id"])
    if set(features["flow_id"]) != expected_ids:
        raise PipelineInvariantError("Aggregated flow coverage differs from released matched flows")
    audit = _add_directional_bytes_fidelity(audit, flows, features)
    return SessionReconstruction(
        features=features,
        assignment_audit=audit,
        packet_rows=packet_rows,
        outer_reordered_adjacent_pairs=sum(item[1] for item in aggregated),
        outer_reordered_flows=sum(item[2] for item in aggregated),
    )
