from __future__ import annotations

ID_COLUMNS = {
    "id",
    "flow_id",
    "application_name",
    "application_category_name",
}

FLOW_REQUIRED_COLUMNS = (
    "id",
    "flow_id",
    "application_name",
    "application_category_name",
    "src_ip",
    "dst_ip",
    "flow_start_ms",
    "flow_end_ms",
    "k5_fwd",
    "k5_rev",
    "matched_packets",
    "bidirectional_packets",
    "src2dst_bytes",
    "dst2src_bytes",
)

PACKET_REQUIRED_COLUMNS = (
    "inner_idx",
    "inner_time",
    "inner_src",
    "inner_dst",
    "inner_proto",
    "inner_sport",
    "inner_dport",
    "inner_length",
    "outer_idx",
    "outer_time",
    "outer_padded_length",
)

INNER_STATS = {
    "packets": "bidirectional_packets",
    "bytes": "bidirectional_bytes",
    "mean_packet_size": "bidirectional_mean_ps",
    "std_packet_size": "bidirectional_stddev_ps",
    "min_iat_ms": "bidirectional_min_piat_ms",
    "mean_iat_ms": "bidirectional_mean_piat_ms",
    "std_iat_ms": "bidirectional_stddev_piat_ms",
    "max_iat_ms": "bidirectional_max_piat_ms",
    "duration_ms": "bidirectional_duration_ms",
    "dst2src_bytes": "dst2src_bytes",
    "src2dst_bytes": "src2dst_bytes",
    "dst2src_min_iat_ms": "dst2src_min_piat_ms",
    "dst2src_mean_iat_ms": "dst2src_mean_piat_ms",
    "dst2src_std_iat_ms": "dst2src_stddev_piat_ms",
    "dst2src_max_iat_ms": "dst2src_max_piat_ms",
    "src2dst_min_iat_ms": "src2dst_min_piat_ms",
    "src2dst_mean_iat_ms": "src2dst_mean_piat_ms",
    "src2dst_std_iat_ms": "src2dst_stddev_piat_ms",
    "src2dst_max_iat_ms": "src2dst_max_piat_ms",
}

OUTER_STATS = {
    "packets": "matched_packets",
    "bytes": "outer_bytes",
    "mean_packet_size": "mean_outer_pkt_size",
    "std_packet_size": "std_outer_pkt_size",
    "min_iat_ms": "outer_min_piat_ms",
    "mean_iat_ms": "outer_mean_piat_ms",
    "std_iat_ms": "outer_stddev_piat_ms",
    "max_iat_ms": "outer_max_piat_ms",
    "duration_ms": "outer_capture_duration_ms",
    "dst2src_bytes": "outer_bytes_in",
    "src2dst_bytes": "outer_bytes_out",
    "dst2src_min_iat_ms": "outer_min_piat_ms_in",
    "dst2src_mean_iat_ms": "outer_mean_piat_ms_in",
    "dst2src_std_iat_ms": "outer_stddev_piat_ms_in",
    "dst2src_max_iat_ms": "outer_max_piat_ms_in",
    "src2dst_min_iat_ms": "outer_min_piat_ms_out",
    "src2dst_mean_iat_ms": "outer_mean_piat_ms_out",
    "src2dst_std_iat_ms": "outer_stddev_piat_ms_out",
    "src2dst_max_iat_ms": "outer_max_piat_ms_out",
}

INNER_SEQUENCES = {
    "direction": "splt_direction",
    "size": "splt_ps",
    "iat_ms": "splt_piat_ms",
}

OUTER_SEQUENCES = {
    "direction": "outer_splt_direction",
    "size": "outer_splt_ps",
    "iat_ms": "outer_splt_piat_ms",
}

DERIVED_STATS = ("packet_rate", "byte_rate")
CANONICAL_STATS = tuple(INNER_STATS) + DERIVED_STATS
if set(CANONICAL_STATS) != set(OUTER_STATS) | set(DERIVED_STATS):
    raise RuntimeError("Inner and outer canonical statistic schemas disagree")

STAT_COLUMNS = tuple(
    f"{domain}_{name}" for domain in ("inner", "outer") for name in CANONICAL_STATS
)

RAW_REQUIRED_COLUMNS = tuple(
    sorted(
        ID_COLUMNS
        | set(INNER_STATS.values())
        | set(OUTER_STATS.values())
        | set(INNER_SEQUENCES.values())
        | set(OUTER_SEQUENCES.values())
    )
)

METADATA_COLUMNS = (
    "pair_id",
    "session",
    "source_id",
    "source_flow_id",
    "application_name",
    "application_category",
)

SEQUENCE_COLUMNS = tuple(
    f"{domain}_{channel}"
    for domain in ("inner", "outer")
    for channel in ("direction", "size", "iat_ms")
)

CANONICAL_COLUMNS = (
    *METADATA_COLUMNS,
    *STAT_COLUMNS,
    *SEQUENCE_COLUMNS,
    "inner_length",
    "outer_length",
)

ABSOLUTE_TIMESTAMP_NAMES = (
    "first_seen",
    "last_seen",
    "first_matched_time",
    "last_matched_time",
)
