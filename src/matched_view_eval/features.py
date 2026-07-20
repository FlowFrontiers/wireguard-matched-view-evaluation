from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from matched_view_eval.errors import PipelineInvariantError
from matched_view_eval.schema import CANONICAL_STATS

DOMAINS = frozenset({"inner", "outer"})
SEQUENCE_CHANNELS = ("direction", "size", "iat_ms")


@dataclass(frozen=True)
class FeatureMatrix:
    values: np.ndarray
    feature_names: tuple[str, ...]
    mask: np.ndarray | None = None


@dataclass(frozen=True)
class SequenceFeatures:
    values: np.ndarray
    mask: np.ndarray
    channels: tuple[str, ...]


def _validate_domain(domain: str) -> None:
    if domain not in DOMAINS:
        raise PipelineInvariantError(f"Unsupported domain: {domain}")


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise PipelineInvariantError(f"Feature input is missing columns: {sorted(missing)}")


def build_matched_flow_stats(frame: pd.DataFrame, *, domain: str) -> FeatureMatrix:
    """Extract statistics over every matched physical packet pair in a flow."""
    _validate_domain(domain)
    columns = tuple(f"{domain}_{name}" for name in CANONICAL_STATS)
    _require_columns(frame, columns)
    values = frame.loc[:, columns].to_numpy(dtype=np.float64, copy=True)
    if np.isinf(values).any():
        raise PipelineInvariantError("MatchedFlowStats contains infinite values")
    return FeatureMatrix(values=values, feature_names=tuple(CANONICAL_STATS))


def _sample_std(values: np.ndarray) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def _gap_summary(values: np.ndarray) -> tuple[float, float, float, float]:
    if len(values) == 0:
        return (0.0, 0.0, 0.0, 0.0)
    return (
        float(np.min(values)),
        float(np.mean(values)),
        _sample_std(values),
        float(np.max(values)),
    )


def _safe_rate(numerator: float, duration_ms: float) -> float:
    return numerator / (duration_ms / 1000.0) if duration_ms > 0 else np.nan


def _prefix_stat_row(
    direction_value: object,
    size_value: object,
    iat_value: object,
    *,
    prefix_length: int,
) -> list[float]:
    direction = np.asarray(direction_value, dtype=np.int8)[:prefix_length]
    size = np.asarray(size_value, dtype=np.float64)[:prefix_length]
    iat = np.asarray(iat_value, dtype=np.float64)[:prefix_length]
    if not (len(direction) == len(size) == len(iat)) or len(size) == 0:
        raise PipelineInvariantError("Canonical SPLT arrays are empty or misaligned")
    if not np.isin(direction, (0, 1)).all():
        raise PipelineInvariantError("Raw SPLT direction must contain only {0, 1}")
    if not np.isfinite(size).all() or not np.isfinite(iat).all():
        raise PipelineInvariantError("SPLT magnitudes must be finite")
    if (size < 0).any() or (iat < 0).any():
        raise PipelineInvariantError("SPLT magnitudes must be nonnegative")

    packets = float(len(size))
    byte_count = float(np.sum(size))
    timestamps = np.cumsum(iat)
    overall_gaps = np.diff(timestamps)
    min_iat, mean_iat, std_iat, max_iat = _gap_summary(overall_gaps)
    duration_ms = float(timestamps[-1] - timestamps[0])

    directional: dict[str, tuple[float, tuple[float, float, float, float]]] = {}
    for raw_direction, name in ((0, "src2dst"), (1, "dst2src")):
        selected = direction == raw_direction
        directional_bytes = float(np.sum(size[selected]))
        directional_gaps = np.diff(timestamps[selected])
        directional[name] = (directional_bytes, _gap_summary(directional_gaps))

    dst_bytes, dst_iat = directional["dst2src"]
    src_bytes, src_iat = directional["src2dst"]
    values = {
        "packets": packets,
        "bytes": byte_count,
        "mean_packet_size": float(np.mean(size)),
        "std_packet_size": _sample_std(size),
        "min_iat_ms": min_iat,
        "mean_iat_ms": mean_iat,
        "std_iat_ms": std_iat,
        "max_iat_ms": max_iat,
        "duration_ms": duration_ms,
        "dst2src_bytes": dst_bytes,
        "src2dst_bytes": src_bytes,
        "dst2src_min_iat_ms": dst_iat[0],
        "dst2src_mean_iat_ms": dst_iat[1],
        "dst2src_std_iat_ms": dst_iat[2],
        "dst2src_max_iat_ms": dst_iat[3],
        "src2dst_min_iat_ms": src_iat[0],
        "src2dst_mean_iat_ms": src_iat[1],
        "src2dst_std_iat_ms": src_iat[2],
        "src2dst_max_iat_ms": src_iat[3],
        "packet_rate": _safe_rate(packets, duration_ms),
        "byte_rate": _safe_rate(byte_count, duration_ms),
    }
    return [values[name] for name in CANONICAL_STATS]


def build_prefix_stats(
    frame: pd.DataFrame,
    *,
    domain: str,
    prefix_length: int,
) -> FeatureMatrix:
    """Compute observation-matched aggregate statistics from an early-flow prefix."""
    _validate_domain(domain)
    if prefix_length < 1:
        raise PipelineInvariantError("prefix_length must be positive")
    columns = tuple(f"{domain}_{channel}" for channel in SEQUENCE_CHANNELS)
    _require_columns(frame, columns)
    values = np.asarray(
        [
            _prefix_stat_row(direction, size, iat, prefix_length=prefix_length)
            for direction, size, iat in frame.loc[:, columns].itertuples(index=False, name=None)
        ],
        dtype=np.float64,
    )
    if np.isinf(values).any():
        raise PipelineInvariantError("PrefixStats contains infinite values")
    return FeatureMatrix(values=values, feature_names=tuple(CANONICAL_STATS))


def build_sequential_splt(
    frame: pd.DataFrame,
    *,
    domain: str,
    prefix_length: int,
    channels: tuple[str, ...] = SEQUENCE_CHANNELS,
    log_transform_magnitudes: bool = True,
) -> SequenceFeatures:
    """Build padded SPLT tensors with ±1 direction and zero reserved for padding."""
    _validate_domain(domain)
    if prefix_length < 1:
        raise PipelineInvariantError("prefix_length must be positive")
    if not channels or not set(channels) <= set(SEQUENCE_CHANNELS):
        raise PipelineInvariantError(f"Unsupported SPLT channels: {channels}")
    columns = tuple(f"{domain}_{channel}" for channel in SEQUENCE_CHANNELS)
    _require_columns(frame, columns)

    values = np.zeros((len(frame), prefix_length, len(channels)), dtype=np.float32)
    mask = np.zeros((len(frame), prefix_length), dtype=bool)
    for row_index, (direction_value, size_value, iat_value) in enumerate(
        frame.loc[:, columns].itertuples(index=False, name=None)
    ):
        raw = {
            "direction": np.asarray(direction_value, dtype=np.int8),
            "size": np.asarray(size_value, dtype=np.float32),
            "iat_ms": np.asarray(iat_value, dtype=np.float32),
        }
        if not (len(raw["direction"]) == len(raw["size"]) == len(raw["iat_ms"])):
            raise PipelineInvariantError("Canonical SPLT arrays are misaligned")
        length = min(len(raw["direction"]), prefix_length)
        if length == 0:
            raise PipelineInvariantError("Canonical SPLT arrays cannot be empty")
        raw_direction = raw["direction"][:length]
        if not np.isin(raw_direction, (0, 1)).all():
            raise PipelineInvariantError("Raw SPLT direction must contain only {0, 1}")
        transformed = {
            "direction": np.where(raw_direction == 0, -1.0, 1.0).astype(np.float32),
            "size": raw["size"][:length],
            "iat_ms": raw["iat_ms"][:length],
        }
        if (transformed["size"] < 0).any() or (transformed["iat_ms"] < 0).any():
            raise PipelineInvariantError("SPLT magnitudes must be nonnegative")
        if log_transform_magnitudes:
            transformed["size"] = np.log1p(transformed["size"])
            transformed["iat_ms"] = np.log1p(transformed["iat_ms"])
        for channel_index, channel in enumerate(channels):
            values[row_index, :length, channel_index] = transformed[channel]
        mask[row_index, :length] = True

    if not np.isfinite(values).all():
        raise PipelineInvariantError("Sequential SPLT contains non-finite values")
    if not np.all(values[~mask] == 0):
        raise PipelineInvariantError("Sequential SPLT padding must be zero")
    return SequenceFeatures(values=values, mask=mask, channels=channels)


def build_flattened_splt(
    frame: pd.DataFrame,
    *,
    domain: str,
    prefix_length: int,
    channels: tuple[str, ...] = SEQUENCE_CHANNELS,
    log_transform_magnitudes: bool = True,
) -> FeatureMatrix:
    """Flatten the exact sequential tensor in timestep-major order."""
    sequence = build_sequential_splt(
        frame,
        domain=domain,
        prefix_length=prefix_length,
        channels=channels,
        log_transform_magnitudes=log_transform_magnitudes,
    )
    names = tuple(
        f"packet_{packet_index + 1}_{channel}"
        for packet_index in range(prefix_length)
        for channel in channels
    )
    return FeatureMatrix(
        values=sequence.values.reshape(len(frame), -1),
        feature_names=names,
        mask=sequence.mask,
    )
