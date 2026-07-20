import numpy as np
import pandas as pd
import pytest

from matched_view_eval.errors import PipelineInvariantError
from matched_view_eval.features import (
    build_flattened_splt,
    build_matched_flow_stats,
    build_prefix_stats,
    build_sequential_splt,
)
from matched_view_eval.schema import CANONICAL_STATS


def _frame() -> pd.DataFrame:
    row = {
        "inner_direction": [0, 1, 0],
        "inner_size": [100.0, 200.0, 300.0],
        "inner_iat_ms": [0.0, 5.0, 7.0],
    }
    for index, name in enumerate(CANONICAL_STATS):
        row[f"inner_{name}"] = float(index)
    return pd.DataFrame([row])


def test_matched_flow_stats_preserves_canonical_order() -> None:
    result = build_matched_flow_stats(_frame(), domain="inner")
    assert result.feature_names == tuple(CANONICAL_STATS)
    np.testing.assert_array_equal(result.values[0], np.arange(21, dtype=float))


def test_prefix_stats_reconstructs_directional_timing() -> None:
    result = build_prefix_stats(_frame(), domain="inner", prefix_length=50)
    values = dict(zip(result.feature_names, result.values[0], strict=True))
    assert values["packets"] == 3
    assert values["bytes"] == 600
    assert values["src2dst_bytes"] == 400
    assert values["dst2src_bytes"] == 200
    assert values["duration_ms"] == 12
    assert values["mean_iat_ms"] == 6
    assert values["std_iat_ms"] == pytest.approx(np.sqrt(2))
    assert values["src2dst_min_iat_ms"] == 12
    assert values["src2dst_mean_iat_ms"] == 12
    assert values["src2dst_std_iat_ms"] == 0
    assert values["dst2src_min_iat_ms"] == 0
    assert values["packet_rate"] == 250
    assert values["byte_rate"] == 50_000


def test_sequential_encoding_and_flattening_share_one_tensor() -> None:
    frame = _frame()
    sequence = build_sequential_splt(
        frame,
        domain="inner",
        prefix_length=5,
        log_transform_magnitudes=True,
    )
    flattened = build_flattened_splt(
        frame,
        domain="inner",
        prefix_length=5,
        log_transform_magnitudes=True,
    )
    np.testing.assert_array_equal(sequence.values[:, :, 0], [[-1, 1, -1, 0, 0]])
    np.testing.assert_allclose(
        sequence.values[0, :3, 1], np.log1p([100.0, 200.0, 300.0])
    )
    np.testing.assert_allclose(sequence.values[0, :3, 2], np.log1p([0.0, 5.0, 7.0]))
    np.testing.assert_array_equal(sequence.mask, [[True, True, True, False, False]])
    np.testing.assert_array_equal(flattened.values, sequence.values.reshape(1, -1))
    assert flattened.feature_names[:6] == (
        "packet_1_direction",
        "packet_1_size",
        "packet_1_iat_ms",
        "packet_2_direction",
        "packet_2_size",
        "packet_2_iat_ms",
    )


def test_feature_builders_reject_invalid_direction() -> None:
    frame = _frame()
    frame.at[0, "inner_direction"] = [0, 2, 1]
    with pytest.raises(PipelineInvariantError, match="direction"):
        build_prefix_stats(frame, domain="inner", prefix_length=50)
    with pytest.raises(PipelineInvariantError, match="direction"):
        build_sequential_splt(frame, domain="inner", prefix_length=50)
