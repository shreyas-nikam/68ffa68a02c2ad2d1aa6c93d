import pytest
import pandas as pd
import numpy as np
from matplotlib.figure import Figure
from definition_efb5818ae1b44ed5a532c1cb7601731c import visualize_test_coverage_and_feedback


def _assert_fig_or_none(obj):
    # Helper to assert return type and close figures to prevent resource leaks
    if isinstance(obj, Figure):
        import matplotlib.pyplot as plt
        plt.close(obj)
        return True
    return obj is None


def test_valid_dataframe_returns_figure_or_none():
    df = pd.DataFrame({
        "task_id": ["t1", "t1", "t2", "t2"],
        "iteration": [1, 2, 1, 2],
        "method": ["PBT", "PBT", "TDD", "TDD"],
        "property_violations": [3, 1, 0, 2],
        "example_failures": [1, 0, 2, 0],
        "input_diversity": [10, 15, 5, 7],
        "timestamp": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-01", "2025-01-02"]),
    })
    res = visualize_test_coverage_and_feedback(df)
    assert _assert_fig_or_none(res)


def test_empty_dataframe_handled_gracefully():
    df_empty = pd.DataFrame(columns=[
        "task_id", "iteration", "method",
        "property_violations", "example_failures",
        "input_diversity", "timestamp"
    ])
    try:
        res = visualize_test_coverage_and_feedback(df_empty)
        assert _assert_fig_or_none(res)
    except Exception as e:
        # Accept reasonable validation error types
        assert isinstance(e, (ValueError, KeyError))


def test_nan_values_handled():
    df_nan = pd.DataFrame({
        "task_id": ["t1", "t2", "t3"],
        "iteration": [1, 2, 3],
        "method": ["PBT", "TDD", "PBT"],
        "property_violations": [np.nan, 2, 0],
        "example_failures": [1, np.nan, 0],
        "input_diversity": [10, 15, np.nan],
        "timestamp": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
    })
    try:
        res = visualize_test_coverage_and_feedback(df_nan)
        assert _assert_fig_or_none(res)
    except Exception as e:
        # Accept reasonable validation error types when NaNs are not supported
        assert isinstance(e, (ValueError, KeyError))


@pytest.mark.parametrize("bad_input, expected_exception", [
    (None, TypeError),
    (123, TypeError),
    ("not a dataframe", TypeError),
])
def test_invalid_inputs_raise_type_errors(bad_input, expected_exception):
    with pytest.raises(expected_exception):
        visualize_test_coverage_and_feedback(bad_input)


def test_missing_required_columns_errors_or_handles():
    df_missing = pd.DataFrame({
        "task_id": ["t1", "t2"],
        # Missing: example_failures, input_diversity, timestamp
        "property_violations": [1, 3],
    })
    try:
        res = visualize_test_coverage_and_feedback(df_missing)
        # If implementation is lenient, it may still return a figure or None
        assert _assert_fig_or_none(res)
    except Exception as e:
        assert isinstance(e, (KeyError, ValueError))

