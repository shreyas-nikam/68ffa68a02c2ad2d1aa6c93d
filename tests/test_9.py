import pytest
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from definition_f1382ca987014774af99ac49ff82df95 import visualize_performance_comparison

def build_valid_df():
    data = [
        {"method": "PBT", "task_id": "task_1", "iteration_count": 1, "success": False, "pass_at_1": 0.2, "repair_success": 0},
        {"method": "PBT", "task_id": "task_1", "iteration_count": 2, "success": True, "pass_at_1": 0.6, "repair_success": 1},
        {"method": "TDD", "task_id": "task_1", "iteration_count": 1, "success": False, "pass_at_1": 0.1, "repair_success": 0},
        {"method": "TDD", "task_id": "task_1", "iteration_count": 2, "success": True, "pass_at_1": 0.4, "repair_success": 1},
        {"method": "PBT", "task_id": "task_2", "iteration_count": 1, "success": False, "pass_at_1": 0.3, "repair_success": 0},
        {"method": "TDD", "task_id": "task_2", "iteration_count": 1, "success": False, "pass_at_1": 0.15, "repair_success": 0},
    ]
    return pd.DataFrame(data)

def build_missing_cols_df():
    df = build_valid_df().drop(columns=["pass_at_1"])
    return df

def build_single_method_df():
    df = build_valid_df()
    return df[df["method"] == "PBT"].reset_index(drop=True)

def build_empty_df():
    cols = ["method", "task_id", "iteration_count", "success", "pass_at_1", "repair_success"]
    return pd.DataFrame(columns=cols)

@pytest.mark.parametrize("input_data, spec", [
    (build_valid_df, {"accept_returns": (type(None), Figure)}),
    (build_missing_cols_df, {"expect_exception": (ValueError, KeyError)}),
    ([1, 2, 3], {"expect_exception": (TypeError,)}),
    (build_single_method_df, {"accept_returns": (type(None), Figure)}),
    (build_empty_df, {"accept_returns": (type(None), Figure), "expect_exception": (ValueError,)}),
])
def test_visualize_performance_comparison(input_data, spec):
    try:
        df_or_input = input_data() if callable(input_data) else input_data
        result = visualize_performance_comparison(df_or_input)
        if "accept_returns" in spec:
            assert isinstance(result, spec["accept_returns"])
        else:
            assert "expect_exception" not in spec, "Expected an exception but function returned successfully."
    except Exception as e:
        if "expect_exception" in spec:
            assert isinstance(e, spec["expect_exception"])
        else:
            raise
    finally:
        plt.close("all")