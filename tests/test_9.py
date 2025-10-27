import pytest
import pandas as pd
from unittest import mock
import matplotlib.pyplot as plt
import seaborn as sns

# Keep the definition_44ae778860f04b9584a0b9a65d6cb17e block as it is. DO NOT REPLACE or REMOVE the block.
from definition_44ae778860f04b9584a0b9a65d6cb17e import visualize_performance_comparison


@pytest.fixture
def sample_results_df():
    """Provides a sample DataFrame for testing visualization."""
    data = {
        'Iteration': [1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3],
        'Method': ['PBT', 'TDD', 'PBT', 'TDD', 'PBT', 'TDD', 'PBT', 'TDD', 'PBT', 'TDD', 'PBT', 'TDD'],
        'Metric': ['Pass Rate', 'Pass Rate', 'Pass Rate', 'Pass Rate', 'Pass Rate', 'Pass Rate',
                   'Repair Success', 'Repair Success', 'Repair Success', 'Repair Success', 'Repair Success', 'Repair Success'],
        'Value': [0.7, 0.5, 0.8, 0.6, 0.9, 0.75, 0.6, 0.4, 0.7, 0.5, 0.8, 0.65]
    }
    return pd.DataFrame(data)

@pytest.fixture
def empty_results_df():
    """Provides an empty DataFrame with expected columns."""
    return pd.DataFrame(columns=['Iteration', 'Method', 'Metric', 'Value'])

@pytest.fixture
def df_missing_columns():
    """Provides a DataFrame with missing expected columns."""
    data = {
        'UnexpectedColumn1': [1, 2, 3],
        'UnexpectedColumn2': ['A', 'B', 'C']
    }
    return pd.DataFrame(data)

def test_visualize_performance_comparison_success(sample_results_df):
    """
    Tests if the visualization function completes successfully and calls expected plotting functions
    with a well-formed DataFrame.
    """
    with mock.patch('seaborn.lineplot') as mock_lineplot, \
         mock.patch('seaborn.barplot') as mock_barplot, \
         mock.patch('matplotlib.pyplot.show') as mock_show, \
         mock.patch('matplotlib.pyplot.figure') as mock_figure: # Mock figure creation
        
        visualize_performance_comparison(sample_results_df)

        # Based on the notebook spec, both line plots (over iterations) and bar charts (summary metrics)
        # are expected.
        assert mock_lineplot.called, "seaborn.lineplot was not called"
        assert mock_barplot.called, "seaborn.barplot was not called"
        assert mock_show.called, "matplotlib.pyplot.show() was not called"
        assert mock_figure.called, "matplotlib.pyplot.figure() was not called"


def test_visualize_performance_comparison_empty_df(empty_results_df):
    """
    Tests if the visualization function handles an empty DataFrame gracefully without raising errors.
    It should not crash, even if no plots are generated.
    """
    with mock.patch('seaborn.lineplot') as mock_lineplot, \
         mock.patch('seaborn.barplot') as mock_barplot, \
         mock.patch('matplotlib.pyplot.show') as mock_show:
        
        try:
            visualize_performance_comparison(empty_results_df)
        except Exception as e:
            pytest.fail(f"visualize_performance_comparison raised an unexpected error with empty DataFrame: {e}")
        
        # Depending on implementation, plotting functions might or might not be called.
        # The key is that it completes without error.
        # For an empty DF, it's reasonable to expect no actual plots, so lineplot/barplot might not be called
        # but plt.show() could still be called if a figure was created.
        assert not mock_lineplot.called or not mock_barplot.called, "Plotting functions were unexpectedly called with empty data"


def test_visualize_performance_comparison_missing_columns(df_missing_columns):
    """
    Tests if the visualization function raises a KeyError when essential columns (e.g., 'Metric', 'Method', 'Value')
    are missing from the input DataFrame, as these are critical for plotting.
    """
    with pytest.raises(KeyError) as excinfo:
        visualize_performance_comparison(df_missing_columns)
    
    # Check if the error message indicates one of the expected missing columns.
    # The function is likely to access 'Metric', 'Method', or 'Value' first.
    error_msg = str(excinfo.value)
    assert any(col in error_msg for col in ['Metric', 'Method', 'Value', 'Iteration']), \
        f"KeyError message did not indicate expected missing columns. Error: {error_msg}"


def test_visualize_performance_comparison_invalid_input_type():
    """
    Tests if the visualization function raises a TypeError for non-DataFrame input,
    ensuring type-safety for its 'results_df' argument.
    """
    with pytest.raises(TypeError):
        visualize_performance_comparison(None)
    with pytest.raises(TypeError):
        visualize_performance_comparison([1, 2, 3])
    with pytest.raises(TypeError):
        visualize_performance_comparison("not a dataframe")
    with pytest.raises(TypeError):
        visualize_performance_comparison(123)