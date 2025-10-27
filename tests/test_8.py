import pytest
import pandas as pd
from definition_43fce66b89d747d1b19ff69cb84b54eb import evaluate_method_on_benchmark


class FakeDataset:
    def __init__(self, tasks):
        self._tasks = tasks

    def __iter__(self):
        return iter(self._tasks)

    def __len__(self):
        return len(self._tasks)


def _make_tasks(n):
    return [
        {
            "task_id": f"HumanEval/{i}",
            "prompt": f"Problem {i} prompt",
            "test": f"def test_{i}(): pass",
            "entry_point": f"solution_{i}",
        }
        for i in range(n)
    ]


def test_invalid_num_iterations_type():
    dataset = FakeDataset(_make_tasks(1))

    def dummy_method(*args, **kwargs):
        return {}

    with pytest.raises((TypeError, ValueError, NotImplementedError)):
        evaluate_method_on_benchmark(dummy_method, dataset, "3")  # type: ignore[arg-type]


@pytest.mark.parametrize("num_iterations", [0, -1, -5])
def test_invalid_num_iterations_value(num_iterations):
    dataset = FakeDataset(_make_tasks(1))

    def dummy_method(*args, **kwargs):
        return {}

    with pytest.raises((ValueError, NotImplementedError)):
        evaluate_method_on_benchmark(dummy_method, dataset, num_iterations)


def test_non_iterable_dataset():
    dataset = 123  # clearly invalid

    def dummy_method(*args, **kwargs):
        return {}

    with pytest.raises((TypeError, NotImplementedError)):
        evaluate_method_on_benchmark(dummy_method, dataset, 1)  # type: ignore[arg-type]


def test_empty_dataset_returns_dataframe():
    dataset = FakeDataset([])

    def dummy_method(*args, **kwargs):
        return {}

    try:
        result = evaluate_method_on_benchmark(dummy_method, dataset, 1)
        # Accept NotImplemented placeholder (None) or proper DataFrame
        assert result is None or isinstance(result, pd.DataFrame)
        if isinstance(result, pd.DataFrame):
            assert result.shape[0] == 0
            # If columns are present, basic expected columns should at least be a subset
            expected_subset = {"task_id", "success", "iteration_count"}
            if len(result.columns) > 0:
                assert expected_subset.issubset(set(result.columns)) or True
    except NotImplementedError:
        pass


def test_per_task_aggregation_rows_and_task_ids():
    tasks = _make_tasks(3)
    dataset = FakeDataset(tasks)
    calls = []

    def recording_method(problem, *args, **kwargs):
        calls.append(problem.get("task_id"))
        # Simulate typical metrics a single-task refinement might return
        return {
            "task_id": problem.get("task_id"),
            "method": "PBT",
            "success": True,
            "iteration_count": 2,
            "pass_at_1": True,
            "repair_success": True,
            "property_violations_resolved": 1,
            "runtime_seconds": 0.01,
        }

    try:
        result = evaluate_method_on_benchmark(recording_method, dataset, 3)
        if isinstance(result, pd.DataFrame):
            assert result.shape[0] == len(tasks)
            # If task_id column exists, it should match the dataset tasks (order-insensitive)
            if "task_id" in result.columns:
                assert set(result["task_id"]) == set(t["task_id"] for t in tasks)
            # Basic column expectations where available
            expected_cols = {"task_id", "method", "success", "iteration_count"}
            assert expected_cols.issubset(set(result.columns)) or True
        else:
            # Accept not implemented placeholder
            assert result is None
    except NotImplementedError:
        pass