import sys
import types
import importlib
import pytest
from definition_bb81617c429e4f7093bd63fbc24f427b import load_humaneval_dataset


class DummyDataset:
    def __init__(self, columns=None, features=None):
        # Either column_names or features may be present
        self.column_names = columns or []
        self.features = features


def _setup_fake_datasets(monkeypatch, return_value=None, side_effect=None):
    fake_module = types.ModuleType("datasets")
    calls = {"args": None, "kwargs": None}

    def load_dataset(name, *args, **kwargs):
        calls["args"] = (name,) + args
        calls["kwargs"] = kwargs
        if side_effect:
            raise side_effect
        return return_value

    fake_module.load_dataset = load_dataset

    # Access the target module to patch its imports if already loaded
    target_module = importlib.import_module("definition_bb81617c429e4f7093bd63fbc24f427b")
    if hasattr(target_module, "load_dataset"):
        # from datasets import load_dataset
        monkeypatch.setattr(target_module, "load_dataset", load_dataset, raising=True)
    elif hasattr(target_module, "datasets"):
        # import datasets
        monkeypatch.setattr(target_module, "datasets", fake_module, raising=True)
    else:
        # Ensure that any future import datasets inside the function resolves here
        monkeypatch.setitem(sys.modules, "datasets", fake_module)
    return fake_module, calls


def test_returns_test_split_and_calls_correct_dataset_name(monkeypatch):
    dummy = DummyDataset(columns=["task_id", "prompt", "canonical_solution", "test", "entry_point"])
    dataset_dict = {"test": dummy}
    _, calls = _setup_fake_datasets(monkeypatch, return_value=dataset_dict)

    result = load_humaneval_dataset()
    assert result is dummy
    assert calls["args"] is not None
    assert calls["args"][0] == "openai_humaneval"


def test_required_fields_present_in_returned_dataset(monkeypatch):
    required = {"task_id", "prompt", "canonical_solution", "test", "entry_point"}
    dummy = DummyDataset(columns=["task_id", "prompt", "canonical_solution", "test", "entry_point", "extra"])
    dataset_dict = {"test": dummy}
    _setup_fake_datasets(monkeypatch, return_value=dataset_dict)

    ds = load_humaneval_dataset()
    cols = set(getattr(ds, "column_names", []) or [])
    if not cols and getattr(ds, "features", None) is not None:
        try:
            cols = set(ds.features.keys())
        except Exception:
            cols = set()
    assert required.issubset(cols)


def test_missing_test_split_raises_error(monkeypatch):
    dummy = DummyDataset(columns=["task_id", "prompt", "canonical_solution", "test", "entry_point"])
    dataset_dict = {"validation": dummy}
    _setup_fake_datasets(monkeypatch, return_value=dataset_dict)

    with pytest.raises(Exception):
        load_humaneval_dataset()


def test_load_dataset_exception_propagates(monkeypatch):
    _setup_fake_datasets(monkeypatch, side_effect=RuntimeError("network error"))

    with pytest.raises(RuntimeError):
        load_humaneval_dataset()