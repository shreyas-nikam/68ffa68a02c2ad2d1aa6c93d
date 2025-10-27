import pytest
from definition_dbe06f375c3249f89c23fd102644778d import iterate_tdd_refinement
import copy

def build_problem(overrides=None):
    problem = {
        "task_id": "HumanEval/000",
        "prompt": "Write a function foo() that returns 42.",
        "entry_point": "foo",
        "test": """
def check(candidate):
    assert candidate() == 42
"""
    }
    if overrides:
        problem.update(overrides)
    return problem

def test_returns_expected_summary_structure_on_valid_problem():
    problem = build_problem()
    result = iterate_tdd_refinement(problem)
    assert isinstance(result, dict), "Expected a dict summary result"
    # required fields
    for key in ["task_id", "final_code", "success", "iteration_count", "failing_examples_over_time"]:
        assert key in result, f"Missing key in summary: {key}"
    assert isinstance(result["task_id"], str)
    assert isinstance(result["final_code"], str)
    assert isinstance(result["success"], bool)
    assert isinstance(result["iteration_count"], int)
    # failing_examples_over_time could be list or similar collection
    assert isinstance(result["failing_examples_over_time"], (list, tuple))
    # timing information: allow flexible key naming, but ensure at least one timing-related key exists
    assert any("time" in k.lower() for k in result.keys()), "Expected timing information in the summary"

@pytest.mark.parametrize("missing_key", ["task_id", "prompt", "entry_point", "test"])
def test_raises_on_missing_required_fields(missing_key):
    problem = build_problem()
    problem.pop(missing_key)
    with pytest.raises(Exception):
        iterate_tdd_refinement(problem)

@pytest.mark.parametrize("bad_input", [None, 123, 3.14, "not a dataset entry", []])
def test_raises_on_invalid_problem_type(bad_input):
    with pytest.raises(Exception):
        iterate_tdd_refinement(bad_input)

def test_handles_empty_test_string():
    problem = build_problem({"test": ""})
    result = iterate_tdd_refinement(problem)
    assert isinstance(result, dict)
    assert "success" in result and isinstance(result["success"], bool)

def test_does_not_mutate_input_problem():
    problem = build_problem()
    original = copy.deepcopy(problem)
    try:
        iterate_tdd_refinement(problem)
    finally:
        assert problem == original, "Function should not mutate the input problem object"