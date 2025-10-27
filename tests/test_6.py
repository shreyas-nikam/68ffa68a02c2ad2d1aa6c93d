import sys
import pytest
from definition_c77da142c5814ab59edce7726652f0b1 import iterate_pbt_refinement

def _get_module():
    return sys.modules[iterate_pbt_refinement.__module__]

def _sample_problem():
    return {
        "task_id": "task_001",
        "prompt": "You are given a function f that should return the input.",
        "entry_point": "f",
        "test": "def test_public(): assert f(1) == 1",
    }

def test_valid_flow_single_iteration(monkeypatch):
    m = _get_module()

    def fake_generate_code_with_llm(prompt: str) -> str:
        return "def f(x):\n    return x"

    def fake_generate_property_checks(problem_spec: str):
        return ["def prop_identity(f, x): assert f(x) == x"]

    def fake_generate_pbt_inputs(property_spec: str):
        return [0, 1, -1, 10**6]

    def fake_run_tests(code: str, tests, inputs):
        return {
            "public_tests_passed": True,
            "property_violations": [],
            "runtime_errors": [],
            "timeouts": False,
            "all_passed": True,
            "feedback": "All checks passed.",
        }

    def fake_refine_code_with_feedback(buggy_code: str, feedback: str) -> str:
        return buggy_code

    monkeypatch.setattr(m, "generate_code_with_llm", fake_generate_code_with_llm, raising=False)
    monkeypatch.setattr(m, "generate_property_checks", fake_generate_property_checks, raising=False)
    monkeypatch.setattr(m, "generate_pbt_inputs", fake_generate_pbt_inputs, raising=False)
    monkeypatch.setattr(m, "run_tests", fake_run_tests, raising=False)
    monkeypatch.setattr(m, "refine_code_with_feedback", fake_refine_code_with_feedback, raising=False)

    result = iterate_pbt_refinement(_sample_problem())
    assert isinstance(result, dict)
    assert result.get("task_id") == "task_001"
    assert isinstance(result.get("final_code"), str) and len(result["final_code"]) > 0
    assert result.get("success") is True
    assert isinstance(result.get("iteration_count"), int) and result["iteration_count"] >= 1
    assert isinstance(result.get("per_iteration_feedback"), list)

def test_handles_property_violation_then_fix(monkeypatch):
    m = _get_module()

    def fake_generate_code_with_llm(prompt: str) -> str:
        return "def f(x):\n    return x if x != -1 else x + 1  # buggy for -1"

    def fake_generate_property_checks(problem_spec: str):
        return ["def prop_identity(f, x): assert f(x) == x"]

    def fake_generate_pbt_inputs(property_spec: str):
        return [0, -1, 2]

    call_count = {"n": 0}
    def fake_run_tests(code: str, tests, inputs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return {
                "public_tests_passed": False,
                "property_violations": [{"input": -1, "property": "identity", "message": "f(-1) != -1"}],
                "runtime_errors": [],
                "timeouts": False,
                "all_passed": False,
                "feedback": "Violation: identity fails at x=-1",
            }
        return {
            "public_tests_passed": True,
            "property_violations": [],
            "runtime_errors": [],
            "timeouts": False,
            "all_passed": True,
            "feedback": "All checks passed after fix.",
        }

    def fake_refine_code_with_feedback(buggy_code: str, feedback: str) -> str:
        return "def f(x):\n    return x  # fixed"

    monkeypatch.setattr(m, "generate_code_with_llm", fake_generate_code_with_llm, raising=False)
    monkeypatch.setattr(m, "generate_property_checks", fake_generate_property_checks, raising=False)
    monkeypatch.setattr(m, "generate_pbt_inputs", fake_generate_pbt_inputs, raising=False)
    monkeypatch.setattr(m, "run_tests", fake_run_tests, raising=False)
    monkeypatch.setattr(m, "refine_code_with_feedback", fake_refine_code_with_feedback, raising=False)

    result = iterate_pbt_refinement(_sample_problem())
    assert isinstance(result, dict)
    assert result.get("success") is True
    assert result.get("iteration_count", 0) >= 2
    feedback = result.get("per_iteration_feedback", [])
    assert isinstance(feedback, list) and len(feedback) >= 2
    assert any(item.get("property_violations") for item in feedback[:-1])

def test_invalid_input_type_raises():
    with pytest.raises((TypeError, ValueError)):
        iterate_pbt_refinement(123)

def test_missing_required_fields_raises():
    problem = _sample_problem()
    problem.pop("prompt")
    with pytest.raises((KeyError, ValueError)):
        iterate_pbt_refinement(problem)

def test_result_schema_types(monkeypatch):
    m = _get_module()

    def fake_generate_code_with_llm(prompt: str) -> str:
        return "def f(x):\n    return x"

    def fake_generate_property_checks(problem_spec: str):
        return ["def prop_identity(f, x): assert f(x) == x"]

    def fake_generate_pbt_inputs(property_spec: str):
        return [1]

    def fake_run_tests(code: str, tests, inputs):
        return {
            "public_tests_passed": True,
            "property_violations": [],
            "runtime_errors": [],
            "timeouts": False,
            "all_passed": True,
            "feedback": "OK",
        }

    def fake_refine_code_with_feedback(buggy_code: str, feedback: str) -> str:
        return buggy_code

    monkeypatch.setattr(m, "generate_code_with_llm", fake_generate_code_with_llm, raising=False)
    monkeypatch.setattr(m, "generate_property_checks", fake_generate_property_checks, raising=False)
    monkeypatch.setattr(m, "generate_pbt_inputs", fake_generate_pbt_inputs, raising=False)
    monkeypatch.setattr(m, "run_tests", fake_run_tests, raising=False)
    monkeypatch.setattr(m, "refine_code_with_feedback", fake_refine_code_with_feedback, raising=False)

    result = iterate_pbt_refinement(_sample_problem())
    assert isinstance(result, dict)
    assert set(["task_id", "final_code", "success", "iteration_count", "per_iteration_feedback"]).issubset(result.keys())
    assert isinstance(result["per_iteration_feedback"], list)
    if "summary" in result:
        summary = result["summary"]
        assert isinstance(summary, dict)
        if "property_violations_resolved" in summary:
            assert isinstance(summary["property_violations_resolved"], int) and summary["property_violations_resolved"] >= 0