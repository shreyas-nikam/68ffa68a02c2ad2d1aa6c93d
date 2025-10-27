import pytest
from definition_752bbc36e8184634bd38063fcb2dac5f import run_tests


def test_run_tests_success_structure():
    code = "def square(x):\n    return x * x\n"
    tests = [
        "def test_square_basic():\n"
        "    assert square(3) == 9\n"
        "    assert square(-2) == 4\n"
    ]
    inputs = []
    result = run_tests(code, tests, inputs)
    assert isinstance(result, dict)
    for key in ["pass_fail", "property_violations", "runtime_errors", "timeouts"]:
        assert key in result

    assert isinstance(result["pass_fail"], dict)
    for v in result["pass_fail"].values():
        assert isinstance(v, bool)
    if result["pass_fail"]:
        assert all(result["pass_fail"].values())

    assert isinstance(result["property_violations"], list)
    assert isinstance(result["runtime_errors"], list)
    timeouts = result["timeouts"]
    if isinstance(timeouts, (list, tuple)):
        assert len(timeouts) == 0
    elif isinstance(timeouts, int):
        assert timeouts == 0


def test_reports_failed_assertion():
    code = "def is_even(x):\n    return x % 2 == 1\n"
    tests = [
        "def test_is_even_for_two():\n"
        "    assert is_even(2) is True\n"
    ]
    res = run_tests(code, tests, [])
    assert isinstance(res, dict)
    pf = res.get("pass_fail", {})
    failed = any(not v for v in pf.values())
    pv = res.get("property_violations", [])
    assert failed or (isinstance(pv, list) and len(pv) > 0)


def test_captures_runtime_error():
    code = "def div(a,b):\n    return a/b\n"
    tests = [
        "def test_zero_division():\n"
        "    div(1,0)\n"
    ]
    res = run_tests(code, tests, [])
    runtime_errors = res.get("runtime_errors")
    pf = res.get("pass_fail", {})
    assert (isinstance(runtime_errors, list) and len(runtime_errors) > 0) or any(not v for v in pf.values())


@pytest.mark.parametrize("code, tests, inputs, expected", [
    (123, ["pass"], [], TypeError),
    ("print('hi')", "not-a-list", [], TypeError),
    ("x=1", [123], [], TypeError),
])
def test_invalid_argument_types(code, tests, inputs, expected):
    with pytest.raises(expected):
        run_tests(code, tests, inputs)


def test_empty_tests_return_structure():
    code = "def foo():\n    return 42\n"
    res = run_tests(code, [], [])
    assert isinstance(res, dict)
    assert "pass_fail" in res and isinstance(res["pass_fail"], dict) and len(res["pass_fail"]) == 0
    assert "property_violations" in res
    assert "runtime_errors" in res
    assert "timeouts" in res