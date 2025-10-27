import pytest
from definition_7bfa10e5a756457e934804e28edb8ce1 import run_tests

@pytest.mark.parametrize(
    "code, tests, inputs, expected_output",
    [
        # Test Case 1: Successful execution - Code runs correctly, all tests pass, no errors.
        (
            "def add(a, b): return a + b",
            ["def test_addition(candidate_func, input_pair): assert candidate_func(*input_pair) == sum(input_pair)"],
            [(1, 2), (0, 0), (-1, 5)],
            {"passed": True, "property_violations": [], "runtime_errors": [], "timeouts": []},
        ),
        # Test Case 2: Runtime error - Code raises an exception during execution (e.g., ZeroDivisionError).
        (
            "def divide(a, b): return a / b",
            ["def test_division(candidate_func, input_pair): candidate_func(*input_pair)"],
            [(4, 2), (10, 0)], # One input (10, 0) should cause ZeroDivisionError
            {"passed": False, "property_violations": [], "runtime_errors_exist": True, "timeouts": []},
        ),
        # Test Case 3: Property violation/Assertion failure - Code produces incorrect results, failing an assertion.
        (
            "def my_sort(arr): return sorted(arr, reverse=True)", # Incorrect sort order (reverse)
            ["def check_is_sorted(candidate_func, input_list): sorted_arr = candidate_func(input_list); for i in range(len(sorted_arr) - 1): assert sorted_arr[i] <= sorted_arr[i+1]"],
            [[3, 1, 2], [5, 4]], # These inputs will expose the reverse sort, failing the 'is sorted' property
            {"passed": False, "property_violations_exist": True, "runtime_errors": [], "timeouts": []},
        ),
        # Test Case 4: Timeout scenario - Code enters an infinite loop or takes too long to execute.
        (
            "def infinite_loop(): import time; time.sleep(100)", # Simulates a long running task
            ["def test_long_running(candidate_func, _): candidate_func()"],
            [None], # Dummy input
            {"passed": False, "property_violations": [], "runtime_errors": [], "timeouts_exist": True},
        ),
        # Test Case 5: Invalid input type for 'code' argument - Expects a TypeError from run_tests itself.
        (
            123, # An integer instead of a string
            [],
            [],
            TypeError,
        ),
    ],
)
def test_run_tests_functionality(code, tests, inputs, expected_output):
    """
    Tests the run_tests function across various scenarios including success,
    runtime errors, property violations, timeouts, and invalid input types.
    """
    if isinstance(expected_output, type) and issubclass(expected_output, Exception):
        # If the expected output is an exception type, we expect run_tests to raise it.
        with pytest.raises(expected_output):
            run_tests(code, tests, inputs)
    else:
        # Otherwise, we expect a dictionary result and check its contents.
        result = run_tests(code, tests, inputs)
        assert isinstance(result, dict), "Result should be a dictionary"

        assert result.get("passed") == expected_output["passed"], \
            f"Expected passed status {expected_output['passed']}, got {result.get('passed')}"

        if "property_violations_exist" in expected_output and expected_output["property_violations_exist"]:
            assert len(result.get("property_violations", [])) > 0, "Expected property violations, but none found"
        else:
            assert result.get("property_violations", []) == [], "Expected no property violations, but found some"

        if "runtime_errors_exist" in expected_output and expected_output["runtime_errors_exist"]:
            assert len(result.get("runtime_errors", [])) > 0, "Expected runtime errors, but none found"
        else:
            assert result.get("runtime_errors", []) == [], "Expected no runtime errors, but found some"

        if "timeouts_exist" in expected_output and expected_output["timeouts_exist"]:
            assert len(result.get("timeouts", [])) > 0, "Expected timeouts, but none found"
        else:
            assert result.get("timeouts", []) == [], "Expected no timeouts, but found some"