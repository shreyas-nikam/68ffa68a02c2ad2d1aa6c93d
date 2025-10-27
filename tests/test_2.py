import pytest
from definition_791c8d4b1ed04258ad34e17ad714f4be import generate_property_checks


def _sample_sort_spec() -> str:
    return (
        "Task: Implement a function sort_list(lst) that returns a new list with the same elements "
        "in non-decreasing order.\n"
        "Properties to capture include:\n"
        "- Output length equals input length.\n"
        "- Output is sorted non-decreasing.\n"
        "- Multiset of elements preserved.\n"
        "- Idempotence: applying the function twice yields the same result.\n"
        "Entry point: sort_list\n"
    )


def test_valid_spec_returns_compilable_properties():
    spec = _sample_sort_spec()
    props = generate_property_checks(spec)
    assert isinstance(props, list), "Expected a list of property strings"
    assert len(props) > 0, "Expected at least one property to be generated"

    for p in props:
        assert isinstance(p, str), "Each property should be a string"
        assert p.strip(), "Property strings should not be empty"
        assert "def " in p, "Each property should define a test function"
        assert "assert" in p, "Properties should contain at least one assertion"
        # Ensure syntactic correctness (name resolution is not required at compile time)
        compile(p, "<property>", "exec")


def test_non_string_input_raises_type_error():
    with pytest.raises(TypeError):
        generate_property_checks(None)  # type: ignore


def test_empty_or_whitespace_string_raises_value_error():
    with pytest.raises(ValueError):
        generate_property_checks("   ")


def test_generated_properties_are_unique():
    spec = _sample_sort_spec()
    props = generate_property_checks(spec)
    assert len(set(props)) == len(props), "Generated properties should be unique (no duplicates)"


def test_large_spec_returns_list_of_strings():
    large_spec = ("Compute Fibonacci numbers.\n" * 500) + "Entry point: fib\n"
    props = generate_property_checks(large_spec)
    assert isinstance(props, list), "Expected a list even for large specs"
    for p in props:
        assert isinstance(p, str), "All returned properties should be strings"
