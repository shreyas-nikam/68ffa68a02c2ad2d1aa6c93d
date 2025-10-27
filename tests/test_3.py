import pytest
from definition_d94a42dd66744f4b8b53392f8a4854a5 import generate_pbt_inputs


def test_returns_list_for_empty_spec():
    outputs = generate_pbt_inputs("")
    assert isinstance(outputs, list)


def test_integer_domain_includes_negative_zero_positive():
    spec = "Generate integers covering edge cases: include negative numbers, zero, and positive numbers."
    outputs = generate_pbt_inputs(spec)
    ints = [x for x in outputs if isinstance(x, int) and not isinstance(x, bool)]
    assert len(ints) > 0
    assert any(i < 0 for i in ints)
    assert any(i == 0 for i in ints)
    assert any(i > 0 for i in ints)


def test_list_of_ints_includes_empty_singleton_and_duplicates():
    spec = "Inputs are lists of integers for sorting; include an empty list, a single-element list, and a list with duplicates."
    outputs = generate_pbt_inputs(spec)
    lists = [x for x in outputs if isinstance(x, list)]
    assert any(l == [] for l in lists)
    int_lists = [l for l in lists if all(isinstance(i, int) and not isinstance(i, bool) for i in l)]
    assert any(len(l) == 1 for l in int_lists)
    assert any(len(l) >= 2 and len(l) != len(set(l)) for l in int_lists)


def test_tuple_pairs_cover_all_orderings():
    spec = "Inputs are tuples (a, b) of integers; include cases where a < b, a == b, and a > b."
    outputs = generate_pbt_inputs(spec)
    pairs = [
        t for t in outputs
        if isinstance(t, tuple) and len(t) == 2 and all(isinstance(v, int) and not isinstance(v, bool) for v in t)
    ]
    assert any(a < b for a, b in pairs)
    assert any(a == b for a, b in pairs)
    assert any(a > b for a, b in pairs)


def test_strings_include_empty_whitespace_and_long():
    spec = "Inputs are strings; include empty string, whitespace-only strings, and longer typical strings."
    outputs = generate_pbt_inputs(spec)
    strings = [s for s in outputs if isinstance(s, str)]
    assert any(s == "" for s in strings)
    assert any(len(s) > 0 and s.strip() == "" for s in strings)
    assert any(len(s) >= 10 for s in strings)