import pytest
from definition_3d166e83f4eb421e9a84a87931664827 import generate_code_with_llm


def test_generate_code_with_valid_add_prompt_executes_correctly():
    prompt = (
        "HumanEval Problem:\n"
        "Write a function add(a: int, b: int) -> int that returns the sum of a and b.\n"
        "The entry point is 'add'. Provide only Python code implementing add and any helpers."
    )
    code = generate_code_with_llm(prompt)
    assert isinstance(code, str), "Expected returned value to be a string of Python code."
    assert "def add" in code, "Generated code must define the entry point function 'add'."
    assert "```" not in code, "Generated code should not include code fences."

    ns = {}
    exec(code, ns)
    assert callable(ns.get("add")), "Function 'add' must be defined in the generated code."
    assert ns["add"](2, 3) == 5
    assert ns["add"](-1, 1) == 0


def test_generate_code_strips_code_fences_and_defines_entry_point():
    prompt = (
        "HumanEval Problem:\n"
        "Implement a function square(x: int) -> int that returns x*x.\n"
        "Entry point is 'square'.\n"
        "Example (for context only):\n"
        "```python\n"
        "def bad_example():\n"
        "    pass\n"
        "```\n"
        "Return only the implementation for 'square'."
    )
    code = generate_code_with_llm(prompt)
    assert isinstance(code, str)
    assert "def square" in code
    assert "```" not in code

    ns = {}
    exec(code, ns)
    assert ns["square"](4) == 16
    assert ns["square"](0) == 0
    assert ns["square"](-3) == 9


def test_empty_prompt_raises_valueerror():
    with pytest.raises(ValueError):
        generate_code_with_llm("")


def test_non_string_prompt_raises_typeerror():
    with pytest.raises(TypeError):
        generate_code_with_llm(123)  # type: ignore[arg-type]


def test_generated_code_is_compilable_and_defines_function():
    prompt = (
        "Write a function reverse_string(s: str) -> str that returns the reverse of s.\n"
        "The entry point is 'reverse_string'. Provide only Python code."
    )
    code = generate_code_with_llm(prompt)
    assert isinstance(code, str)
    assert "def reverse_string" in code
    # Ensure code is at least syntactically valid
    compile(code, filename="<generated>", mode="exec")
    ns = {}
    exec(code, ns)
    assert ns["reverse_string"]("abc") == "cba"
    assert ns["reverse_string"]("") == ""