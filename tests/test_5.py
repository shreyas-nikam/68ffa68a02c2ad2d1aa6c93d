import pytest
from definition_234a5a3454b04e01a27733070f4d830f import refine_code_with_feedback


def test_refinement_changes_code_on_bug_feedback():
    buggy_code = "def add(a, b):\n    return a - b\n"
    feedback = "Bug: add subtracts instead of adding. Replace '-' with '+'."
    refined = refine_code_with_feedback(buggy_code, feedback)
    assert isinstance(refined, str)
    assert refined.strip() != ""
    assert refined != buggy_code, "Refinement should modify the buggy code when actionable feedback is provided."
    assert "```" not in refined, "Refined code should not include Markdown code fences."
    compile(refined, "<refined>", "exec")


def test_handles_empty_feedback():
    buggy_code = "def noop(x):\n    return x\n"
    feedback = ""
    refined = refine_code_with_feedback(buggy_code, feedback)
    assert isinstance(refined, str)
    assert refined.strip() != ""
    assert "```" not in refined
    compile(refined, "<refined>", "exec")


@pytest.mark.parametrize("buggy_code, feedback", [
    (None, "some feedback"),
    ("print('hello')", None),
    (123, "feedback"),
])
def test_invalid_input_types_raise(buggy_code, feedback):
    with pytest.raises((TypeError, ValueError)):
        refine_code_with_feedback(buggy_code, feedback)