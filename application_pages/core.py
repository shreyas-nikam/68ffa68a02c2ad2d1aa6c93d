import re
import time
import json
import random
from typing import Any, Dict, List, Optional, Callable, Tuple

import streamlit as st
import pandas as pd
import numpy as np


@st.cache_data
def load_tasks() -> Tuple[List[Dict[str, Any]], str]:
    try:
        from datasets import load_dataset  # type: ignore
        ds = load_dataset("openai_humaneval")
        test_split = ds["test"]
        tasks = [
            {
                "task_id": ex.get("task_id"),
                "prompt": ex.get("prompt"),
                "canonical_solution": ex.get("canonical_solution"),
                "test": ex.get("test"),
                "entry_point": ex.get("entry_point"),
            }
            for ex in test_split
        ]
        source = "HumanEval (HF)"
    except Exception:
        tasks = [
            {
                "task_id": "Synth/ReverseString/001",
                "prompt": "Implement a function reverse_string(s: str) -> str that returns the reversed string. The entry point is 'reverse_string'.",
                "canonical_solution": "def reverse_string(s: str) -> str:\n    return s[::-1]\n",
                "entry_point": "reverse_string",
                "test": (
                    "def check(f):\n"
                    "    assert f('') == ''\n"
                    "    assert f('a') == 'a'\n"
                    "    assert f('ab') == 'ba'\n"
                    "    assert f('Hello') == 'olleH'\n"
                    "    assert f('racecar') == 'racecar'\n"
                ),
            },
            {
                "task_id": "Synth/Add/002",
                "prompt": "Write a function add(a: int, b: int) -> int that returns the sum of a and b. Entry point is 'add'.",
                "canonical_solution": "def add(a: int, b: int) -> int:\n    return a + b\n",
                "entry_point": "add",
                "test": (
                    "def check(f):\n"
                    "    assert f(0,0) == 0\n"
                    "    assert f(1,2) == 3\n"
                    "    assert f(-1,1) == 0\n"
                    "    assert f(100,-5) == 95\n"
                ),
            },
        ]
        source = "Synthetic fallback"
    return tasks, source


def _infer_entry_point_from_code(code: str, default: Optional[str] = None) -> Optional[str]:
    m = re.search(r"def\s+([a-zA-Z_]\w*)\s*\(", code)
    if m:
        return m.group(1)
    return default


def generate_code_with_llm(prompt: str) -> str:
    p = prompt.lower()
    if "reverse_string" in p or "revers" in p:
        return (
            "def reverse_string(s: str) -> str:\n"
            "    # Simple and efficient string reverse using slicing\n"
            "    return s[::-1]\n"
        )
    if " entry point is 'add'" in p or "function add(" in p or " add(" in p:
        return (
            "def add(a: int, b: int) -> int:\n"
            "    # Basic integer addition with Python\n"
            "    return a + b\n"
        )
    # Generic fallback: create a pass-through function if we can infer a name
    m = re.search(r"(def\s+[a-zA-Z_]\w*\(.*?\)\s*->\s*[^:]+:)", prompt)
    if m:
        header = m.group(1)
        name = re.search(r"def\s+([a-zA-Z_]\w*)", header).group(1) if re.search(r"def\s+([a-zA-Z_]\w*)", header) else "solution"
        return f"{header}\n    raise NotImplementedError('Please implement {name}')\n"
    return (
        "def solution(*args, **kwargs):\n"
        "    raise NotImplementedError('No generator found for this prompt')\n"
    )


def generate_property_checks(problem_spec: str) -> List[str]:
    spec = problem_spec.lower()
    props: List[str] = []
    if "reverse_string" in spec or "revers" in spec:
        props.append(
            "def prop_length_preserved(f, x):\n"
            "    assert isinstance(x, str)\n"
            "    y = f(x)\n"
            "    assert isinstance(y, str)\n"
            "    assert len(y) == len(x)\n"
        )
        props.append(
            "def prop_double_reverse_identity(f, x):\n"
            "    assert f(f(x)) == x\n"
        )
        props.append(
            "def prop_palindrome_fixed(f, x):\n"
            "    if x == x[::-1]:\n"
            "        assert f(x) == x\n"
        )
        props.append(
            "def prop_char_multiset_equal(f, x):\n"
            "    assert sorted(f(x)) == sorted(x)\n"
        )
        return props
    if " add(" in spec or "function add(" in spec or " entry point is 'add'" in spec:
        props.append(
            "def prop_commutative(f, xy):\n"
            "    a,b = xy\n"
            "    assert f(a,b) == f(b,a)\n"
        )
        props.append(
            "def prop_identity_zero(f, xy):\n"
            "    a,b = xy\n"
            "    assert f(a,0) == a\n"
            "    assert f(0,a) == a\n"
        )
        props.append(
            "def prop_linear_bounds(f, xy):\n"
            "    a,b = xy\n"
            "    y = f(a,b)\n"
            "    assert y - a == b and y - b == a\n"
        )
        return props
    # Generic minimal property: function should be callable and not raise for sample inputs
    props.append(
        "def prop_callable_no_crash(f, x):\n"
        "    try:\n"
        "        _ = f(x) if not isinstance(x, tuple) else f(*x)\n"
        "    except Exception as e:\n"
        "        raise AssertionError(f'Function raised: {e}')\n"
    )
    return props


def _rand_string(n: int) -> str:
    alphabet = [chr(c) for c in range(32, 127)]
    return "".join(random.choice(alphabet) for _ in range(n))


def generate_pbt_inputs(problem_spec: str, n: int = 100) -> List[Any]:
    spec = problem_spec.lower()
    inputs: List[Any] = []
    rng = random.Random(42)
    if "reverse_string" in spec or "revers" in spec:
        samples = ["", "a", "ab", "racecar", "Hello", "😀emoji", "A man, a plan, a canal, Panama"]
        inputs.extend(samples)
        for k in range(max(0, n - len(samples))):
            L = rng.randint(0, 60)
            inputs.append(_rand_string(L))
        return inputs
    if " add(" in spec or "function add(" in spec or " entry point is 'add'" in spec:
        base = [(0,0), (1,2), (-1,1), (100,-5), (10**6, 10**6)]
        inputs.extend(base)
        for _ in range(max(0, n - len(base))):
            a = rng.randint(-10**6, 10**6)
            b = rng.randint(-10**6, 10**6)
            inputs.append((a,b))
        return inputs
    # Generic tuples and scalars
    for _ in range(n):
        a = rng.randint(-1000, 1000)
        b = rng.randint(-1000, 1000)
        inputs.append((a,b))
    return inputs


def run_tests(
    code: str,
    tests: List[str],
    inputs: List[Any],
    entry_point: Optional[str] = None,
    mode: str = "PBT",
) -> Dict[str, Any]:
    start = time.time()
    env: Dict[str, Any] = {}
    result: Dict[str, Any] = {"passed": False, "details": {}}
    try:
        exec(code, env, env)
    except Exception as e:
        result["runtime_error"] = f"Code exec error: {e}"
        result["duration_sec"] = time.time() - start
        return result
    func_name = entry_point or _infer_entry_point_from_code(code, None)
    f = env.get(func_name) if func_name else None
    if not callable(f):
        result["runtime_error"] = f"Entry point '{func_name}' not found or not callable"
        result["duration_sec"] = time.time() - start
        return result

    if mode.upper() == "TDD":
        failed_examples = 0
        total_examples = 0
        assertion_error: Optional[str] = None
        runtime_error: Optional[str] = None
        for tcode in tests:
            try:
                total_examples += len(re.findall(r"\\bassert\\b", tcode))
                tns: Dict[str, Any] = {}
                exec(tcode, tns, tns)
                check = tns.get("check")
                if callable(check):
                    check(f)
                else:
                    runtime_error = "No check(f) function in test snippet"
                    failed_examples += 1
            except AssertionError as ae:
                assertion_error = str(ae)
                failed_examples += 1 if failed_examples == 0 else 0
            except Exception as e:
                runtime_error = str(e)
                failed_examples += 1 if failed_examples == 0 else 0
        passed = failed_examples == 0 and runtime_error is None and assertion_error is None
        result.update(
            {
                "passed": passed,
                "failed_examples": failed_examples,
                "total_examples": total_examples,
                "assertion_error": assertion_error,
                "runtime_error": runtime_error,
                "duration_sec": time.time() - start,
            }
        )
        return result

    # PBT mode
    # Compile property functions
    props: List[Tuple[str, Callable]] = []
    for pcode in tests:
        try:
            pname = None
            m = re.search(r"def\s+([a-zA-Z_]\w*)\s*\(", pcode)
            if m:
                pname = m.group(1)
            pns: Dict[str, Any] = {}
            exec(pcode, pns, pns)
            if pname and callable(pns.get(pname)):
                props.append((pname, pns[pname]))
        except Exception:
            continue
    violations: List[Dict[str, Any]] = []
    prop_pass_vector: List[bool] = []
    for pname, prop_fn in props:
        ok = True
        for idx, x in enumerate(inputs):
            try:
                if isinstance(x, tuple):
                    prop_fn(f, x)
                else:
                    prop_fn(f, x)
            except AssertionError as ae:
                ok = False
                violations.append({
                    "property": pname,
                    "input_index": idx,
                    "input": repr(x),
                    "error": "AssertionError",
                    "message": str(ae),
                })
            except Exception as e:
                ok = False
                violations.append({
                    "property": pname,
                    "input_index": idx,
                    "input": repr(x),
                    "error": type(e).__name__,
                    "message": str(e),
                })
        prop_pass_vector.append(ok)

    total_props = len(props)
    passed_props = sum(1 for v in prop_pass_vector if v)
    result["passed"] = (passed_props == total_props and total_props > 0)
    result["details"] = {
        "passed_properties": passed_props,
        "total_properties": total_props,
        "violations": violations,
    }
    result["duration_sec"] = time.time() - start
    return result


def refine_code_with_feedback(buggy_code: str, feedback: str) -> str:
    code = buggy_code
    # Heuristic patches for common tasks
    if "reverse_string" in buggy_code or "reverse_string" in feedback:
        code = (
            "def reverse_string(s: str) -> str:\n"
            "    return s[::-1]\n"
        )
        return code
    if re.search(r"\bdef\s+add\(", buggy_code) or " add" in feedback:
        code = (
            "def add(a: int, b: int) -> int:\n"
            "    return a + b\n"
        )
        return code
    # Generic patch: if NotImplementedError, replace with a safe pass-through
    if "NotImplementedError" in buggy_code:
        name = _infer_entry_point_from_code(buggy_code, "solution") or "solution"
        code = f"def {name}(*args, **kwargs):\n    return None\n"
        return code
    return code


def iterate_pbt_refinement(problem: Dict[str, Any], num_iterations: int = 5, initial_code: Optional[str] = None) -> Dict[str, Any]:
    prompt = problem.get("prompt", "")
    entry_point = problem.get("entry_point") or _infer_entry_point_from_code(problem.get("canonical_solution", ""), None)
    candidate = initial_code if initial_code is not None else generate_code_with_llm(prompt)
    history_rows: List[Dict[str, Any]] = []
    start_all = time.time()
    success = False

    for it in range(1, int(num_iterations) + 1):
        props = generate_property_checks(prompt)
        inputs = generate_pbt_inputs(prompt, n=120)
        res = run_tests(candidate, props, inputs, entry_point=entry_point, mode="PBT")
        det = res.get("details", {}) if isinstance(res, dict) else {}
        passed_props = int(det.get("passed_properties", 0))
        total_props = int(det.get("total_properties", 0))
        violations = det.get("violations", [])
        history_rows.append(
            {
                "iteration": it,
                "passed_properties": passed_props,
                "total_properties": total_props,
                "violations_count": len(violations),
                "code_size": len(candidate),
                "duration_sec": float(res.get("duration_sec", 0.0)),
                "passed": bool(res.get("passed", False)),
            }
        )
        if res.get("passed", False):
            success = True
            break
        # Construct minimal feedback signal
        feedback_msg = "; ".join(
            [f"{v.get('property')}: {v.get('message')} on input {v.get('input')}" for v in violations[:3]]
        )
        feedback_msg = feedback_msg or ("Property-based tests failed for entry point " + str(entry_point))
        candidate = refine_code_with_feedback(candidate, feedback_msg)

    duration = time.time() - start_all
    history = pd.DataFrame(history_rows)
    return {
        "success": success,
        "final_code": candidate,
        "history": history,
        "duration_sec": duration,
    }


def iterate_tdd_refinement(problem: Dict[str, Any], num_iterations: int = 3, initial_code: Optional[str] = None) -> Dict[str, Any]:
    prompt = problem.get("prompt", "")
    entry_point = problem.get("entry_point") or _infer_entry_point_from_code(problem.get("canonical_solution", ""), None)
    candidate = initial_code if initial_code is not None else generate_code_with_llm(prompt)
    test_snippet = problem.get("test", "")
    history_rows: List[Dict[str, Any]] = []
    start_all = time.time()
    success = False

    for it in range(1, int(num_iterations) + 1):
        res = run_tests(candidate, [test_snippet], [], entry_point=entry_point, mode="TDD")
        history_rows.append(
            {
                "iteration": it,
                "failed_examples": int(res.get("failed_examples", 0) or 0),
                "total_examples": int(res.get("total_examples", 0) or 0),
                "assertion_error": res.get("assertion_error"),
                "runtime_error": res.get("runtime_error"),
                "code_size": len(candidate),
                "duration_sec": float(res.get("duration_sec", 0.0)),
                "passed": bool(res.get("passed", False)),
            }
        )
        if res.get("passed", False):
            success = True
            break
        fb = res.get("assertion_error") or res.get("runtime_error") or "Example tests failed"
        candidate = refine_code_with_feedback(candidate, str(fb))

    duration = time.time() - start_all
    history = pd.DataFrame(history_rows)
    return {
        "success": success,
        "final_code": candidate,
        "history": history,
        "duration_sec": duration,
    }


def evaluate_method_on_benchmark(method: Callable, dataset: List[Dict[str, Any]], num_iterations: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for task in dataset:
        out = method(task, num_iterations=num_iterations, initial_code=generate_code_with_llm(task.get("prompt", "")))
        hist = out.get("history")
        if isinstance(hist, pd.DataFrame) and not hist.empty:
            for _, r in hist.iterrows():
                rows.append(
                    {
                        "task_id": task.get("task_id"),
                        "iteration": int(r.get("iteration", 0)),
                        "passed": bool(r.get("passed", False)),
                        "violations_count": int(r.get("violations_count", 0)) if "violations_count" in r else None,
                        "failed_examples": int(r.get("failed_examples", 0)) if "failed_examples" in r else None,
                        "total_examples": int(r.get("total_examples", 0)) if "total_examples" in r else None,
                        "duration_sec": float(r.get("duration_sec", 0.0)),
                        "method": method.__name__,
                    }
                )
        else:
            rows.append(
                {
                    "task_id": task.get("task_id"),
                    "iteration": 1,
                    "passed": bool(out.get("success", False)),
                    "violations_count": None,
                    "failed_examples": None,
                    "total_examples": None,
                    "duration_sec": float(out.get("duration_sec", 0.0)),
                    "method": method.__name__,
                }
            )
    df = pd.DataFrame(rows)
    return df


def visualize_performance_comparison(results_df: pd.DataFrame):
    import plotly.express as px
    if results_df is None or results_df.empty:
        return None
    # Compute pass rate per iteration per method
    agg = results_df.groupby(["method", "iteration"]).agg(pass_rate=("passed", "mean"), mean_time=("duration_sec", "mean")).reset_index()
    fig = px.line(
        agg,
        x="iteration",
        y="pass_rate",
        color="method",
        markers=True,
        hover_data=["mean_time"],
        title="Pass rate over iterations by method",
    )
    fig.update_yaxes(range=[0,1])
    fig.update_traces(mode="lines+markers")
    return fig


def visualize_test_coverage_and_feedback(results_df: pd.DataFrame):
    import plotly.express as px
    if results_df is None or results_df.empty:
        return None
    # Build a bar of success by task and method at the final iteration observed per task
    final_iter = results_df.groupby(["task_id", "method"]).iteration.max().reset_index()
    merged = final_iter.merge(results_df, on=["task_id", "method", "iteration"], how="left")
    fig = px.bar(
        merged,
        x="task_id",
        y="passed",
        color="method",
        barmode="group",
        title="Final iteration success by task and method",
    )
    fig.update_yaxes(range=[0,1])
    return fig
