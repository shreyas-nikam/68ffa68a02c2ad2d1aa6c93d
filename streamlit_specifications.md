
# Streamlit Application Requirements Specification

## 1. Application Overview

**Learning Goals:**
- Explain and demonstrate the fundamental differences between Property-Based Testing (PBT) and Traditional Test-Driven Development (TDD) as applied to refining Large Language Model (LLM)-generated code.
- Illustrate why PBT provides stronger semantic guarantees than example-based TDD by validating invariants across broad input domains.
- Provide an end-to-end interactive workflow for loading HumanEval tasks, generating initial LLM code candidates, defining properties, running tests and properties, and iteratively refining the code.
- Quantitatively compare PBT and TDD iterative refinement strategies using metrics such as pass@1, repair success rate, property violations, and runtime.
- Visualize comparative performance statistics to guide practical adoption and CI/CD policy decisions.
- Demonstrate practical code snippets and workflows implemented in Python that embody the PBT versus TDD comparison.

## 2. User Interface Requirements

### Layout and Navigation Structure
- The app should have a sidebar for user controls and navigation.
- The main body will display content including problem descriptions, generated code, property definitions, test results, iterative refinement feedback, and comparative visualizations.
- Tabs or sections to toggle between:
  - Task selection and problem details
  - Generated code and test running interface
  - Iterative refinement process and feedback
  - Quantitative performance comparison visualizations
  - Summary insights and learnings

### Input Widgets and Controls
- **Task Selection**: Dropdown to select a HumanEval (or synthetic fallback) programming problem.
- **Iteration Controls**: Slider or numeric input controlling the number of refinement iterations.
- **Method Selector**: Radio buttons or dropdown to switch between PBT refinement and TDD refinement workflows.
- **Code Editor/Text Area**: Display and optionally allow editing of the generated/refined candidate code.
- **Run Tests Button**: Trigger execution of tests/properties on the current code.
- **Refresh Dataset Button**: Reload or reset dataset/tasks.
- Possible checkboxes to toggle display options for feedback details (assertion errors, runtime errors, etc.).

### Visualization Components
- Display generated problem prompt and description in markdown format.
- Display code snippets with syntax highlighting.
- Show property-based test code snippets and traditional test code snippets.
- Results tables summarizing:
  - Per-iteration pass/fail counts
  - Property violations and runtime errors
- Comparative plots:
  - Line chart showing pass rate over refinement iterations by method (PBT, TDD)
  - Bar charts for pass@1 and repair success rates by method
  - Heatmaps or line plots summarizing property violations, example failures, and input diversity over iterations
- Logs/feedback area showing human-readable feedback messages per iteration.

### Interactive Elements and Feedback Mechanisms
- Interactive execution of code generation, testing, and refinement with feedback displayed in real-time.
- Feedback summaries showing which properties failed, error messages, and suggestions for refinement.
- Visual feedback when candidate code passes all tests/properties.
- Visual cues for success/failure states (green/red highlights).
- Cumulative statistics updating as iterations progress.

## 3. Additional Requirements

### Annotation and Tooltip Specifications
- Mathematical formulas should be displayed using LaTeX rendering inline and in display mode according to the format rules:
  - Inline math enclosed by single dollar signs: $ ... $
  - Display math enclosed by double dollar signs:
    $$
    ...
    $$
- Tooltips for explanations of terms like "pass@1", "property violation", "repair success", "input diversity".
- Code blocks should have tooltips or information buttons explaining what the code is doing (e.g., property generation heuristics, test runner logic).
- Annotations on plots explaining axes, legends, and key metrics.

### Save State
- Preserve selected task, method, iteration count, and current candidate code during session reruns.
- Cache dataset loading to minimize delays on app startup.
- Allow downloading or exporting of final refined code and test/property summaries.

## 4. Notebook Content and Code Requirements

This section maps the key notebook content to Streamlit app components including the code stubs required to implement them. These cover all notebook elements and ensure markdown content is included appropriately.

### 4.1 Environment Setup and Imports
```python
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap
import time
import json
import re
from typing import Any, Dict, List, Optional, Callable

sns.set_theme(context="notebook", style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 4)
```
(Use this to setup the app environment)

### 4.2 Data Loading (HumanEval Dataset with Fallback)
```python
@st.cache_data
def load_tasks():
    try:
        from datasets import load_dataset
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
```

### 4.3 Code Generation Agent (Generator)
```python
def generate_code_with_llm(prompt: str) -> str:
    # Implementation of heuristic code generation from prompt as given
    # Returns Python code string implementing initial candidate solution
    # [Code same as notebook Section 3]
```
(Use in app to generate candidate code on problem selection or refresh)

### 4.4 Property Definition and Generation
```python
def generate_property_checks(problem_spec: str) -> List[str]:
    # Generate property-based test snippets as strings based on problem spec heuristics
    # [Code same as notebook Section 4]
```
(Display generated property code snippets in UI)

### 4.5 Property-Based Input Generation
```python
def generate_pbt_inputs(property_spec: str) -> List[Any]:
    # Generate diverse inputs to stress test properties and candidate code
    # [Code same as notebook Section 5]
```
(Used when running PBT refinement iterations)

### 4.6 Test Runner for Properties and Traditional Tests
```python
def run_tests(code: str, tests: List[str], inputs: List[Any]) -> Dict[str, Any]:
    # Executes code + test snippets on inputs, returning pass/fail, violations, errors
    # [Code same as notebook Section 6]
```
(Used to run property checks or TDD example tests)

### 4.7 Refinement Loop for Property-Based Testing (PBT)
```python
def refine_code_with_feedback(buggy_code: str, feedback: str) -> str:
    # Simple heuristic refinements to buggy code based on feedback text
    # [Code same as notebook Section 7]

def iterate_pbt_refinement(problem: Dict[str, Any], num_iterations: int = 5) -> Dict[str, Any]:
    # Iterative PBT refinement of a task's code candidate, collecting feedback and recording progress
    # [Code same as notebook Section 7]
```
(Implement the PBT workflow with iteration control input)

### 4.8 Refinement Loop for Traditional Test-Driven Development (TDD)
```python
def iterate_tdd_refinement(problem: Dict[str, Any], num_iterations: int = 1) -> Dict[str, Any]:
    # Iterative TDD refinement baseline implementation running example tests with a candidate
    # [Code same as notebook Section 8]
```
(Implement TDD workflow with iteration and method selection in UI)

### 4.9 Quantitative Benchmark Evaluation
```python
def evaluate_method_on_benchmark(
    method: Callable, dataset: List[Dict[str, Any]], num_iterations: int
) -> pd.DataFrame:
    # Runs refinement method over dataset, returns aggregated results for visualization
    # [Code equivalent to notebook Section 9]
```
(Use internally for comparative metrics presentation)

### 4.10 Visualization of Results
- Implement interactive rendering of:

```python
def visualize_performance_comparison(results_df: pd.DataFrame) -> plt.Figure:
    # Plot pass rate over iterations, pass@1 and repair success by method
    # [Code same as notebook Section 10]
```

```python
def visualize_test_coverage_and_feedback(results_df: pd.DataFrame) -> Optional[plt.Figure]:
    # Show heatmap/bar/line plots for property violations, failures, and input diversity
    # [Code same as notebook Section 10]
```

- Display these plots in main UI area.

### 4.11 Markdown Content and Explanations

Reuse the notebook's rich markdown content to provide:

- Executive summaries
- Definitions of PBT and TDD distinctions
- Equations like:

$$
\forall (I_j, O_j) \in T_h, \quad C(I_j) = O_j
$$

$$
\forall I \in \mathcal{D}, \quad P(C, I) = \text{True}
$$

$$
\text{Pass} := \bigwedge_{\forall i} C(I_i) = O_i \,\wedge\, \bigwedge_{\forall P_k} P_k(C, I_k) = \text{True}
$$

- Business impacts and practical guidance sections
- Explanations of the evaluation metrics (pass@1, Repair Success Rate)
- Summaries of findings and recommendations for CI integration

Ensure all math content follows LaTeX formatting with $$...$$ for display and $...$ for inline.

---

This requirements specification captures all interactive and code elements needed to faithfully reproduce the notebook's exploratory and demonstration content as a Streamlit app for user-driven experimentation and visualization of PBT vs TDD iterative refinement for LLM code generation.

