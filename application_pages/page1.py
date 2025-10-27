import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from .core import (
    load_tasks,
    generate_code_with_llm,
    generate_property_checks,
    generate_pbt_inputs,
    run_tests,
)


def _init_state():
    if "selected_task_id" not in st.session_state:
        st.session_state.selected_task_id = None
    if "candidate_code" not in st.session_state:
        st.session_state.candidate_code = None
    if "pbt_last_result" not in st.session_state:
        st.session_state.pbt_last_result = None
    if "tdd_last_result" not in st.session_state:
        st.session_state.tdd_last_result = None


def run_page1():
    _init_state()
    st.subheader("Task Explorer")
    st.caption("Browse tasks, generate initial code, inspect properties and tests, and run quick checks.")

    tasks, source = load_tasks()
    st.info(f"Task source: {source}")

    task_options = [t["task_id"] for t in tasks]
    default_index = 0 if st.session_state.selected_task_id is None else max(0, task_options.index(st.session_state.selected_task_id))
    selected_task_id = st.selectbox(
        "Select a task",
        options=task_options,
        index=default_index,
        help="Choose a programming task (HumanEval or synthetic fallback)",
    )
    task = next(t for t in tasks if t["task_id"] == selected_task_id)
    st.session_state.selected_task_id = selected_task_id

    st.markdown("Problem prompt:")
    st.code(task.get("prompt", ""))

    with st.expander("Math foundations and business logic", expanded=False):
        st.markdown(
            "In example-based TDD, we verify labeled examples $(I_j, O_j)$ such that $C(I_j)=O_j$. "
            "In PBT, we verify that invariants hold for many inputs from a domain $\\mathcal{D}$."
        )
        st.latex(r"\forall (I_j, O_j) \in T_h,\ C(I_j)=O_j")
        st.latex(r"\forall I \in \mathcal{D},\ P(C,I)=\text{True}")
        st.latex(r"\text{Pass} := \bigwedge_i C(I_i)=O_i\ \wedge\ \bigwedge_k P_k(C, I_k)=\text{True}")
        st.markdown(
            "Business rationale: PBT increases input diversity, reducing escaped defects and improving reliability metrics."
        )

    # Generate initial code
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button(
            "Generate candidate code",
            type="primary",
            help="Create an initial solution with a heuristic LLM-like generator",
        ):
            st.session_state.candidate_code = generate_code_with_llm(task.get("prompt", ""))
    with col2:
        if st.button("Reset to canonical solution", help="Load the reference canonical solution for this task"):
            st.session_state.candidate_code = task.get("canonical_solution", "")
    with col3:
        if st.button("Clear code", help="Reset the code area to empty"):
            st.session_state.candidate_code = ""

    if st.session_state.candidate_code is None:
        st.session_state.candidate_code = generate_code_with_llm(task.get("prompt", ""))

    st.markdown("Candidate code (editable):")
    code = st.text_area(
        "Code",
        value=st.session_state.candidate_code,
        height=240,
        label_visibility="collapsed",
        help="You may edit code before running checks",
    )
    st.session_state.candidate_code = code

    # Show traditional tests and properties with tabs
    tab_tests, tab_props = st.tabs(["TDD tests", "PBT properties"])
    with tab_tests:
        st.code(task.get("test", ""), language="python")
        st.caption("These are example-based assertions used by the TDD baseline.")
    with tab_props:
        props = generate_property_checks(task.get("prompt", ""))
        for s in props:
            st.code(s, language="python")
        st.caption("Heuristically generated properties that should hold for broad inputs.")

    # Run buttons
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Run PBT checks", help="Execute property functions over a diverse input set"):
            inputs = generate_pbt_inputs(task.get("prompt", ""))
            res = run_tests(
                code,
                props,
                inputs,
                entry_point=task.get("entry_point", "solution"),
                mode="PBT",
            )
            st.session_state.pbt_last_result = res
    with c2:
        if st.button("Run TDD tests", help="Execute example-based unit tests"):
            res = run_tests(
                code,
                [task.get("test", "")],
                [],
                entry_point=task.get("entry_point", "solution"),
                mode="TDD",
            )
            st.session_state.tdd_last_result = res

    # Results rendering
    if st.session_state.pbt_last_result:
        res = st.session_state.pbt_last_result
        det = res.get("details", {}) if isinstance(res, dict) else {}
        total = int(det.get("total_properties", 0))
        passed = int(det.get("passed_properties", 0))
        rate = (passed / total) if total else 0.0
        st.metric("PBT pass rate", f"{rate:.2%}")
        st.write("Violations (sample):")
        viol = det.get("violations", []) if det else []
        dfv = pd.DataFrame(viol)
        st.dataframe(dfv.head(50))
        # Visualization: approximate per-property pass trend by input index (synthetic)
        n_inputs = len(generate_pbt_inputs(task.get("prompt", "")))
        pass_counts = max(0, total - len(viol))
        df_line = pd.DataFrame({"input_idx": list(range(n_inputs)), "pass_count": [pass_counts] * n_inputs})
        fig = px.line(df_line, x="input_idx", y="pass_count", title="Approximate per-input passed properties")
        st.plotly_chart(fig, use_container_width=True)
        if res.get("passed", False):
            st.success("All properties passed. Congratulations!")
        else:
            st.warning("Some properties failed. Consider refining the code.")
        csv = dfv.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download PBT violations (CSV)",
            data=csv,
            file_name="pbt_violations.csv",
            mime="text/csv",
        )

    if st.session_state.tdd_last_result:
        res = st.session_state.tdd_last_result
        df = pd.DataFrame(
            [
                {
                    "failed_examples": res.get("failed_examples", 0),
                    "total_examples": res.get("total_examples", 0),
                    "runtime_error": res.get("runtime_error", None),
                    "assertion_error": res.get("assertion_error", None),
                }
            ]
        )
        st.dataframe(df)
        if res.get("passed", False):
            st.success("All example tests passed.")
        else:
            st.error("Some example tests failed.")

    st.markdown(
        "Tips: Toggle between PBT and TDD to see complementary perspectives on correctness. "
        "Use the Refinement Playground to iterate and compare methods quantitatively."
    )
