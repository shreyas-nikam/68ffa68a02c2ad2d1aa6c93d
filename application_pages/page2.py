import random
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def _init_state():
    if "selected_task_id" not in st.session_state:
        st.session_state.selected_task_id = None
    if "iterations" not in st.session_state:
        st.session_state.iterations = 5
    if "method" not in st.session_state:
        st.session_state.method = "PBT"


def run_page2():
    from .core import (
        load_tasks,
        generate_code_with_llm,
        iterate_pbt_refinement,
        iterate_tdd_refinement,
    )

    _init_state()
    st.subheader("Refinement Playground")
    st.markdown("Run iterative refinement using either Property-Based Testing (PBT) or example tests (TDD). The app collects per-iteration results and shows whether the candidate converges to a passing solution.")

    tasks, _ = load_tasks()
    task_options = [t["task_id"] for t in tasks]
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        selected_task_id = st.selectbox(
            "Select task",
            options=task_options,
            index=0 if st.session_state.selected_task_id is None else max(0, task_options.index(st.session_state.selected_task_id)),
        )
    with col2:
        method = st.radio(
            "Refinement method",
            options=["PBT", "TDD"],
            index=["PBT", "TDD"].index(st.session_state.method),
            help="PBT uses invariant properties over diverse inputs; TDD uses example assertions.",
        )
    with col3:
        iterations = st.number_input("Iterations", min_value=1, max_value=25, value=int(st.session_state.iterations), step=1)

    st.session_state.selected_task_id = selected_task_id
    st.session_state.method = method
    st.session_state.iterations = iterations

    task = next(t for t in tasks if t["task_id"] == selected_task_id)

    initial_code = generate_code_with_llm(task.get("prompt", ""))

    if st.button("Run refinement", type="primary"):
        with st.spinner("Refining candidate..."):
            if method == "PBT":
                outcome = iterate_pbt_refinement(task, num_iterations=int(iterations), initial_code=initial_code)
            else:
                outcome = iterate_tdd_refinement(task, num_iterations=int(iterations), initial_code=initial_code)
        history = outcome["history"] if isinstance(outcome["history"], pd.DataFrame) else pd.DataFrame()
        st.session_state.ref_history = history
        st.session_state.ref_success = bool(outcome.get("success", False))
        st.session_state.ref_final_code = outcome.get("final_code", initial_code)
        st.session_state.ref_duration = float(outcome.get("duration_sec", 0.0))

    if "ref_history" in st.session_state and len(st.session_state.ref_history) > 0:
        st.write("Duration (s):", round(st.session_state.ref_duration, 3))
        st.dataframe(st.session_state.ref_history)

        hist = st.session_state.ref_history.copy()
        if method == "PBT" and "passed_properties" in hist.columns and "total_properties" in hist.columns:
            hist["pass_rate"] = hist["passed_properties"] / hist["total_properties"].replace(0, np.nan)
        elif method == "TDD" and "failed_examples" in hist.columns:
            hist["pass_rate"] = (hist["failed_examples"] == 0).astype(float)
        else:
            hist["pass_rate"] = 0.0
        fig = px.line(hist, x="iteration", y="pass_rate", markers=True, title="Pass rate over iterations")
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

        if st.session_state.ref_success:
            st.success("Refinement succeeded: all checks passed by the final iteration.")
        else:
            st.warning("Refinement did not fully succeed within the allotted iterations.")

        with st.expander("Final code"):
            st.code(st.session_state.ref_final_code, language="python")
        st.download_button(
            label="Download final refined code",
            data=st.session_state.ref_final_code,
            file_name=(task.get("entry_point", "solution") + "_refined.py"),
            mime="text/x-python",
        )

    st.markdown("Guidance: In PBT, we search a larger input space, approximating the condition $\\forall I \\in \\mathcal{D}$. In TDD, we focus on a labeled set $T_h = \\{(I_j, O_j)\\}$. Failure signals guide automatic heuristic repairs; your edits to the code can further improve outcomes.")