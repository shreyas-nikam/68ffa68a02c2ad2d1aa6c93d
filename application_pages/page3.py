import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from .core import (
    load_tasks,
    generate_code_with_llm,
    iterate_pbt_refinement,
    iterate_tdd_refinement,
    evaluate_method_on_benchmark,
    visualize_performance_comparison,
    visualize_test_coverage_and_feedback,
)


def _init_state():
    if "bench_iterations" not in st.session_state:
        st.session_state.bench_iterations = 5
    if "bench_subset" not in st.session_state:
        st.session_state.bench_subset = 5
    if "bench_results" not in st.session_state:
        st.session_state.bench_results = None


def _compute_metrics(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df is None or results_df.empty:
        return pd.DataFrame()
    df = results_df.copy()
    df["method"] = df["method"].map({
        "iterate_pbt_refinement": "PBT",
        "iterate_tdd_refinement": "TDD",
    }).fillna(df["method"]) 
    pass1 = (
        df[df["iteration"] == 1]
        .groupby("method")
        .agg(pass_at_1=("passed", "mean"))
        .reset_index()
    )
    final_iter = df.groupby(["task_id", "method"]).iteration.max().reset_index()
    merged = final_iter.merge(df, on=["task_id", "method", "iteration"], how="left")
    repair = merged.groupby("method").agg(repair_success=("passed", "mean")).reset_index()
    metrics = pass1.merge(repair, on="method", how="outer")
    return metrics


def run_page3():
    _init_state()
    st.subheader("Benchmark & Insights")
    st.markdown(
        "Run a small benchmark to compare PBT and TDD across tasks. Visualize pass rate over iterations, pass@1, and repair success."
    )

    tasks, source = load_tasks()
    st.caption("Dataset source: " + str(source))

    c1, c2 = st.columns([1, 1])
    with c1:
        iters = st.slider(
            "Iterations per method",
            min_value=1,
            max_value=15,
            value=int(st.session_state.bench_iterations),
            step=1,
            help="Controls depth of refinement loop for each method",
        )
    with c2:
        subset = st.slider(
            "Number of tasks",
            min_value=1,
            max_value=max(1, len(tasks)),
            value=min(int(st.session_state.bench_subset), len(tasks)),
            step=1,
            help="Size of the task subset used for quick benchmarking",
        )

    st.session_state.bench_iterations = iters
    st.session_state.bench_subset = subset

    selected_tasks = tasks[:subset]

    if st.button("Run benchmark", type="primary"):
        with st.spinner("Evaluating methods across tasks..."):
            df_pbt = evaluate_method_on_benchmark(
                iterate_pbt_refinement, selected_tasks, num_iterations=int(iters)
            )
            df_tdd = evaluate_method_on_benchmark(
                iterate_tdd_refinement, selected_tasks, num_iterations=int(iters)
            )
            results_df = pd.concat([df_pbt, df_tdd], ignore_index=True)
            st.session_state.bench_results = results_df

    results_df = st.session_state.bench_results
    if isinstance(results_df, pd.DataFrame) and not results_df.empty:
        fig1 = visualize_performance_comparison(results_df)
        if fig1 is not None:
            st.plotly_chart(fig1, use_container_width=True)
        fig2 = visualize_test_coverage_and_feedback(results_df)
        if fig2 is not None:
            st.plotly_chart(fig2, use_container_width=True)

        metrics = _compute_metrics(results_df)
        st.markdown("Key metrics:")
        st.dataframe(metrics)

        if not metrics.empty:
            fig_bar1 = px.bar(
                metrics,
                x="method",
                y="pass_at_1",
                title="pass@1 by method",
                text=metrics["pass_at_1"].round(3),
            )
            fig_bar1.update_yaxes(range=[0, 1])
            st.plotly_chart(fig_bar1, use_container_width=True)
            fig_bar2 = px.bar(
                metrics,
                x="method",
                y="repair_success",
                title="Repair success rate by method",
                text=metrics["repair_success"].round(3),
            )
            fig_bar2.update_yaxes(range=[0, 1])
            st.plotly_chart(fig_bar2, use_container_width=True)

        csv = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download benchmark results (CSV)", data=csv, file_name="benchmark_results.csv", mime="text/csv"
        )

    with st.expander("Definitions and formulas"):
        st.markdown(
            "- pass@1: Probability that the first refinement attempt passes. Formally, $\\mathbb{P}[\\text{pass at } i=1]$.\n"
            "- Repair success rate: Fraction of tasks where any iteration achieves pass by the end.\n"
            "- TDD correctness criterion: $C(I_j)=O_j$ for labeled examples $(I_j,O_j)$.\n"
            "- PBT criterion across domain $\\mathcal{D}$: $P(C, I)=\\text{True}$."
        )
        st.latex(r"\\forall (I_j, O_j) \\in T_h,\\ C(I_j) = O_j")
        st.latex(r"\\forall I \\in \\mathcal{D},\\ P(C, I) = \\text{True}")
        st.latex(r"\\text{Pass} := \\bigwedge_{\\forall i} C(I_i) = O_i \\wedge \\bigwedge_{\\forall P_k} P_k(C, I_k) = \\text{True}")

    st.info(
        "Insights: PBT often detects invariant violations on corner cases that TDD misses, improving robustness and reducing escaped defects in production."
    )
