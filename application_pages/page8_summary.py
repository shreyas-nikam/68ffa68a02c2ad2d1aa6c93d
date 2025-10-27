
import streamlit as st
import pandas as pd
import plotly.express as px

def run_page():
    st.header("8. Summary and Insights")

    st.markdown("""
    This section summarizes the key findings and insights gained from comparing Property-Based Testing (PBT) and Traditional Test-Driven Development (TDD) for refining LLM-generated code.

    ### Key Takeaways:

    1.  **PBT's Strength in Edge Case Discovery:** PBT consistently demonstrates a superior ability to uncover subtle bugs and edge cases that are often missed by fixed, example-based TDD tests. The generative nature of PBT's input exploration leads to more robust code.

    2.  **Faster Refinement with PBT-driven Feedback:** The rich, semantic feedback provided by PBT (i.e., minimal failing examples from shrinking) significantly accelerates the LLM's refinement process. This leads to fewer iterations required to achieve a correct and robust solution, as observed in the "Iterations to Solution" metric.

    3.  **Improved Code Quality and Robustness:** While TDD helps in achieving functional correctness for known examples, PBT pushes the LLM to generate code that adheres to fundamental properties, leading to code that is not just "correct" for specific inputs, but also semantically more robust and resilient to unexpected inputs.

    4.  **Challenges and Considerations:**
        *   **Property Definition:** Defining effective properties for complex functions can be challenging and requires a deep understanding of the problem domain. Poorly defined properties can lead to ineffective testing.
        *   **LLM Interpretation of Feedback:** The effectiveness of PBT also depends on the LLM's ability to interpret and act upon property violation feedback. Future LLMs will likely be even better at this.
        *   **Performance Overhead:** PBT can sometimes involve running a large number of generated tests, which might have a higher computational overhead compared to a small set of TDD unit tests, especially for very complex properties or slower code.

    ### Visualizing Overall Trends
    Based on our mock benchmark and the principles discussed, we can generally expect trends similar to the following (refer to "7. Visualization of Results" for interactive plots):
    """))

    # Display summary visualizations if benchmark data exists
    if "combined_eval_df" in st.session_state and st.session_state.combined_eval_df is not None:
        combined_eval_df = st.session_state.combined_eval_df
        avg_performance = combined_eval_df.groupby("method").mean(numeric_only=True).reset_index()

        st.subheader("Average Correctness Rate (PBT vs TDD)")
        if "correctness_rate" in avg_performance.columns:
            fig_correctness = px.bar(avg_performance, x="method", y="correctness_rate",
                                      title="Average Correctness Rate",
                                      labels={"method": "Testing Method", "correctness_rate": "Correctness Rate"})
            st.plotly_chart(fig_correctness, use_container_width=True)

        st.subheader("Average Iterations to Solution (PBT vs TDD)")
        if "iterations_to_solution" in avg_performance.columns:
            fig_iterations = px.bar(avg_performance, x="method", y="iterations_to_solution",
                                    title="Average Iterations to Solution",
                                    labels={"method": "Testing Method", "iterations_to_solution": "Iterations"})
            st.plotly_chart(fig_iterations, use_container_width=True)

        st.subheader("Average Test Coverage Score (PBT vs TDD)")
        # Filter out NaN values from TDD for plotting coverage if it's not applicable or mock-empty
        coverage_df = avg_performance.dropna(subset=["test_coverage_score"])
        if not coverage_df.empty:
            fig_coverage = px.bar(coverage_df, x="method", y="test_coverage_score",
                                  title="Average Test Coverage Score",
                                  labels={"method": "Testing Method", "test_coverage_score": "Coverage Score"})
            st.plotly_chart(fig_coverage, use_container_width=True)
        else:
            st.info("Test coverage data not available for visualization.")


    st.markdown("""
    ### Conclusion:
    This lab highlights the significant potential of Property-Based Testing as a powerful tool for developing and refining robust code, especially when working with LLM-generated outputs.
    By focusing on the *behavior* of the code rather than just specific examples, PBT fosters a deeper understanding of correctness and leads to more resilient software systems.

    As LLMs continue to advance, integrating sophisticated testing paradigms like PBT will be crucial for ensuring the reliability and trustworthiness of AI-generated code.
    """))
