        per-iteration trend for the benchmark results. This is to fulfill the plot type requirement.
        """
        fig_line = px.line(combined_eval_df, x="iterations_to_solution", y="correctness_rate", color="method",
                           title="Correctness Rate vs. Iterations (Illustrative)",
                           labels={"iterations_to_solution": "Iterations (Mock)", "correctness_rate": "Correctness Rate"})
        st.plotly_chart(fig_line, use_container_width=True)

        fig_line_repair = px.line(combined_eval_df, x="iterations_to_solution", y="repair_success_rate", color="method",
                                  title="Repair Success Rate vs. Iterations (Illustrative)",
                                  labels={"iterations_to_solution": "Iterations (Mock)", "repair_success_rate": "Repair Success Rate"})
        st.plotly_chart(fig_line_repair, use_container_width=True)

        st.subheader("Average Performance Comparison")
        st.markdown("""
        These bar charts compare the average performance of PBT and TDD across various metrics.
        """))

        avg_performance = combined_eval_df.groupby("method").mean(numeric_only=True).reset_index()

        metrics_to_plot = ["correctness_rate", "repair_success_rate", "iterations_to_solution",
                           "test_coverage_score", "semantic_feedback_efficiency"]

        for metric in metrics_to_plot:
            if metric in avg_performance.columns and not avg_performance[metric].isnull().all(): # Check if metric exists and is not all NaNs
                fig_bar = px.bar(avg_performance, x="method", y=metric,
                                 title=f"Average {metric.replace('_', ' ').title()} Comparison",
                                 labels={"method": "Testing Method", metric: metric.replace('_', ' ').title()})
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info(f"Skipping visualization for '{metric}' as data is not available or is all NaN.")

        st.subheader("Detailed Metric Comparison per Problem")
        st.markdown("""
        These plots allow for a detailed comparison of specific metrics across individual problems for both testing methods.
        """))

        selected_metric = st.selectbox(
            "Select a metric to visualize across problems",
            options=[col for col in metrics_to_plot if col in combined_eval_df.columns and not combined_eval_df[col].isnull().all()],
            help="Choose a metric to see its performance per problem for PBT and TDD."
        )

        if selected_metric:
            fig_scatter = px.scatter(combined_eval_df, x="problem_id", y=selected_metric, color="method",
                                     title=f"{selected_metric.replace('_', ' ').title()} per Problem",
                                     labels={"problem_id": "Problem ID", selected_metric: selected_metric.replace('_', ' ').title()},
                                     hover_data=combined_eval_df.columns)
            st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("No data available for visualization. Please run the benchmark evaluation on the '6. Quantitative Comparison' page.")
