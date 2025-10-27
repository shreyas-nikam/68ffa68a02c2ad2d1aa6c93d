import streamlit as st
st.set_page_config(page_title="QuLab", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab: Property-Based Testing vs TDD for LLM Code Refinement")
st.divider()
st.markdown("""
This interactive lab demonstrates how Property-Based Testing (PBT) compares with example-based Test-Driven Development (TDD) for refining Large Language Model (LLM) generated code.

Business logic and learning goals:
- Explain and demonstrate differences between PBT and TDD for iteratively improving LLM-generated code.
- Show that PBT validates invariants on broad input domains, often catching edge cases missed by example tests.
- Provide an end-to-end workflow: load HumanEval-like tasks, generate candidate code, define properties, run tests/properties, and iteratively refine.
- Quantitatively compare PBT and TDD via metrics and visualizations to guide CI/CD policy.

Key formal ideas:
- TDD validates example pairs $(I_j, O_j)$:

$$
\forall (I_j, O_j) \in T_h, \quad C(I_j) = O_j
$$

- PBT validates properties $P$ across a domain $\mathcal{D}$:

$$
\forall I \in \mathcal{D}, \quad P(C, I) = \text{True}
$$

- Combined notion of pass outcome:

$$
\text{Pass} := \bigwedge_{\forall i} C(I_i) = O_i \,\wedge\, \bigwedge_{\forall P_k} P_k(C, I_k) = \text{True}
$$

Use the sidebar to navigate across pages. Each page includes interactive controls, real-time feedback, and Plotly visualizations. You can download refined code and review iterative logs to understand how each method converges.
""")

# Your code starts here
page = st.sidebar.selectbox(label="Navigation", options=["Task Explorer", "Refinement Playground", "Benchmark & Insights"], index=0)
if page == "Task Explorer":
    from application_pages.page1 import run_page1
    run_page1()
elif page == "Refinement Playground":
    from application_pages.page2 import run_page2
    run_page2()
elif page == "Benchmark & Insights":
    from application_pages.page3 import run_page3
    run_page3()
# Your code ends here
