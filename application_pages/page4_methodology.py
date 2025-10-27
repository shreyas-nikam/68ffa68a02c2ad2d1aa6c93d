
import streamlit as st

def run_page():
    st.header("4. Methodology Overview")

    st.markdown("""
    This section delves into the core methodologies employed in this lab: **Property-Based Testing (PBT)** and **Traditional Test-Driven Development (TDD)**.
    We will also introduce the **PGS (Propose, Generate, Synthesize)** framework, which orchestrates the interaction between LLM agents.

    ### Traditional Test-Driven Development (TDD)
    In TDD, the development cycle is as follows:

    1.  **Write a Failing Test:** A unit test is written for a specific piece of functionality, expecting it to fail initially.
    2.  **Write Code:** The minimum amount of code is written to make the test pass.
    3.  **Refactor:** The code is refactored to improve its design without changing its behavior.

    When applied to LLM-generated code, TDD typically involves providing specific input-output examples as tests. The LLM then attempts to generate code that passes these examples.

    ### Property-Based Testing (PBT)
    PBT shifts the focus from specific examples to **properties** that the code should always uphold. A property is a high-level invariant that should hold true for all valid inputs.

    The PBT workflow involves:

    1.  **Define Properties:** Identify fundamental truths or invariants about the function's behavior.
    2.  **Generate Inputs:** A PBT framework (like Hypothesis) generates a wide range of random, yet valid, inputs.
    3.  **Verify Properties:** The code under test is executed with the generated inputs, and its output is checked against the defined properties.
    4.  **Shrink Failing Cases:** If a property fails, the framework attempts to "shrink" the failing input to the simplest possible example, making debugging easier.

    For LLM-generated code, PBT is particularly powerful because it can uncover edge cases and subtle bugs that might be missed by a limited set of hand-crafted TDD examples.

    ### The PGS (Propose, Generate, Synthesize) Framework
    The PGS framework is an iterative process involving two distinct LLM-powered agents:

    1.  **Generator Agent (Propose & Generate):**
        *   **Role:** Synthesizes initial code based on the problem prompt and refines it based on feedback.
        *   **Input:** Problem description, previous code attempts, and feedback from the Tester Agent.
        *   **Output:** New or refined code.

    2.  **Tester Agent (Synthesize):**
        *   **Role:** Generates test inputs (for both TDD and PBT) and validates the correctness of the Generator's code.
        *   **Input (TDD):** Code from the Generator, specific test cases (if any).
        *   **Input (PBT):** Code from the Generator, defined properties.
        *   **Output:** Test results, error reports, failing inputs (for PBT, these are property violations).

    The interaction can be visualized as a loop:

    $$ \text{Generator (Code)} \xrightarrow{\text{submit}} \text{Tester (Feedback)} \xrightarrow{\text{refine}} \text{Generator (New Code)} $$

    This feedback loop drives the refinement process, allowing the LLM-generated code to become more robust and accurate over successive iterations.

    ### Key Metrics for Comparison
    When comparing PBT and TDD, we will focus on metrics such as:

    *   **Correctness Rate:** The percentage of problems for which the LLM eventually produces a correct solution.
        $$ \text{Correctness Rate} = \frac{\text{Number of Correct Solutions}}{\text{Total Number of Problems}} $$
    *   **Repair Success Rate:** How often the LLM successfully fixes issues identified by the testing method.
        $$ \text{Repair Success Rate} = \frac{\text{Number of Successful Repairs}}{\text{Number of Identified Issues}} $$
    *   **Number of Iterations to Solution:** The average number of refinement steps required to achieve a correct solution.
    *   **Test Coverage (PBT specific):** How effectively PBT explores the input space.
    *   **Semantic Feedback Efficiency:** How well the feedback from each testing method guides the LLM towards semantically correct solutions.

    These metrics will help us quantitatively assess the effectiveness of PBT in improving LLM-generated code compared to TDD.
    """))
