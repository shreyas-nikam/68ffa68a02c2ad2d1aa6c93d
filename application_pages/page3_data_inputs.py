
import streamlit as st
import pandas as pd
from datasets import load_dataset

@st.cache_data
def load_humaneval_dataset():
    """Loads the HumanEval dataset and returns it."""
    with st.spinner("Loading HumanEval dataset... This might take a moment."):
        try:
            dataset = load_dataset("openai_humaneval")
            st.success("HumanEval dataset loaded successfully!")
            return dataset
        except Exception as e:
            st.error(f"Error loading HumanEval dataset: {e}")
            return None

def run_page():
    st.header("3. Data/Inputs Overview")

    st.markdown("""
    The **HumanEval dataset** serves as a benchmark for evaluating the code generation capabilities of Large Language Models.
    It consists of 164 programming problems, each with a problem description, a canonical solution, and a set of unit tests.

    ### Dataset Structure
    Each problem in the HumanEval dataset typically includes the following fields:

    *   `task_id`: A unique identifier for the problem (e.g., "HumanEval/0").
    *   `prompt`: The natural language problem description that an LLM would receive.
    *   `entry_point`: The name of the function to be implemented.
    *   `canonical_solution`: A correct Python solution to the problem.
    *   `test`: Python code containing unit tests to verify the correctness of a submitted solution.
    *   `declaration`: The function signature for the problem.

    ### Loading the Dataset
    Use the button below to load the HumanEval dataset into the application's memory.
    Once loaded, a sample of the dataset will be displayed.
    """))

    if st.button("Load HumanEval Dataset"):
        st.session_state.humaneval_dataset = load_humaneval_dataset()

    if "humaneval_dataset" in st.session_state and st.session_state.humaneval_dataset is not None:
        st.subheader("Sample of HumanEval Dataset")
        st.dataframe(st.session_state.humaneval_dataset["test"].to_pandas().head())

        humaneval_dataset_task_ids = st.session_state.humaneval_dataset["test"]["task_id"].tolist()
        if humaneval_dataset_task_ids:
            st.session_state.humaneval_dataset_task_ids = humaneval_dataset_task_ids

        st.markdown(f"""
        The dataset contains {len(st.session_state.humaneval_dataset["test"])} problems.
        You can select a specific problem in the "5. Sectioned Implementation" page.
        """))
    else:
        st.info("Dataset not loaded yet. Click the 'Load HumanEval Dataset' button above.")
