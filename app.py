import streamlit as st
import pandas as pd
import openai
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Streamlit page settings
st.set_page_config(
    page_title="ChatGPT-Powered Data Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

@st.cache_data(ttl=3600)
def load_data(uploaded_file) -> pd.DataFrame:
    """
    Load and preprocess dataset from the uploaded file.

    Args:
        uploaded_file: The uploaded file object.
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    try:
        if uploaded_file.name.endswith(".xlsx") or uploaded_file.name.endswith(".xls"):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            raise ValueError("Unsupported file format. Please upload an Excel or CSV file.")

        # Clean up column names for easier processing
        df.columns = [col.strip().replace(" ", "_").lower() for col in df.columns]

        logger.info(f"Successfully loaded dataset with {len(df)} rows and {len(df.columns)} columns.")
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        logger.error(f"Data loading failed: {str(e)}")
        raise e

def query_gpt_for_analysis(df: pd.DataFrame, query: str, model="gpt-3.5-turbo") -> str:
    """
    Query GPT API for analysis based on dataset context and user query.

    Args:
        df (pd.DataFrame): The uploaded dataset.
        query (str): User's natural language query.
        model (str): The OpenAI model to use (e.g., "gpt-4" or "gpt-3.5-turbo").
    Returns:
        str: GPT's response with analysis.
    """
    # Extract dataset context
    column_names = df.columns.tolist()
    sample_data = df.head(5).to_dict(orient="records")
    dataset_summary = f"""
    The dataset has {len(df)} rows and {len(column_names)} columns.
    Column names: {column_names}.
    Here are the first 5 rows of the dataset:
    {sample_data}
    """

    # Construct the GPT prompt
    prompt = f"""
    I have the following dataset:
    {dataset_summary}

    User query: '{query}'.

    Please analyze the dataset and answer the query. Provide your response in plain text or as a Markdown table.
    If needed, perform calculations based on the dataset. Always base your response on the provided data and context.
    """

    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a data analysis assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.7
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        st.error(f"Error querying GPT: {e}")
        logger.error(f"GPT query error: {e}")
        return "Error processing your query."

def main():
    st.title("ChatGPT-Powered Data Analysis")

    # File uploader
    uploaded_file = st.file_uploader("Upload your dataset (Excel or CSV)", type=["xlsx", "xls", "csv"])
    if uploaded_file:
        with st.spinner("Processing your file..."):
            df = load_data(uploaded_file)

        if df is not None:
            # Display dataset preview
            st.write("### Dataset Preview")
            st.dataframe(df)

            # Query input
            query = st.text_input("Ask a question about your dataset", placeholder="e.g., Show total impressions for November vs October")
            if query:
                st.write(f"**Your Query:** {query}")

                # Query GPT for analysis
                with st.spinner("Analyzing your query..."):
                    response = query_gpt_for_analysis(df, query, model="gpt-3.5-turbo")

                # Display GPT's response
                st.write("### Analysis Results")
                st.markdown(response)

if __name__ == "__main__":
    main()
