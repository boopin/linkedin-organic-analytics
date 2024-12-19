import streamlit as st
import pandas as pd
import openai
import plotly.express as px
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Streamlit page settings
st.set_page_config(
    page_title="Dynamic Dataset Analysis with GPT",
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
        # Automatically detect Excel or CSV
        if uploaded_file.name.endswith(".xlsx") or uploaded_file.name.endswith(".xls"):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            raise ValueError("Unsupported file format. Please upload an Excel or CSV file.")

        logger.info(f"Successfully loaded dataset with {len(df)} rows and {len(df.columns)} columns.")
        return df

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        logger.error(f"Data loading failed: {str(e)}")
        raise e

def query_gpt(prompt: str) -> str:
    """
    Query GPT API with a prompt using the ChatCompletion method.

    Args:
        prompt (str): The user's query.
    Returns:
        str: GPT-generated Python code or response.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Use "gpt-3.5-turbo" if preferred
            messages=[
                {"role": "system", "content": "You are a helpful assistant skilled at analyzing datasets."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.7
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        st.error(f"Error querying GPT: {e}")
        logger.error(f"GPT query error: {e}")
        return ""

def convert_to_monthly(df: pd.DataFrame, date_column: str, value_columns: list) -> pd.DataFrame:
    """
    Converts daily data into monthly aggregated data.

    Args:
        df (pd.DataFrame): The dataset containing daily data.
        date_column (str): The column containing the dates.
        value_columns (list): Columns to aggregate (e.g., numeric data).
    Returns:
        pd.DataFrame: Monthly aggregated data.
    """
    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
    df["Month"] = df[date_column].dt.to_period("M")  # Extract month
    monthly_df = df.groupby("Month")[value_columns].sum().reset_index()
    return monthly_df

def analyze_query_with_gpt(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """
    Use GPT to interpret the query and process the DataFrame dynamically.

    Args:
        df (pd.DataFrame): The uploaded dataset.
        query (str): User's natural language query.
    Returns:
        pd.DataFrame: Processed DataFrame based on GPT-generated logic.
    """
    column_names = df.columns.tolist()
    prompt = f"""
    I have a dataset with the following columns: {column_names}.
    User query: '{query}'.
    Write Python code to filter, sort, or process the dataset to answer the query.
    If the query involves 'month', aggregate daily data into monthly data.
    The dataset is stored in a DataFrame called 'df', and the result should be stored in a variable called 'result'.
    """
    gpt_response = query_gpt(prompt)
    st.write("### GPT-Generated Code")
    st.code(gpt_response, language="python")

    # Execute the GPT-generated code safely
    local_context = {"df": df}
    try:
        exec(gpt_response, {}, local_context)
        result = local_context.get("result", None)
        if result is None:
            st.error("GPT did not generate a valid result.")
            return df
        return result
    except Exception as e:
        st.error(f"Error executing GPT-generated logic: {e}")
        logger.error(f"Execution error: {e}")
        return df

def main():
    st.title("Dynamic Dataset Analysis with GPT")

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
            query = st.text_input("Ask a question about your dataset", placeholder="e.g., Show month-on-month sales comparison")
            if query:
                st.write(f"**Your Query:** {query}")

                # Analyze the query using GPT
                processed_df = analyze_query_with_gpt(df, query)

                # Display the processed results
                st.write("### Query Results")
                st.dataframe(processed_df)

                # Export results as CSV
                if not processed_df.empty:
                    csv_data = processed_df.to_csv(index=False)
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv_data,
                        file_name="query_results.csv",
                        mime="text/csv"
                    )

                # Visualization options
                st.write("### Visualization")
                chart_type = st.selectbox("Select Chart Type", ["Bar Chart", "Line Chart", "Scatter Plot"])
                if chart_type:
                    x_axis = st.selectbox("Select X-Axis", processed_df.columns)
                    y_axis = st.selectbox("Select Y-Axis", processed_df.columns)
                    if chart_type == "Bar Chart":
                        fig = px.bar(processed_df, x=x_axis, y=y_axis, title=f"{chart_type}")
                    elif chart_type == "Line Chart":
                        fig = px.line(processed_df, x=x_axis, y=y_axis, title=f"{chart_type}")
                    elif chart_type == "Scatter Plot":
                        fig = px.scatter(processed_df, x=x_axis, y=y_axis, title=f"{chart_type}")
                    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
