import streamlit as st
import pandas as pd
import openai
import plotly.express as px
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

def query_gpt(prompt: str, model="gpt-3.5-turbo") -> str:
    """
    Query GPT API with a prompt using the ChatCompletion method.

    Args:
        prompt (str): The user's query.
        model (str): The OpenAI model to use (e.g., "gpt-4" or "gpt-3.5-turbo").
    Returns:
        str: GPT-generated Python code or response.
    """
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a data analysis assistant. Interpret user queries to analyze the dataset and generate the desired output (table or visualization)."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        st.error(f"Error querying GPT: {e}")
        logger.error(f"GPT query error: {e}")
        return ""

def analyze_query_with_gpt(df: pd.DataFrame, query: str, model="gpt-3.5-turbo") -> pd.DataFrame:
    """
    Use GPT to interpret the query and process the DataFrame dynamically.

    Args:
        df (pd.DataFrame): The uploaded dataset.
        query (str): User's natural language query.
        model (str): The OpenAI model to use (e.g., "gpt-4" or "gpt-3.5-turbo").
    Returns:
        pd.DataFrame: Processed DataFrame based on GPT-generated logic.
    """
    column_names = df.columns.tolist()
    prompt = f"""
    I have a dataset with the following columns: {column_names}.
    User query: '{query}'.
    Generate Python code to process the dataset stored in a DataFrame called 'df' to fulfill the query. 
    The code should:
    - Dynamically reference the column names from the provided dataset.
    - Handle potential errors gracefully.
    - Output the processed DataFrame into a variable named 'result'.
    - Avoid hardcoding specific column names or data values.
    Ensure the Python code is syntactically valid and concise.
    """
    gpt_response = query_gpt(prompt, model=model)

    # Execute GPT-generated Python code
    local_context = {"df": df}
    try:
        exec(gpt_response, {}, local_context)
        result = local_context.get("result", None)
        if result is None or not isinstance(result, pd.DataFrame):
            st.error("GPT did not generate a valid DataFrame.")
            st.write("### GPT-Generated Code with Error")
            st.code(gpt_response, language="python")
            return pd.DataFrame()
        return result
    except SyntaxError as e:
        st.error(f"Syntax error in GPT-generated code: {e}")
        logger.error(f"Syntax error: {e}")
        st.write("### GPT-Generated Code with Error")
        st.code(gpt_response, language="python")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error executing GPT-generated logic: {e}")
        logger.error(f"Execution error: {e}")
        st.write("### GPT-Generated Code with Error")
        st.code(gpt_response, language="python")
        return pd.DataFrame()

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
            query = st.text_input("Ask a question about your dataset", placeholder="e.g., Show total impressions for November vs October")
            if query:
                st.write(f"**Your Query:** {query}")

                # Analyze the query using GPT
                processed_df = analyze_query_with_gpt(df, query, model="gpt-3.5-turbo")

                # Display the processed results
                if not processed_df.empty:
                    st.write("### Query Results")
                    st.dataframe(processed_df)

                    # Export results as CSV
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
                else:
                    st.warning("No results generated from your query.")

if __name__ == "__main__":
    main()
