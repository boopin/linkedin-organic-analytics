# App Version: 1.0.8
import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
import logging
import difflib
import re
from datetime import datetime

# Configure logging
logging.basicConfig(filename="workflow.log", level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

EXAMPLE_SQL_QUERIES = [
    "SELECT * FROM uploaded_data LIMIT 5;",
    "SELECT date, impressions_total FROM uploaded_data ORDER BY impressions_total DESC LIMIT 5;",
    "SELECT post_type, SUM(clicks) as total_clicks FROM uploaded_data GROUP BY post_type;"
]

EXAMPLE_PROMPTS = [
    "Show me the top 5 dates with the highest total impressions.",
    "What are the top 5 posts with the most clicks?",
    "Generate a bar graph of clicks grouped by post type."
]

class PreprocessingPipeline:
    @staticmethod
    def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [col.lower().strip().replace(" ", "_").replace("(", "").replace(")", "") for col in df.columns]
        return df

    @staticmethod
    def handle_missing_dates(df: pd.DataFrame) -> pd.DataFrame:
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df

    @staticmethod
    def fix_arrow_incompatibility(df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].astype("string", errors="ignore")
        return df

    @staticmethod
    def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
        df = PreprocessingPipeline.clean_column_names(df)
        df = PreprocessingPipeline.handle_missing_dates(df)
        df = PreprocessingPipeline.fix_arrow_incompatibility(df)
        return df

def preprocess_dataframe_for_arrow(df):
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]):
            df[col] = df[col].astype("string")
        elif pd.api.types.is_categorical_dtype(df[col]):
            df[col] = df[col].astype("string")
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].astype("datetime64[ns]")
    return df

def extract_data(query, database_connection):
    try:
        df = pd.read_sql_query(query, database_connection)
        return preprocess_dataframe_for_arrow(df)
    except Exception as e:
        return {"error": str(e)}

def generate_sql_from_prompt(prompt, schema):
    """Generates an SQL query from a natural language prompt using GPT."""
    try:
        chat_model = ChatOpenAI(model="gpt-4")
        schema_description = ", ".join([f"{col}" for col in schema])
        prompt_message = (
            f"You are an expert SQL assistant. The database schema is as follows: {schema_description}.\n"
            "Based on this schema, convert the following prompt into an SQL query: \n"
            f"{prompt}"
        )
        response = chat_model([HumanMessage(content=prompt_message)])
        return response.content.strip()
    except Exception as e:
        logger.error(f"Error generating SQL: {e}")
        return None

def main():
    st.title("AI Reports Analyzer with LangChain Workflow")

    uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
    if not uploaded_file:
        st.info("Please upload a file.")
        return

    try:
        if uploaded_file.name.endswith('.xlsx'):
            excel_data = pd.ExcelFile(uploaded_file)
            sheet_names = excel_data.sheet_names
            logger.info(f"Excel file loaded with sheets: {sheet_names}")

            sheet = st.selectbox("Select a sheet to load:", sheet_names)
            df = pd.read_excel(excel_data, sheet_name=sheet)
            logger.info(f"Sheet '{sheet}' loaded successfully.")
        elif uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            logger.info("CSV file loaded successfully.")
        else:
            raise ValueError("Unsupported file type. Please upload a CSV or Excel file.")

        df = preprocess_dataframe_for_arrow(df)

        conn = sqlite3.connect(":memory:")
        df.to_sql("uploaded_data", conn, index=False, if_exists="replace")

        st.success("Data successfully loaded into the database!")

        st.write("### Example SQL Queries")
        for example in EXAMPLE_SQL_QUERIES:
            st.code(example, language="sql")

        st.write("### Example Prompts")
        for example in EXAMPLE_PROMPTS:
            st.markdown(f"- {example}")

        query_type = st.radio("Choose query type:", ("SQL", "Natural Language"))
        user_query = st.text_area("Enter your query or prompt:", "")

        if st.button("Run Query"):
            if not user_query.strip():
                st.error("Please enter a valid query or prompt.")
                return

            try:
                if query_type == "SQL":
                    query_result = extract_data(user_query, conn)
                else:
                    schema = df.columns.tolist()
                    generated_sql = generate_sql_from_prompt(user_query, schema)
                    if not generated_sql:
                        st.error("Failed to generate SQL from the prompt.")
                        return

                    st.write("### Generated SQL Query")
                    st.code(generated_sql, language="sql")
                    query_result = extract_data(generated_sql, conn)

                if isinstance(query_result, dict) and "error" in query_result:
                    st.error(f"Query failed: {query_result['error']}")
                else:
                    st.write("### Query Results")
                    st.dataframe(query_result)
            except Exception as e:
                st.error(f"An error occurred while processing your query: {e}")
    except Exception as e:
        logger.error(f"Error: {e}")
        st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
