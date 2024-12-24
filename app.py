# App Version: 1.1.0
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

EXAMPLE_QUERIES = [
    "Show me the top 5 dates with the highest total impressions.",
    "Show me the posts with the most clicks.",
    "What is the average engagement rate of all posts?",
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
    """
    Preprocess the dataframe to ensure all columns are compatible with Arrow serialization.
    """
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]):
            df[col] = df[col].astype("string")  # Convert object columns to string
        elif pd.api.types.is_categorical_dtype(df[col]):
            df[col] = df[col].astype("string")  # Convert categorical columns to string
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].astype("datetime64[ns]")  # Ensure proper datetime format
    return df

def extract_data(query, database_connection):
    """Extracts data from the database using SQL queries."""
    try:
        df = pd.read_sql_query(query, database_connection)
        return preprocess_dataframe_for_arrow(df)  # Ensure compatibility for Arrow serialization
    except Exception as e:
        return {"error": str(e)}

def generate_sql_from_prompt(prompt, table_name):
    """Generate SQL query from user prompt."""
    if "top 5 dates" in prompt.lower() and "impressions" in prompt.lower():
        return f"SELECT date, SUM(impressions) as total_impressions FROM {table_name} GROUP BY date ORDER BY total_impressions DESC LIMIT 5;"
    return None

def main():
    st.title("AI Reports Analyzer with LangChain Workflow")

    uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
    if not uploaded_file:
        st.info("Please upload a file.")
        return

    try:
        # Load and preprocess the uploaded file
        conn = sqlite3.connect(":memory:")
        table_names = []

        if uploaded_file.name.endswith('.xlsx'):
            excel_data = pd.ExcelFile(uploaded_file)
            sheet_names = excel_data.sheet_names
            logger.info(f"Excel file loaded with sheets: {sheet_names}")

            for sheet in sheet_names:
                df = pd.read_excel(excel_data, sheet_name=sheet)
                df = preprocess_dataframe_for_arrow(df)
                table_name = sheet.lower().replace(" ", "_")
                df.to_sql(table_name, conn, index=False, if_exists="replace")
                table_names.append(table_name)

        elif uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            df = preprocess_dataframe_for_arrow(df)
            table_name = uploaded_file.name.split('.')[0].lower().replace(" ", "_")
            df.to_sql(table_name, conn, index=False, if_exists="replace")
            table_names.append(table_name)

        else:
            raise ValueError("Unsupported file type. Please upload a CSV or Excel file.")

        st.success("Data successfully loaded into the database!")

        # Table selection dropdown
        selected_table = st.selectbox("Select a table to query:", table_names)

        # Provide example prompts
        st.write("### Example Prompts")
        st.write("- Show me the top 5 dates with the highest total impressions.")
        st.write("- Show me the posts with the most clicks.")
        st.write("- What is the average engagement rate of all posts?")
        st.write("- Generate a bar graph of clicks grouped by post type.")

        # Provide a text box for SQL query input
        user_prompt = st.text_area("Enter your prompt or SQL query", "")

        if st.button("Run Query"):
            if not user_prompt.strip():
                st.error("Please enter a valid prompt or query.")
                return

            # Check if the prompt can be converted to SQL
            sql_query = generate_sql_from_prompt(user_prompt, selected_table)
            if not sql_query:
                sql_query = user_prompt  # Assume the user entered a valid SQL query

            # Execute the query and display results
            try:
                query_result = extract_data(sql_query, conn)
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
