# App Version: 1.0.6
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

def analyze_data(data):
    """Analyzes and interprets the extracted data."""
    if isinstance(data, dict) and "error" in data:
        return data["error"]
    # Perform sample analysis
    return {
        "rows": len(data),
        "columns": list(data.columns),
        "sample_data": data.head(5).to_dict()
    }

def execute_workflow(query, db_connection):
    """
    Execute a simplified workflow using LangChain for task orchestration.
    """
    logger.info("Starting workflow execution.")

    # Step 1: Extract Data
    logger.info("Executing data extraction task.")
    extracted_data = extract_data(query, db_connection)
    if isinstance(extracted_data, dict) and "error" in extracted_data:
        logger.error(f"Data extraction failed: {extracted_data['error']}")
        return {"error": extracted_data["error"]}

    # Step 2: Analyze Data
    logger.info("Executing data analysis task.")
    analysis_result = analyze_data(extracted_data)

    logger.info("Workflow execution completed.")
    return {
        "extracted_data": extracted_data,
        "analysis_result": analysis_result
    }

def load_file(uploaded_file):
    """Load CSV or Excel and return a pandas DataFrame."""
    try:
        if uploaded_file.name.endswith('.xlsx'):
            excel_data = pd.ExcelFile(uploaded_file)
            sheet_names = excel_data.sheet_names
            logger.info(f"Excel file loaded with sheets: {sheet_names}")

            # Let the user select a sheet
            sheet = st.selectbox("Select a sheet to load:", sheet_names)
            df = pd.read_excel(excel_data, sheet_name=sheet)
            logger.info(f"Sheet '{sheet}' loaded successfully.")
        elif uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            logger.info("CSV file loaded successfully.")
        else:
            raise ValueError("Unsupported file type. Please upload a CSV or Excel file.")
        return df
    except Exception as e:
        logger.error(f"Error loading file: {e}")
        st.error(f"Error loading file: {e}")
        raise

def main():
    st.title("AI Reports Analyzer with LangChain Workflow")

    uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
    if not uploaded_file:
        st.info("Please upload a file.")
        return

    try:
        df = load_file(uploaded_file)

        # Preprocess data for SQLite
        df = preprocess_dataframe_for_arrow(df)

        # Load data into SQLite
        conn = sqlite3.connect(":memory:")
        df.to_sql("uploaded_data", conn, index=False, if_exists="replace")

        # Define a query for extraction
        query = "SELECT * FROM uploaded_data LIMIT 10;"

        # Execute the workflow
        results = execute_workflow(query, conn)

        if "error" in results:
            st.error(f"Workflow failed: {results['error']}")
        else:
            st.write("### Extracted Data")
            st.dataframe(results["extracted_data"])

            st.write("### Analysis Results")
            st.json(results["analysis_result"])

    except Exception as e:
        st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
