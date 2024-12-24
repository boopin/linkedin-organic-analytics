# App Version: 1.0.5
import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from crewai import Agent, Workflow, Task
import logging
import difflib
import re
from datetime import datetime

# Configure logging
logging.basicConfig(filename="crew.log", level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
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

def extract_data_task(database_connection, query):
    """Task to extract data from the SQLite database."""
    try:
        df = pd.read_sql_query(query, database_connection)
        return preprocess_dataframe_for_arrow(df)  # Ensure compatibility for Arrow serialization
    except Exception as e:
        return {"error": str(e)}

def analyze_data_task(data):
    """Task to analyze extracted data."""
    if isinstance(data, dict) and "error" in data:
        return data["error"]
    # Perform sample analysis
    return {
        "rows": len(data),
        "columns": list(data.columns),
        "sample_data": data.head(5).to_dict()
    }

def main():
    st.title("AI Reports Analyzer with Crew.ai")

    uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
    if not uploaded_file:
        st.info("Please upload a file.")
        return

    try:
        if uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)

        # Preprocess data for SQLite
        df = preprocess_dataframe_for_arrow(df)

        # Load data into SQLite
        conn = sqlite3.connect(":memory:")
        df.to_sql("uploaded_data", conn, index=False, if_exists="replace")

        # Define agents
        sql_dev = Agent(name="sql_dev", role="Handles SQL queries for data extraction.")
        data_analyst = Agent(name="data_analyst", role="Analyzes extracted data.")

        # Create workflow
        workflow = Workflow(
            name="AI Data Analysis Workflow",
            tasks=[
                Task(name="extract_data", func=extract_data_task, inputs={"query": "SELECT * FROM uploaded_data", "database_connection": conn}),
                Task(name="analyze_data", func=analyze_data_task, inputs={"data": "{extract_data}"})
            ],
            agents=[sql_dev, data_analyst],
            process="sequential",
            verbose=2,
            memory=False
        )

        # Execute the workflow
        results = workflow.run()
        st.write("### Workflow Results")
        for task, result in results.items():
            st.write(f"**{task}:**", result)

    except Exception as e:
        st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
