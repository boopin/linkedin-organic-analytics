# App Version: 1.2.0
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
from dateutil.relativedelta import relativedelta

# Configure logging
logging.basicConfig(filename="workflow.log", level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

DEFAULT_COLUMNS = {
    "all_posts": ["post_title", "post_link", "posted_by", "likes", "engagement_rate", "date"],
    "metrics": ["date", "impressions", "clicks", "engagement_rate"],
}

QUERY_SHORTCUTS = {
    "top_liked_posts": {
        "description": "Top 5 posts with the most likes.",
        "sql_template": "SELECT post_title, post_link, likes FROM all_posts ORDER BY likes DESC LIMIT 5"
    },
    "high_engagement": {
        "description": "Top 10 posts with the highest engagement rate.",
        "sql_template": "SELECT post_title, post_link, engagement_rate FROM all_posts ORDER BY engagement_rate DESC LIMIT 10"
    },
    "impressions_by_date": {
        "description": "Daily impressions sorted by highest to lowest.",
        "sql_template": "SELECT date, impressions FROM metrics ORDER BY impressions DESC LIMIT 10"
    },
    "recent_clicks": {
        "description": "Most clicked posts in the last week.",
        "sql_template": (
            "SELECT post_title, post_link, clicks FROM all_posts "
            "WHERE date BETWEEN '{start_date}' AND '{end_date}' ORDER BY clicks DESC LIMIT 5"
        )
    }
}

EXAMPLE_QUERIES = [
    "Show me the top 5 dates with the highest total impressions.",
    "Show me the posts with the most clicks.",
    "What is the average engagement rate of all posts?",
    "Generate a bar graph of clicks grouped by post type.",
    "Show me the top 10 posts with the most likes, displaying post title, post link, posted by, and likes.",
    "What are the engagement rates for Q3 2024?",
    "Show impressions by day for last week.",
    "Show me the top 5 posts with the highest engagement rate."
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

def replace_date_placeholders(sql_template):
    today = datetime.today()
    start_date = (today - relativedelta(weeks=1)).strftime("%Y-%m-%d")
    end_date = today.strftime("%Y-%m-%d")
    return sql_template.format(start_date=start_date, end_date=end_date)

def extract_data(query, database_connection):
    """Extracts data from the database using SQL queries."""
    try:
        df = pd.read_sql_query(query, database_connection)
        return preprocess_dataframe_for_arrow(df)  # Ensure compatibility for Arrow serialization
    except Exception as e:
        return {"error": str(e)}

def main():
    st.title("AI Reports Analyzer with LangChain Workflow")

    uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
    if not uploaded_file:
        st.info("Please upload a file.")
        return

    try:
        conn = sqlite3.connect(":memory:")
        table_names = []

        # Load and preprocess the uploaded file
        if uploaded_file.name.endswith('.xlsx'):
            excel_data = pd.ExcelFile(uploaded_file)
            sheet_names = excel_data.sheet_names
            logger.info(f"Excel file loaded with sheets: {sheet_names}")

            for sheet in sheet_names:
                df = pd.read_excel(excel_data, sheet_name=sheet)
                df = PreprocessingPipeline.preprocess_data(df)
                table_name = sheet.lower().replace(" ", "_").replace("-", "_")
                df.to_sql(table_name, conn, index=False, if_exists="replace")
                table_names.append(table_name)
                logger.info(f"Sheet '{sheet}' loaded into table '{table_name}'.")

        elif uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            df = PreprocessingPipeline.preprocess_data(df)
            table_name = uploaded_file.name.lower().replace(".csv", "").replace(" ", "_").replace("-", "_")
            df.to_sql(table_name, conn, index=False, if_exists="replace")
            table_names.append(table_name)
            logger.info("CSV file loaded successfully.")

        else:
            raise ValueError("Unsupported file type. Please upload a CSV or Excel file.")

        st.success("Data successfully loaded into the database!")

        # Let the user select a table
        selected_table = st.selectbox("Select a table to query:", table_names)

        st.write("### Query Shortcuts")
        for shortcut, details in QUERY_SHORTCUTS.items():
            st.markdown(f"**{shortcut}**: {details['description']}")

        user_query = st.text_area("Enter your query or shortcut", "")

        if st.button("Run Query"):
            if not user_query.strip():
                st.error("Please enter a valid query or shortcut.")
                return

            # Check if the query matches a shortcut
            sql_query = None
            if user_query in QUERY_SHORTCUTS:
                sql_query = QUERY_SHORTCUTS[user_query]["sql_template"]
                if "{start_date}" in sql_query or "{end_date}" in sql_query:
                    sql_query = replace_date_placeholders(sql_query)

            if not sql_query:
                # Assume the user entered a custom SQL command
                sql_query = user_query

            try:
                query_result = extract_data(sql_query, conn)
                if isinstance(query_result, dict) and "error" in query_result:
                    st.error(f"Query failed: {query_result['error']}")
                else:
                    st.write("### Query Results")
                    st.dataframe(query_result)

            except Exception as e:
                st.error(f"An error occurred: {e}")

    except Exception as e:
        logger.error(f"Error: {e}")
        st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
