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
from dateutil.relativedelta import relativedelta

# Configure logging
logging.basicConfig(filename="workflow.log", level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

DEFAULT_COLUMNS = {
    "all_posts": ["post_title", "post_link", "posted_by", "likes", "date"],
    "metrics": ["date", "impressions", "clicks", "engagement_rate"],
}

EXAMPLE_QUERIES = [
    "Show me the top 5 dates with the highest total impressions.",
    "Show me the posts with the most clicks.",
    "What is the average engagement rate of all posts?",
    "Generate a bar graph of clicks grouped by post type.",
    "Show me the top 10 posts with the most likes, displaying post title, post link, posted by, and likes.",
    "What are the engagement rates for Q3 2024?",
    "Show impressions by day for last week."
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

def parse_date_range(query):
    """Extracts and converts date ranges from natural language to SQL-compatible formats."""
    today = datetime.today()
    if "last week" in query.lower():
        start_date = (today - relativedelta(weeks=1)).strftime("%Y-%m-%d")
        end_date = today.strftime("%Y-%m-%d")
    elif "last month" in query.lower():
        start_date = (today - relativedelta(months=1)).strftime("%Y-%m-%d")
        end_date = today.strftime("%Y-%m-%d")
    elif match := re.search(r"Q(\d) (\d{4})", query):
        quarter = int(match.group(1))
        year = int(match.group(2))
        start_date = f"{year}-{'01' if quarter == 1 else '04' if quarter == 2 else '07' if quarter == 3 else '10'}-01"
        end_date = f"{year}-{'03-31' if quarter == 1 else '06-30' if quarter == 2 else '09-30' if quarter == 3 else '12-31'}"
    else:
        start_date, end_date = None, None
    return start_date, end_date

def parse_columns_and_filters(query, available_columns):
    """Extracts desired columns and additional filters from the user query."""
    columns = re.findall(r"\b(" + "|".join(re.escape(col) for col in available_columns) + r")\b", query, re.IGNORECASE)
    start_date, end_date = parse_date_range(query)
    return list(set(columns)), start_date, end_date

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
                df = preprocess_dataframe_for_arrow(df)
                table_name = sheet.lower().replace(" ", "_").replace("-", "_")
                df.to_sql(table_name, conn, index=False, if_exists="replace")
                table_names.append(table_name)
                logger.info(f"Sheet '{sheet}' loaded into table '{table_name}'.")

        elif uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            df = preprocess_dataframe_for_arrow(df)
            table_name = uploaded_file.name.lower().replace(".csv", "").replace(" ", "_").replace("-", "_")
            df.to_sql(table_name, conn, index=False, if_exists="replace")
            table_names.append(table_name)
            logger.info("CSV file loaded successfully.")

        else:
            raise ValueError("Unsupported file type. Please upload a CSV or Excel file.")

        st.success("Data successfully loaded into the database!")

        # Let the user select a table
        selected_table = st.selectbox("Select a table to query:", table_names)

        user_query = st.text_area("Enter your query or prompt", "")

        if st.button("Run Query"):
            if not user_query.strip():
                st.error("Please enter a valid query or prompt.")
                return

            try:
                # Extract available columns for validation
                columns_query = f"PRAGMA table_info({selected_table});"
                columns_info = pd.read_sql_query(columns_query, conn)
                available_columns = [col['name'] for col in columns_info.to_dict(orient='records')]

                # Parse columns and date filters
                desired_columns, start_date, end_date = parse_columns_and_filters(user_query, available_columns)
                if not desired_columns:
                    desired_columns = DEFAULT_COLUMNS.get(selected_table, [])  # Use default columns if none are specified

                # Construct SQL query
                where_clause = f"WHERE date BETWEEN '{start_date}' AND '{end_date}'" if start_date and end_date else ""
                sql_query = f"SELECT {', '.join(desired_columns)} FROM {selected_table} {where_clause} ORDER BY {desired_columns[-1]} DESC LIMIT 10"
                st.info(f"Generated SQL Query:\n{sql_query}")

                # Execute the query
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
