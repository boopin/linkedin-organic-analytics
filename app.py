# App Version: 1.2.0
import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
from langchain_openai import ChatOpenAI  # Updated import for langchain-openai
from datetime import datetime
from dateutil.relativedelta import relativedelta
import logging
import re

# Configure logging
logging.basicConfig(filename="workflow.log", level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

DEFAULT_COLUMNS = {
    "metrics": ["date", "impressions_total", "clicks", "engagement_rate"],
    "all_posts": ["post_title", "post_link", "posted_by", "likes", "engagement_rate", "date"],
}

EXAMPLE_QUERIES = [
    "Show me the top 5 dates with the highest total impressions.",
    "Show me the posts with the most clicks.",
    "What is the average engagement rate of all posts?",
    "Generate a bar graph of clicks grouped by post type.",
    "Show me the top 10 posts with the most likes, displaying post title, post link, posted by, and likes.",
]

class PreprocessingPipeline:
    @staticmethod
    def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [col.lower().strip().replace(" ", "_").replace("(", "").replace(")", "") for col in df.columns]
        return df

    @staticmethod
    def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
        df = PreprocessingPipeline.clean_column_names(df)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df

def parse_date_range(query):
    today = datetime.today()
    if "last week" in query.lower():
        start_date = (today - relativedelta(weeks=1)).strftime("%Y-%m-%d")
        end_date = today.strftime("%Y-%m-%d")
    elif "last month" in query.lower():
        start_date = (today - relativedelta(months=1)).strftime("%Y-%m-%d")
        end_date = today.strftime("%Y-%m-%d")
    else:
        start_date, end_date = None, None
    return start_date, end_date

def generate_sql_query(query, table_name, available_columns):
    try:
        if "top" in query.lower():
            match = re.search(r"top (\d+)", query, re.IGNORECASE)
            limit = match.group(1) if match else "10"
            columns = DEFAULT_COLUMNS.get(table_name, [])
            sql_query = f"SELECT {', '.join(columns)} FROM {table_name} ORDER BY {columns[-1]} DESC LIMIT {limit}"
            return sql_query
        return None
    except Exception as e:
        logger.error(f"Error generating SQL query: {e}")
        return None

def main():
    st.title("AI Reports Analyzer with LangChain Workflow")
    st.markdown("## Upload CSV or Excel Files")
    uploaded_file = st.file_uploader("Upload a file", type=["csv", "xlsx"])

    if not uploaded_file:
        st.info("Please upload a file.")
        return

    try:
        conn = sqlite3.connect(":memory:")
        table_names = []

        if uploaded_file.name.endswith(".xlsx"):
            excel_data = pd.ExcelFile(uploaded_file)
            for sheet_name in excel_data.sheet_names:
                df = pd.read_excel(excel_data, sheet_name=sheet_name)
                df = PreprocessingPipeline.preprocess_data(df)
                table_name = sheet_name.lower().replace(" ", "_")
                df.to_sql(table_name, conn, index=False, if_exists="replace")
                table_names.append(table_name)
                logger.info(f"Sheet '{sheet_name}' loaded into table '{table_name}'.")
        elif uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            df = PreprocessingPipeline.preprocess_data(df)
            table_name = uploaded_file.name.split(".")[0].lower()
            df.to_sql(table_name, conn, index=False, if_exists="replace")
            table_names.append(table_name)
            logger.info(f"CSV file '{uploaded_file.name}' loaded into table '{table_name}'.")
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return

        st.success("File successfully processed and saved to the database!")
        selected_table = st.selectbox("Select a table to query:", table_names)

        st.write("### Example Queries")
        for example in EXAMPLE_QUERIES:
            st.markdown(f"- {example}")

        user_query = st.text_area("Enter your query or prompt")

        if st.button("Run Query"):
            if not user_query.strip():
                st.error("Please enter a valid query or prompt.")
                return

            try:
                columns_query = f"PRAGMA table_info({selected_table});"
                columns_info = pd.read_sql_query(columns_query, conn)
                available_columns = [col['name'] for col in columns_info.to_dict(orient='records')]

                sql_query = generate_sql_query(user_query, selected_table, available_columns)
                if not sql_query:
                    st.error("Could not generate a valid SQL query.")
                    return

                st.info(f"Generated SQL Query:\n{sql_query}")
                df_result = pd.read_sql_query(sql_query, conn)
                st.write("### Query Results")
                st.dataframe(df_result)
            except Exception as e:
                st.error(f"Error: {e}")
                logger.error(f"Query execution error: {e}")

    except Exception as e:
        st.error(f"Error: {e}")
        logger.error(f"App error: {e}")

if __name__ == "__main__":
    main()
