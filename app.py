# App Version: 1.2.1
import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
from datetime import datetime
from dateutil.relativedelta import relativedelta
import logging
import re

# Configure logging
logging.basicConfig(filename="workflow.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

DEFAULT_COLUMNS = {
    "all_posts": ["post_title", "post_link", "posted_by", "likes", "engagement_rate", "date"],
    "metrics": ["date", "impressions", "clicks", "engagement_rate"],
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

def preprocess_data(df):
    """Preprocess the dataframe for compatibility."""
    df.columns = [col.lower().strip().replace(" ", "_").replace("(", "").replace(")", "") for col in df.columns]
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df

def precompute_aggregations(conn, table_name):
    """Precompute weekly, monthly, and quarterly aggregates."""
    query = f"SELECT * FROM {table_name};"
    df = pd.read_sql_query(query, conn)

    if "date" in df.columns:
        df["week"] = df["date"].dt.to_period("W").dt.start_time
        df["month"] = df["date"].dt.to_period("M").dt.start_time
        df["quarter"] = df["date"].dt.to_period("Q").dt.start_time

        # Save aggregates
        for period in ["week", "month", "quarter"]:
            agg_table = f"{table_name}_{period}"
            df_agg = df.groupby(period).sum().reset_index()
            df_agg.to_sql(agg_table, conn, if_exists="replace", index=False)
            logger.info(f"Precomputed {period} aggregation saved to table '{agg_table}'.")

def main():
    st.title("AI Reports Analyzer with Optimized Workflow")

    uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
    if not uploaded_file:
        st.info("Please upload a file.")
        return

    conn = sqlite3.connect(":memory:")
    table_names = []

    try:
        # Load and preprocess the uploaded file
        if uploaded_file.name.endswith('.xlsx'):
            excel_data = pd.ExcelFile(uploaded_file)
            for sheet in excel_data.sheet_names:
                df = pd.read_excel(excel_data, sheet_name=sheet)
                df = preprocess_data(df)
                table_name = sheet.lower().replace(" ", "_").replace("-", "_")
                df.to_sql(table_name, conn, index=False, if_exists="replace")
                precompute_aggregations(conn, table_name)
                table_names.append(table_name)
                logger.info(f"Sheet '{sheet}' loaded into table '{table_name}'.")

        elif uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            df = preprocess_data(df)
            table_name = uploaded_file.name.lower().replace(".csv", "").replace(" ", "_").replace("-", "_")
            df.to_sql(table_name, conn, index=False, if_exists="replace")
            precompute_aggregations(conn, table_name)
            table_names.append(table_name)
            logger.info("CSV file loaded successfully.")

        else:
            raise ValueError("Unsupported file type. Please upload a CSV or Excel file.")

        st.success("File successfully processed and saved to the database!")

        # Display table options
        selected_table = st.selectbox("Select a table to query:", table_names)
        schema_info = pd.read_sql_query(f"PRAGMA table_info({selected_table});", conn)
        st.write("### Available Columns in Selected Table")
        st.dataframe(schema_info[["name", "type"]])

        # Example queries
        st.write("### Example Queries")
        for example in EXAMPLE_QUERIES:
            st.markdown(f"- {example}")

        user_query = st.text_area("Enter your query or prompt", "")
        if st.button("Run Query"):
            if not user_query.strip():
                st.error("Please enter a valid query or prompt.")
                return

            # Delay LLM loading until needed
            from langchain_openai import ChatOpenAI
            from langchain.schema import HumanMessage

            openai_api_key = st.secrets["OPENAI_API_KEY"]
            llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=openai_api_key)

            # Parse query
            prompt = f"Convert this natural language query into SQL for the table '{selected_table}': {user_query}"
            ai_response = llm([HumanMessage(content=prompt)])
            sql_query = ai_response.content.strip()

            st.info(f"Generated SQL Query:\n{sql_query}")

            try:
                query_result = pd.read_sql_query(sql_query, conn)
                st.write("### Query Results")
                st.dataframe(query_result)
            except Exception as e:
                st.error(f"Query failed: {e}")
    except Exception as e:
        logger.error(f"Error: {e}")
        st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
