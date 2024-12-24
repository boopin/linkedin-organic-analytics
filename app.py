# App Version: 1.2.0
import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import logging

# Configure logging
logging.basicConfig(filename="app.log", level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

DEFAULT_COLUMNS = {
    "all_posts": ["post_title", "post_link", "posted_by", "likes", "engagement_rate", "date"],
    "metrics": ["date", "impressions_total", "clicks", "engagement_rate"],
}

EXAMPLE_QUERIES = [
    "Show me the top 5 dates with the highest total impressions.",
    "Show me the posts with the most clicks.",
    "What is the average engagement rate of all posts?",
    "Generate a bar graph of clicks grouped by post type.",
    "Show me the top 10 posts with the most likes, displaying post title, post link, posted by, and likes.",
    "What are the engagement rates for Q3 2024?",
    "Show impressions by day for last week.",
    "Show me the top 5 posts with the highest engagement rate.",
]

def create_schemas(df: pd.DataFrame, table_name: str, conn):
    """Prepares weekly, monthly, and quarterly schemas."""
    schemas = []
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["month"] = df["date"].dt.to_period("M").astype(str)
        df["week"] = df["date"].dt.to_period("W").astype(str)
        df["quarter"] = df["date"].dt.to_period("Q").astype(str)
        
        for schema in ["month", "week", "quarter"]:
            schema_name = f"{table_name}_{schema}"
            schema_df = df.groupby(schema).sum().reset_index()
            schema_df.to_sql(schema_name, conn, index=False, if_exists="replace")
            schemas.append(schema_name)
            logger.info(f"Schema '{schema_name}' created.")
    return schemas

def process_file(uploaded_file, conn):
    """Processes the uploaded file and creates database tables."""
    table_names = []
    schemas = []

    if uploaded_file.name.endswith(".xlsx"):
        excel_data = pd.ExcelFile(uploaded_file)
        for sheet in excel_data.sheet_names:
            df = pd.read_excel(excel_data, sheet_name=sheet)
            table_name = sheet.lower().replace(" ", "_").replace("-", "_")
            df.to_sql(table_name, conn, index=False, if_exists="replace")
            table_names.append(table_name)
            schemas.extend(create_schemas(df, table_name, conn))
            logger.info(f"Sheet '{sheet}' processed into table '{table_name}'.")

    elif uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        table_name = uploaded_file.name.lower().replace(".csv", "").replace(" ", "_").replace("-", "_")
        df.to_sql(table_name, conn, index=False, if_exists="replace")
        table_names.append(table_name)
        schemas.extend(create_schemas(df, table_name, conn))
        logger.info(f"CSV file processed into table '{table_name}'.")

    return table_names, schemas

def main():
    st.title("AI-Driven Data Analyzer")
    uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
    
    if not uploaded_file:
        st.info("Please upload a file to get started.")
        return
    
    conn = sqlite3.connect(":memory:")
    table_names, schemas = [], []
    
    if st.button("Process File"):
        try:
            table_names, schemas = process_file(uploaded_file, conn)
            st.success("File successfully processed and saved to the database!")
            st.write("### Available Tables:")
            st.write(table_names)
        except Exception as e:
            st.error(f"Error during processing: {e}")
            logger.error(f"Processing error: {e}")
            return

    if schemas:
        st.write("### Schema Selection")
        selected_schema = st.selectbox("Select a schema to query:", schemas)
        st.write("### Example Queries")
        for example in EXAMPLE_QUERIES:
            st.markdown(f"- {example}")

        user_query = st.text_area("Enter your query or prompt", "")

        if st.button("Run Query"):
            try:
                if not user_query.strip():
                    st.error("Please enter a valid query.")
                    return
                
                # Execute user query
                query_result = pd.read_sql_query(user_query, conn)
                st.write("### Query Results")
                st.dataframe(query_result)

            except Exception as e:
                st.error(f"Query execution failed: {e}")
                logger.error(f"Query execution error: {e}")

if __name__ == "__main__":
    main()
