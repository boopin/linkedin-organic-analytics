# App Version: 1.2.0
import streamlit as st
import pandas as pd
import sqlite3
import re
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(filename="app.log", level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

DEFAULT_COLUMNS = {
    "all_posts": ["post_title", "post_link", "posted_by", "likes", "engagement_rate", "date"],
    "metrics": ["date", "impressions_total", "clicks", "reactions_total", "engagement_rate"],
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
    def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
        df = PreprocessingPipeline.clean_column_names(df)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df

def generate_sql_query(query, table_name, available_columns):
    """Generate SQL query from user query and schema."""
    try:
        mentioned_columns = [col for col in available_columns if col in query.lower()]
        if not mentioned_columns:
            mentioned_columns = DEFAULT_COLUMNS.get(table_name, [])
        
        if "top" in query.lower():
            match = re.search(r"top (\d+)", query, re.IGNORECASE)
            limit = match.group(1) if match else "10"
            order_column = mentioned_columns[0] if mentioned_columns else DEFAULT_COLUMNS.get(table_name, [])[0]
            sql_query = f"SELECT {', '.join(mentioned_columns)} FROM {table_name} ORDER BY {order_column} DESC LIMIT {limit}"
            return sql_query
        
        return f"SELECT * FROM {table_name} LIMIT 10"
    except Exception as e:
        logger.error(f"Error generating SQL query: {e}")
        return None

def main():
    st.title("AI-Powered Data Analysis Tool")

    uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
    if not uploaded_file:
        st.info("Please upload a file.")
        return

    try:
        conn = sqlite3.connect(":memory:")
        table_names = []

        # Load and preprocess the uploaded file
        if uploaded_file.name.endswith(".xlsx"):
            excel_data = pd.ExcelFile(uploaded_file)
            for sheet_name in excel_data.sheet_names:
                df = pd.read_excel(excel_data, sheet_name=sheet_name)
                df = PreprocessingPipeline.preprocess_data(df)
                table_name = sheet_name.lower().replace(" ", "_").replace("-", "_")
                df.to_sql(table_name, conn, index=False, if_exists="replace")
                table_names.append(table_name)

        elif uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            df = PreprocessingPipeline.preprocess_data(df)
            table_name = uploaded_file.name.lower().replace(".csv", "").replace(" ", "_").replace("-", "_")
            df.to_sql(table_name, conn, index=False, if_exists="replace")
            table_names.append(table_name)

        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return

        st.success("File successfully processed and saved to the database!")

        # Allow user to select a table
        selected_table = st.selectbox("Select a table for analysis:", table_names)

        # Display schema for the selected table
        schema_query = f"PRAGMA table_info({selected_table});"
        schema_info = pd.read_sql_query(schema_query, conn)
        st.write("### Table Schema")
        st.dataframe(schema_info)

        # Example queries
        st.write("### Example Queries")
        for example in EXAMPLE_QUERIES:
            st.markdown(f"- {example}")

        user_query = st.text_area("Enter your query or prompt", "")

        if st.button("Run Query"):
            if not user_query.strip():
                st.error("Please enter a valid query or prompt.")
                return

            try:
                # Extract available columns dynamically
                available_columns = schema_info["name"].str.lower().tolist()

                # Generate SQL query
                sql_query = generate_sql_query(user_query, selected_table, available_columns)
                if not sql_query:
                    st.error("Could not generate a valid SQL query.")
                    return

                st.info(f"Generated SQL Query:\n{sql_query}")
                df_result = pd.read_sql_query(sql_query, conn)
                st.write("### Query Results")
                st.dataframe(df_result)

            except Exception as e:
                st.error(f"Query execution failed: {e}")
                logger.error(f"Query execution error: {e}")

    except Exception as e:
        st.error(f"Error: {e}")
        logger.error(f"Error: {e}")

if __name__ == "__main__":
    main()
