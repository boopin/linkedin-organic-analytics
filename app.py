# App Version: 1.2.0
import streamlit as st
import pandas as pd
import sqlite3
import logging
import re
from datetime import datetime
from difflib import get_close_matches

# Configure logging
logging.basicConfig(filename="app.log", level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
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
]

def get_schema(table_name, conn):
    """Fetches the schema (column names) of the selected table."""
    query = f"PRAGMA table_info({table_name});"
    try:
        schema = pd.read_sql_query(query, conn)
        return schema["name"].tolist()
    except Exception as e:
        logger.error(f"Error fetching schema for table {table_name}: {e}")
        return []

def map_column_name(user_input, available_columns):
    """Maps user-friendly column names to actual column names using fuzzy matching."""
    matched_column = get_close_matches(user_input.lower(), available_columns, n=1, cutoff=0.6)
    return matched_column[0] if matched_column else None

def preprocess_dataframe(df):
    """Preprocess the dataframe to standardize column names."""
    df.columns = [col.lower().strip().replace(" ", "_").replace("(", "").replace(")", "") for col in df.columns]
    return df

def main():
    st.title("AI-Powered SQL Query App")

    uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
    if not uploaded_file:
        st.info("Please upload a file.")
        return

    try:
        conn = sqlite3.connect(":memory:")
        table_names = []

        # Load the uploaded file
        if uploaded_file.name.endswith(".xlsx"):
            excel_data = pd.ExcelFile(uploaded_file)
            for sheet in excel_data.sheet_names:
                df = pd.read_excel(excel_data, sheet_name=sheet)
                df = preprocess_dataframe(df)
                table_name = sheet.lower().replace(" ", "_").replace("-", "_")
                df.to_sql(table_name, conn, index=False, if_exists="replace")
                table_names.append(table_name)

        elif uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            df = preprocess_dataframe(df)
            table_name = uploaded_file.name.lower().replace(".csv", "").replace(" ", "_").replace("-", "_")
            df.to_sql(table_name, conn, index=False, if_exists="replace")
            table_names.append(table_name)

        else:
            raise ValueError("Unsupported file type. Please upload a CSV or Excel file.")

        st.success("Data successfully loaded into the database!")

        # Let the user select a table
        selected_table = st.selectbox("Select a table to query:", table_names)
        schema = get_schema(selected_table, conn)

        st.write("### Table Schema")
        st.write(schema)

        # Allow users to select columns for the query
        user_query = st.text_area("Enter your query or prompt", "")

        if st.button("Run Query"):
            if not user_query.strip():
                st.error("Please enter a valid query or prompt.")
                return

            try:
                # Replace user-friendly aliases with actual column names
                query_columns = re.findall(r"\b(" + "|".join(re.escape(col) for col in schema) + r")\b", user_query, re.IGNORECASE)
                if not query_columns:
                    st.warning("No matching columns found in the query. Please revise your query.")
                    return

                # Construct the SQL query
                sql_query = f"SELECT {', '.join(query_columns)} FROM {selected_table} ORDER BY {query_columns[-1]} DESC LIMIT 10"
                st.info(f"Generated SQL Query:\n{sql_query}")

                # Execute the query
                query_result = pd.read_sql_query(sql_query, conn)
                st.write("### Query Results")
                st.dataframe(query_result)

            except Exception as e:
                logger.error(f"Error executing query: {e}")
                st.error(f"An error occurred: {e}")

    except Exception as e:
        logger.error(f"Error: {e}")
        st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
