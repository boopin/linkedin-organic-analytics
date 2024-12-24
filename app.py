# App Version: 1.2.0
import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
from datetime import datetime
from dateutil.relativedelta import relativedelta
import logging

# Configure logging
logging.basicConfig(filename="app.log", level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

# Default columns for each schema
DEFAULT_COLUMNS = {
    "raw": ["date", "impressions", "clicks", "engagement_rate"],
    "monthly": ["month", "impressions", "clicks", "engagement_rate"],
    "weekly": ["week", "impressions", "clicks", "engagement_rate"],
    "quarterly": ["quarter", "impressions", "clicks", "engagement_rate"],
}

def preprocess_and_create_schemas(df: pd.DataFrame, conn: sqlite3.Connection, table_name: str):
    """Preprocess the dataframe and create schemas for raw, monthly, weekly, and quarterly data."""
    # Raw schema
    df.to_sql(f"{table_name}_raw", conn, index=False, if_exists="replace")

    # Add additional time-based columns
    df["month"] = df["date"].dt.to_period("M")
    df["week"] = df["date"].dt.to_period("W")
    df["quarter"] = df["date"].dt.to_period("Q")

    # Monthly schema
    monthly_df = df.groupby("month").sum().reset_index()
    monthly_df.to_sql(f"{table_name}_monthly", conn, index=False, if_exists="replace")

    # Weekly schema
    weekly_df = df.groupby("week").sum().reset_index()
    weekly_df.to_sql(f"{table_name}_weekly", conn, index=False, if_exists="replace")

    # Quarterly schema
    quarterly_df = df.groupby("quarter").sum().reset_index()
    quarterly_df.to_sql(f"{table_name}_quarterly", conn, index=False, if_exists="replace")

    logger.info(f"Schemas created for table: {table_name}")
    return [f"{table_name}_raw", f"{table_name}_monthly", f"{table_name}_weekly", f"{table_name}_quarterly"]

def display_schema_columns(selected_schema: str, conn: sqlite3.Connection):
    """Display available columns for the selected schema."""
    columns_query = f"PRAGMA table_info({selected_schema});"
    columns_info = pd.read_sql_query(columns_query, conn)
    available_columns = [col["name"] for col in columns_info.to_dict(orient="records")]
    st.write("### Available Columns in the Selected Schema")
    st.write(", ".join(available_columns))
    return available_columns

def main():
    st.title("AI Reports Analyzer with Enhanced Schema Support")

    uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
    if not uploaded_file:
        st.info("Please upload a file.")
        return

    try:
        conn = sqlite3.connect(":memory:")
        schemas = []

        # Load and preprocess uploaded file
        if uploaded_file.name.endswith(".xlsx"):
            excel_data = pd.ExcelFile(uploaded_file)
            for sheet_name in excel_data.sheet_names:
                df = pd.read_excel(excel_data, sheet_name=sheet_name)
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                table_name = sheet_name.lower().replace(" ", "_")
                schemas.extend(preprocess_and_create_schemas(df, conn, table_name))

        elif uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            table_name = uploaded_file.name.lower().replace(".csv", "").replace(" ", "_")
            schemas.extend(preprocess_and_create_schemas(df, conn, table_name))

        else:
            raise ValueError("Unsupported file type. Please upload a CSV or Excel file.")

        st.success("Data successfully loaded and schemas created!")

        # Allow the user to select a schema
        selected_schema = st.selectbox("Select a schema to query:", schemas)

        # Display schema-specific columns
        available_columns = display_schema_columns(selected_schema, conn)

        # Provide example queries
        st.write("### Example Queries")
        st.write("- Show me impressions_total for July 2024")
        st.write("- Compare engagement rate between Q3 and Q2 2024")

        # Input query
        user_query = st.text_area("Enter your query or prompt", "")

        if st.button("Run Query"):
            if not user_query.strip():
                st.error("Please enter a valid query or prompt.")
                return

            try:
                # Construct SQL query
                if "compare" in user_query.lower():
                    st.warning("Comparison functionality is under development.")
                else:
                    sql_query = f"SELECT * FROM {selected_schema} WHERE impressions_total > 0 LIMIT 10"  # Simplified example
                    st.info(f"Generated SQL Query:\n{sql_query}")

                    # Execute and display the query
                    query_result = pd.read_sql_query(sql_query, conn)
                    st.write("### Query Results")
                    st.dataframe(query_result)

            except Exception as e:
                logger.error(f"Query Error: {e}")
                st.error(f"An error occurred: {e}")

    except Exception as e:
        logger.error(f"Error: {e}")
        st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
