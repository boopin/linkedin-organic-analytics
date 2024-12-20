import streamlit as st
import pandas as pd
import plotly.express as px
import sqlite3
from typing import Tuple
from datetime import datetime
import logging
import openai
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import OpenAI  # Updated import

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure OpenAI API key
openai.api_key = "your-openai-api-key"  # Replace with your OpenAI API key

class DataAnalyzer:
    def __init__(self):
        self.conn = None
        self.current_table = None
        self.llm = OpenAI(temperature=0)  # LangChain LLM setup
        logger.info("DataAnalyzer initialized.")

    def get_connection(self):
        """Ensure the database connection is open and return it."""
        if self.conn is None or not self.conn:
            self.conn = sqlite3.connect(':memory:', check_same_thread=False)
        return self.conn

    def close_connection(self):
        """Close the database connection to avoid memory leaks."""
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.info("Database connection closed.")

    def load_data(self, file, sheet_name=None) -> Tuple[bool, str]:
        """Load data from uploaded file into SQLite database and compute derived columns."""
        try:
            logger.info("Loading data...")
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
            else:
                excel_file = pd.ExcelFile(file)
                if sheet_name is None:
                    sheet_name = excel_file.sheet_names[0]
                df = pd.read_excel(file, sheet_name=sheet_name)

            # Clean column names
            df.columns = [
                c.lower().strip().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
                for c in df.columns
            ]

            # Check if 'date' column exists
            if 'date' in df.columns:
                logger.info("Processing date column...")
                df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y', errors='coerce')
                if not df['date'].isnull().all():
                    # Compute derived time-based fields
                    df['week'] = df['date'].dt.to_period('W-SUN').astype(str)
                    df['year_month'] = df['date'].dt.to_period('M').astype(str)
                    df['quarter'] = 'Q' + df['date'].dt.quarter.astype(str) + ' ' + df['date'].dt.year.astype(str)
                    df['year'] = df['date'].dt.year.astype(str)
                    df['week_start'] = df['date'] - pd.to_timedelta(df['date'].dt.weekday, unit='d')
                    df['week_start'] = df['week_start'].dt.to_period('W').astype(str)
                else:
                    raise ValueError("The 'date' column contains no valid dates. Please check the dataset format.")
            else:
                logger.warning("The dataset is missing a 'date' column. Time-based analyses will be unavailable.")

            logger.info("Saving processed data to SQLite...")
            # Save the processed dataset into SQLite
            self.current_table = 'data_table'
            conn = self.get_connection()
            df.to_sql(self.current_table, conn, index=False, if_exists='replace')

            # Validate derived columns if 'date' exists
            if 'date' in df.columns:
                cursor = conn.cursor()
                logger.info("Validating derived columns...")
                distinct_quarters = cursor.execute(f"SELECT DISTINCT quarter FROM {self.current_table}").fetchall()
                logger.info(f"Distinct quarters in the data: {distinct_quarters}")

            # Return schema information for user feedback
            cursor = conn.cursor()
            schema_info = cursor.execute(f"PRAGMA table_info({self.current_table})").fetchall()
            logger.info("Data loaded successfully.")
            return True, self.format_schema_info(schema_info)

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False, str(e)

    def format_schema_info(self, schema_info) -> str:
        """Format schema information for display"""
        columns = [f"- {col[1]} ({col[2]})" for col in schema_info]
        return "Table columns:\n" + "\n".join(columns)

    def get_table_columns(self) -> list:
        """Fetch the list of columns from the current table"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            columns = [row[1] for row in cursor.execute(f"PRAGMA table_info({self.current_table})").fetchall()]
            logger.info(f"Fetched table columns: {columns}")
            return columns
        except Exception as e:
            logger.error(f"Error fetching table columns: {str(e)}")
            return []

    def validate_data_availability(self, period: str, column: str) -> bool:
        """Validate if data is available for a specific period and column."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            query = f"SELECT COUNT(*) FROM {self.current_table} WHERE {column} = ?"
            cursor.execute(query, (period,))
            count = cursor.fetchone()[0]
            if count > 0:
                logger.info(f"Data availability check for {period} in column {column}: {count} rows found.")
                return True
            else:
                logger.warning(f"No data found for {period} in column {column}.")
                return False
        except sqlite3.OperationalError as e:
            logger.error(f"OperationalError during data validation: {str(e)}. This could indicate a missing column or invalid SQL.")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during data validation: {str(e)}")
            return False

    def analyze(self, user_query: str, schema_info: str) -> Tuple[pd.DataFrame, str]:
        """Generate and execute SQL query based on user input"""
        try:
            if not self.current_table:
                raise Exception("No data loaded. Please upload a dataset first.")

            logger.info("Analyzing user query...")
            user_query = self.generate_monthly_filter(user_query)
            if "quarter" in user_query.lower():
                user_query = user_query.replace(
                    "quarter",
                    """
                    CASE
                        WHEN strftime('%m', date) BETWEEN '01' AND '03' THEN 'Q1'
                        WHEN strftime('%m', date) BETWEEN '04' AND '06' THEN 'Q2'
                        WHEN strftime('%m', date) BETWEEN '07' AND '09' THEN 'Q3'
                        WHEN strftime('%m', date) BETWEEN '10' AND '12' THEN 'Q4'
                    END || ' ' || strftime('%Y', date)
                    """
                )
            sql_query = self.generate_sql_with_langchain(user_query, schema_info)

            logger.info(f"Executing SQL query: {sql_query}")
            conn = self.get_connection()
            df_result = pd.read_sql_query(sql_query, conn)

            # Verify if the query returned any data
            if df_result.empty:
                raise Exception("The query returned no data. Ensure the dataset has relevant data for the requested period.")

            logger.info("Analysis completed successfully.")
            return df_result, sql_query
        except Exception as e:
            logger.error(f"Analysis error: {str(e)}")
            raise Exception(f"Analysis failed: {str(e)}. Ensure the 'date' column is correctly formatted and the necessary columns exist.")

    def generate_monthly_filter(self, user_query: str) -> str:
        """Map user-specified months to year_month values"""
        month_mapping = {
            "january": "01", "february": "02", "march": "03",
            "april": "04", "may": "05", "june": "06",
            "july": "07", "august": "08", "september": "09",
            "october": "10", "november": "11", "december": "12"
        }
        for month_name, month_code in month_mapping.items():
            if month_name in user_query.lower():
                year = "2024"  # Default year if not specified
                if "2023" in user_query:
                    year = "2023"
                user_query = user_query.replace(
                    month_name,
                    f"{year}-{month_code}"
                )
        logger.info(f"User query after monthly filter mapping: {user_query}")
        return user_query

    def generate_sql_with_langchain(self, user_query: str, schema_info: str) -> str:
        """Generate SQL query using LangChain"""
        # Fetch available columns
        conn = self.get_connection()
        cursor = conn.cursor()
        available_columns = [row[1] for row in cursor.execute(f"PRAGMA table_info({self.current_table})").fetchall()]
        logger.info(f"Available columns in the table: {available_columns}")

        # Dynamically map user-specified columns to actual columns in the database
        column_mapping = {}
        for col in available_columns:
            normalized_col = col.replace('_', ' ').lower()
            column_mapping[normalized_col] = col
        logger.info(f"Column mapping: {column_mapping}")

        # Normalize user query by replacing user-friendly terms with actual column names
        normalized_query = user_query.lower()
        for user_col, actual_col in column_mapping.items():
            normalized_query = normalized_query.replace(user_col, actual_col)
        logger.info(f"Normalized user query: {normalized_query}")

        # LangChain prompt for SQL generation
        prompt_template = PromptTemplate(
            input_variables=["user_query", "columns"],
            template=(
                "You are a SQL query generator. Based on the user's request, generate a valid SQL query. "
                "The table is named '{table_name}' and has the following columns: {columns}. "
                "If the user query references quarters, dynamically calculate them from the 'date' column as follows: "
                "CASE WHEN strftime('%m', date) BETWEEN '01' AND '03' THEN 'Q1' "
                "WHEN strftime('%m', date) BETWEEN '04' AND '06' THEN 'Q2' "
                "WHEN strftime('%m', date) BETWEEN '07' AND '09' THEN 'Q3' "
                "WHEN strftime('%m', date) BETWEEN '10' AND '12' THEN 'Q4'. "
                "Combine the quarter and year (e.g., 'Q1 2024'). "
                "User request: '{user_query}'."
            ),
        )

        # Generate SQL using LangChain
        chain = LLMChain(llm=self.llm, prompt=prompt_template)
        sql_query = chain.run(user_query=normalized_query, columns=", ".join(available_columns), table_name=self.current_table)
        logger.info(f"Generated SQL query: {sql_query}")
        return sql_query

def main():
    st.set_page_config(page_title="AI Data Analyzer", layout="wide")
    st.title("🔹 AI-Powered Data Analyzer")
    st.write("Upload your data and analyze it with your own queries!")

    # Notices for query structure
    st.info("""
        **Query Guidelines:**
        - Specify the time period for analysis (e.g., "Compare Q3 2024 vs Q2 2024").
        - Use clear column names like "impressions", "clicks", etc.
        - Add "chart" or "table" to indicate the output format.
        """)

    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = DataAnalyzer()

    # File upload
    uploaded_file = st.file_uploader("Upload your data (Excel or CSV)", type=['xlsx', 'xls', 'csv'])
    selected_sheet = None

    if uploaded_file:
        # Reinitialize the analyzer to reset the database connection for new files
        st.session_state.analyzer = DataAnalyzer()

        if uploaded_file.name.endswith(('xls', 'xlsx')):
            excel_file = pd.ExcelFile(uploaded_file)
            sheet_names = excel_file.sheet_names
            selected_sheet = st.selectbox("Select a sheet to analyze", sheet_names)

        success, schema_info = st.session_state.analyzer.load_data(uploaded_file, sheet_name=selected_sheet)

        if success:
            st.success("Data loaded successfully!")

            with st.expander("View Data Schema"):
                st.code(schema_info)

            with st.expander("View Data Columns"):
                conn = st.session_state.analyzer.get_connection()
                cursor = conn.cursor()
                columns = [row[1] for row in cursor.execute(f"PRAGMA table_info({st.session_state.analyzer.current_table})").fetchall()]
                st.write(columns)

            # Input for user query
            user_query = st.text_area(
                "Enter your query about the data",
                placeholder="e.g., 'Show the monthly
