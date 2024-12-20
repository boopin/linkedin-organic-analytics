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
                else:
                    raise ValueError("The 'date' column contains no valid dates. Please check the dataset format.")
            else:
                logger.warning("The dataset is missing a 'date' column. Time-based analyses will be unavailable.")

            logger.info("Saving processed data to SQLite...")
            # Save the processed dataset into SQLite
            self.current_table = 'data_table'
            conn = self.get_connection()
            df.to_sql(self.current_table, conn, index=False, if_exists='replace')

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

    def analyze(self, user_query: str, schema_info: str) -> Tuple[pd.DataFrame, str]:
        """Generate and execute SQL query based on user input"""
        try:
            if not self.current_table:
                raise Exception("No data loaded. Please upload a dataset first.")

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
            raise Exception(f"Analysis failed: {str(e)}. Ensure the 'date' column is correctly formatted.")

    def generate_sql_with_langchain(self, user_query: str, schema_info: str) -> str:
        """Generate SQL query using LangChain"""
        prompt_template = PromptTemplate(
            input_variables=["user_query", "columns"],
            template=(
                "You are a SQL query generator. Based on the user's request, generate a valid SQL query. "
                "User request: '{user_query}'."
            ),
        )
        chain = LLMChain(llm=self.llm, prompt=prompt_template)
        sql_query = chain.run(user_query=user_query, columns=", ".join(self.get_table_columns()))
        return sql_query


def main():
    st.set_page_config(page_title="AI Data Analyzer", layout="wide")
    st.title("🔹 AI-Powered Data Analyzer")
    st.write("Upload your data and analyze it with your own queries!")

    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = DataAnalyzer()

    # File upload
    uploaded_file = st.file_uploader("Upload your data (Excel or CSV)", type=['xlsx', 'xls', 'csv'])

    if uploaded_file:
        success, schema_info = st.session_state.analyzer.load_data(uploaded_file)

        if success:
            st.success("Data loaded successfully!")
            with st.expander("View Data Schema"):
                st.code(schema_info)

            # Input for user query
            user_query = st.text_area("Enter your query about the data", height=100)
            if st.button("Analyze"):
                try:
                    df_result, sql_query = st.session_state.analyzer.analyze(user_query, schema_info)
                    st.write("### Query Result")
                    st.write(df_result)
                except Exception as e:
                    st.error(str(e))
        else:
            st.error("Failed to load the data. Check the file format and content.")

if __name__ == "__main__":
    main()
