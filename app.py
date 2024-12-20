import streamlit as st
import pandas as pd
import sqlite3
from typing import Tuple
import logging
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataAnalyzer:
    def __init__(self):
        self.conn = sqlite3.connect(':memory:', check_same_thread=False)
        self.current_table = 'data_table'
        self.llm = ChatOpenAI(model="gpt-4")  # Use GPT-4 for dynamic query generation

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean column names and handle missing or invalid data."""
        # Clean column names
        df.columns = [
            c.lower().strip().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
            for c in df.columns
        ]

        # Drop entirely empty columns
        df = df.dropna(how='all', axis=1)

        # Fill missing numeric values with 0 and ensure consistent types
        for col in df.select_dtypes(include=['float64', 'int64']).columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Ensure all column names and data types are compatible with SQLite
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).fillna('')

        if df.empty:
            raise ValueError("The dataset is empty after preprocessing. Ensure the sheet contains valid data.")

        return df

    def load_data(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Load preprocessed DataFrame into SQLite database."""
        try:
            # Preprocess the dataset
            processed_df = self.preprocess_data(df)
            logger.info(f"Dataset after preprocessing: {processed_df.head()}")

            # Drop existing table
            cursor = self.conn.cursor()
            cursor.execute(f"DROP TABLE IF EXISTS {self.current_table}")

            # Save the processed dataset into SQLite
            processed_df.to_sql(self.current_table, self.conn, index=False, if_exists='replace')
            logger.info("Dataset loaded into SQLite successfully.")

            # Return schema information for user feedback
            schema_info = cursor.execute(f"PRAGMA table_info({self.current_table})").fetchall()
            return True, self.format_schema_info(schema_info)

        except Exception as e:
            logger.error(f"Error loading data into SQLite: {e}")
            return False, str(e)

    def format_schema_info(self, schema_info) -> str:
        """Format schema information for display."""
        return "\n".join([f"- {col[1]} ({col[2]})" for col in schema_info])

    def analyze(self, user_query: str, df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """Perform analysis based on user query."""
        try:
            logger.info(f"User query: {user_query}")

            # Extract the requested metric from the query
            metric = self.extract_metric_from_query(user_query, df)
            logger.info(f"Extracted metric: {metric}")

            # Generate the SQL query using GPT-4
            sql_query = self.generate_sql_with_gpt4(user_query, df)
            logger.info(f"Generated SQL query: {sql_query}")

            # Execute the query
            df_result = pd.read_sql_query(sql_query, self.conn)
            logger.info(f"Query execution result: {df_result.head()}")

            # Check for empty results
            if df_result.empty:
                raise ValueError(f"The query returned no data. Ensure the dataset has valid entries for '{metric}'.")

            return df_result, sql_query
        except Exception as e:
            logger.error(f"Analysis error: {e}")

            # Provide a fallback query using the extracted metric
            fallback_query = (
                f"SELECT post_title, post_link, post_type, {metric} "
                f"FROM {self.current_table} "
                f"ORDER BY {metric} DESC LIMIT 5;"
            )
            logger.info(f"Using fallback query: {fallback_query}")

            try:
                df_result = pd.read_sql_query(fallback_query, self.conn)
                logger.info(f"Fallback query result: {df_result.head()}")
                return df_result, fallback_query
            except Exception as fallback_error:
                logger.error(f"Fallback query also failed: {fallback_error}")
                raise Exception("Analysis failed: Both primary and fallback queries failed.")

    def extract_metric_from_query(self, user_query: str, df: pd.DataFrame) -> str:
        """Extract the ranking metric from the user's query."""
        available_columns = [col.lower() for col in df.columns]
        query_keywords = user_query.lower().split()

        for keyword in query_keywords:
            if keyword in available_columns:
                return keyword

        return 'clicks'

    def generate_sql_with_gpt4(self, user_query: str, df: pd.DataFrame) -> str:
        """Generate SQL query dynamically using GPT-4."""
        schema = self.extract_schema_and_sample(df)
        metric = self.extract_metric_from_query(user_query, df)
        prompt = (
            f"You are an expert in data analysis. Based on the following dataset schema and sample data, "
            f"generate a valid SQL query for a SQLite database that matches the user's intent.\n\n"
            f"{schema}\n\nUser Query: {user_query}\n"
        )
        response = self.llm([HumanMessage(content=prompt)])
        sql_query = response.content.strip()

        if not sql_query.lower().startswith("select"):
            raise ValueError("Generated query is not a valid SELECT statement.")

        return sql_query

def main():
    st.title("📊 AI-Powered Data Analyzer")

    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = DataAnalyzer()

    uploaded_file = st.file_uploader("Upload your data (Excel or CSV)", type=['xlsx', 'xls', 'csv'])
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        success, schema_info = st.session_state.analyzer.load_data(df)

        if success:
            st.success("Data loaded successfully!")
            user_query = st.text_area("Enter your query about the data", "")
            if st.button("Analyze"):
                try:
                    result, query = st.session_state.analyzer.analyze(user_query, df)
                    st.write(result)
                except Exception as e:
                    st.error(str(e))

if __name__ == "__main__":
    main()
