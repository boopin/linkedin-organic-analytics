import streamlit as st
import pandas as pd
import sqlite3
import logging
from typing import Tuple  # Fixed import for Tuple
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class DataAnalyzer:
    def __init__(self):
        self.conn = None
        self.current_table = 'data_table'
        self.llm = ChatOpenAI(model="gpt-4")

    def initialize_database(self):
        """Initialize SQLite database only when required."""
        if not self.conn:
            self.conn = sqlite3.connect(':memory:', check_same_thread=False)
            logger.info("SQLite database initialized.")

    def extract_schema(self, df: pd.DataFrame) -> str:
        """Extract schema dynamically from the dataset."""
        schema = [f"{col} ({dtype})" for col, dtype in zip(df.columns, df.dtypes)]
        return " | ".join(schema)

    def load_data(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Load dataset into SQLite."""
        try:
            self.initialize_database()
            # Clean column names for SQLite compatibility
            df.columns = [c.lower().strip().replace(' ', '_') for c in df.columns]

            # Drop existing table
            cursor = self.conn.cursor()
            cursor.execute(f"DROP TABLE IF EXISTS {self.current_table}")

            # Load the DataFrame into SQLite
            df.to_sql(self.current_table, self.conn, index=False, if_exists='replace')

            # Extract schema for passing to GPT-4
            schema = self.extract_schema(df)
            return True, schema
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return False, str(e)

    def generate_sql_with_gpt4(self, user_query: str, df: pd.DataFrame) -> str:
        """Generate SQL query dynamically using GPT-4."""
        schema = self.extract_schema(df)
        prompt = (
            f"You are an expert SQL data analyst. Based on the following schema:\n\n"
            f"{schema}\n\n"
            f"Generate an SQL query that matches this user query:\n'{user_query}'. "
            f"If the query is invalid or the dataset does not support it, explain why."
        )
        response = self.llm([HumanMessage(content=prompt)])
        sql_query = response.content.strip()
        if not sql_query.lower().startswith("select"):
            raise ValueError("Generated query is not a valid SELECT statement.")
        return sql_query

    def analyze(self, user_query: str, df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """Perform analysis with GPT-4 as the sole SQL generator."""
        try:
            sql_query = self.generate_sql_with_gpt4(user_query, df)
            logger.info(f"Generated SQL query: {sql_query}")

            # Execute the SQL query
            df_result = pd.read_sql_query(sql_query, self.conn)
            return df_result, sql_query
        except Exception as e:
            raise Exception(f"Analysis failed: {e}")

def main():
    st.title("Simplified AI Data Analyzer")
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = DataAnalyzer()

    # File upload functionality
    uploaded_file = st.file_uploader("Upload data (CSV or Excel)", type=['csv', 'xlsx'])
    if not uploaded_file:
        st.info("Please upload a file to begin analysis.")
        st.stop()

    # Load the dataset
    df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('xlsx') else pd.read_csv(uploaded_file)
    success, schema = st.session_state.analyzer.load_data(df)

    if success:
        st.success("Data loaded successfully!")
        st.write(f"**Dataset Schema:** {schema}")
        
        # User query input
        user_query = st.text_area("Enter your query", placeholder="e.g., Show me top 5 posts by clicks.")
        if st.button("Analyze"):
            try:
                with st.spinner("Analyzing your data..."):
                    result, query = st.session_state.analyzer.analyze(user_query, df)
                st.write("**Analysis Result:**")
                st.dataframe(result)
                st.write("**SQL Query Used:**")
                st.code(query, language='sql')
            except Exception as e:
                st.error(str(e))
    else:
        st.error(f"Failed to load data: {schema}")

if __name__ == "__main__":
    main()
