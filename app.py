import streamlit as st
import pandas as pd
import sqlite3
import logging
from typing import Tuple
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

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the dataset to clean columns and create time-based groupings."""
        df.columns = [c.lower().strip().replace(' ', '_') for c in df.columns]

        # Create time-based groupings if date exists
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        else:
            logger.warning("No 'date' column found; skipping date-based preprocessing.")
        return df

    def load_data(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Load the preprocessed DataFrame into SQLite."""
        try:
            self.initialize_database()
            df = self.preprocess_data(df)
            cursor = self.conn.cursor()
            cursor.execute(f"DROP TABLE IF EXISTS {self.current_table}")
            df.to_sql(self.current_table, self.conn, index=False, if_exists='replace')
            schema = [f"{col} ({dtype})" for col, dtype in zip(df.columns, df.dtypes)]
            return True, " | ".join(schema)
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return False, str(e)

    def extract_metric_from_query(self, user_query: str, df: pd.DataFrame) -> str:
        """Extract metric dynamically from user query."""
        query_keywords = user_query.lower().split()
        available_columns = [col.lower() for col in df.columns]
        for keyword in query_keywords:
            if keyword in available_columns:
                return keyword
        raise ValueError(f"Metric not found in the dataset. Available columns: {', '.join(df.columns)}")

    def generate_sql(self, user_query: str, df: pd.DataFrame) -> str:
        """Generate SQL dynamically based on dataset schema and user query."""
        metric = self.extract_metric_from_query(user_query, df)
        if 'date' in df.columns:
            return (
                f"SELECT date, SUM({metric}) AS total_{metric} "
                f"FROM {self.current_table} "
                f"GROUP BY date "
                f"ORDER BY total_{metric} DESC LIMIT 5;"
            )
        else:
            raise ValueError("This query requires a 'date' column, but the dataset does not include one.")

    def analyze(self, user_query: str, df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """Perform SQL-based analysis dynamically."""
        try:
            metric = self.extract_metric_from_query(user_query, df)
            sql_query = self.generate_sql(user_query, df)
            df_result = pd.read_sql_query(sql_query, self.conn)
            return df_result, sql_query
        except Exception as e:
            raise Exception(f"Analysis failed: {e}")

def main():
    st.title("AI Data Analyzer")
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = DataAnalyzer()

    uploaded_file = st.file_uploader("Upload data (CSV or Excel)", type=['csv', 'xlsx'])
    if not uploaded_file:
        st.stop()

    df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('xlsx') else pd.read_csv(uploaded_file)
    success, schema = st.session_state.analyzer.load_data(df)

    if success:
        user_query = st.text_area("Enter your query")
        if st.button("Analyze"):
            try:
                result, query = st.session_state.analyzer.analyze(user_query, df)
                st.dataframe(result)
                st.code(query, language='sql')
            except Exception as e:
                st.error(e)
