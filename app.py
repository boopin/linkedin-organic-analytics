import streamlit as st
import pandas as pd
import sqlite3
import logging
from typing import Tuple
from langchain_openai.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
import plotly.express as px
from urllib.parse import urlparse
import re

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Column mapping
COLUMN_MAPPING = {
    "total impressions": "impressions",
    "total clicks": "clicks",
    "total likes": "likes",
    "total comments": "comments",
    "total reposts": "reposts",
    "engagement rate": "engagement_rate",
    "date": "date",
}

class SQLQueryAgent:
    """Handles SQL query generation."""
    def __init__(self, llm):
        self.llm = llm

    def generate_sql(self, user_query: str, schema: str) -> str:
        """Generate SQL query using GPT-4."""
        prompt = (
            f"Schema: {schema}\n"
            f"Query: {user_query}\n"
            f"Generate a valid SQL query for SQLite. Use the table name 'data_table'."
        )
        response = self.llm.invoke([HumanMessage(content=prompt)])
        sql_query = response.content.strip()
        return sql_query

class DataAnalyzer:
    """Analyzes data using SQLite and AI."""
    def __init__(self):
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self.llm = ChatOpenAI(model="gpt-4")
        self.sql_agent = SQLQueryAgent(self.llm)

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data."""
        df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df

    def load_data(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Load data into SQLite."""
        try:
            df = self.preprocess_data(df)
            df.to_sql("data_table", self.conn, index=False, if_exists="replace")
            schema = ", ".join([f"{col} ({dtype})" for col, dtype in zip(df.columns, df.dtypes)])
            return True, schema
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False, str(e)

    def analyze(self, user_query: str, schema: str) -> Tuple[pd.DataFrame, str]:
        """Perform analysis."""
        try:
            sql_query = self.sql_agent.generate_sql(user_query, schema)
            logger.info(f"Generated SQL query: {sql_query}")
            result = pd.read_sql_query(sql_query, self.conn)
            return result, sql_query
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise ValueError(f"Analysis failed: {e}")

def main():
    st.title("Social Media Analytics Tool")
    analyzer = DataAnalyzer()

    uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
    if not uploaded_file:
        st.info("Please upload a file.")
        return

    try:
        if uploaded_file.name.endswith(".xlsx"):
            excel_file = pd.ExcelFile(uploaded_file)
            sheet_name = st.selectbox("Select Sheet", excel_file.sheet_names)
            df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
        else:
            df = pd.read_csv(uploaded_file)

        success, schema = analyzer.load_data(df)
        if not success:
            st.error(f"Failed to load data: {schema}")
            return

        st.success("Data loaded successfully!")
        st.write("Schema:", schema)

        user_query = st.text_input("Enter your query")
        if st.button("Analyze"):
            try:
                result, sql_query = analyzer.analyze(user_query, schema)
                st.write("Results:")
                st.dataframe(result)
                st.write("SQL Query Used:")
                st.code(sql_query, language="sql")
            except Exception as e:
                st.error(f"Error: {e}")

    except Exception as e:
        st.error(f"Failed to process file: {e}")

if __name__ == "__main__":
    main()
