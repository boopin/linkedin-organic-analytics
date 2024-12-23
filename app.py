import streamlit as st
import pandas as pd
import sqlite3
import logging
from typing import Tuple
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
import difflib

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Column mapping for user-friendly queries
COLUMN_MAPPING = {
    "total impressions": "impressions_(total)",
    "total clicks": "clicks_(total)",
    "total reactions": "reactions_(total)",
    "total comments": "comments_(total)",
    "total reposts": "reposts_(total)",
    "engagement rate": "engagement_rate_(total)"
}

class MetadataExtractionAgent:
    """Extracts schema and metadata from the dataset."""
    @staticmethod
    def extract_schema(df: pd.DataFrame) -> str:
        schema = [f"{col} ({dtype})" for col, dtype in zip(df.columns, df.dtypes)]
        return " | ".join(schema)


class SQLQueryAgent:
    """Handles prompt-to-SQL query conversion using GPT-4."""
    def __init__(self, llm):
        self.llm = llm

    def generate_sql(self, user_query: str, schema: str, column_mapping: dict) -> str:
        """Converts a natural language query into SQL using GPT-4."""
        # Map user terms to actual column names
        for user_term, column_name in column_mapping.items():
            user_query = user_query.replace(user_term, column_name)

        # Generate the SQL query
        prompt = (
            f"You are an expert SQL data analyst. Based on the following schema:\n\n"
            f"{schema}\n\n"
            f"Generate a valid SQL query for a SQLite database that matches this user query:\n"
            f"'{user_query}'.\n"
            f"Ensure the following:\n"
            f"- If the query references dates, use appropriate SQL functions like `GROUP BY`.\n"
            f"- Use `ORDER BY` to sort results by the metric specified in the query.\n"
            f"- Return a valid `SELECT` statement. If the query cannot be executed, explain why."
        )
        response = self.llm([HumanMessage(content=prompt)])
        sql_query = response.content.strip()

        # Validate and log the query
        logger.warning(f"Generated SQL query: {sql_query}")
        if not sql_query.lower().startswith("select"):
            explanation_prompt = (
                f"The following SQL query could not be generated for this user query: '{user_query}'.\n\n"
                f"The dataset schema is: {schema}.\n"
                f"Explain why the query failed and suggest an alternative query."
            )
            explanation_response = self.llm([HumanMessage(content=explanation_prompt)])
            raise ValueError(f"Query generation failed: {explanation_response.content.strip()}")
        
        return sql_query


class DataAnalyzer:
    def __init__(self):
        self.conn = None
        self.current_table = 'data_table'
        self.llm = ChatOpenAI(model="gpt-4")
        self.sql_agent = SQLQueryAgent(self.llm)

    def initialize_database(self):
        """Initialize SQLite database only when required."""
        if not self.conn:
            self.conn = sqlite3.connect(':memory:', check_same_thread=False)
            logger.info("SQLite database initialized.")

    def load_data(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Load dataset into SQLite."""
        try:
            self.initialize_database()
            df.columns = [c.lower().strip().replace(' ', '_') for c in df.columns]
            cursor = self.conn.cursor()
            cursor.execute(f"DROP TABLE IF EXISTS {self.current_table}")
            df.to_sql(self.current_table, self.conn, index=False, if_exists='replace')
            schema = MetadataExtractionAgent.extract_schema(df)
            return True, schema
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return False, str(e)

    def analyze(self, user_query: str, df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """Perform analysis with AI agents."""
        try:
            # Extract schema
            schema = MetadataExtractionAgent.extract_schema(df)

            # Generate SQL query using SQL Query Agent
            sql_query = self.sql_agent.generate_sql(user_query, schema, COLUMN_MAPPING)

            # Execute the SQL query
            df_result = pd.read_sql_query(sql_query, self.conn)
            return df_result, sql_query
        except Exception as e:
            raise Exception(f"Analysis failed: {e}")

def main():
    st.title("AI Data Analyzer with SQL Query Agent")
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = DataAnalyzer()

    uploaded_file = st.file_uploader("Upload data (CSV or Excel)", type=['csv', 'xlsx'])
    if not uploaded_file:
        st.info("Please upload a file to begin analysis.")
        st.stop()

    if uploaded_file.name.endswith('xlsx'):
        excel_file = pd.ExcelFile(uploaded_file)
        sheet_names = excel_file.sheet_names
        sheet_name = st.selectbox("Select the sheet to analyze", sheet_names)
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
    else:
        df = pd.read_csv(uploaded_file)

    success, schema = st.session_state.analyzer.load_data(df)
    if success:
        st.success("Data loaded successfully!")
        st.write(f"**Dataset Schema:** {schema}")
        
        user_query = st.text_area("Enter your query", placeholder="Show me the top 5 dates with the highest total impressions.")
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
