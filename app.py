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
    "total impressions": "impressions",
    "total clicks": "clicks",
    "total reactions": "reactions",
    "total comments": "comments",
    "total reposts": "reposts",
    "engagement rate": "engagement_rate"
}

class MetadataExtractionAgent:
    """Extracts schema and metadata from the dataset."""
    @staticmethod
    def extract_schema(df: pd.DataFrame) -> str:
        schema = [f"{col} ({dtype})" for col, dtype in zip(df.columns, df.dtypes)]
        return " | ".join(schema)


class ColumnMappingAgent:
    """Handles dynamic mapping of user terms to dataset column names."""
    @staticmethod
    def map_columns(user_query: str, df: pd.DataFrame, column_mapping: dict) -> str:
        """Map user-friendly terms to actual dataset columns using fuzzy matching."""
        available_columns = [col.lower() for col in df.columns]  # Lowercase for consistency
        for user_term, expected_column in column_mapping.items():
            # Use fuzzy matching to find the closest match
            match = difflib.get_close_matches(expected_column.lower(), available_columns, n=1, cutoff=0.6)
            if match:
                # Replace the user-friendly term with the actual column name
                actual_column = df.columns[available_columns.index(match[0])]  # Original case
                user_query = user_query.replace(user_term, actual_column)
        return user_query


class SQLQueryAgent:
    """Handles prompt-to-SQL query conversion using GPT-4."""
    def __init__(self, llm):
        self.llm = llm

    def generate_sql(self, user_query: str, schema: str, df: pd.DataFrame) -> str:
        """Converts a natural language query into SQL using GPT-4."""
        # Validate columns before generating the SQL query
        user_query = ColumnMappingAgent.map_columns(user_query, df, COLUMN_MAPPING)

        # Generate the SQL query
        prompt = (
            f"You are an expert SQL data analyst. Based on the following schema:\n\n"
            f"{schema}\n\n"
            f"Generate a valid SQL query for a SQLite database that matches this user query:\n"
            f"'{user_query}'.\n"
            f"Ensure the following:\n"
            f"- Use the actual column names from the schema.\n"
            f"- Replace placeholder table names with 'data_table'.\n"
            f"- Return a valid `SELECT` statement. If the query cannot be executed, explain why."
        )
        response = self.llm([HumanMessage(content=prompt)])
        sql_query = response.content.strip()

        # Replace placeholder table names with the actual table name
        sql_query = sql_query.replace("TABLENAME", "data_table").replace("table_name", "data_table")

        # Log the generated query
        logger.warning(f"Generated SQL query: {sql_query}")
        if not sql_query.lower().startswith("select"):
            explanation_prompt = (
                f"The SQL query generated for the user query: '{user_query}' failed.\n\n"
                f"The dataset schema is: {schema}.\n"
                f"Explain why the query failed and provide an alternative SQL query."
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

    def verify_table_existence(self):
        """Check if the table exists in SQLite before querying."""
        if not self.conn:
            raise ValueError("SQLite database is not initialized.")
        cursor = self.conn.cursor()
        tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
        if "data_table" not in [table[0] for table in tables]:
            raise ValueError("The required table 'data_table' does not exist in the database.")

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

            # Validate table existence
            self.verify_table_existence()

            # Generate SQL query using SQL Query Agent
            sql_query = self.sql_agent.generate_sql(user_query, schema, df)

            # Log and execute the SQL query
            logger.info(f"Executing SQL query: {sql_query}")
            df_result = pd.read_sql_query(sql_query, self.conn)
            return df_result, sql_query
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            raise Exception(f"Analysis failed: {e}")


def main():
    st.title("AI Data Analyzer with Dynamic Column Mapping")
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

        # Display dataset schema in a dropdown
        with st.expander("View Dataset Schema"):
            schema_list = schema.split(" | ")
            for col in schema_list:
                st.write(col)

        # Show example queries
        st.write("### Example Queries")
        st.write("- Show me the top 5 dates with the highest total impressions.")
        st.write("- Show me the top 5 posts with the highest impressions.")

        user_query = st.text_area("Enter your query", placeholder="e.g., Show me the top 5 posts with the highest impressions.")
        if st.button("Analyze"):
            try:
                with st.spinner("Analyzing your data..."):
                    result, query = st.session_state.analyzer.analyze(user_query, df)
                st.write("**Analysis Result:**")
                st.dataframe(result)
                st.write("**SQL Query Used:**")
                st.code(query, language='sql')
            except Exception as e:
                st.error(f"Analysis failed: {e}")
    else:
        st.error(f"Failed to load data: {schema}")


if __name__ == "__main__":
    main()
