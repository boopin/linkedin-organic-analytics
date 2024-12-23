import streamlit as st
import pandas as pd
import sqlite3
import logging
from typing import Tuple
from langchain_openai.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
import difflib
import re

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Column mapping for user-friendly queries
COLUMN_MAPPING = {
    "total impressions": ["impressions", "total impressions", "impressions_(total)"],
    "total clicks": ["clicks", "total clicks", "clicks_(total)"],
    "total likes": ["likes", "total likes"],
    "total comments": ["comments", "total comments"],
    "total reposts": ["reposts", "total reposts"],
    "engagement rate": ["engagement_rate", "engagement rate"],
    "date": ["date"]
}

# Example queries for user guidance
EXAMPLE_QUERIES = [
    "Show me the top 5 dates with the highest total impressions.",
    "Show me the posts with the most clicks.",
    "What is the average engagement rate of all posts?",
    "Generate a bar graph of clicks grouped by post type."
]

class ColumnMappingAgent:
    """Handles dynamic mapping of user terms to dataset column names."""
    @staticmethod
    def preprocess_query(user_query: str) -> str:
        """Preprocess query to remove filler words and focus on meaningful terms."""
        filler_words = {"show", "me", "the", "top", "with", "highest", "and", "or", "by"}
        query_terms = [word for word in user_query.lower().split() if word not in filler_words]
        return " ".join(query_terms)

    @staticmethod
    def map_columns(user_query: str, df: pd.DataFrame, column_mapping: dict) -> str:
        """Map user-friendly terms to actual dataset columns."""
        preprocessed_query = ColumnMappingAgent.preprocess_query(user_query)
        available_columns = [col.lower() for col in df.columns]
        mapped_query = preprocessed_query
        for user_term, synonyms in column_mapping.items():
            for synonym in synonyms:
                match = difflib.get_close_matches(synonym.lower(), available_columns, n=1, cutoff=0.6)
                if match:
                    actual_column = df.columns[available_columns.index(match[0])]
                    mapped_query = mapped_query.replace(user_term, actual_column)
                    break
        return mapped_query

    @staticmethod
    def validate_query_columns(mapped_query: str, df: pd.DataFrame):
        """Validate if all referenced columns in the query exist in the dataset."""
        referenced_columns = re.findall(r"[a-zA-Z0-9_()]+", mapped_query)
        missing_columns = [col for col in referenced_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(
                f"The following columns referenced in the query do not exist in the dataset: {', '.join(missing_columns)}. "
                f"Available columns: {list(df.columns)}"
            )

class SQLQueryAgent:
    """Handles SQL query generation."""
    def __init__(self, llm):
        self.llm = llm

    def preprocess_query(self, user_query: str, column_mapping: dict, df: pd.DataFrame) -> str:
        """Preprocess and map columns in the query."""
        mapped_query = ColumnMappingAgent.map_columns(user_query, df, column_mapping)
        ColumnMappingAgent.validate_query_columns(mapped_query, df)
        return mapped_query

    def generate_sql(self, user_query: str, schema: str, df: pd.DataFrame) -> str:
        """Generate SQL query using GPT-4."""
        mapped_query = self.preprocess_query(user_query, COLUMN_MAPPING, df)

        # Generate the SQL query
        prompt = (
            f"Schema: {schema}\n"
            f"Query: {mapped_query}\n"
            f"Generate a valid SQL query for SQLite. Use the table name 'data_table'."
        )
        response = self.llm.invoke([HumanMessage(content=prompt)])
        sql_query = response.content.strip()

        sql_query = sql_query.replace("TABLENAME", "data_table").replace("table_name", "data_table")
        logger.warning(f"Generated SQL query: {sql_query}")
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

    def analyze(self, user_query: str, schema: str, df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """Perform analysis."""
        mapped_query = ColumnMappingAgent.map_columns(user_query, df, COLUMN_MAPPING)
        sql_query = self.sql_agent.generate_sql(mapped_query, schema, df)
        result = pd.read_sql_query(sql_query, self.conn)
        return result, sql_query

def main():
    st.title("AI Reports Analyzer")
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

        # Display schema dropdown
        st.write("### Dataset Schema")
        schema_columns = pd.DataFrame({"Column": df.columns, "Data Type": df.dtypes})
        selected_column = st.selectbox("Select a column to view details", schema_columns['Column'])
        selected_details = schema_columns[schema_columns['Column'] == selected_column]
        st.write("Details:", selected_details)

        st.write("### Example Queries")
        for query in EXAMPLE_QUERIES:
            st.markdown(f"- {query}")

        user_query = st.text_input("Enter your query")
        if st.button("Analyze"):
            try:
                result, sql_query = analyzer.analyze(user_query, schema, df)
                st.write("**Results:**")
                st.dataframe(result)
                st.write("**SQL Query Used:**")
                st.code(sql_query, language="sql")
            except ValueError as ve:
                st.error(f"Validation Error: {ve}")
            except Exception as e:
                st.error(f"Error: {e}")

    except Exception as e:
        st.error(f"Failed to process file: {e}")

if __name__ == "__main__":
    main()
