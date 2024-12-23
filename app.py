import streamlit as st
import pandas as pd
import sqlite3
import logging
from typing import Tuple
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.schema import HumanMessage

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class MetadataExtractionAgent:
    """Extracts schema and metadata from the dataset."""
    @staticmethod
    def extract_schema(df: pd.DataFrame) -> str:
        schema = [f"{col} ({dtype})" for col, dtype in zip(df.columns, df.dtypes)]
        return " | ".join(schema)


class DataValidationAgent:
    """Validates dataset compatibility with user queries."""
    @staticmethod
    def validate_schema(df: pd.DataFrame, required_columns: list) -> None:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"The dataset is missing required columns: {', '.join(missing_columns)}")


class SQLQueryAgent:
    """Generates SQL queries dynamically using GPT-4."""
    def __init__(self, llm):
        self.llm = llm

    def generate_sql(self, user_query: str, schema: str) -> str:
        prompt = (
            f"You are an expert SQL data analyst. Based on the following schema:\n\n"
            f"{schema}\n\n"
            f"Generate an SQL query for a SQLite database that matches this user query:\n"
            f"'{user_query}'. Ensure the query uses appropriate SQL functions for time-based analysis "
            f"if the query references dates. If the query cannot be executed, explain why."
        )
        response = self.llm([HumanMessage(content=prompt)])
        sql_query = response.content.strip()
        if not sql_query.lower().startswith("select"):
            raise ValueError("Generated query is not a valid SELECT statement.")
        return sql_query


class InsightsGenerationAgent:
    """Generates insights from SQL query results."""
    @staticmethod
    def generate_insights(results: pd.DataFrame) -> str:
        if 'clicks' in results.columns:
            top_clicks = results['clicks'].max()
            avg_clicks = results['clicks'].mean()
            return f"The top post had {top_clicks} clicks. The average clicks across posts were {avg_clicks:.2f}."
        return "No actionable insights available for this dataset."


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
            # Extract schema and validate
            schema = MetadataExtractionAgent.extract_schema(df)
            DataValidationAgent.validate_schema(df, required_columns=['date', 'total_impressions'])

            # Generate SQL query
            sql_query = self.sql_agent.generate_sql(user_query, schema)
            df_result = pd.read_sql_query(sql_query, self.conn)
            return df_result, sql_query
        except Exception as e:
            raise Exception(f"Analysis failed: {e}")

def main():
    st.title("AI Data Analyzer with Agents")
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = DataAnalyzer()

    uploaded_file = st.file_uploader("Upload data (CSV or Excel)", type=['csv', 'xlsx'])
    if not uploaded_file:
        st.info("Please upload a file to begin analysis.")
        st.stop()

    # Handle multi-sheet Excel files
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
        
        user_query = st.text_area("Enter your query", placeholder="Show me top 5 posts by clicks.")
        if st.button("Analyze"):
            try:
                with st.spinner("Analyzing your data..."):
                    result, query = st.session_state.analyzer.analyze(user_query, df)
                    insights = InsightsGenerationAgent.generate_insights(result)
                st.write("**Analysis Result:**")
                st.dataframe(result)
                st.write("**Generated Insights:**")
                st.write(insights)
                st.write("**SQL Query Used:**")
                st.code(query, language='sql')
            except Exception as e:
                st.error(str(e))
    else:
        st.error(f"Failed to load data: {schema}")

if __name__ == "__main__":
    main()
