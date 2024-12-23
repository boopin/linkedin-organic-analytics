import streamlit as st
import pandas as pd
import sqlite3
import logging
from typing import Tuple
from langchain_openai.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
import difflib
import plotly.express as px

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Column mapping for user-friendly queries
COLUMN_MAPPING = {
    "total impressions": ["impressions", "total impressions"],
    "total clicks": ["clicks", "total clicks"],
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
    "Show the total likes grouped by month."
]

class ColumnMappingAgent:
    """Handles dynamic mapping of user terms to dataset column names."""
    @staticmethod
    def map_columns(user_query: str, df: pd.DataFrame, column_mapping: dict) -> str:
        """Map user-friendly terms to actual dataset columns using fuzzy matching."""
        available_columns = [col.lower() for col in df.columns]
        for user_term, synonyms in column_mapping.items():
            for synonym in synonyms:
                match = difflib.get_close_matches(synonym.lower(), available_columns, n=1, cutoff=0.6)
                if match:
                    actual_column = df.columns[available_columns.index(match[0])]
                    user_query = user_query.replace(user_term, actual_column)
                    return user_query
        return user_query

class SQLQueryAgent:
    """Handles SQL query generation."""
    def __init__(self, llm):
        self.llm = llm

    def generate_sql(self, user_query: str, schema: str, df: pd.DataFrame) -> str:
        """Generate SQL query using GPT-4."""
        user_query = ColumnMappingAgent.map_columns(user_query, df, COLUMN_MAPPING)

        # Generate the SQL query
        prompt = (
            f"Schema: {schema}\n"
            f"Query: {user_query}\n"
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
        user_query = ColumnMappingAgent.map_columns(user_query, df, COLUMN_MAPPING)
        sql_query = self.sql_agent.generate_sql(user_query, schema, df)
        result = pd.read_sql_query(sql_query, self.conn)
        return result, sql_query

def create_plot(df: pd.DataFrame, x: str, y: str, chart_type: str):
    """Create a Plotly chart."""
    if chart_type == "Bar":
        fig = px.bar(df, x=x, y=y, title=f"{y} vs {x}")
    elif chart_type == "Line":
        fig = px.line(df, x=x, y=y, title=f"{y} vs {x}")
    elif chart_type == "Scatter":
        fig = px.scatter(df, x=x, y=y, title=f"{y} vs {x}", size=y, hover_data=df.columns)
    else:
        fig = px.histogram(df, x=x, title=f"Distribution of {x}")
    return fig

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

        # Display schema dropdown
        schema_columns = schema.split(", ")
        st.write("### Dataset Schema")
        schema_selection = st.selectbox("Select a column to view details", schema_columns)

        # Show example queries
        st.write("### Example Queries")
        for query in EXAMPLE_QUERIES:
            st.markdown(f"- {query}")

        # User query input
        user_query = st.text_input("Enter your query")
        if st.button("Analyze"):
            try:
                result, sql_query = analyzer.analyze(user_query, schema, df)
                st.write("**Results:**")
                st.dataframe(result)
                st.write("**SQL Query Used:**")
                st.code(sql_query, language="sql")
            except Exception as e:
                st.error(f"Error: {e}")

        # Visualization Options
        st.write("### Generate Visualizations")
        x_axis = st.selectbox("Select X-axis", options=df.columns)
        y_axis = st.selectbox("Select Y-axis", options=df.columns)
        chart_type = st.selectbox("Select Chart Type", ["Bar", "Line", "Scatter", "Histogram"])
        if st.button("Generate Chart"):
            try:
                fig = create_plot(df, x=x_axis, y=y_axis, chart_type=chart_type)
                st.plotly_chart(fig)
            except Exception as e:
                st.error(f"Error generating chart: {e}")

    except Exception as e:
        st.error(f"Failed to process file: {e}")

if __name__ == "__main__":
    main()
