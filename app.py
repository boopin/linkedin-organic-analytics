import streamlit as st
import pandas as pd
import sqlite3
from typing import Tuple
import logging
from datetime import datetime
import plotly.express as px
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage
from langchain.chains import LLMChain

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataAnalyzer:
    def __init__(self):
        self.conn = sqlite3.connect(':memory:', check_same_thread=False)
        self.current_table = 'data_table'
        self.llm = ChatOpenAI(model="gpt-3.5-turbo")  # Updated for chat-based model

    def load_data(self, file, sheet_name=None) -> Tuple[bool, str]:
        """Load data from uploaded file into SQLite database and handle optional date-based processing."""
        try:
            # Load data
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

            # Optional: Process the date column if present
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')  # Parse dates
                if not df['date'].isnull().all():
                    # Compute derived time-based fields
                    df['week'] = df['date'].dt.to_period('W-SUN').astype(str)
                    df['year_month'] = df['date'].dt.to_period('M').astype(str)
                    df['quarter'] = 'Q' + df['date'].dt.quarter.astype(str) + ' ' + df['date'].dt.year.astype(str)
                    df['year'] = df['date'].dt.year.astype(str)
                else:
                    st.warning("The 'date' column contains no valid dates. Time-based fields will not be generated.")
            else:
                st.warning("No 'date' column detected. Time-based analyses will be unavailable for this sheet.")

            # Drop the existing table if it exists
            cursor = self.conn.cursor()
            cursor.execute(f"DROP TABLE IF EXISTS {self.current_table}")

            # Save the processed dataset into SQLite
            df.to_sql(self.current_table, self.conn, index=False, if_exists='replace')

            # Return schema information for user feedback
            schema_info = cursor.execute(f"PRAGMA table_info({self.current_table})").fetchall()
            return True, self.format_schema_info(schema_info)

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False, str(e)

    def format_schema_info(self, schema_info) -> str:
        """Format schema information for display."""
        return "\n".join([f"- {col[1]} ({col[2]})" for col in schema_info])

    def generate_sql_with_langchain(self, user_query: str) -> str:
        """Generate SQL query dynamically using LangChain and ChatOpenAI."""
        cursor = self.conn.cursor()
        available_columns = [row[1] for row in cursor.execute(f"PRAGMA table_info({self.current_table})").fetchall()]
        logger.info(f"Available columns: {available_columns}")

        # Define ChatLangChain prompt template
        prompt_template = ChatPromptTemplate.from_messages([
            HumanMessage(
                content=(
                    "You are an SQL expert. Based on the user's request, generate a valid SQL SELECT query. "
                    "The table is named 'data_table' and has the following columns: {columns}. "
                    "User's request: {user_query}. "
                    "The query should start with SELECT, include valid SQL syntax, and must return a table of results."
                    " If filtering by a month, ensure you filter correctly by the month name or number."
                )
            )
        ])

        # Generate SQL query using LangChain
        chain = LLMChain(llm=self.llm, prompt=prompt_template)
        sql_query = chain.run({"user_query": user_query, "columns": ", ".join(available_columns)}).strip()

        # Log the generated query
        logger.info(f"Generated SQL query: {sql_query}")

        # Validate the query
        if not sql_query.lower().startswith("select"):
            raise ValueError("Generated query is not a valid SELECT statement.")

        return sql_query

    def analyze(self, user_query: str) -> Tuple[pd.DataFrame, str]:
        """Perform analysis based on user query."""
        try:
            # Generate the SQL query using LangChain
            sql_query = self.generate_sql_with_langchain(user_query)
            
            # Execute the query
            df_result = pd.read_sql_query(sql_query, self.conn)

            # Check for empty results
            if df_result.empty:
                raise ValueError("The query returned no data. Ensure the dataset has relevant information.")

            return df_result, sql_query
        except Exception as e:
            logger.error(f"Analysis error: {e}")

            # Provide fallback query example
            fallback_query = (
                "SELECT strftime('%m', date) AS month, SUM(impressions_total) AS total_impressions "
                "FROM data_table "
                "WHERE strftime('%m', date) IN ('10', '11') "
                "GROUP BY month;"
            )
            logger.info(f"Using fallback query: {fallback_query}")

            try:
                df_result = pd.read_sql_query(fallback_query, self.conn)
                return df_result, fallback_query
            except Exception as fallback_error:
                logger.error(f"Fallback query also failed: {fallback_error}")
                raise Exception("Analysis failed: Both primary and fallback queries failed.")

def main():
    st.set_page_config(page_title="AI Data Analyzer", layout="wide")
    st.title("📊 AI-Powered Data Analyzer")
    st.write("Upload your dataset and analyze it with AI-driven insights!")

    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = DataAnalyzer()

    uploaded_file = st.file_uploader("Upload your data (Excel or CSV)", type=['xlsx', 'xls', 'csv'])
    selected_sheet = None

    if uploaded_file:
        if uploaded_file.name.endswith(('xls', 'xlsx')):
            excel_file = pd.ExcelFile(uploaded_file)
            selected_sheet = st.selectbox("Select a sheet to analyze", excel_file.sheet_names)

        success, schema_info = st.session_state.analyzer.load_data(uploaded_file, selected_sheet)

        if success:
            st.success("Data loaded successfully!")
            with st.expander("View Data Schema"):
                st.code(schema_info)

            user_query = st.text_area(
                "Enter your query about the data",
                placeholder="e.g., 'Compare Q3 and Q2 impressions' or 'Monthly trends for 2024'."
            )
            if st.button("Analyze"):
                try:
                    with st.spinner("Analyzing your data..."):
                        df_result, sql_query = st.session_state.analyzer.analyze(user_query)

                    # Display Results
                    st.write("### Analysis Results")
                    st.dataframe(df_result)

                    # Display SQL Query
                    st.code(sql_query, language='sql')

                    # Visualization
                    if "month" in df_result.columns or "quarter" in df_result.columns:
                        x_axis = "month" if "month" in df_result.columns else "quarter"
                        fig = px.bar(df_result, x=x_axis, y=df_result.columns[1], title="Comparison Results")
                    else:
                        fig = px.bar(df_result, x=df_result.columns[0], y=df_result.columns[1], title="Analysis Results")
                    st.plotly_chart(fig)
                except Exception as e:
                    st.error(str(e))
        else:
            st.error(f"Error loading data: {schema_info}")

    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        - Supports time-based analyses (weekly, monthly, quarterly, yearly).
        - Dynamically generates insights from user queries using LangChain and ChatOpenAI.
        - Provides interactive visualizations for key metrics.
        """)

if __name__ == "__main__":
    main()
