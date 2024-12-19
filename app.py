import streamlit as st
import pandas as pd
import plotly.express as px
import sqlite3
from typing import Tuple
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataAnalyzer:
    def __init__(self):
        if 'db_conn' not in st.session_state:
            st.session_state.db_conn = sqlite3.connect(':memory:', check_same_thread=False)
        self.conn = st.session_state.db_conn
        self.current_table = None

    def load_data(self, file, sheet_name=None) -> Tuple[bool, str]:
        """Load data from uploaded file into SQLite database"""
        try:
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
            else:
                # Load specific sheet or first sheet by default
                if sheet_name is None:
                    excel_file = pd.ExcelFile(file)
                    sheet_name = excel_file.sheet_names[0]
                df = pd.read_excel(file, sheet_name=sheet_name)

            # Clean column names for SQL compatibility
            original_columns = df.columns.tolist()
            df.columns = [
                c.lower()
                .replace(' ', '_')
                .replace('(', '')
                .replace(')', '')
                .replace('-', '_')
                for c in df.columns
            ]
            logger.info(f"Original columns: {original_columns}")
            logger.info(f"Cleaned columns: {df.columns.tolist()}")

            # Process date column
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y', errors='coerce')
                df['year_month'] = df['date'].dt.to_period('M').astype(str)  # e.g., '2023-10'
                df['quarter'] = 'Q' + df['date'].dt.quarter.astype(str) + ' ' + df['date'].dt.year.astype(str)

            # Store table name
            self.current_table = 'data_table'

            # Save to SQLite
            df.to_sql(self.current_table, self.conn, index=False, if_exists='replace')

            # Get schema info
            cursor = self.conn.cursor()
            schema_info = cursor.execute(f"PRAGMA table_info({self.current_table})").fetchall()
            return True, self.format_schema_info(schema_info)
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return False, str(e)

    def format_schema_info(self, schema_info) -> str:
        """Format schema information for display"""
        columns = [f"- {col[1]} ({col[2]})" for col in schema_info]
        return "Table columns:\n" + "\n".join(columns)

    def analyze(self, user_query: str, schema_info: str) -> Tuple[pd.DataFrame, str]:
        """Generate and execute SQL query based on user input"""
        try:
            # Generate SQL query
            sql_query = self.generate_sql(user_query, schema_info)

            # Execute SQL and fetch results
            df_result = pd.read_sql_query(sql_query, self.conn)
            return df_result, sql_query
        except Exception as e:
            logger.error(f"Analysis error: {str(e)}")
            raise Exception(f"Analysis failed: {str(e)}")

    def generate_sql(self, user_query: str, schema_info: str) -> str:
        """Generate SQL query using the user's prompt"""
        prompt = f"""
        Given a database table with the following schema:
        {schema_info}

        Generate a SQL query to answer this question: "{user_query}"

        Requirements:
        - Use only the columns shown above
        - Return results that can be visualized
        - Use proper SQL syntax for SQLite
        - Return only the SQL query, no explanations

        SQL Query:
        """
        # Mocking AI response for the purpose of local execution
        if "monthly" in user_query.lower():
            return f"SELECT year_month AS month, SUM(impressions_total) AS total_impressions FROM {self.current_table} GROUP BY year_month ORDER BY year_month;"
        elif "quarterly" in user_query.lower():
            return f"SELECT quarter AS quarter, SUM(impressions_total) AS total_impressions FROM {self.current_table} GROUP BY quarter ORDER BY quarter;"
        else:
            raise Exception("Unsupported query type. Try using 'monthly' or 'quarterly' in your query.")

def main():
    st.set_page_config(page_title="AI Data Analyzer", layout="wide")
    st.title("🔹 AI-Powered Data Analyzer")
    st.write("Upload your data and analyze it with your own queries!")

    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = DataAnalyzer()

    # File upload
    uploaded_file = st.file_uploader("Upload your data (Excel or CSV)", type=['xlsx', 'xls', 'csv'])
    selected_sheet = None

    if uploaded_file:
        if uploaded_file.name.endswith(('xls', 'xlsx')):
            excel_file = pd.ExcelFile(uploaded_file)
            sheet_names = excel_file.sheet_names
            selected_sheet = st.selectbox("Select a sheet to analyze", sheet_names)

        success, schema_info = st.session_state.analyzer.load_data(uploaded_file, sheet_name=selected_sheet)

        if success:
            st.success("Data loaded successfully!")

            with st.expander("View Data Schema"):
                st.code(schema_info)

            with st.expander("View Data Columns"):
                cursor = st.session_state.analyzer.conn.cursor()
                columns = [row[1] for row in cursor.execute(f"PRAGMA table_info({st.session_state.analyzer.current_table})").fetchall()]
                st.write(columns)

            # Input for user query
            user_query = st.text_area(
                "Enter your query about the data",
                placeholder="e.g., 'Show the monthly trend of impressions' or 'What is the total for each quarter?'",
                height=100
            )

            # Analyze button
            analyze_button = st.button("🔍 Analyze")

            if analyze_button:
                if not user_query:
                    st.warning("Please enter a query before clicking Analyze.")
                else:
                    try:
                        with st.spinner("Analyzing your data..."):
                            df_result, sql_query = st.session_state.analyzer.analyze(user_query, schema_info)

                        # Display results
                        tab1, tab2 = st.tabs(["🔹 Visualization", "🔍 Query"])

                        with tab1:
                            fig = px.bar(df_result, x=df_result.columns[0], y=df_result.columns[1], title="Analysis Results")
                            st.plotly_chart(fig, use_container_width=True)

                        with tab2:
                            st.code(sql_query, language='sql')

                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")

        else:
            st.error(f"Error loading data: {schema_info}")

    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This app uses:
        - SQLite for data analysis
        - Plotly for visualizations
        - Streamlit for the UI

        Upload any Excel or CSV file, and analyze it with natural language queries!
        """)

if __name__ == "__main__":
    main()
