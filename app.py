import streamlit as st
import pandas as pd
import sqlite3
from typing import Tuple
import logging
from datetime import datetime
import plotly.express as px

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataAnalyzer:
    def __init__(self):
        self.conn = sqlite3.connect(':memory:', check_same_thread=False)
        self.current_table = None

    def load_data(self, file, sheet_name=None) -> Tuple[bool, str]:
        """Load data from the uploaded file into SQLite database and compute derived columns."""
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

            # Check and process the date column
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')  # Parse dates
                if not df['date'].isnull().all():
                    # Compute derived time-based fields
                    df['week'] = df['date'].dt.to_period('W-SUN').astype(str)
                    df['year_month'] = df['date'].dt.to_period('M').astype(str)
                    df['quarter'] = 'Q' + df['date'].dt.quarter.astype(str) + ' ' + df['date'].dt.year.astype(str)
                    df['year'] = df['date'].dt.year.astype(str)
                else:
                    raise ValueError("The 'date' column contains no valid dates. Please check the dataset.")
            else:
                raise ValueError("The dataset is missing a 'date' column.")

            # Save the processed dataset into SQLite
            self.current_table = 'data_table'
            df.to_sql(self.current_table, self.conn, index=False, if_exists='replace')

            # Return schema information for user feedback
            cursor = self.conn.cursor()
            schema_info = cursor.execute(f"PRAGMA table_info({self.current_table})").fetchall()
            return True, self.format_schema_info(schema_info)

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False, str(e)

    def format_schema_info(self, schema_info) -> str:
        """Format schema information for display."""
        return "\n".join([f"- {col[1]} ({col[2]})" for col in schema_info])

    def analyze(self, user_query: str) -> Tuple[pd.DataFrame, str]:
        """Perform analysis based on user query."""
        try:
            cursor = self.conn.cursor()
            # Validate quarter data existence
            available_quarters = [row[0] for row in cursor.execute("SELECT DISTINCT quarter FROM data_table").fetchall()]
            if not set(['Q2 2024', 'Q3 2024']).issubset(set(available_quarters)):
                raise ValueError("No data available for Q2 2024 or Q3 2024. Please check your dataset.")

            # Generate SQL query for quarterly comparison
            sql_query = """
                SELECT quarter, SUM(impressions_total) AS total_impressions
                FROM data_table
                WHERE quarter IN ('Q2 2024', 'Q3 2024')
                GROUP BY quarter;
            """
            df_result = pd.read_sql_query(sql_query, self.conn)

            if df_result.empty:
                raise ValueError("The query returned no data. Ensure the dataset has relevant information.")

            return df_result, sql_query
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            raise Exception(f"Analysis failed: {str(e)}")

def main():
    st.set_page_config(page_title="AI Data Analyzer", layout="wide")
    st.title("📊 AI-Powered Data Analyzer")
    st.write("Upload your dataset and analyze it with time-based insights!")

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
                    st.write("### Analysis Results")
                    st.dataframe(df_result)
                    st.code(sql_query, language='sql')

                    # Visualization
                    fig = px.bar(df_result, x='quarter', y='total_impressions', title="Quarterly Comparison")
                    st.plotly_chart(fig)
                except Exception as e:
                    st.error(str(e))
        else:
            st.error(f"Error loading data: {schema_info}")

    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        - Supports time-based analyses (weekly, monthly, quarterly, yearly).
        - Dynamically generates insights from user queries.
        """)

if __name__ == "__main__":
    main()
