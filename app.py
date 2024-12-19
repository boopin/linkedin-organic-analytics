import streamlit as st
import pandas as pd
import plotly.express as px
import sqlite3
from typing import Tuple  # Add this import
from datetime import datetime
import logging


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataAnalyzer:
    def __init__(self):
        self.conn = sqlite3.connect(':memory:')
        self.current_table = None
        
    def load_data(self, file, sheet_name=None) -> Tuple[bool, str]:
        """Load data from uploaded file into SQLite database"""
        try:
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
            else:
                # Load specific sheet or default to the first sheet
                if sheet_name is None:
                    excel_file = pd.ExcelFile(file)
                    sheet_name = excel_file.sheet_names[0]
                df = pd.read_excel(file, sheet_name=sheet_name)
            
            # Clean column names for SQL compatibility
            df.columns = [c.lower().replace(' ', '_').replace('(', '_').replace(')', '').replace('-', '_')[:50]
                          for c in df.columns]
            
            # Convert date column to yyyy-mm-dd if it exists
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y', errors='coerce')
                df['year_month'] = df['date'].dt.to_period('M').astype(str)  # e.g., '2023-10'
                df['quarter'] = 'Q' + df['date'].dt.quarter.astype(str) + ' ' + df['date'].dt.year.astype(str)

            # Store table name
            self.current_table = 'data_table'
            
            # Save to SQLite
            df.to_sql(self.current_table, self.conn, index=False, if_exists='replace')
            
            # Get schema information
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

    def analyze(self, analysis_type: str) -> Tuple[pd.DataFrame, str]:
        """Perform analysis based on analysis type (Monthly/Quarterly)"""
        try:
            # Generate SQL based on analysis type
            if analysis_type == "Monthly":
                sql_query = f"""
                SELECT 
                    year_month AS month, 
                    SUM(impressions_total) AS total_impressions
                FROM 
                    {self.current_table}
                GROUP BY 
                    year_month
                ORDER BY 
                    year_month;
                """
            elif analysis_type == "Quarterly":
                sql_query = f"""
                SELECT 
                    quarter AS quarter, 
                    SUM(impressions_total) AS total_impressions
                FROM 
                    {self.current_table}
                GROUP BY 
                    quarter
                ORDER BY 
                    quarter;
                """
            else:
                raise ValueError("Invalid analysis type")

            # Execute SQL and fetch results
            df_result = pd.read_sql_query(sql_query, self.conn)
            return df_result, sql_query
        except Exception as e:
            logger.error(f"Analysis error: {str(e)}")
            raise Exception(f"Analysis failed: {str(e)}")

def main():
    st.set_page_config(page_title="AI Data Analyzer", layout="wide")
    st.title("📊 AI-Powered Data Analyzer")
    st.write("Upload your data and analyze it with monthly or quarterly trends!")

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

            # Select analysis type
            analysis_type = st.selectbox("Select Analysis Type", ["Monthly", "Quarterly"])

            # Analyze button
            if st.button("🔍 Analyze"):
                try:
                    with st.spinner("Analyzing your data..."):
                        df_result, sql_query = st.session_state.analyzer.analyze(analysis_type)

                    # Display results
                    tab1, tab2 = st.tabs(["📈 Visualization", "🔍 Query"])

                    with tab1:
                        fig = px.bar(df_result, 
                                     x='month' if analysis_type == "Monthly" else 'quarter', 
                                     y='total_impressions', 
                                     title=f"{analysis_type} Trend of Impressions")
                        st.plotly_chart(fig, use_container_width=True)

                    with tab2:
                        st.code(sql_query, language='sql')
                except Exception as e:
                    st.error(str(e))
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

        Upload any Excel or CSV file, and analyze monthly or quarterly trends easily!
        """)

if __name__ == "__main__":
    main()
