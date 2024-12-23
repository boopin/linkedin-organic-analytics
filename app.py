import streamlit as st
import pandas as pd
import sqlite3
import logging
import plotly.express as px
from typing import Tuple

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
        try:
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
            else:
                excel_file = pd.ExcelFile(file)
                if sheet_name is None:
                    sheet_name = excel_file.sheet_names[0]
                df = pd.read_excel(file, sheet_name=sheet_name)

            df.columns = [
                c.lower().strip().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
                for c in df.columns
            ]

            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y', errors='coerce')
                if not df['date'].isnull().all():
                    df['week'] = df['date'].dt.to_period('W-SUN').astype(str)
                    df['year_month'] = df['date'].dt.to_period('M').astype(str)
                    df['quarter'] = 'Q' + df['date'].dt.quarter.astype(str) + ' ' + df['date'].dt.year.astype(str)
                    df['year'] = df['date'].dt.year.astype(str)
                else:
                    raise ValueError("Invalid dates in 'date' column.")
            else:
                raise ValueError("Dataset missing 'date' column.")

            self.current_table = 'data_table'
            df.to_sql(self.current_table, self.conn, index=False, if_exists='replace')

            cursor = self.conn.cursor()
            schema_info = cursor.execute(f"PRAGMA table_info({self.current_table})").fetchall()
            return True, "\n".join([f"- {col[1]} ({col[2]})" for col in schema_info])

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False, str(e)

    def analyze(self, sql_query: str) -> Tuple[pd.DataFrame, str]:
        try:
            df_result = pd.read_sql_query(sql_query, self.conn)
            if df_result.empty:
                raise ValueError("Query returned no data.")
            return df_result, sql_query
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            raise Exception(f"Analysis failed: {str(e)}")

def main():
    st.title("Data Analysis Tool")
    analyzer = DataAnalyzer()

    uploaded_file = st.file_uploader("Upload a dataset (Excel or CSV)", type=["csv", "xlsx"])
    if uploaded_file:
        sheet_name = None
        if uploaded_file.name.endswith('.xlsx'):
            excel_file = pd.ExcelFile(uploaded_file)
            sheet_name = st.selectbox("Select sheet", excel_file.sheet_names)

        success, schema_info = analyzer.load_data(uploaded_file, sheet_name)
        if success:
            st.success("Data loaded successfully!")
            st.text("Schema:")
            st.text(schema_info)

            user_query = st.text_area("Enter your SQL query")
            if st.button("Run Query"):
                try:
                    result, query = analyzer.analyze(user_query)
                    st.dataframe(result)
                    st.code(query, language="sql")
                    if len(result.columns) >= 2:
                        st.plotly_chart(px.bar(result, x=result.columns[0], y=result.columns[1], title="Query Results"))
                except Exception as e:
                    st.error(e)
        else:
            st.error("Failed to load data.")

if __name__ == "__main__":
    main()
