import streamlit as st
import pandas as pd
import sqlite3
from typing import Tuple
import logging
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataAnalyzer:
    def __init__(self):
        self.conn = sqlite3.connect(':memory:', check_same_thread=False)
        self.current_table = 'data_table'
        self.llm = ChatOpenAI(model="gpt-4")  # Use GPT-4 for dynamic query generation

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean column names and handle missing or invalid data."""
        # Clean column names
        df.columns = [
            c.lower().strip().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
            for c in df.columns
        ]

        # Drop entirely empty columns
        df = df.dropna(how='all', axis=1)

        # Fill missing numeric values with 0 and ensure consistent types
        for col in df.select_dtypes(include=['float64', 'int64']).columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Ensure all column names and data types are compatible with SQLite
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).fillna('')

        if df.empty:
            raise ValueError("The dataset is empty after preprocessing. Ensure the sheet contains valid data.")

        return df

    def load_data(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Load preprocessed DataFrame into SQLite database."""
        try:
            # Preprocess the dataset
            processed_df = self.preprocess_data(df)

            # Log cleaned column names and data types
            logger.info(f"Cleaned columns: {processed_df.columns}")
            logger.info(f"Column data types:\n{processed_df.dtypes}")

            # Drop existing table
            cursor = self.conn.cursor()
            cursor.execute(f"DROP TABLE IF EXISTS {self.current_table}")

            # Save the processed dataset into SQLite
            processed_df.to_sql(self.current_table, self.conn, index=False, if_exists='replace')

            # Return schema information for user feedback
            schema_info = cursor.execute(f"PRAGMA table_info({self.current_table})").fetchall()
            return True, self.format_schema_info(schema_info)

        except Exception as e:
            logger.error(f"Error loading data into SQLite: {e}")
            return False, str(e)

    def format_schema_info(self, schema_info) -> str:
        """Format schema information for display."""
        return "\n".join([f"- {col[1]} ({col[2]})" for col in schema_info])

    # Other methods for building prompts, querying GPT-4, and analysis remain unchanged...

def main():
    st.set_page_config(page_title="AI Data Analyzer", layout="wide")
    st.title("📊 AI-Powered Data Analyzer")

    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = DataAnalyzer()

    uploaded_file = st.file_uploader("Upload your data (Excel or CSV)", type=['xlsx', 'xls', 'csv'])
    selected_sheet = None

    if uploaded_file:
        try:
            # Load dataset into DataFrame
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                excel_file = pd.ExcelFile(uploaded_file)
                selected_sheet = st.selectbox("Select a sheet to analyze", excel_file.sheet_names)
                df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)

            # Load data into SQLite and analyze
            success, schema_info = st.session_state.analyzer.load_data(df)
            if success:
                st.success("Data loaded successfully!")
                with st.expander("View Data Schema"):
                    st.code(schema_info)

                user_query = st.text_area(
                    "Enter your query about the data",
                    placeholder="e.g., 'Show me total impressions by date.'"
                )
                if st.button("Analyze"):
                    try:
                        with st.spinner("Analyzing your data..."):
                            # Analysis logic here...
                            pass
                    except Exception as e:
                        st.error(str(e))
            else:
                st.error("Error loading dataset into SQLite.")
        except Exception as e:
            st.error(f"Error processing uploaded file: {str(e)}")

if __name__ == "__main__":
    main()
