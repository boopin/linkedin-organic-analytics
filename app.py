import streamlit as st
import pandas as pd
import plotly.express as px
import sqlite3
from typing import Tuple
from datetime import datetime
import logging
import openai
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure OpenAI API key
openai.api_key = "your-openai-api-key"  # Replace with your OpenAI API key

class DataAnalyzer:
    def __init__(self):
        if 'db_conn' not in st.session_state:
            st.session_state.db_conn = sqlite3.connect(':memory:', check_same_thread=False)
        else:
            # Reset the connection to avoid stale state
            st.session_state.db_conn.close()
            st.session_state.db_conn = sqlite3.connect(':memory:', check_same_thread=False)
        self.conn = st.session_state.db_conn
        self.current_table = None
        self.llm = OpenAI(temperature=0)  # LangChain LLM setup

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
                .strip()
                .replace(' ', '_')
                .replace('(', '')
                .replace(')', '')
                .replace('-', '_')
                for c in df.columns
            ]
            logger.info(f"Original columns: {original_columns}")
            logger.info(f"Normalized columns: {df.columns.tolist()}")

            # Process date column
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                if df['date'].isnull().all():
                    logger.warning("The 'date' column contains no valid dates. Please check the uploaded file.")
                    st.warning("The 'date' column in your file contains no valid dates. Please upload a file with properly formatted dates.")
                else:
                    df['year_month'] = df['date'].dt.to_period('M').astype(str)  # e.g., '2023-10'
                    df['quarter'] = 'Q' + df['date'].dt.quarter.astype(str) + ' ' + df['date'].dt.year.astype(str)
            else:
                logger.warning("No 'date' column found in the uploaded data.")
                st.warning("No 'date' column found in the uploaded data. Columns 'year_month' and 'quarter' cannot be generated.")

            # Store table name
            self.current_table = 'data_table'

            # Save to SQLite (replace existing table)
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
            # Generate SQL query using LangChain and OpenAI
            sql_query = self.generate_sql_with_langchain(user_query, schema_info)

            # Execute SQL and fetch results
            df_result = pd.read_sql_query(sql_query, self.conn)

            # Verify if the query returned any data
            if df_result.empty:
                raise Exception("The query returned no data. Ensure the dataset has relevant data for the requested period.")

            return df_result, sql_query
        except Exception as e:
            logger.error(f"Analysis error: {str(e)}")
            raise Exception(f"Analysis failed: {str(e)}. Ensure the 'date' column is correctly formatted and the necessary columns exist.")

    def generate_sql_with_langchain(self, user_query: str, schema_info: str) -> str:
        """Generate SQL query using LangChain"""
        # Fetch available columns
        cursor = self.conn.cursor()
        available_columns = [row[1] for row in cursor.execute(f"PRAGMA table_info({self.current_table})").fetchall()]
        logger.info(f"Available columns in the table: {available_columns}")

        # Dynamically map user-specified columns to actual columns in the database
        column_mapping = {}
        for col in available_columns:
            normalized_col = col.replace('_', ' ').lower()
            column_mapping[normalized_col] = col
        logger.info(f"Column mapping: {column_mapping}")

        # Normalize user query by replacing user-friendly terms with actual column names
        normalized_query = user_query.lower()
        for user_col, actual_col in column_mapping.items():
            normalized_query = normalized_query.replace(user_col, actual_col)
        logger.info(f"Normalized user query: {normalized_query}")

        # LangChain prompt for SQL generation
        prompt_template = PromptTemplate(
            input_variables=["user_query", "columns"],
            template=(
                "You are a SQL query generator. Based on the user's request, generate a valid SQL query. "
                "The table is named '{table_name}' and has the following columns: {columns}. "
                "User request: '{user_query}'."
            ),
        )

        # Generate SQL using LangChain
        chain = LLMChain(llm=self.llm, prompt=prompt_template)
        sql_query = chain.run(user_query=normalized_query, columns=", ".join(available_columns), table_name=self.current_table)
        logger.info(f"Generated SQL query: {sql_query}")
        return sql_query

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
        # Reinitialize the analyzer to reset the database connection for new files
        st.session_state.analyzer = DataAnalyzer()

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

                        # Determine if the user wants a table or chart
                        if "table" in user_query.lower():
                            st.dataframe(df_result)  # Display as table
                        else:
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
        - OpenAI and LangChain for natural language to SQL translation
        - Plotly for visualizations
        - Streamlit for the UI

        Upload any Excel or CSV file, and analyze it with natural language queries!
        """)

if __name__ == "__main__":
    main()
