import streamlit as st
import pandas as pd
import sqlite3
from typing import Tuple
import logging
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
        self.llm = ChatOpenAI(model="gpt-3.5-turbo")

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

            # Drop existing table
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
                    "You are an SQL expert. Generate a valid SQL SELECT query based on the user's request. "
                    "The table is named 'data_table' and has the following columns: {columns}. "
                    "User's request: {user_query}. "
                    "The query must start with SELECT, use valid SQL syntax, and return the desired table. "
                    "If ranking is required (e.g., top 5), use the ORDER BY clause."
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

            # Provide fallback query for top posts
            fallback_query = (
                "SELECT post_title, posted_by, post_type, post_link, likes "
                "FROM data_table "
                "ORDER BY likes DESC LIMIT 5;"
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
                placeholder="e.g., 'Generate a table showcasing top 5 posts by likes.'"
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
                except Exception as e:
                    st.error(str(e))
        else:
            st.error(f"Error loading data: {schema_info}")

if __name__ == "__main__":
    main()
