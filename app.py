import streamlit as st
import pandas as pd
import sqlite3
from typing import Tuple
import logging
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import plotly.express as px

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataAnalyzer:
    def __init__(self):
        if 'db_conn' not in st.session_state:
            st.session_state.db_conn = sqlite3.connect(':memory:', check_same_thread=False)
        self.conn = st.session_state.db_conn
        self.current_table = None
        self.llm = OpenAI(model="text-davinci-003", temperature=0)

    def load_data(self, file, sheet_name=None) -> Tuple[bool, str]:
        """Load data from uploaded file into SQLite database and compute derived columns."""
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
                # Parse date with MM/DD/YYYY format
                df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y', errors='coerce')
                if not df['date'].isnull().all():
                    # Compute derived time-based fields
                    df['week'] = df['date'].dt.to_period('W-SUN').astype(str)
                    df['year_month'] = df['date'].dt.to_period('M').astype(str)
                    df['quarter'] = 'Q' + df['date'].dt.quarter.astype(str) + ' ' + df['date'].dt.year.astype(str)
                    df['year'] = df['date'].dt.year.astype(str)
                else:
                    raise ValueError("The 'date' column contains no valid dates. Please check the dataset format.")
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
        columns = [f"- {col[1]} ({col[2]})" for col in schema_info]
        return "\n".join(columns)

    def generate_sql_with_langchain(self, user_query: str) -> str:
        """Generate SQL query using LangChain and OpenAI."""
        try:
            cursor = self.conn.cursor()
            available_columns = [row[1] for row in cursor.execute(f"PRAGMA table_info({self.current_table})").fetchall()]
            logger.info(f"Available columns: {available_columns}")

            # Create a LangChain prompt template
            prompt_template = PromptTemplate(
                input_variables=["user_query", "columns"],
                template=(
                    "You are a SQL query generator. Based on the user's request, generate a valid SQL query. "
                    "The table has the following columns: {columns}. "
                    "User request: {user_query}."
                )
            )

            # Run the LangChain model
            chain = LLMChain(llm=self.llm, prompt=prompt_template)
            sql_query = chain.run({
                "user_query": user_query,
                "columns": ", ".join(available_columns),
            })

            logger.info(f"Generated SQL query: {sql_query}")
            return sql_query.strip()
        except Exception as e:
            logger.error(f"Error generating SQL: {e}")
            raise Exception("Failed to generate SQL query.")

    def analyze(self, user_query: str) -> Tuple[pd.DataFrame, str]:
        """Generate and execute SQL query based on user input."""
        try:
            # Generate SQL query
            sql_query = self.generate_sql_with_langchain(user_query)
            logger.info(f"Executing query: {sql_query}")

            # Execute query
            df_result = pd.read_sql_query(sql_query, self.conn)

            if df_result.empty:
                raise ValueError("The query returned no data. Ensure the dataset has relevant information.")

            return df_result, sql_query
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            raise Exception(f"Analysis failed: {str(e)}")

def main():
    st.title("AI Data Analyzer")
    analyzer = DataAnalyzer()

    # File upload
    uploaded_file = st.file_uploader("Upload your dataset (Excel or CSV)", type=["csv", "xlsx"])

    if uploaded_file:
        sheet_name = None
        if uploaded_file.name.endswith('.xlsx'):
            excel_file = pd.ExcelFile(uploaded_file)
            sheet_name = st.selectbox("Select a sheet to analyze", excel_file.sheet_names)

        success, schema_info = analyzer.load_data(uploaded_file, sheet_name=sheet_name)

        if success:
            st.success("Data loaded successfully!")
            st.text("Schema:")
            st.text(schema_info)

            # User query input
            user_query = st.text_input("Enter your query", "Show quarterly comparison of Q3 vs Q2 for total impressions")

            if st.button("Run Analysis"):
                try:
                    result, query = analyzer.analyze(user_query)
                    st.dataframe(result)
                    st.code(query, language='sql')

                    # Plotly visualization
                    st.plotly_chart(
                        px.bar(
                            result, x=result.columns[0], y=result.columns[1],
                            title="Quarterly Comparison of Total Impressions"
                        ),
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(e)
        else:
            st.error("Failed to load data. Please check the file and try again.")

if __name__ == "__main__":
    main()
