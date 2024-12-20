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

        # Ensure 'clicks' column exists and is numeric
        if 'clicks' in df.columns:
            df['clicks'] = pd.to_numeric(df['clicks'], errors='coerce')
            df = df.dropna(subset=['clicks'])  # Drop rows with missing clicks
        else:
            raise ValueError("The dataset does not contain a 'clicks' column.")

        return df

    def load_data(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Load preprocessed DataFrame into SQLite database."""
        try:
            # Preprocess the dataset
            processed_df = self.preprocess_data(df)

            # Drop existing table
            cursor = self.conn.cursor()
            cursor.execute(f"DROP TABLE IF EXISTS {self.current_table}")

            # Save the processed dataset into SQLite
            processed_df.to_sql(self.current_table, self.conn, index=False, if_exists='replace')

            # Return schema information for user feedback
            schema_info = cursor.execute(f"PRAGMA table_info({self.current_table})").fetchall()
            return True, self.format_schema_info(schema_info)

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False, str(e)

    def format_schema_info(self, schema_info) -> str:
        """Format schema information for display."""
        return "\n".join([f"- {col[1]} ({col[2]})" for col in schema_info])

    def extract_schema_and_sample(self, df: pd.DataFrame) -> str:
        """Extract schema (column names and types) and sample data from the DataFrame."""
        schema = [f"{col} ({dtype})" for col, dtype in zip(df.columns, df.dtypes)]
        schema_description = " | ".join(schema)

        # Extract sample data
        sample_data = df.head(3).to_dict(orient="records")

        # Format schema and sample data for GPT-4
        schema_text = f"Schema: {schema_description}\nSample Data: {sample_data}"
        return schema_text

    def build_prompt(self, schema: str, user_query: str) -> str:
        """Build a context-rich prompt for GPT-4."""
        prompt = (
            f"You are an expert in data analysis. Based on the following dataset schema and sample data, "
            f"generate a valid SQL query that matches the user's intent.\n\n"
            f"{schema}\n\n"
            f"User Query: {user_query}\n\n"
            f"The query should return the top 5 posts ranked by the 'clicks' column, and include the following columns: "
            f"'post_title', 'post_link', 'post_type', and 'clicks'. Ensure the query is written for a SQLite database."
        )
        return prompt

    def generate_sql_with_gpt4(self, user_query: str, df: pd.DataFrame) -> str:
        """Generate SQL query dynamically using GPT-4."""
        schema = self.extract_schema_and_sample(df)
        prompt = self.build_prompt(schema, user_query)

        # Use GPT-4 to generate the query
        response = self.llm([HumanMessage(content=prompt)])
        sql_query = response.content.strip()

        # Log the generated query
        logger.info(f"Generated SQL query: {sql_query}")

        # Validate the query
        if not sql_query.lower().startswith("select"):
            raise ValueError("Generated query is not a valid SELECT statement.")

        return sql_query

    def analyze(self, user_query: str, df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """Perform analysis based on user query."""
        try:
            # Generate the SQL query using GPT-4
            sql_query = self.generate_sql_with_gpt4(user_query, df)

            # Log and execute the query
            logger.info(f"Executing SQL query: {sql_query}")
            df_result = pd.read_sql_query(sql_query, self.conn)

            # Check for empty results
            if df_result.empty:
                raise ValueError("The query returned no data. Ensure the dataset has valid entries.")

            return df_result, sql_query
        except Exception as e:
            logger.error(f"Analysis error: {e}")

            # Provide a fallback query for top 5 posts
            fallback_query = (
                "SELECT post_title, post_link, post_type, clicks "
                "FROM data_table "
                "ORDER BY clicks DESC LIMIT 5;"
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
                    placeholder="e.g., 'Show me top 5 posts by clicks.'"
                )
                if st.button("Analyze"):
                    try:
                        with st.spinner("Analyzing your data..."):
                            df_result, sql_query = st.session_state.analyzer.analyze(user_query, df)

                        # Display Results
                        st.write("### Analysis Results")
                        st.dataframe(df_result)

                        # Display SQL Query
                        st.code(sql_query, language='sql')
                    except Exception as e:
                        st.error(str(e))
            else:
                st.error("Error loading dataset into SQLite.")
        except Exception as e:
            st.error(f"Error processing uploaded file: {str(e)}")

if __name__ == "__main__":
    main()
