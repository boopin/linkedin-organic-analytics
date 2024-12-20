import streamlit as st
import pandas as pd
import sqlite3
import logging
from typing import Tuple
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataAnalyzer:
    def __init__(self):
        self.conn = sqlite3.connect(':memory:', check_same_thread=False)
        self.current_table = 'data_table'
        self.llm = ChatOpenAI(model="gpt-4")

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean column names, handle missing or invalid data, and create a 'month' column if applicable."""
        df.columns = [c.lower().strip().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_') for c in df.columns]

        # Parse the date column and create a 'month' column if 'date' exists
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['month'] = df['date'].dt.to_period('M').astype(str)
        else:
            logger.warning("No 'date' column found; skipping 'month' column generation.")

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
            processed_df = self.preprocess_data(df)
            logger.info(f"Processed dataset: {processed_df.head()}")

            # Drop existing table
            cursor = self.conn.cursor()
            cursor.execute(f"DROP TABLE IF EXISTS {self.current_table}")

            # Save the processed dataset into SQLite
            processed_df.to_sql(self.current_table, self.conn, index=False, if_exists='replace')
            logger.info(f"Table '{self.current_table}' created successfully in SQLite.")

            # Return schema information for user feedback
            schema_info = cursor.execute(f"PRAGMA table_info({self.current_table})").fetchall()
            return True, "\n".join([f"- {col[1]} ({col[2]})" for col in schema_info])
        except Exception as e:
            logger.error(f"Error loading data into SQLite: {e}")
            return False, str(e)

    def verify_table_existence(self):
        """Check if the table exists in SQLite before querying."""
        cursor = self.conn.cursor()
        tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
        logger.info(f"Existing tables in SQLite: {tables}")
        if not tables or self.current_table not in [table[0] for table in tables]:
            raise ValueError(f"The table '{self.current_table}' does not exist. Please upload a valid dataset.")

    def analyze(self, user_query: str, df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """Perform analysis based on user query."""
        metric = None
        try:
            self.verify_table_existence()
            metric = self.extract_metric_from_query(user_query, df)
            sql_query = self.generate_sql_with_gpt4(user_query, df).replace("your_data_table", self.current_table).replace("[Table]", self.current_table)
            logger.info(f"Generated SQL query: {sql_query}")

            # Execute the generated query
            df_result = pd.read_sql_query(sql_query, self.conn)
            return df_result, sql_query
        except Exception as e:
            logger.error(f"Primary query failed: {e}")
            
            # Handle time-based query failures gracefully
            if "month" in user_query.lower() and "month" not in df.columns:
                raise ValueError("The dataset does not have a 'month' or 'date' column required for time-based queries.")

            # Dynamic fallback query for non-time-based queries
            if not metric:
                numeric_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
                if numeric_columns:
                    metric = numeric_columns[0]
                else:
                    raise Exception("No numeric columns available for fallback query.")

            try:
                fallback_query = f"SELECT post_title, post_link, post_type, {metric} FROM {self.current_table} ORDER BY {metric} DESC LIMIT 5;"
                logger.info(f"Using fallback query: {fallback_query}")
                df_result = pd.read_sql_query(fallback_query, self.conn)
                return df_result, fallback_query
            except Exception as fallback_error:
                raise Exception(f"Analysis failed: {e}\nFallback query error: {fallback_error}")

    def extract_metric_from_query(self, user_query: str, df: pd.DataFrame) -> str:
        """Extract the ranking metric from the user's query."""
        available_columns = [col.lower() for col in df.columns]
        for word in user_query.lower().split():
            if word in available_columns:
                return word
        numeric_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        if numeric_columns:
            return numeric_columns[0]
        raise ValueError("Requested metric not found in the dataset and no numeric columns available.")

    def generate_sql_with_gpt4(self, user_query: str, df: pd.DataFrame) -> str:
        """Generate SQL query dynamically using GPT-4."""
        schema = self.extract_schema_and_sample(df)
        prompt = (
            f"You are an expert in data analysis. Based on the following dataset schema and sample data, "
            f"generate a valid SQL query for a SQLite database that matches the user's intent.\n\n"
            f"{schema}\n\nUser Query: {user_query}\n"
        )
        response = self.llm([HumanMessage(content=prompt)])
        sql_query = response.content.strip()
        if not sql_query.lower().startswith("select"):
            raise ValueError("Generated query is not a valid SELECT statement.")
        return sql_query

    def extract_schema_and_sample(self, df: pd.DataFrame) -> str:
        """Extract schema and sample data for GPT-4."""
        schema = [f"{col} ({dtype})" for col, dtype in zip(df.columns, df.dtypes)]
        schema_description = " | ".join(schema)
        sample_data = df.head(3).to_dict(orient="records")
        return f"Schema: {schema_description}\nSample Data: {sample_data}"

def main():
    st.title("AI Data Analyzer")
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = DataAnalyzer()

    uploaded_file = st.file_uploader("Upload data (CSV or Excel)", type=['csv', 'xlsx'])
    if not uploaded_file:
        st.error("Please upload a dataset before analyzing.")
        st.stop()

    if uploaded_file.name.endswith('xlsx'):
        excel_file = pd.ExcelFile(uploaded_file)
        sheet_names = excel_file.sheet_names
        sheet_name = st.selectbox("Select the sheet to analyze", sheet_names)
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
    else:
        df = pd.read_csv(uploaded_file)

    success, schema_info = st.session_state.analyzer.load_data(df)
    if success:
        st.success("Data loaded successfully!")
        user_query = st.text_area("Enter your query", placeholder="Show me top 5 posts by clicks.")
        if st.button("Analyze"):
            try:
                with st.spinner("Analyzing your data..."):
                    result, query = st.session_state.analyzer.analyze(user_query, df)
                st.write("**Analysis Result:**")
                st.dataframe(result)
                st.write("**SQL Query Used:**")
                st.code(query, language='sql')
            except Exception as e:
                st.error(str(e))
    else:
        st.error(f"Failed to load data: {schema_info}")

if __name__ == "__main__":
    main()
