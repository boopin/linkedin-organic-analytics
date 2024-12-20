import streamlit as st
import pandas as pd
import sqlite3
import logging
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
        """Clean column names and handle missing or invalid data."""
        df.columns = [c.lower().strip().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_') for c in df.columns]
        df = df.dropna(how='all', axis=1)
        for col in df.select_dtypes(include=['float64', 'int64']).columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).fillna('')
        if df.empty:
            raise ValueError("The dataset is empty after preprocessing.")
        return df

    def load_data(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Load preprocessed DataFrame into SQLite database."""
        try:
            processed_df = self.preprocess_data(df)
            logger.info(f"Processed dataset: {processed_df.head()}")
            cursor = self.conn.cursor()
            cursor.execute(f"DROP TABLE IF EXISTS {self.current_table}")
            processed_df.to_sql(self.current_table, self.conn, index=False, if_exists='replace')
            schema_info = cursor.execute(f"PRAGMA table_info({self.current_table})").fetchall()
            return True, "\n".join([f"- {col[1]} ({col[2]})" for col in schema_info])
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False, str(e)

    def analyze(self, user_query: str, df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """Perform analysis based on user query."""
        try:
            self.verify_table_existence()
            metric = self.extract_metric_from_query(user_query, df)
            sql_query = self.generate_sql_with_gpt4(user_query, df).replace("table_name", self.current_table)
            df_result = pd.read_sql_query(sql_query, self.conn)
            return df_result, sql_query
        except Exception as e:
            fallback_query = f"SELECT post_title, post_link, post_type, {metric} FROM {self.current_table} ORDER BY {metric} DESC LIMIT 5;"
            try:
                df_result = pd.read_sql_query(fallback_query, self.conn)
                return df_result, fallback_query
            except Exception as fallback_error:
                raise Exception(f"Analysis failed: {e}, fallback query error: {fallback_error}")

    def verify_table_existence(self):
        """Check if the table exists in SQLite."""
        cursor = self.conn.cursor()
        tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
        if self.current_table not in [table[0] for table in tables]:
            raise ValueError(f"Table '{self.current_table}' does not exist.")

def main():
    st.title("AI Data Analyzer")
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = DataAnalyzer()

    uploaded_file = st.file_uploader("Upload data (CSV or Excel)", type=['csv', 'xlsx'])
    if uploaded_file:
        df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('xlsx') else pd.read_csv(uploaded_file)
        success, schema_info = st.session_state.analyzer.load_data(df)
        if success:
            st.success("Data loaded successfully!")
            user_query = st.text_area("Enter your query", placeholder="Show me top 5 posts by comments.")
            if st.button("Analyze"):
                try:
                    result, query = st.session_state.analyzer.analyze(user_query, df)
                    st.write(result)
                except Exception as e:
                    st.error(str(e))

if __name__ == "__main__":
    main()
