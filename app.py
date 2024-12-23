import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
from langchain_openai.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
import logging
import difflib
import re

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

EXAMPLE_QUERIES = [
    "Show me the top 5 dates with the highest total impressions.",
    "Show me the posts with the most clicks.",
    "What is the average engagement rate of all posts?",
    "Generate a bar graph of clicks grouped by post type."
]

class PreprocessingPipeline:
    @staticmethod
    def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [col.lower().strip().replace(" ", "_").replace("(", "").replace(")", "") for col in df.columns]
        return df

    @staticmethod
    def handle_missing_dates(df: pd.DataFrame) -> pd.DataFrame:
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df

    @staticmethod
    def fix_arrow_incompatibility(df: pd.DataFrame) -> pd.DataFrame:
        for col in df.select_dtypes(include=["datetime", "object"]).columns:
            df[col] = df[col].astype("string", errors="ignore")
        return df

    @staticmethod
    def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
        df = PreprocessingPipeline.clean_column_names(df)
        df = PreprocessingPipeline.handle_missing_dates(df)
        df = PreprocessingPipeline.fix_arrow_incompatibility(df)
        return df

class DynamicQueryParser:
    @staticmethod
    def singularize_term(term: str) -> str:
        if term.endswith("s"):
            return term[:-1]
        return term

    @staticmethod
    def preprocess_query(user_query: str) -> str:
        filler_words = {"show", "me", "the", "top", "with", "highest", "most", "posts", "and", "or", "by", "in", "a", "table", "along"}
        query_terms = [word for word in user_query.lower().split() if word not in filler_words and not word.isdigit()]
        return " ".join(query_terms)

    @staticmethod
    def map_composite_terms(mapped_query: str) -> str:
        composite_mappings = {
            "total impressions": "impressions_total",
            "most clicks": "clicks_total"
        }
        for composite, column in composite_mappings.items():
            if composite in mapped_query:
                mapped_query = mapped_query.replace(composite, column)
        return mapped_query

    @staticmethod
    def map_query_to_columns(user_query: str, df: pd.DataFrame) -> str:
        preprocessed_query = DynamicQueryParser.preprocess_query(user_query)
        preprocessed_query = DynamicQueryParser.map_composite_terms(preprocessed_query)
        available_columns = [col.lower() for col in df.columns]
        mapped_query = preprocessed_query

        for term in preprocessed_query.split():
            singular_term = DynamicQueryParser.singularize_term(term)
            match = difflib.get_close_matches(singular_term, available_columns, n=1, cutoff=0.6)
            if match:
                actual_column = df.columns[available_columns.index(match[0])]
                mapped_query = mapped_query.replace(term, actual_column)

        return mapped_query

    @staticmethod
    def validate_query(mapped_query: str, df: pd.DataFrame):
        referenced_columns = re.findall(r"[a-zA-Z_]+", mapped_query)  # Ignore numbers in column validation
        missing_columns = [col for col in referenced_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(
                f"The following columns referenced in the query do not exist in the dataset: {', '.join(missing_columns)}. "
                f"Available columns: {list(df.columns)}"
            )

class SQLQueryAgent:
    def __init__(self, llm):
        self.llm = llm

    def generate_sql_with_function_calling(self, user_query: str, schema: str, df: pd.DataFrame) -> str:
        """Generate SQL query using OpenAI function calling."""
        available_columns = [col for col in df.columns]

        def sql_generation_function(query: str, columns: list):
            """Structured function for SQL query generation."""
            if any(term not in columns for term in query.split()):
                raise ValueError("Query references non-existent columns.")
            # Example implementation (this would be enhanced with LLM)
            return f"SELECT * FROM data_table WHERE {query}"

        function_call_prompt = {
            "name": "sql_generation",
            "description": "Generate an SQL query for the given schema and user query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "columns": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["query", "columns"]
            }
        }

        response = self.llm.invoke(
            [HumanMessage(content=user_query)],
            functions=[function_call_prompt],
            function_call="sql_generation"
        )

        sql_query = response.get("content", {}).get("query", "")
        return sql_query

    def generate_sql(self, user_query: str, schema: str, df: pd.DataFrame) -> str:
        mapped_query = DynamicQueryParser.map_query_to_columns(user_query, df)
        DynamicQueryParser.validate_query(mapped_query, df)

        return self.generate_sql_with_function_calling(mapped_query, schema, df)

class DataAnalyzer:
    def __init__(self):
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self.llm = ChatOpenAI(model="gpt-4")
        self.sql_agent = SQLQueryAgent(self.llm)

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        return PreprocessingPipeline.preprocess_data(df)

    def load_data(self, df: pd.DataFrame):
        try:
            df = self.preprocess_data(df)
            df.to_sql("data_table", self.conn, index=False, if_exists="replace")
            schema = ", ".join([f"{col} ({dtype})" for col, dtype in zip(df.columns, df.dtypes)])
            return True, schema
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False, str(e)

    def analyze(self, user_query: str, schema: str, df: pd.DataFrame):
        try:
            mapped_query = DynamicQueryParser.map_query_to_columns(user_query, df)
            sql_query = self.sql_agent.generate_sql(mapped_query, schema, df)
            result = pd.read_sql_query(sql_query, self.conn)
            return result, sql_query
        except ValueError as ve:
            raise ValueError(f"Validation Error: {ve}\nProcessed Query: {mapped_query}\nOriginal Query: {user_query}")
        except Exception as e:
            raise Exception(f"Analysis Error: {str(e)}\nProcessed Query: {mapped_query}\nOriginal Query: {user_query}")

def main():
    st.title("AI Reports Analyzer")
    analyzer = DataAnalyzer()

    uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
    if not uploaded_file:
        st.info("Please upload a file.")
        return

    try:
        if uploaded_file.name.endswith(".xlsx"):
            try:
                import openpyxl
            except ImportError:
                st.error("Missing dependency: openpyxl. Please install it using 'pip install openpyxl'.")
                return

            excel_file = pd.ExcelFile(uploaded_file)
            sheet_name = st.selectbox("Select Sheet", excel_file.sheet_names)
            df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
        else:
            df = pd.read_csv(uploaded_file)

        success, schema = analyzer.load_data(df)
        if not success:
            st.error(f"Failed to load data: {schema}")
            return

        st.success("Data loaded successfully!")

        st.write("### Dataset Schema")
        schema_columns = pd.DataFrame({"Column": df.columns, "Data Type": df.dtypes})
        st.dataframe(schema_columns)

        st.write("### Example Queries")
        for query in EXAMPLE_QUERIES:
            st.markdown(f"- {query}")

        user_query = st.text_input("Enter your query")
        if st.button("Analyze"):
            try:
                result, sql_query = analyzer.analyze(user_query, schema, df)
                st.write("**Results (Table Format):**")
                st.dataframe(result)
                st.write("**SQL Query Used:**")
                st.code(sql_query, language="sql")
            except ValueError as ve:
                st.error(f"Validation Error: {ve}")
            except Exception as e:
                st.error(f"Error: {e}")

    except Exception as e:
        st.error(f"Failed to process file: {e}")

if __name__ == "__main__":
    main()
