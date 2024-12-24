# App Version: 1.0.5
import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from crewai import Agent, Workflow, Task
import logging
import difflib
import re
from datetime import datetime

# Configure logging
logging.basicConfig(filename="crew.log", level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

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
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].astype("string", errors="ignore")
        return df

    @staticmethod
    def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
        df = PreprocessingPipeline.clean_column_names(df)
        df = PreprocessingPipeline.handle_missing_dates(df)
        df = PreprocessingPipeline.fix_arrow_incompatibility(df)
        return df

class DynamicQueryParser:
    NUMBER_MAPPING = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
    }

    @staticmethod
    def convert_words_to_numbers(query: str) -> str:
        words = query.split()
        converted_words = [
            str(DynamicQueryParser.NUMBER_MAPPING[word]) if word in DynamicQueryParser.NUMBER_MAPPING else word
            for word in words
        ]
        return " ".join(converted_words)

    @staticmethod
    def preprocess_query(user_query: str) -> str:
        filler_words = {"show", "me", "the", "top", "with", "highest", "most", "posts", "and", "or", "by", "in", "a", "table", "along"}
        query_terms = [word for word in user_query.lower().split() if word not in filler_words]
        processed_query = DynamicQueryParser.convert_words_to_numbers(" ".join(query_terms))
        return processed_query

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
                actual_column = df.columns[available_columns.index(match[0])].strip()
                mapped_query = mapped_query.replace(term, actual_column)

        return mapped_query

    @staticmethod
    def validate_query(mapped_query: str, df: pd.DataFrame):
        referenced_columns = re.findall(r"[a-zA-Z_]+", mapped_query)
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
        available_columns = [col for col in df.columns]

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

        prompt_content = (
            f"Schema: {schema}\n"
            "Examples:\n"
            "1. User Query: Show me the top 5 dates with the highest total impressions.\n"
            "   SQL Query: SELECT date, impressions_total FROM data_table ORDER BY impressions_total DESC LIMIT 5;\n"
            "2. User Query: Show me the posts with the most clicks.\n"
            "   SQL Query: SELECT * FROM data_table ORDER BY clicks DESC LIMIT 5;\n"
            f"User Query: {user_query}\n"
            "Generate the SQL query based on the examples above."
        )

        response = self.llm.invoke(
            [HumanMessage(content=prompt_content)],
            functions=[function_call_prompt],
            function_call="auto"
        )

        logger.warning(f"Raw LLM Response: {response}")

        if isinstance(response, AIMessage):
            query_data = response.additional_kwargs.get("function_call", {}).get("query", "")
            if not query_data:
                logger.error("No valid SQL query returned by the LLM. Returning default SQL.")
                return "SELECT * FROM data_table LIMIT 10"
            return query_data.strip()

        raise Exception("Unexpected response type received from OpenAI.")

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

def extract_data_task(database_connection, query):
    return extract_data(query, database_connection)

def analyze_data_task(data):
    return analyze_data(data)

def main():
    st.title("AI Reports Analyzer with Crew.ai")

    uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
    if not uploaded_file:
        st.info("Please upload a file.")
        return

    try:
        if uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)

        # Load data into SQLite
        conn = sqlite3.connect(":memory:")
        df.to_sql("uploaded_data", conn, index=False, if_exists="replace")

        # Define agents
        sql_dev = Agent(name="sql_dev", role="Handles SQL queries for data extraction.")
        data_analyst = Agent(name="data_analyst", role="Analyzes extracted data.")

        # Create workflow
        workflow = Workflow(
            name="AI Data Analysis Workflow",
            tasks=[
                Task(name="extract_data", func=extract_data_task, inputs={"query": "SELECT * FROM uploaded_data", "database_connection": conn}),
                Task(name="analyze_data", func=analyze_data_task, inputs={"data": "{extract_data}"})
            ],
            agents=[sql_dev, data_analyst],
            process="sequential",
            verbose=2,
            memory=False
        )

        # Execute the workflow
        results = workflow.run()
        st.write("### Workflow Results")
        for task, result in results.items():
            st.write(f"**{task}:**", result)

    except Exception as e:
        st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
