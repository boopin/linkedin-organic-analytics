import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Configure SQLite database
class DataAnalyzer:
    def __init__(self):
        if 'db_conn' not in st.session_state:
            st.session_state.db_conn = sqlite3.connect(':memory:', check_same_thread=False)
        self.conn = st.session_state.db_conn
        self.current_table = None
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    def load_data(self, file, sheet_name=None):
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
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                if not df['date'].isnull().all():
                    df['week'] = df['date'].dt.to_period('W').astype(str)
                    df['year_month'] = df['date'].dt.to_period('M').astype(str)
                    df['quarter'] = 'Q' + df['date'].dt.quarter.astype(str) + ' ' + df['date'].dt.year.astype(str)
                    df['year'] = df['date'].dt.year.astype(str)
                else:
                    raise ValueError("Invalid or missing dates in the 'date' column.")
            else:
                raise ValueError("The dataset is missing a 'date' column.")

            self.current_table = 'data_table'
            df.to_sql(self.current_table, self.conn, index=False, if_exists='replace')

            cursor = self.conn.cursor()
            schema_info = cursor.execute(f"PRAGMA table_info({self.current_table})").fetchall()
            return True, "\n".join([f"- {col[1]} ({col[2]})" for col in schema_info])

        except Exception as e:
            return False, str(e)

    def generate_sql(self, user_query: str):
        try:
            cursor = self.conn.cursor()
            columns = [row[1] for row in cursor.execute(f"PRAGMA table_info({self.current_table})").fetchall()]
            prompt_template = PromptTemplate(
                input_variables=["user_query", "columns"],
                template="Generate an SQL query for the following request: {user_query}. The table has these columns: {columns}."
            )
            chain = LLMChain(llm=self.llm, prompt=prompt_template)
            sql_query = chain.run({"user_query": user_query, "columns": ", ".join(columns)})
            return sql_query.strip()
        except Exception as e:
            raise Exception("SQL generation failed.")

    def analyze(self, user_query: str):
        try:
            sql_query = self.generate_sql(user_query)
            df_result = pd.read_sql_query(sql_query, self.conn)
            if df_result.empty:
                raise ValueError("Query returned no data.")
            return df_result, sql_query
        except Exception as e:
            raise Exception(f"Analysis failed: {str(e)}")

def main():
    st.title("AI-Driven Data Analyzer")
    analyzer = DataAnalyzer()

    uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])
    if uploaded_file:
        sheet_name = None
        if uploaded_file.name.endswith('.xlsx'):
            excel_file = pd.ExcelFile(uploaded_file)
            sheet_name = st.selectbox("Select a sheet", excel_file.sheet_names)

        success, schema_info = analyzer.load_data(uploaded_file, sheet_name)
        if success:
            st.success("Data loaded successfully!")
            st.text(f"Schema:\n{schema_info}")

            user_query = st.text_area("Enter your query", placeholder="Show total impressions by quarter.")
            if st.button("Run Query"):
                try:
                    result, sql_query = analyzer.analyze(user_query)
                    st.dataframe(result)
                    st.code(sql_query, language="sql")
                    if len(result.columns) >= 2:
                        st.plotly_chart(px.bar(result, x=result.columns[0], y=result.columns[1], title="Query Results"))
                except Exception as e:
                    st.error(e)
        else:
            st.error(f"Failed to load data: {schema_info}")

if __name__ == "__main__":
    main()
