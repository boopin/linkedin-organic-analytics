import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
import sqlite3
import json
from typing import Dict, List, Tuple, Optional
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

class DataAnalyzer:
    def __init__(self):
        self.conn = sqlite3.connect(':memory:')
        self.current_table = None
        
    def load_data(self, file) -> Tuple[bool, str]:
        """Load data from uploaded file into SQLite database"""
        try:
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)
                
            # Clean column names for SQL compatibility
            df.columns = [c.lower().replace(' ', '_').replace('-', '_') 
                         for c in df.columns]
            
            # Store table name
            self.current_table = 'data_table'
            
            # Save to SQLite
            df.to_sql(self.current_table, self.conn, index=False, if_exists='replace')
            
            # Get schema information
            cursor = self.conn.cursor()
            schema_info = cursor.execute(f"PRAGMA table_info({self.current_table})").fetchall()
            
            return True, self.format_schema_info(schema_info)
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return False, str(e)
    
    def format_schema_info(self, schema_info: List) -> str:
        """Format schema information for GPT context"""
        columns = []
        for col in schema_info:
            name, dtype = col[1], col[2]
            columns.append(f"- {name} ({dtype})")
        return "Table columns:\n" + "\n".join(columns)

    def generate_sql(self, user_query: str, schema_info: str) -> str:
        """Generate SQL query using GPT"""
        prompt = f"""
        Given a database table with the following schema:
        {schema_info}
        
        Generate a SQL query to answer this question: "{user_query}"
        
        Requirements:
        - Use only the columns shown above
        - Return results that can be visualized
        - Include relevant aggregations and grouping
        - Use proper SQL syntax for SQLite
        - Return only the SQL query, no explanations
        
        SQL Query:
        """
        
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "You are a SQL expert. Generate SQL queries that answer user questions about their data."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=300
        )
        
        return response.choices[0].message.content.strip()

    def suggest_visualization(self, sql_query: str, df: pd.DataFrame) -> Dict:
        """Use GPT to suggest appropriate visualization"""
        prompt = f"""
        Given this SQL query and resulting data:
        Query: {sql_query}
        Columns: {', '.join(df.columns)}
        Data sample: {df.head(2).to_dict()}
        
        Suggest the best visualization type and configuration. Consider:
        1. The nature of the data (temporal, categorical, numerical)
        2. The number of variables
        3. The relationship we want to show

        Return a JSON object with these keys:
        - chart_type: one of [line, bar, scatter, pie, area, histogram]
        - x_column: column for x-axis
        - y_column: column(s) for y-axis
        - title: suggested title
        - color_column (optional): column for color encoding
        
        JSON response:
        """
        
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "You are a data visualization expert. Suggest the best chart type and configuration for given data."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=300
        )
        
        return json.loads(response.choices[0].message.content.strip())

    def create_visualization(self, df: pd.DataFrame, viz_config: Dict) -> go.Figure:
        """Create visualization based on configuration"""
        chart_type = viz_config['chart_type']
        
        if chart_type == 'bar':
            fig = px.bar(df, x=viz_config['x_column'], y=viz_config['y_column'],
                        title=viz_config['title'],
                        color=viz_config.get('color_column'))
            
        elif chart_type == 'line':
            fig = px.line(df, x=viz_config['x_column'], y=viz_config['y_column'],
                         title=viz_config['title'],
                         color=viz_config.get('color_column'))
            
        elif chart_type == 'scatter':
            fig = px.scatter(df, x=viz_config['x_column'], y=viz_config['y_column'],
                           title=viz_config['title'],
                           color=viz_config.get('color_column'))
            
        elif chart_type == 'pie':
            fig = px.pie(df, values=viz_config['y_column'], names=viz_config['x_column'],
                        title=viz_config['title'])
            
        elif chart_type == 'area':
            fig = px.area(df, x=viz_config['x_column'], y=viz_config['y_column'],
                         title=viz_config['title'],
                         color=viz_config.get('color_column'))
            
        else:  # histogram
            fig = px.histogram(df, x=viz_config['x_column'],
                             title=viz_config['title'],
                             color=viz_config.get('color_column'))
        
        fig.update_layout(
            template='plotly_white',
            title_x=0.5,
            margin=dict(t=50, l=0, r=0, b=0)
        )
        
        return fig

    def analyze(self, user_query: str, schema_info: str) -> Tuple[pd.DataFrame, Dict, str]:
        """Perform complete analysis pipeline"""
        try:
            # Generate SQL
            sql_query = self.generate_sql(user_query, schema_info)
            
            # Execute query
            df_result = pd.read_sql_query(sql_query, self.conn)
            
            # Get visualization suggestion
            viz_config = self.suggest_visualization(sql_query, df_result)
            
            return df_result, viz_config, sql_query
            
        except Exception as e:
            logger.error(f"Analysis error: {str(e)}")
            raise Exception(f"Analysis failed: {str(e)}")

def main():
    st.set_page_config(page_title="AI Data Analyzer", layout="wide")
    
    st.title("📊 AI-Powered Data Analyzer")
    st.write("Upload your data and ask questions in natural language!")
    
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = DataAnalyzer()
    
    # File upload
    uploaded_file = st.file_uploader("Upload your data (Excel or CSV)", 
                                   type=['xlsx', 'xls', 'csv'])
    
    if uploaded_file:
        success, schema_info = st.session_state.analyzer.load_data(uploaded_file)
        
        if success:
            st.success("Data loaded successfully!")
            
            with st.expander("View Data Schema"):
                st.code(schema_info)
            
            # Query input
            user_query = st.text_area(
                "What would you like to know about your data?",
                placeholder="e.g., 'Show me the trend of engagement over time' or 'What are the top 5 posts by comments?'"
            )
            
            if user_query:
                try:
                    with st.spinner("Analyzing your data..."):
                        df_result, viz_config, sql_query = st.session_state.analyzer.analyze(
                            user_query, schema_info
                        )
                    
                    # Display results in tabs
                    tab1, tab2, tab3 = st.tabs(["📈 Visualization", "📊 Data", "🔍 Query"])
                    
                    with tab1:
                        fig = st.session_state.analyzer.create_visualization(
                            df_result, viz_config
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                    with tab2:
                        st.dataframe(
                            df_result.style.background_gradient(cmap='Blues'),
                            use_container_width=True
                        )
                        
                    with tab3:
                        st.code(sql_query, language='sql')
                        
                except Exception as e:
                    st.error(str(e))
        else:
            st.error(f"Error loading data: {schema_info}")
    
    # Sidebar
    with st.sidebar:
        st.header("💡 Sample Questions")
        st.info("""
        Try asking questions like:
        - What's the overall trend of engagement over time?
        - Show me the distribution of likes across different post types
        - Which days of the week get the most engagement?
        - What are the top 10 posts by total engagement?
        - How does the average comment count vary by month?
        """)
        
        st.header("ℹ️ About")
        st.markdown("""
        This app uses:
        - GPT-4 for natural language understanding
        - SQL for data analysis
        - Plotly for visualizations
        
        Upload any Excel or CSV file and ask questions about your data!
        """)

if __name__ == "__main__":
    main()
