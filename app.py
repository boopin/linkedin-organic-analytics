import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import re
import time
from typing import Tuple, Optional, Dict, List
import logging
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure page settings
st.set_page_config(
    page_title="LinkedIn Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_timestamp' not in st.session_state:
    st.session_state.data_timestamp = None
if 'processing_history' not in st.session_state:
    st.session_state.processing_history = []
if 'error_count' not in st.session_state:
    st.session_state.error_count = 0

class DataValidationError(Exception):
    """Custom exception for data validation errors"""
    pass

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_data(uploaded_file) -> Optional[pd.DataFrame]:
    """
    Load and preprocess LinkedIn data with caching and validation
    
    Args:
        uploaded_file: Streamlit uploaded file object
    Returns:
        Preprocessed DataFrame or None if validation fails
    """
    try:
        df = pd.read_excel(uploaded_file)
        
        # Validate required columns
        required_columns = ['Date', 'Likes', 'Comments', 'Shares']
        missing_columns = [col for col in required_columns if col.lower() not in 
                         [c.lower() for c in df.columns]]
        
        if missing_columns:
            raise DataValidationError(
                f"Missing required columns: {', '.join(missing_columns)}"
            )
            
        # Standardize column names
        df.columns = [col.lower().strip() for col in df.columns]
        
        # Convert date columns
        date_columns = [col for col in df.columns if 'date' in col]
        for col in date_columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            
        # Handle missing values
        numeric_columns = ['likes', 'comments', 'shares']
        df[numeric_columns] = df[numeric_columns].fillna(0)
        
        # Add engagement score
        df['engagement_score'] = (
            df['likes'] * 1 + 
            df['comments'] * 2 + 
            df['shares'] * 3
        )
        
        # Log successful load
        logger.info(f"Successfully loaded data with {len(df)} rows")
        st.session_state.data_timestamp = time.time()
        
        return df
        
    except DataValidationError as e:
        st.error(f"Data Validation Error: {str(e)}")
        logger.error(f"Data validation failed: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        logger.error(f"Data loading failed: {str(e)}")
        st.session_state.error_count += 1
        return None

@st.cache_data(ttl=3600)
def parse_time_period(query: str) -> datetime:
    """
    Extract time period from query with improved pattern matching
    
    Args:
        query: Natural language query string
    Returns:
        Start date for filtering
    """
    time_mapping = {
        'day': 1,
        'week': 7,
        'month': 30,
        'months': 30,
        'year': 365,
        'quarter': 90
    }
    
    patterns = [
        r'last\s+(\d+)\s+(day|week|month|months|year|quarter)s?',
        r'past\s+(\d+)\s+(day|week|month|months|year|quarter)s?',
        r'previous\s+(\d+)\s+(day|week|month|months|year|quarter)s?'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, query.lower())
        if match:
            number = int(match.group(1))
            period = match.group(2)
            days = number * time_mapping[period]
            return datetime.now() - timedelta(days=days)
    
    # Default to last 30 days if no time period specified
    return datetime.now() - timedelta(days=30)

@st.cache_data(ttl=3600)
def analyze_query(df: pd.DataFrame, query: str) -> Tuple[pd.DataFrame, str, Dict]:
    """
    Analyze data based on natural language query with enhanced features
    
    Args:
        df: Input DataFrame
        query: Natural language query string
    Returns:
        Tuple of (results DataFrame, metric name, analysis metadata)
    """
    query = query.lower()
    
    # Extract time period
    start_date = parse_time_period(query)
    df_filtered = df[df['date'] >= start_date].copy()
    
    # Define metrics with weights
    metrics = {
        'engagement': {
            'columns': ['engagement_score'],
            'weight': 1.0
        },
        'comments': {
            'columns': ['comments'],
            'weight': 2.0
        },
        'likes': {
            'columns': ['likes'],
            'weight': 1.0
        },
        'shares': {
            'columns': ['shares'],
            'weight': 3.0
        }
    }
    
    # Determine metrics to analyze
    selected_metrics = []
    weights = []
    
    for metric, config in metrics.items():
        if metric in query:
            selected_metrics.extend(config['columns'])
            weights.extend([config['weight']] * len(config['columns']))
    
    if not selected_metrics:
        selected_metrics = metrics['engagement']['columns']
        weights = [metrics['engagement']['weight']]
    
    # Determine sort direction
    ascending = any(word in query for word in ['lowest', 'worst', 'bottom', 'least'])
    
    # Get number of results
    num_pattern = r'top\s+(\d+)|bottom\s+(\d+)|(\d+)\s+posts'
    num_match = re.search(num_pattern, query)
    num_results = int(num_match.group(1) if num_match else 5)
    
    # Calculate weighted score if multiple metrics
    if len(selected_metrics) > 1:
        df_filtered['weighted_score'] = sum(
            df_filtered[metric] * weight 
            for metric, weight in zip(selected_metrics, weights)
        )
        sort_column = 'weighted_score'
    else:
        sort_column = selected_metrics[0]
    
    # Sort and get results
    results = df_filtered.sort_values(
        sort_column, 
        ascending=ascending
    ).head(num_results)
    
    # Prepare metadata
    metadata = {
        'time_period': f"{start_date.strftime('%Y-%m-%d')} to {datetime.now().strftime('%Y-%m-%d')}",
        'metrics_analyzed': selected_metrics,
        'total_posts': len(df_filtered),
        'direction': 'ascending' if ascending else 'descending'
    }
    
    return results, sort_column, metadata

def create_visualization(df: pd.DataFrame, metric: str, title: str = None) -> px.Figure:
    """
    Create enhanced visualization for results
    
    Args:
        df: Input DataFrame
        metric: Metric to visualize
        title: Optional custom title
    Returns:
        Plotly figure object
    """
    if not title:
        title = f'{metric.replace("_", " ").title()} by Post'
        
    fig = px.bar(
        df,
        x=df.index,
        y=metric,
        title=title,
        labels={'x': 'Post', 'y': metric.replace('_', ' ').title()},
        template="plotly_white"
    )
    
    fig.update_layout(
        hoverlabel=dict(bgcolor="white"),
        hovermode='x unified',
        showlegend=True,
        height=500
    )
    
    return fig

def display_metrics_summary(df: pd.DataFrame):
    """Display key metrics summary in a clean layout"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Posts",
            len(df),
            delta=None
        )
    
    with col2:
        st.metric(
            "Avg Engagement",
            f"{df['engagement_score'].mean():.1f}",
            delta=f"{df['engagement_score'].std():.1f} σ"
        )
    
    with col3:
        st.metric(
            "Total Likes",
            df['likes'].sum(),
            delta=f"{df['likes'].mean():.1f} avg"
        )
    
    with col4:
        st.metric(
            "Total Comments",
            df['comments'].sum(),
            delta=f"{df['comments'].mean():.1f} avg"
        )

def main():
    # Page header with custom styling
    st.markdown("""
        <style>
        .title {
            font-size: 42px;
            font-weight: bold;
            color: #0A66C2;
            margin-bottom: 20px;
        }
        .subtitle {
            font-size: 24px;
            color: #666;
            margin-bottom: 30px;
        }
        </style>
        <div class="title">LinkedIn Posts Analytics</div>
        <div class="subtitle">Upload your data and analyze your posts with natural language queries</div>
    """, unsafe_allow_html=True)
    
    # File uploader with error handling
    uploaded_file = st.file_uploader(
        "Upload your LinkedIn data export (Excel file)", 
        type=['xlsx', 'xls'],
        help="Export your LinkedIn posts data as an Excel file"
    )
    
    if uploaded_file is not None:
        with st.spinner('Loading and processing your data...'):
            df = load_data(uploaded_file)
            
        if df is not None:
            # Display metrics summary
            display_metrics_summary(df)
            
            # Query input with advanced features
            col1, col2 = st.columns([3, 1])
            with col1:
                query = st.text_input(
                    "Enter your question",
                    placeholder="e.g., Show me the top 5 posts by engagement in the last 3 months",
                    help="Try asking about engagement, likes, comments, or shares over different time periods"
                )
            
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)
                clear_cache = st.button("Clear Cache")
                if clear_cache:
                    st.cache_data.clear()
                    st.success("Cache cleared!")
            
            if query:
                try:
                    # Process query
                    with st.spinner('Analyzing your request...'):
                        results, metric, metadata = analyze_query(df, query)
                    
                    # Display results in tabs
                    tab1, tab2, tab3 = st.tabs(["📊 Visualization", "📋 Data", "ℹ️ Analysis Details"])
                    
                    with tab1:
                        fig = create_visualization(
                            results, 
                            metric,
                            f"Analysis Results: {query}"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with tab2:
                        st.dataframe(
                            results.style.background_gradient(
                                subset=[metric],
                                cmap='Blues'
                            ),
                            use_container_width=True
                        )
                    
                    with tab3:
                        st.json(metadata)
                    
                    # Log successful query
                    st.session_state.processing_history.append({
                        'timestamp': time.time(),
                        'query': query,
                        'success': True
                    })
                    
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
                    logger.error(f"Query processing failed: {str(e)}")
                    st.session_state.error_count += 1
    
    # Sidebar with enhanced features
    with st.sidebar:
        st.header("📝 Sample Queries")
        st.info("""
        Try these example queries:
        - Show me the top 5 posts by engagement in the last 3 months
        - What are the lowest performing posts by comments in the last week?
        - Show top 10 posts by likes in the last month
        - Which posts had the highest shares in the last quarter?
        """)
        
        st.header("📊 App Stats")
        if st.session_state.data_timestamp:
            st.text(f"Last data update: {datetime.fromtimestamp(st.session_state.data_timestamp).strftime('%Y-%m-%d %H:%M:%S')}")
        st.text(f"Errors encountered: {st.session_state.error_count}")
        
        st.header("ℹ️ About")
        st.markdown("""
        This app helps you analyze your LinkedIn posts data using natural language queries.
        
        **Features:**
        - Data validation and preprocessing
        - Natural language query processing
        - Interactive visualizations
        - Performance caching
        - Error handling and logging
        
        Made with ❤️ using Streamlit
        """)

if __name__ == "__main__":
    main()