import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go  # Updated import
from datetime import datetime, timedelta
import logging
import time
from typing import Optional

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
if 'error_count' not in st.session_state:
    st.session_state.error_count = 0

class DataValidationError(Exception):
    """Custom exception for data validation errors."""
    pass

@st.cache_data(ttl=3600)
def load_data(uploaded_file) -> Optional[pd.DataFrame]:
    """
    Load and preprocess LinkedIn data with validation.

    Args:
        uploaded_file: Streamlit uploaded file object.
    Returns:
        Preprocessed DataFrame or None if validation fails.
    """
    try:
        df = pd.read_excel(uploaded_file)

        # Validate required columns
        required_columns = ['Date', 'Likes', 'Comments', 'Shares']
        missing_columns = [col for col in required_columns if col.lower() not in [c.lower() for c in df.columns]]

        if missing_columns:
            raise DataValidationError(f"Missing required columns: {', '.join(missing_columns)}")

        # Standardize column names
        df.columns = [col.lower().strip() for col in df.columns]

        # Convert date columns
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

        # Handle missing values
        numeric_columns = ['likes', 'comments', 'shares']
        df[numeric_columns] = df[numeric_columns].fillna(0)

        # Add engagement score
        df['engagement_score'] = df['likes'] * 1 + df['comments'] * 2 + df['shares'] * 3

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

def create_visualization(df: pd.DataFrame, metric: str, title: str = None) -> go.Figure:
    """
    Create enhanced visualization for results.

    Args:
        df: Input DataFrame
        metric: Metric to visualize
        title: Optional custom title.
    Returns:
        Plotly figure object.
    """
    if not title:
        title = f'{metric.replace("_", " ").title()} by Post'

    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' is not in the DataFrame columns: {df.columns}")

    if df.empty:
        raise ValueError("The DataFrame is empty. No data to visualize.")

    fig = px.bar(
        df,
        x='Post Text' if 'post text' in df.columns else df.index,
        y=metric,
        title=title,
        labels={'x': 'Post', 'y': metric.replace('_', ' ').title()},
        template="plotly_white"
    )

    fig.update_layout(
        hoverlabel=dict(bgcolor="white"),
        hovermode='x unified',
        showlegend=False,
        height=500
    )

    return fig

def display_metrics_summary(df: pd.DataFrame):
    """Display key metrics summary in a clean layout."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Posts", len(df), delta=None)

    with col2:
        st.metric("Avg Engagement", f"{df['engagement_score'].mean():.1f}", delta=f"{df['engagement_score'].std():.1f} σ")

    with col3:
        st.metric("Total Likes", df['likes'].sum(), delta=f"{df['likes'].mean():.1f} avg")

    with col4:
        st.metric("Total Comments", df['comments'].sum(), delta=f"{df['comments'].mean():.1f} avg")

def main():
    # Page header
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

    # File uploader
    uploaded_file = st.file_uploader("Upload your LinkedIn data export (Excel file)", type=['xlsx', 'xls'])

    if uploaded_file is not None:
        with st.spinner('Loading and processing your data...'):
            df = load_data(uploaded_file)

        if df is not None:
            # Display metrics summary
            display_metrics_summary(df)

            # Top 5 posts by engagement
            top_posts = df.nlargest(5, 'engagement_score')[['date', 'likes', 'comments', 'shares', 'engagement_score']]
            st.write("### Top 5 Posts by Engagement Score")
            st.dataframe(top_posts)

            # Visualization
            st.write("### Engagement Score Visualization")
            try:
                fig = create_visualization(top_posts, 'engagement_score', "Top 5 Posts by Engagement Score")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating visualization: {str(e)}")
                logger.error(f"Visualization error: {str(e)}")

if __name__ == "__main__":
    main()
