import streamlit as st
import pandas as pd
import sqlite3
import logging
from typing import Tuple, Optional
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
import plotly.express as px
import plotly.graph_objects as go
from urllib.parse import urlparse
import re

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Enhanced column mapping for post metrics
COLUMN_MAPPING = {
    "title": ["post title", "title"],
    "type": ["post type", "type"],
    "campaign": ["campaign name", "campaign"],
    "impressions": ["impressions", "total impressions"],
    "views": ["views", "total views"],
    "clicks": ["clicks", "total clicks"],
    "ctr": ["click through rate", "ctr", "click through rate (ctr)"],
    "likes": ["likes", "total likes"],
    "comments": ["comments", "total comments"],
    "reposts": ["reposts", "total reposts"],
    "follows": ["follows", "total follows"],
    "engagement_rate": ["engagement rate", "total engagement rate"],
    "content_type": ["content type", "content_type"]
}

class ContentAnalyzer:
    """Analyzes post content and extracts insights."""
    
    @staticmethod
    def extract_hashtags(text: str) -> list:
        """Extract hashtags from post content."""
        return re.findall(r'#(\w+)', text) if text else []
    
    @staticmethod
    def extract_urls(text: str) -> list:
        """Extract URLs from post content."""
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
        return [url for url in urls if 'linkedin.com' not in url]
    
    @staticmethod
    def analyze_content(df: pd.DataFrame) -> dict:
        """Analyze post content for insights."""
        all_hashtags = []
        external_links = []
        content_lengths = []
        
        for text in df['post title']:
            if isinstance(text, str):
                hashtags = ContentAnalyzer.extract_hashtags(text)
                urls = ContentAnalyzer.extract_urls(text)
                all_hashtags.extend(hashtags)
                external_links.extend(urls)
                content_lengths.append(len(text))
        
        return {
            "hashtag_usage": {
                "total_hashtags": len(all_hashtags),
                "unique_hashtags": len(set(all_hashtags)),
                "top_hashtags": pd.Series(all_hashtags).value_counts().head(5).to_dict()
            },
            "content_stats": {
                "avg_length": sum(content_lengths) / len(content_lengths) if content_lengths else 0,
                "max_length": max(content_lengths) if content_lengths else 0,
                "min_length": min(content_lengths) if content_lengths else 0
            },
            "links": {
                "total_external_links": len(external_links),
                "unique_domains": len(set([urlparse(url).netloc for url in external_links]))
            }
        }

class PostMetricsAnalyzer:
    """Analyzes post performance metrics."""
    
    @staticmethod
    def calculate_metrics(df: pd.DataFrame) -> dict:
        """Calculate key post performance metrics."""
        return {
            "engagement": {
                "total_impressions": df['impressions'].sum(),
                "total_views": df['views'].sum(),
                "total_clicks": df['clicks'].sum(),
                "avg_ctr": df['click through rate (ctr)'].mean(),
                "total_likes": df['likes'].sum(),
                "total_comments": df['comments'].sum(),
                "total_reposts": df['reposts'].sum(),
                "total_follows": df['follows'].sum(),
                "avg_engagement_rate": df['engagement rate'].mean()
            },
            "performance": {
                "best_post_by_impressions": df.loc[df['impressions'].idxmax(), 'post title'][:100] + "...",
                "best_post_by_engagement": df.loc[df['engagement rate'].idxmax(), 'post title'][:100] + "..."
            }
        }

class DataVisualizationAgent:
    """Creates visualizations for post performance."""
    
    @staticmethod
    def create_engagement_metrics_chart(df: pd.DataFrame) -> go.Figure:
        """Create bar chart of engagement metrics."""
        metrics = ['likes', 'comments', 'reposts', 'follows']
        values = [df[metric].sum() for metric in metrics]
        
        fig = go.Figure([go.Bar(x=metrics, y=values)])
        fig.update_layout(
            title='Engagement Metrics Distribution',
            xaxis_title='Metric Type',
            yaxis_title='Count'
        )
        return fig
    
    @staticmethod
    def create_performance_scatter(df: pd.DataFrame) -> go.Figure:
        """Create scatter plot of impressions vs engagement rate."""
        fig = go.Figure(data=go.Scatter(
            x=df['impressions'],
            y=df['engagement rate'],
            mode='markers+text',
            text=df['post title'].str[:30] + '...',
            textposition='top center'
        ))
        
        fig.update_layout(
            title='Impressions vs Engagement Rate',
            xaxis_title='Impressions',
            yaxis_title='Engagement Rate',
            showlegend=False
        )
        return fig

def main():
    st.set_page_config(page_title="Social Media Posts Analyzer", page_icon="📊", layout="wide")
    
    st.title("📊 Social Media Posts Performance Analyzer")
    
    uploaded_file = st.file_uploader("Upload your posts data (CSV or Excel)", type=['csv', 'xlsx'])
    if not uploaded_file:
        st.info("👋 Welcome! Please upload your social media posts data to begin analysis.")
        return

    try:
        if uploaded_file.name.endswith('xlsx'):
            excel_file = pd.ExcelFile(uploaded_file)
            sheet_name = st.selectbox("Select sheet", excel_file.sheet_names)
            df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
        else:
            df = pd.read_csv(uploaded_file)

        # Clean column names
        df.columns = [col.lower().strip() for col in df.columns]
        
        # Calculate metrics
        post_metrics = PostMetricsAnalyzer.calculate_metrics(df)
        content_analysis = ContentAnalyzer.analyze_content(df)
        
        # Layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Visualization tabs
            tab1, tab2 = st.tabs(["📊 Engagement Metrics", "🎯 Performance Analysis"])
            
            with tab1:
                fig_engagement = DataVisualizationAgent.create_engagement_metrics_chart(df)
                st.plotly_chart(fig_engagement, use_container_width=True)
            
            with tab2:
                fig_performance = DataVisualizationAgent.create_performance_scatter(df)
                st.plotly_chart(fig_performance, use_container_width=True)
            
            # Post Analysis
            st.write("### 📝 Post Analysis")
            
            # Filter and sort options
            metric_options = ["impressions", "engagement rate", "clicks", "likes"]
            sort_by = st.selectbox("Sort posts by:", metric_options)
            show_top = st.slider("Show top N posts:", 1, len(df), 5)
            
            # Display sorted posts
            sorted_df = df.sort_values(by=sort_by, ascending=False).head(show_top)
            st.dataframe(
                sorted_df[['post title', 'impressions', 'engagement rate', 'clicks', 'likes']],
                hide_index=True
            )
        
        with col2:
            # Key Metrics
            st.write("### 📈 Performance Summary")
            
            # Engagement metrics
            metrics = post_metrics['engagement']
            st.write("**Engagement Totals**")
            cols = st.columns(2)
            with cols[0]:
                st.metric("Impressions", f"{metrics['total_impressions']:,}")
                st.metric("Views", f"{metrics['total_views']:,}")
                st.metric("Clicks", f"{metrics['total_clicks']:,}")
            with cols[1]:
                st.metric("Likes", f"{metrics['total_likes']:,}")
                st.metric("Comments", f"{metrics['total_comments']:,}")
                st.metric("Reposts", f"{metrics['total_reposts']:,}")
            
            # Content Analysis
            st.write("### 📑 Content Analysis")
            
            # Hashtag analysis
            st.write("**Hashtag Usage**")
            st.write(f"- Total hashtags: {content_analysis['hashtag_usage']['total_hashtags']}")
            st.write(f"- Unique hashtags: {content_analysis['hashtag_usage']['unique_hashtags']}")
            
            if content_analysis['hashtag_usage']['top_hashtags']:
                st.write("**Top Hashtags**")
                for tag, count in content_analysis['hashtag_usage']['top_hashtags'].items():
                    st.write(f"- #{tag}: {count}")
            
            # Content stats
            st.write("**Content Statistics**")
            st.write(f"- Average length: {content_analysis['content_stats']['avg_length']:.0f} characters")
            st.write(f"- External links: {content_analysis['links']['total_external_links']}")

    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")

if __name__ == "__main__":
    main()
