import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
from datetime import datetime
import numpy as np
from typing import List, Dict, Any, Tuple
import re
import io
import base64

class CampaignAnalyzer:
    """Handles advanced campaign analysis."""
    
    @staticmethod
    def analyze_campaign_performance(df: pd.DataFrame) -> pd.DataFrame:
        """Analyze campaign performance metrics."""
        campaign_metrics = df.groupby('Campaign name').agg({
            'Impressions': 'sum',
            'Clicks': 'sum',
            'Engagement rate': 'mean',
            'Campaign duration': 'first',
            'Content Type': lambda x: list(x.unique()),
            'Post title': 'count'
        }).round(2)
        
        campaign_metrics.rename(columns={'Post title': 'Number of Posts'}, inplace=True)
        campaign_metrics['CTR'] = (campaign_metrics['Clicks'] / campaign_metrics['Impressions'] * 100).round(2)
        
        return campaign_metrics

    @staticmethod
    def create_campaign_timeline(df: pd.DataFrame) -> go.Figure:
        """Create campaign timeline visualization."""
        campaigns = df['Campaign name'].unique()
        
        fig = go.Figure()
        
        for campaign in campaigns:
            campaign_data = df[df['Campaign name'] == campaign]
            
            fig.add_trace(go.Bar(
                name=campaign,
                x=[campaign],
                y=[(campaign_data['Campaign end date'].max() - 
                    campaign_data['Campaign start date'].min()).days],
                text=campaign_data['Engagement rate'].mean().round(2),
                customdata=np.array([
                    [campaign_data['Impressions'].sum(),
                     campaign_data['Clicks'].sum(),
                     len(campaign_data)]
                ]),
                hovertemplate="Campaign: %{x}<br>" +
                             "Duration: %{y} days<br>" +
                             "Avg Engagement Rate: %{text}%<br>" +
                             "Total Impressions: %{customdata[0]}<br>" +
                             "Total Clicks: %{customdata[1]}<br>" +
                             "Number of Posts: %{customdata[2]}"
            ))
        
        fig.update_layout(
            title="Campaign Duration and Performance",
            xaxis_title="Campaign",
            yaxis_title="Duration (days)",
            barmode='group',
            height=500
        )
        
        return fig

class ContentAnalyzer:
    """Handles content type analysis and comparisons."""
    
    @staticmethod
    def analyze_content_performance(df: pd.DataFrame) -> pd.DataFrame:
        """Analyze performance by content type."""
        content_metrics = df.groupby('Content Type').agg({
            'Impressions': ['sum', 'mean'],
            'Clicks': ['sum', 'mean'],
            'Engagement rate': ['mean', 'std'],
            'Post title': 'count'
        }).round(2)
        
        content_metrics.columns = [
            'Total Impressions', 'Avg Impressions',
            'Total Clicks', 'Avg Clicks',
            'Avg Engagement Rate', 'Engagement Rate Std',
            'Number of Posts'
        ]
        
        return content_metrics

    @staticmethod
    def create_content_comparison_plot(df: pd.DataFrame) -> go.Figure:
        """Create content type comparison visualization."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Engagement by Content Type',
                          'Impressions by Content Type',
                          'Clicks by Content Type',
                          'Post Distribution')
        )
        
        content_metrics = ContentAnalyzer.analyze_content_performance(df)
        
        # Engagement plot
        fig.add_trace(
            go.Bar(x=content_metrics.index,
                  y=content_metrics['Avg Engagement Rate'],
                  name='Avg Engagement Rate'),
            row=1, col=1
        )
        
        # Impressions plot
        fig.add_trace(
            go.Bar(x=content_metrics.index,
                  y=content_metrics['Avg Impressions'],
                  name='Avg Impressions'),
            row=1, col=2
        )
        
        # Clicks plot
        fig.add_trace(
            go.Bar(x=content_metrics.index,
                  y=content_metrics['Avg Clicks'],
                  name='Avg Clicks'),
            row=2, col=1
        )
        
        # Post distribution plot
        fig.add_trace(
            go.Pie(labels=content_metrics.index,
                  values=content_metrics['Number of Posts'],
                  name='Post Distribution'),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=False)
        return fig

class DataExporter:
    """Handles data export functionality."""
    
    @staticmethod
    def generate_excel_download_link(df: pd.DataFrame, filename: str) -> str:
        """Generate a download link for Excel file."""
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Analysis Results')
        excel_data = output.getvalue()
        b64 = base64.b64encode(excel_data).decode()
        return f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">Download Excel file</a>'

    @staticmethod
    def generate_csv_download_link(df: pd.DataFrame, filename: str) -> str:
        """Generate a download link for CSV file."""
        csv = df.to_csv(index=True)
        b64 = base64.b64encode(csv.encode()).decode()
        return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV file</a>'

def create_specialized_visualizations(df: pd.DataFrame, metric: str) -> go.Figure:
    """Create specialized visualizations for specific metrics."""
    
    if 'engagement' in metric.lower():
        # Create engagement funnel
        fig = go.Figure(go.Funnel(
            name='Engagement Funnel',
            y=['Impressions', 'Clicks', 'Reactions', 'Comments', 'Reposts'],
            x=[df[col].sum() for col in 
               ['Impressions (total)', 'Clicks (total)', 
                'Reactions (total)', 'Comments (total)', 'Reposts (total)']]
        ))
        fig.update_layout(title='Engagement Funnel')
        
    elif 'impression' in metric.lower():
        # Create stacked area chart for impressions
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['Impressions (organic)'],
            name='Organic',
            stackgroup='one'
        ))
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['Impressions (sponsored)'],
            name='Sponsored',
            stackgroup='one'
        ))
        fig.update_layout(title='Organic vs Sponsored Impressions')
        
    else:
        # Create default line chart with trend
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df[metric],
            name=metric
        ))
        # Add trend line
        z = np.polyfit(range(len(df)), df[metric], 1)
        p = np.poly1d(z)
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=p(range(len(df))),
            name='Trend',
            line=dict(dash='dash')
        ))
        fig.update_layout(title=f'{metric} Over Time')
    
    return fig

def main():
    st.set_page_config(page_title="Enhanced LinkedIn Data Analyzer", layout="wide")
    st.title("Enhanced LinkedIn Data Analyzer")
    
    # Sidebar for navigation
    with st.sidebar:
        st.header("📊 Analysis Options")
        analysis_type = st.radio(
            "Select Analysis Type:",
            ["Metrics Analysis", 
             "Campaign Analysis",
             "Content Analysis",
             "Custom Analysis"]
        )
    
    # File upload section
    metrics_file = st.file_uploader("Upload Metrics Sheet", type=['xlsx', 'csv'], key='metrics')
    posts_file = st.file_uploader("Upload Posts Sheet", type=['xlsx', 'csv'], key='posts')
    
    if metrics_file and posts_file:
        # Load and process data
        metrics_df = pd.read_excel(metrics_file) if metrics_file.name.endswith('xlsx') else pd.read_csv(metrics_file)
        posts_df = pd.read_excel(posts_file) if posts_file.name.endswith('xlsx') else pd.read_csv(posts_file)
        
        # Main analysis section
        if analysis_type == "Metrics Analysis":
            st.header("Metrics Analysis")
            
            metric = st.selectbox("Select Metric:", metrics_df.columns[1:])
            period = st.selectbox("Select Time Period:", ['Daily', 'Weekly', 'Monthly', 'Quarterly'])
            
            # Create specialized visualization
            fig = create_specialized_visualizations(metrics_df, metric)
            st.plotly_chart(fig, use_container_width=True)
            
            # Export options
            st.subheader("Export Options")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(DataExporter.generate_excel_download_link(
                    metrics_df, 'metrics_analysis.xlsx'), unsafe_allow_html=True)
            with col2:
                st.markdown(DataExporter.generate_csv_download_link(
                    metrics_df, 'metrics_analysis.csv'), unsafe_allow_html=True)
        
        elif analysis_type == "Campaign Analysis":
            st.header("Campaign Analysis")
            
            # Campaign performance analysis
            campaign_metrics = CampaignAnalyzer.analyze_campaign_performance(posts_df)
            st.dataframe(campaign_metrics)
            
            # Campaign timeline visualization
            fig = CampaignAnalyzer.create_campaign_timeline(posts_df)
            st.plotly_chart(fig, use_container_width=True)
            
            # Export options
            st.subheader("Export Options")
            st.markdown(DataExporter.generate_excel_download_link(
                campaign_metrics, 'campaign_analysis.xlsx'), unsafe_allow_html=True)
        
        elif analysis_type == "Content Analysis":
            st.header("Content Analysis")
            
            # Content performance analysis
            content_metrics = ContentAnalyzer.analyze_content_performance(posts_df)
            st.dataframe(content_metrics)
            
            # Content comparison visualization
            fig = ContentAnalyzer.create_content_comparison_plot(posts_df)
            st.plotly_chart(fig, use_container_width=True)
            
            # Export options
            st.markdown(DataExporter.generate_excel_download_link(
                content_metrics, 'content_analysis.xlsx'), unsafe_allow_html=True)
        
        else:  # Custom Analysis
            st.header("Custom Analysis")
            
            # Let user select metrics and dimensions
            metrics = st.multiselect("Select Metrics:", metrics_df.columns[1:])
            dimension = st.selectbox("Select Dimension:", 
                                   ['Content Type', 'Campaign name', 'Posted by'])
            
            if metrics and dimension:
                custom_analysis = posts_df.groupby(dimension)[metrics].agg(['mean', 'sum']).round(2)
                st.dataframe(custom_analysis)
                
                # Export options
                st.markdown(DataExporter.generate_excel_download_link(
                    custom_analysis, 'custom_analysis.xlsx'), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
