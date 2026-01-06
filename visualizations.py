import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, List, Any

class Visualizations:
    """Generate interactive visualizations for telecom data"""
    
    def __init__(self):
        self.color_scheme = px.colors.qualitative.Set2
    
    def create_subscriber_distribution(self, df: pd.DataFrame, column: str) -> go.Figure:
        """Create subscriber distribution chart"""
        dist = df[column].value_counts()
        
        fig = px.pie(
            values=dist.values,
            names=dist.index,
            title=f"Distribution by {column}",
            color_discrete_sequence=self.color_scheme
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        return fig
    
    def create_usage_trend(self, df: pd.DataFrame, date_col: str, metric_col: str) -> go.Figure:
        """Create usage trend over time"""
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df_sorted = df.sort_values(date_col)
        
        daily = df_sorted.groupby(df_sorted[date_col].dt.date)[metric_col].sum()
        
        fig = px.line(
            x=daily.index,
            y=daily.values,
            title=f"{metric_col} Over Time",
            labels={'x': 'Date', 'y': metric_col}
        )
        fig.update_traces(line_color='#1f77b4', line_width=2)
        fig.update_layout(hovermode='x unified')
        return fig
    
    def create_regional_heatmap(self, df: pd.DataFrame, region_col: str, metric_col: str) -> go.Figure:
        """Create regional performance heatmap"""
        regional_data = df.groupby(region_col)[metric_col].sum().sort_values(ascending=False)
        
        fig = px.bar(
            x=regional_data.index,
            y=regional_data.values,
            title=f"{metric_col} by {region_col}",
            labels={'x': region_col, 'y': metric_col},
            color=regional_data.values,
            color_continuous_scale='Blues'
        )
        fig.update_layout(showlegend=False, xaxis_tickangle=-45)
        return fig
    
    def create_device_comparison(self, device_metrics: Dict) -> go.Figure:
        """Create device format comparison chart"""
        if 'device_distribution' in device_metrics:
            dist = device_metrics['device_distribution']
            
            fig = go.Figure(data=[
                go.Bar(
                    x=list(dist.keys()),
                    y=list(dist.values()),
                    marker_color=self.color_scheme[:len(dist)]
                )
            ])
            fig.update_layout(
                title="Device Format Usage",
                xaxis_title="Device Type",
                yaxis_title="Count",
                showlegend=False
            )
            return fig
        return None
    
    def create_metrics_dashboard(self, metrics: Dict) -> go.Figure:
        """Create comprehensive metrics dashboard"""
        # This would create a subplot with multiple metrics
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Subscribers", "Usage", "Devices", "Regions")
        )
        
        # Add traces as needed based on available metrics
        # This is a placeholder structure
        
        return fig
    
    def create_comparison_chart(self, df: pd.DataFrame, category_col: str, 
                              value_cols: List[str]) -> go.Figure:
        """Create comparison chart for multiple metrics"""
        fig = go.Figure()
        
        for col in value_cols:
            data = df.groupby(category_col)[col].mean()
            fig.add_trace(go.Bar(name=col, x=data.index, y=data.values))
        
        fig.update_layout(
            title=f"Comparison by {category_col}",
            barmode='group',
            xaxis_title=category_col,
            yaxis_title="Value"
        )
        return fig
    
    def create_box_plot(self, df: pd.DataFrame, column: str, category: str = None) -> go.Figure:
        """Create box plot for distribution analysis"""
        if category:
            fig = px.box(df, x=category, y=column, title=f"{column} Distribution by {category}")
        else:
            fig = px.box(df, y=column, title=f"{column} Distribution")
        
        return fig
    
    def create_correlation_heatmap(self, df: pd.DataFrame) -> go.Figure:
        """Create correlation heatmap for numeric columns"""
        numeric_df = df.select_dtypes(include=['int64', 'float64'])
        corr_matrix = numeric_df.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0
        ))
        
        fig.update_layout(
            title="Correlation Matrix",
            xaxis_title="Features",
            yaxis_title="Features"
        )
        return fig
