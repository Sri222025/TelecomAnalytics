"""
Telecom & SaaS Data Intelligence Tool with Groq Llama 3 AI
A Streamlit application for analyzing Telecom and SaaS data with AI-powered insights
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from datetime import datetime
import io
import os
from dotenv import load_dotenv

# Try to import seaborn, but don't fail if it's not available
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# Import analysis engine - try Groq version first, fallback to standard
try:
    from analysis_engine_groq import TelecomSaaSAnalyzerWithAI as TelecomSaaSAnalyzer
    HAS_GROQ = True
except ImportError:
    try:
        from analysis_engine import TelecomSaaSAnalyzer
        HAS_GROQ = False
    except ImportError:
        st.error("Error: analysis_engine module not found. Please ensure analysis_engine.py or analysis_engine_groq.py is in the project directory.")
        st.stop()

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Telecom & SaaS Data Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    .insight-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-left: 4px solid #667eea;
        border-radius: 5px;
        margin: 10px 0;
    }
    .ai-insight-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 15px 0;
    }
    .header-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .ai-badge {
        background-color: #FFD700;
        color: #000;
        padding: 5px 10px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = None
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'groq_api_key' not in st.session_state:
    st.session_state.groq_api_key = None


def load_excel_file(uploaded_file):
    """
    Load Excel file with multiple worksheets
    """
    try:
        excel_file = pd.ExcelFile(uploaded_file)
        dataframes = {}
        for sheet in excel_file.sheet_names:
            dataframes[sheet] = pd.read_excel(uploaded_file, sheet_name=sheet)
        return dataframes, None
    except Exception as e:
        return None, str(e)


def create_correlation_heatmap(corr_matrix, sheet_name):
    """
    Create an interactive correlation heatmap
    """
    if corr_matrix.empty:
        return None
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text:.2f}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title=f"Correlation Matrix - {sheet_name}",
        xaxis_title="Variables",
        yaxis_title="Variables",
        height=600,
        width=800
    )
    
    return fig


def create_distribution_plots(df, numeric_cols):
    """
    Create distribution plots for numeric columns
    """
    if not numeric_cols:
        return None
    
    n_cols = min(len(numeric_cols), 4)
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=numeric_cols[:n_cols*n_rows],
        specs=[[{"type": "histogram"}] * n_cols for _ in range(n_rows)]
    )
    
    for idx, col in enumerate(numeric_cols[:n_cols*n_rows]):
        row = (idx // n_cols) + 1
        col_pos = (idx % n_cols) + 1
        
        fig.add_trace(
            go.Histogram(x=df[col].dropna(), name=col, nbinsx=30),
            row=row,
            col=col_pos
        )
    
    fig.update_layout(height=300*n_rows, showlegend=False, title_text="Distribution Analysis")
    return fig


def create_growth_trend_chart(df, numeric_cols):
    """
    Create growth trend visualization
    """
    if not numeric_cols or len(df) < 2:
        return None
    
    fig = go.Figure()
    
    for col in numeric_cols[:5]:
        if col in df.columns:
            values = df[col].dropna()
            if len(values) > 1:
                fig.add_trace(go.Scatter(
                    y=values.values,
                    mode='lines+markers',
                    name=col,
                    line=dict(width=2)
                ))
    
    fig.update_layout(
        title="Growth Trends Over Time",
        xaxis_title="Index",
        yaxis_title="Value",
        hovermode='x unified',
        height=400
    )
    
    return fig


def create_ai_insights_display(ai_insights):
    """
    Create a visual display of AI-powered insights
    """
    if not ai_insights:
        st.info("No AI insights available. Please configure Groq API key.")
        return
    
    for insight in ai_insights:
        priority_color = {
            'High': '#FF6B6B',
            'Medium': '#FFD93D',
            'Low': '#6BCB77'
        }.get(insight.get('priority', 'Medium'), '#667eea')
        
        st.markdown(f"""
        <div class="ai-insight-box" style="border-left: 5px solid {priority_color};">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h4>{insight.get('title', 'AI Insight')}</h4>
                <span class="ai-badge">🤖 AI</span>
            </div>
            <p><strong>Analysis:</strong> {insight.get('description', 'N/A')}</p>
            <p><strong>Recommendation:</strong> {insight.get('recommendation', 'N/A')}</p>
            <p><strong>Priority:</strong> {insight.get('priority', 'Medium')}</p>
        </div>
        """, unsafe_allow_html=True)


def create_insights_dashboard(insights_dict):
    """
    Create a visual dashboard of insights
    """
    for idx, (sheet_name, insights) in enumerate(insights_dict.items()):
        if not insights:
            continue
        
        with st.expander(f"📈 Insights - {sheet_name}", expanded=(idx == 0)):
            for insight in insights[:6]:
                st.markdown(f"""
                <div class="insight-box">
                    <strong>{insight['type']}</strong><br>
                    <span style="color: #667eea; font-size: 1.2rem;">{insight['value']}</span><br>
                    <small>{insight['insight']}</small>
                </div>
                """, unsafe_allow_html=True)


def main():
    # Header
    st.markdown('<div class="header-title">📊 Telecom & SaaS Data Intelligence Tool</div>', unsafe_allow_html=True)
    
    if HAS_GROQ:
        st.markdown("Analyze patterns, correlations, and generate **AI-powered insights** from your Telecom and SaaS data using Groq Llama 3")
    else:
        st.markdown("Analyze patterns, correlations, and generate insights from your Telecom and SaaS data")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Groq API Key (only show if Groq support is available)
        if HAS_GROQ:
            st.subheader("🤖 AI Configuration")
            api_key_input = st.text_input(
                "Groq API Key (Optional)",
                type="password",
                help="Get your API key from https://console.groq.com",
                value=st.session_state.groq_api_key or os.getenv('GROQ_API_KEY', '')
            )
            
            if api_key_input:
                st.session_state.groq_api_key = api_key_input
            
            st.divider()
        
        # File upload
        st.subheader("📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Upload Excel File",
            type=['xlsx', 'xls'],
            help="Upload an Excel file with one or multiple worksheets"
        )
        
        if uploaded_file:
            st.session_state.uploaded_file = uploaded_file
            st.success("✅ File uploaded successfully")
    
    # Main content
    if st.session_state.uploaded_file:
        # Load data
        with st.spinner("Loading and analyzing data..."):
            dataframes, error = load_excel_file(st.session_state.uploaded_file)
            
            if error:
                st.error(f"Error loading file: {error}")
                return
            
            # Initialize analyzer with Groq API key if available
            if HAS_GROQ:
                analyzer = TelecomSaaSAnalyzer(
                    dataframes,
                    groq_api_key=st.session_state.groq_api_key
                )
            else:
                analyzer = TelecomSaaSAnalyzer(dataframes)
            
            analyzer.analyze_all()
            st.session_state.analyzer = analyzer
        
        # Create tabs for different views
        if HAS_GROQ:
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "📊 Overview",
                "🔗 Correlations",
                "📈 Trends & Patterns",
                "💡 Insights",
                "🤖 AI Analysis",
                "📋 Raw Data"
            ])
        else:
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Overview",
                "🔗 Correlations",
                "📈 Trends & Patterns",
                "💡 Insights",
                "📋 Raw Data"
            ])
        
        # Tab 1: Overview
        with tab1:
            st.header("Data Overview")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Worksheets", len(dataframes))
            
            with col2:
                total_rows = sum(df.shape[0] for df in dataframes.values())
                st.metric("Total Rows", f"{total_rows:,}")
            
            with col3:
                total_cols = sum(df.shape[1] for df in dataframes.values())
                st.metric("Total Columns", total_cols)
            
            st.divider()
            
            # Dataset summary
            for sheet_name, df in dataframes.items():
                with st.expander(f"📑 {sheet_name} Details", expanded=True):
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Rows", df.shape[0])
                    with col2:
                        st.metric("Columns", df.shape[1])
                    with col3:
                        numeric_cols = df.select_dtypes(include=[np.number]).shape[1]
                        st.metric("Numeric Columns", numeric_cols)
                    with col4:
                        missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
                        st.metric("Missing Data", f"{missing_pct:.1f}%")
                    
                    # Column info
                    st.subheader("Column Information")
                    col_info = pd.DataFrame({
                        'Column': df.columns,
                        'Data Type': df.dtypes.astype(str),
                        'Non-Null Count': df.count(),
                        'Null Count': df.isnull().sum()
                    })
                    st.dataframe(col_info, use_container_width=True)
        
        # Tab 2: Correlations
        with tab2:
            st.header("Correlation Analysis")
            
            sheet_selection = st.selectbox(
                "Select Dataset",
                list(dataframes.keys()),
                key="corr_sheet"
            )
            
            if sheet_selection:
                corr_matrix = analyzer.get_correlation_matrix(sheet_selection)
                
                if not corr_matrix.empty:
                    # Heatmap
                    fig_heatmap = create_correlation_heatmap(corr_matrix, sheet_selection)
                    if fig_heatmap:
                        st.plotly_chart(fig_heatmap, use_container_width=True)
                    
                    st.divider()
                    
                    # Top correlations table
                    st.subheader("Top Correlations")
                    top_corr = analyzer.get_top_correlations(sheet_selection, top_n=15)
                    
                    if not top_corr.empty:
                        def color_correlation(val):
                            if val > 0.7:
                                return 'background-color: #90EE90'
                            elif val < -0.7:
                                return 'background-color: #FFB6C6'
                            else:
                                return ''
                        
                        styled_df = top_corr.style.applymap(
                            color_correlation,
                            subset=['Correlation']
                        )
                        st.dataframe(styled_df, use_container_width=True)
                else:
                    st.info("No numeric columns found for correlation analysis")
        
        # Tab 3: Trends & Patterns
        with tab3:
            st.header("Trends & Pattern Analysis")
            
            sheet_selection = st.selectbox(
                "Select Dataset",
                list(dataframes.keys()),
                key="trend_sheet"
            )
            
            if sheet_selection:
                df = dataframes[sheet_selection]
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                
                if numeric_cols:
                    # Growth trends
                    st.subheader("📈 Growth Trends")
                    fig_trends = create_growth_trend_chart(df, numeric_cols)
                    if fig_trends:
                        st.plotly_chart(fig_trends, use_container_width=True)
                    
                    st.divider()
                    
                    # Distribution analysis
                    st.subheader("📊 Distribution Analysis")
                    selected_cols = st.multiselect(
                        "Select columns for distribution analysis",
                        numeric_cols,
                        default=numeric_cols[:3]
                    )
                    
                    if selected_cols:
                        fig_dist = create_distribution_plots(df, selected_cols)
                        if fig_dist:
                            st.plotly_chart(fig_dist, use_container_width=True)
                    
                    st.divider()
                    
                    # Anomalies
                    st.subheader("⚠️ Anomalies Detected")
                    patterns = analyzer.patterns.get(sheet_selection, {})
                    anomalies = patterns.get('anomalies', {})
                    
                    if anomalies:
                        for col, anom_data in anomalies.items():
                            st.warning(f"**{col}**: {anom_data['count']} anomalies detected ({anom_data['percentage']:.1f}% of data)")
                    else:
                        st.success("No significant anomalies detected")
                else:
                    st.info("No numeric columns found for trend analysis")
        
        # Tab 4: Insights
        with tab4:
            st.header("💡 Domain-Specific Insights")
            
            # Create insights dashboard
            create_insights_dashboard(analyzer.insights)
        
        # Tab 5: AI Analysis (only if Groq is available)
        if HAS_GROQ:
            with tab5:
                st.header("🤖 AI-Powered Analysis with Groq Llama 3")
                
                if not st.session_state.groq_api_key:
                    st.warning("⚠️ Groq API key not configured. Please add your API key in the sidebar to enable AI insights.")
                    st.info("Get your free API key from [Groq Console](https://console.groq.com)")
                else:
                    st.success("✅ Groq API configured")
                    
                    # Display AI insights for each sheet
                    for sheet_name, ai_insights in analyzer.ai_insights.items():
                        with st.expander(f"🤖 AI Analysis - {sheet_name}", expanded=True):
                            if ai_insights:
                                create_ai_insights_display(ai_insights)
                            else:
                                st.info("Generating AI insights...")
        
        # Tab 6 or 5: Raw Data
        raw_data_tab = tab6 if HAS_GROQ else tab5
        with raw_data_tab:
            st.header("Raw Data Explorer")
            
            sheet_selection = st.selectbox(
                "Select Dataset",
                list(dataframes.keys()),
                key="raw_sheet"
            )
            
            if sheet_selection:
                df = dataframes[sheet_selection]
                
                # Display options
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    show_rows = st.slider("Rows to display", 5, min(100, len(df)), 20)
                
                with col2:
                    if st.checkbox("Show statistics"):
                        st.dataframe(df.describe(), use_container_width=True)
                
                with col3:
                    if st.checkbox("Show data types"):
                        st.dataframe(df.dtypes.astype(str), use_container_width=True)
                
                st.subheader("Data Preview")
                st.dataframe(df.head(show_rows), use_container_width=True)
                
                # Download option
                st.subheader("Export Data")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download as CSV",
                        data=csv,
                        file_name=f"{sheet_name}_export.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        df.to_excel(writer, sheet_name=sheet_selection, index=False)
                    buffer.seek(0)
                    st.download_button(
                        label="Download as Excel",
                        data=buffer,
                        file_name=f"{sheet_selection}_export.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
    
    else:
        # Welcome screen
        st.info("👈 Upload an Excel file to get started!")
        
        with st.expander("📖 How to use this tool", expanded=True):
            st.markdown("""
            ### Getting Started
            
            1. **Configure AI** (Optional): Add your Groq API key for AI-powered insights
            2. **Upload Your File**: Click the file uploader in the sidebar to upload an Excel file
            3. **Explore Overview**: View basic statistics about your datasets
            4. **Analyze Correlations**: Discover relationships between variables
            5. **Identify Trends**: Detect patterns and anomalies in your data
            6. **Get Insights**: Receive domain-specific insights
            7. **AI Analysis**: Get AI-powered recommendations using Groq Llama 3
            
            ### Supported Data Types
            
            - **Fiber/AirFiber**: Bandwidth, speed, latency, HSI data
            - **Voice/Call**: CSSR, ASR, MoU, call attempts, quality metrics
            - **Penetration**: User segments, penetration rates, customer counts
            - **JioJoin App**: Platform usage (Android, iOS, STB), video streaming, engagement
            - **Circle-wise**: Regional or geographic distribution analysis
            
            ### Features
            
            - ✅ Automatic pattern detection
            - ✅ Correlation analysis with heatmaps
            - ✅ Growth trend visualization
            - ✅ Anomaly detection
            - ✅ Distribution analysis
            - ✅ Domain-specific insights
            - ✅ **AI-powered analysis with Groq Llama 3** (if configured)
            - ✅ Data export capabilities
            """)
        
        if HAS_GROQ:
            with st.expander("🤖 About AI Integration"):
                st.markdown("""
                ### Groq Llama 3 Integration
                
                This tool uses **Groq's Llama 3 model** to provide intelligent, context-aware analysis of your telecom data.
                
                **Features:**
                - Automatic insight generation based on data patterns
                - Actionable recommendations for network optimization
                - Priority-based findings (High/Medium/Low)
                - Natural language analysis of complex metrics
                
                **Getting Started:**
                1. Visit [Groq Console](https://console.groq.com)
                2. Create a free account
                3. Generate an API key
                4. Add it in the sidebar configuration
                
                **Note:** Groq offers free API access with generous rate limits for development and testing.
                """)
        
        with st.expander("📊 Sample Data Structure"):
            st.markdown("""
            ### Example: Penetration & Voice Data
            | Circle | Penetration (%) | Non User | Low | Moderate | Heavy | CSSR (%) | ASR (%) | MoU (Mins) |
            |--------|-----------------|----------|-----|----------|-------|----------|---------|-----------|
            | Delhi | 25.9% | 154666 | 133681 | 43211 | 20538 | 99.2 | 99.6 | 46.91 |
            | Mumbai | 24.5% | 145000 | 125000 | 40000 | 19000 | 99.1 | 99.5 | 48.50 |
            
            ### Example: JioJoin App Data
            | Platform | Date | Android_Users | iOS_Users | STB_Users | Video_Streams | Engagement |
            |----------|------|---------------|-----------|-----------|---------------|-----------|
            | Android | 2024-01-01 | 150000 | 80000 | 20000 | 500000 | 0.85 |
            | iOS | 2024-01-01 | 150000 | 85000 | 22000 | 520000 | 0.88 |
            """)


if __name__ == "__main__":
    main()
