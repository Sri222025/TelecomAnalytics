import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
import traceback

# Import our custom modules
from file_processor import FileProcessor
from data_merger import DataMerger
import ai_insights_engine

# Must be FIRST Streamlit command
st.set_page_config(
    page_title="Telecom Analytics Platform",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .critical-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []
if 'merged_data' not in st.session_state:
    st.session_state.merged_data = None
if 'merge_summary' not in st.session_state:
    st.session_state.merge_summary = None
if 'ai_insights' not in st.session_state:
    st.session_state.ai_insights = None
if 'relationships' not in st.session_state:
    st.session_state.relationships = []
if 'ai_error' not in st.session_state:
    st.session_state.ai_error = None
if 'last_uploaded_files' not in st.session_state:
    st.session_state.last_uploaded_files = []

# Initialize processors
file_processor = FileProcessor()
data_merger = DataMerger()

# Check for API keys
has_groq = 'GROQ_API_KEY' in st.secrets

# Title
st.markdown('<h1 class="main-header">📡 Telecom Analytics Platform</h1>', unsafe_allow_html=True)
st.markdown("### Simple, Powerful Analytics for Telecom Data | Upload → Analyze → Get Insights")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=80)
    st.title("Navigation")
    
    page = st.radio(
        "Select Page",
        ["🏠 Home", "📤 Upload & Process", "📊 AI Insights", "📈 Visualizations", "💾 Export"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Status indicators
    st.subheader("Status")
    if st.session_state.merged_data is not None:
        st.success("✅ Data Loaded")
        st.metric("Records", f"{len(st.session_state.merged_data):,}")
        
        if st.session_state.ai_insights is not None:
            st.success("✅ AI Analysis Done")
        elif st.session_state.ai_error:
            st.error("❌ AI Failed")
    else:
        st.info("⏳ No data loaded")
    
    if has_groq:
        st.success("✅ AI Enabled")
    else:
        st.warning("⚠️ AI Disabled")
    
    st.markdown("---")
    st.caption("V10 Final - Datetime Filtering Active 🚀")

# ============================================================================
# HOME PAGE
# ============================================================================
if page == "🏠 Home":
    st.header("Welcome to Telecom Analytics Platform")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 What This Platform Does
        
        - **📁 Process Multiple Files** - Upload Excel/CSV files
        - **🔗 Auto-Merge Data** - Intelligent relationship detection
        - **🤖 AI Analysis** - Deep telecom insights
        - **📊 Visual Analytics** - Interactive charts
        
        ### 🚀 Quick Start
        1. Go to **📤 Upload & Process**
        2. Upload your Excel/CSV files
        3. Click **Process & Analyze**
        4. Review **AI Insights**
        """)
    
    with col2:
        st.markdown("### 📊 Current Status")
        
        if st.session_state.merged_data is not None:
            st.success("✅ Data Ready")
            st.metric("Total Records", f"{len(st.session_state.merged_data):,}")
            st.metric("Columns", len(st.session_state.merged_data.columns))
            
            if st.button("🔄 Process New Files", type="primary"):
                st.session_state.processed_files = []
                st.session_state.merged_data = None
                st.session_state.ai_insights = None
                st.session_state.ai_error = None
                st.rerun()
        else:
            st.info("👈 Upload files to get started")

# ============================================================================
# UPLOAD & PROCESS PAGE
# ============================================================================
elif page == "📤 Upload & Process":
    st.header("📤 Upload and Process Files")
    
    uploaded_files = st.file_uploader(
        "Upload your Excel or CSV files",
        type=['xlsx', 'xls', 'csv'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        # Detect if files changed
        current_files = [f.name for f in uploaded_files]
        
        if current_files != st.session_state.last_uploaded_files:
            st.session_state.merged_data = None
            st.session_state.ai_insights = None
            st.session_state.ai_error = None
            st.session_state.processed_files = []
            st.session_state.relationships = []
            st.session_state.last_uploaded_files = current_files
        
        st.info(f"📁 **{len(uploaded_files)} file(s) selected**")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if st.button("🚀 Process & Analyze", type="primary", use_container_width=True):
                # Clear state
                st.session_state.merged_data = None
                st.session_state.ai_insights = None
                st.session_state.ai_error = None
                
                with st.spinner("🔄 Processing..."):
                    try:
                        # Process files
                        processed_files = []
                        progress_bar = st.progress(0)
                        
                        for idx, file in enumerate(uploaded_files):
                            st.text(f"Processing: {file.name}")
                            file_info = file_processor.process_file(file)
                            processed_files.append(file_info)
                            progress_bar.progress((idx + 1) / len(uploaded_files))
                        
                        st.session_state.processed_files = processed_files
                        
                        # Merge data
                        relationships = data_merger.detect_relationships(processed_files)
                        merged_data, merge_summary = data_merger.merge_files(processed_files, relationships)
                        st.session_state.merged_data = merged_data
                        st.session_state.merge_summary = merge_summary
                        
                        st.success(f"✅ Merged: {len(merged_data):,} records")
                        
                        # AI Analysis
                        if has_groq:
                            st.text("🤖 AI analyzing...")
                            try:
                                groq_key = st.secrets.get('GROQ_API_KEY')
                                insights = ai_insights_engine.analyze_data(
                                    merged_data, 
                                    groq_api_key=groq_key,
                                    context=merge_summary
                                )
                                st.session_state.ai_insights = insights
                                st.success("✅ AI complete!")
                            except Exception as e:
                                error_msg = str(e)
                                st.session_state.ai_error = error_msg
                                st.error(f"⚠️ AI failed: {error_msg}")
                        else:
                            st.warning("⚠️ AI disabled (no API key)")
                        
                        progress_bar.progress(1.0)
                        st.success("✅ Done!")
                        st.balloons()
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        st.exception(e)
        
        with col2:
            if st.button("Clear All"):
                st.session_state.processed_files = []
                st.session_state.merged_data = None
                st.session_state.ai_insights = None
                st.session_state.ai_error = None
                st.rerun()
    
    # Show results
    if st.session_state.merged_data is not None:
        st.markdown("---")
        st.subheader("✅ Processing Summary")
        
        summary = st.session_state.merge_summary
        col1, col2, col3 = st.columns(3)
        col1.metric("Files", summary.get('files_processed', 0))
        col2.metric("Records", f"{summary.get('total_records', 0):,}")
        col3.metric("Columns", summary.get('columns', 0))
        
        with st.expander("👀 Data Preview"):
            st.dataframe(st.session_state.merged_data.head(20), use_container_width=True)

# ============================================================================
# AI INSIGHTS PAGE - FIXED FOR V9
# ============================================================================
elif page == "📊 AI Insights":
    st.header("📊 AI-Powered Insights")
    
    if st.session_state.ai_insights:
        ai_insights = st.session_state.ai_insights
        
        # Executive Summary
        st.markdown("### 📝 Executive Summary")
        if 'executive_summary' in ai_insights:
            st.text(ai_insights['executive_summary'])
        
        # Critical issues overview
        st.markdown("### 🚨 Critical Issues Overview")
        
        problems = ai_insights.get('problems', [])
        critical_count = len([p for p in problems if p.get('severity') == 'critical'])
        high_count = len([p for p in problems if p.get('severity') == 'high'])
        medium_count = len([p for p in problems if p.get('severity') == 'medium'])
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🔴 Critical", critical_count)
        col2.metric("🟠 High", high_count)
        col3.metric("🟡 Medium", medium_count)
        col4.metric("Total", len(problems))
        
        # Key Insights
        st.markdown("### 💡 Key Business Insights")
        st.info("💼 Board-Ready Insights - Ready to share with management")
        
        if 'key_insights' in ai_insights and ai_insights['key_insights']:
            st.markdown(ai_insights['key_insights'])
        
        # Recommendations - FIXED FOR V9
        st.markdown("### 🎯 Recommendations")
        
        recommendations = ai_insights.get('recommendations', [])
        
        if isinstance(recommendations, str):
            # String format
            st.markdown(recommendations)
        elif isinstance(recommendations, list):
            # List format - FIXED
            for idx, rec in enumerate(recommendations, 1):
                priority = rec.get('priority', 'MEDIUM')
                
                if priority == 'CRITICAL':
                    emoji = '🔴'
                elif priority == 'HIGH':
                    emoji = '🟡'
                else:
                    emoji = '⚪'
                
                with st.expander(f"{emoji} Recommendation {idx}: {rec.get('title', 'N/A')} [{priority}]", expanded=(idx==1)):
                    st.markdown(f"**Problem:** {rec.get('problem', 'N/A')}")
                    
                    affected = rec.get('affected_circles', [])
                    if affected:
                        st.markdown(f"**Affected Circles:** {', '.join(affected[:5])}")
                    
                    st.markdown(f"**Impact:** {rec.get('impact', 'N/A')}")
                    
                    st.markdown("**Actions:**")
                    actions = rec.get('actions', [])
                    if isinstance(actions, list):
                        for action in actions:
                            st.markdown(f"- {action}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("💰 Investment", rec.get('investment', 'TBD'))
                    with col2:
                        st.metric("⏱️ ROI", rec.get('roi', 'TBD'))
                    
                    st.success(f"**Expected:** {rec.get('expected_result', 'TBD')}")
        
        # Circle Analysis
        st.markdown("### 📍 Circle-by-Circle Analysis")
        
        circle_data = ai_insights.get('circle_analysis', [])
        if circle_data:
            # Priority filter
            all_priorities = list(set([c.get('priority', 'normal') for c in circle_data]))
            priority_filter = st.multiselect(
                "Filter by Priority",
                options=all_priorities,
                default=all_priorities
            )
            
            filtered = [c for c in circle_data if c.get('priority', 'normal') in priority_filter]
            st.info(f"Showing {len(filtered)} of {len(circle_data)} circles")
            
            for circle in filtered[:10]:
                priority = circle.get('priority', 'normal').upper()
                circle_name = circle.get('circle', 'Unknown')
                
                priority_emoji = {'CRITICAL': '🔴', 'HIGH': '🟡', 'NORMAL': '🟢'}.get(priority, '⚪')
                
                with st.expander(f"{priority_emoji} **{circle_name}** [{priority}]", expanded=(priority=='CRITICAL')):
                    # Metrics
                    metrics = circle.get('metrics', {})
                    if metrics:
                        st.markdown("**📊 Metrics:**")
                        metric_items = list(metrics.items())
                        for i in range(0, len(metric_items), 3):
                            cols = st.columns(3)
                            for j, (name, value) in enumerate(metric_items[i:i+3]):
                                with cols[j]:
                                    if isinstance(value, dict):
                                        display = value.get('value', 'N/A')
                                    else:
                                        display = value
                                    st.metric(str(name)[:25], str(display))
                    
                    # Problems
                    circle_problems = circle.get('problems', [])
                    if circle_problems:
                        st.markdown("**⚠️ Issues:**")
                        for prob in circle_problems:
                            severity = prob.get('severity', 'medium')
                            sev_emoji = {'critical': '🔴', 'high': '🟡', 'medium': '🟠'}.get(severity, '🔵')
                            
                            metric = prob.get('metric', 'Unknown')
                            value = prob.get('value', 'N/A')
                            target = prob.get('target', 'N/A')
                            
                            text = f"{sev_emoji} **{metric}**: {value} (Target: {target})"
                            
                            if severity == 'critical':
                                st.error(text)
                            elif severity == 'high':
                                st.warning(text)
                            else:
                                st.info(text)
                    else:
                        st.success("✅ No critical issues")
        
        # Problems Table
        if problems:
            st.markdown("### 📋 All Problems")
            
            problems_data = []
            for p in problems:
                problems_data.append({
                    'Circle': p.get('circle', 'Unknown'),
                    'Metric': p.get('metric', 'Unknown'),
                    'Current': f"{p.get('value', 'N/A')}",
                    'Target': f"{p.get('target', 'N/A')}",
                    'Gap': f"{p.get('gap', 'N/A')}",
                    'Severity': p.get('severity', 'medium').upper()
                })
            
            if problems_data:
                df_problems = pd.DataFrame(problems_data)
                st.dataframe(df_problems, use_container_width=True, height=400)
        
        # Network Summary
        if 'network_summary' in ai_insights:
            st.markdown("### 🌐 Network Summary")
            network = ai_insights['network_summary']
            
            cols = st.columns(4)
            col_idx = 0
            for key, value in network.items():
                if isinstance(value, (int, float)) and key != 'total_circles':
                    with cols[col_idx % 4]:
                        label = key.replace('_', ' ').title()
                        if isinstance(value, float):
                            display = f"{value:,.1f}" if value > 1000 else f"{value:.1f}"
                        else:
                            display = f"{value:,}"
                        st.metric(label, display)
                        col_idx += 1
    
    elif st.session_state.ai_error:
        st.error(f"AI Error: {st.session_state.ai_error}")
        st.info("Please try processing again")
    
    else:
        st.info("👆 Please upload and process files first")

# ============================================================================
# VISUALIZATIONS PAGE
# ============================================================================
elif page == "📈 Visualizations":
    st.header("📈 Visualizations")
    
    if st.session_state.merged_data is None:
        st.warning("⚠️ Please upload and process files first")
    else:
        df = st.session_state.merged_data
        
        numeric_cols = [c for c in df.select_dtypes(include=['number']).columns if not c.startswith('_')]
        categorical_cols = [c for c in df.select_dtypes(include=['object']).columns if not c.startswith('_')]
        
        if not numeric_cols and not categorical_cols:
            st.info("No columns for visualization")
        else:
            chart_type = st.selectbox("Chart Type", ["Bar Chart", "Pie Chart", "Histogram"])
            
            try:
                if chart_type == "Bar Chart" and categorical_cols and numeric_cols:
                    col1, col2 = st.columns(2)
                    with col1:
                        x_col = st.selectbox("Category", categorical_cols)
                    with col2:
                        y_col = st.selectbox("Metric", numeric_cols)
                    
                    if x_col and y_col:
                        data = df.groupby(x_col)[y_col].sum().reset_index()
                        data = data.nlargest(15, y_col)
                        fig = px.bar(data, x=x_col, y=y_col)
                        st.plotly_chart(fig, use_container_width=True)
                
                elif chart_type == "Pie Chart" and categorical_cols:
                    cat_col = st.selectbox("Category", categorical_cols)
                    if cat_col:
                        data = df[cat_col].value_counts().head(10)
                        fig = px.pie(values=data.values, names=data.index)
                        st.plotly_chart(fig, use_container_width=True)
                
                elif chart_type == "Histogram" and numeric_cols:
                    num_col = st.selectbox("Column", numeric_cols)
                    if num_col:
                        fig = px.histogram(df, x=num_col)
                        st.plotly_chart(fig, use_container_width=True)
            
            except Exception as e:
                st.error(f"Error: {str(e)}")

# ============================================================================
# EXPORT PAGE
# ============================================================================
elif page == "💾 Export":
    st.header("💾 Export Data")
    
    if st.session_state.merged_data is None:
        st.warning("⚠️ Please upload and process files first")
    else:
        df = st.session_state.merged_data
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv = df.to_csv(index=False)
            st.download_button(
                "📄 Download CSV",
                csv,
                f"telecom_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                use_container_width=True
            )
        
        with col2:
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df.to_excel(writer, index=False)
            
            st.download_button(
                "📊 Download Excel",
                buffer.getvalue(),
                f"telecom_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
