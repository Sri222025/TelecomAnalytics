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
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
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
if 'processing_log' not in st.session_state:
    st.session_state.processing_log = []

# Initialize processors
file_processor = FileProcessor()
data_merger = DataMerger()

# Check for API keys
has_groq = 'GROQ_API_KEY' in st.secrets if hasattr(st, 'secrets') else False
has_gemini = 'GEMINI_API_KEY' in st.secrets if hasattr(st, 'secrets') else False

# Title
st.markdown('<h1 class="main-header">📡 Telecom Analytics Platform</h1>', unsafe_allow_html=True)
st.markdown("### AI-Powered Data Analysis for Telecom Operations")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=80)
    st.title("Navigation")
    
    page = st.radio(
        "Select Page",
        ["🏠 Home", "📤 Upload & Process", "🤖 AI Insights", "🚨 Alerts & Anomalies", 
         "📊 Data Explorer", "📈 Visualizations", "💾 Export"],
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
            insights_count = len(st.session_state.ai_insights.get('key_insights', []))
            if insights_count > 0:
                st.metric("Insights", insights_count)
        elif st.session_state.ai_error:
            st.error("❌ AI Failed")
    else:
        st.info("⏳ No data loaded")
    
    if has_groq:
        st.success("✅ AI Enabled (Groq)")
    else:
        st.warning("⚠️ AI Disabled")
    
    st.markdown("---")
    st.caption("Powered by Groq Llama 3.3 🚀")
    st.caption("v6.0 - Enhanced Edition")

# ============================================================================
# HOME PAGE
# ============================================================================
if page == "🏠 Home":
    st.header("Welcome to Telecom Analytics Platform")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 What This Platform Does
        
        This AI-powered platform helps you:
        - **📁 Process Multiple Files** - Upload 3-4 Excel/CSV files with multiple sheets
        - **🔗 Auto-Merge Data** - Intelligent detection of relationships between files
        - **🤖 AI Analysis** - Deep telecom insights powered by enhanced analytics engine
        - **🚨 Anomaly Detection** - Business-critical issues identification
        - **📊 Visual Analytics** - Interactive charts and dashboards
        - **💡 Smart Recommendations** - Specific, actionable business plans
        
        ### 🚀 Quick Start
        1. Go to **📤 Upload & Process**
        2. Upload your Excel/CSV files (3-4 files recommended)
        3. Click **Process & Analyze**
        4. Review **AI Insights** and **Alerts**
        5. Explore visualizations and export results
        
        ### ✨ New in V6
        - Enhanced structure detection
        - Improved problem identification
        - Better data cleaning
        - More robust error handling
        """)
    
    with col2:
        st.markdown("### 📊 Current Status")
        
        if st.session_state.merged_data is not None:
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.success("✅ Data Ready")
            st.metric("Total Records", f"{len(st.session_state.merged_data):,}")
            st.metric("Columns", len(st.session_state.merged_data.columns))
            st.metric("Files Processed", len(st.session_state.processed_files))
            
            if st.session_state.ai_insights:
                insights = st.session_state.ai_insights
                st.metric("AI Insights", len(insights.get('key_insights', [])))
                problems = insights.get('problems', [])
                if problems:
                    critical = len([p for p in problems if p.get('severity') == 'critical'])
                    if critical > 0:
                        st.error(f"🔴 {critical} Critical Issues")
            elif st.session_state.ai_error:
                st.error(f"AI Error: {st.session_state.ai_error[:50]}...")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            if st.button("🔄 Process New Files", type="primary"):
                st.session_state.processed_files = []
                st.session_state.merged_data = None
                st.session_state.merge_summary = None
                st.session_state.ai_insights = None
                st.session_state.ai_error = None
                st.session_state.relationships = []
                st.rerun()
        else:
            st.info("👈 Upload files to get started")

# ============================================================================
# UPLOAD & PROCESS PAGE
# ============================================================================
elif page == "📤 Upload & Process":
    st.header("📤 Upload and Process Files")
    
    uploaded_files = st.file_uploader(
        "Upload your Excel or CSV files (3-4 files recommended)",
        type=['xlsx', 'xls', 'csv'],
        accept_multiple_files=True,
        help="Upload multiple files. The system will auto-detect relationships."
    )
    
    if uploaded_files:
        # Detect if files changed - clear old results
        current_files = [f.name for f in uploaded_files]
        if 'last_uploaded_files' not in st.session_state:
            st.session_state.last_uploaded_files = []
        
        if current_files != st.session_state.last_uploaded_files:
            # Files changed - clear ALL state
            st.session_state.merged_data = None
            st.session_state.ai_insights = None
            st.session_state.ai_error = None
            st.session_state.processed_files = []
            st.session_state.relationships = []
            st.session_state.last_uploaded_files = current_files
        
        st.info(f"📁 **{len(uploaded_files)} file(s) selected:** {', '.join([f.name for f in uploaded_files])}")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if st.button("🚀 Process & Analyze", type="primary", use_container_width=True):
                
                # CLEAR ALL OLD STATE
                st.session_state.merged_data = None
                st.session_state.ai_insights = None
                st.session_state.ai_error = None
                st.session_state.processed_files = []
                st.session_state.relationships = []
                st.session_state.processing_log = []
                
                with st.spinner("🔄 Processing files..."):
                    try:
                        # Process each file
                        processed_files = []
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for idx, file in enumerate(uploaded_files):
                            status_text.text(f"Processing: {file.name}")
                            try:
                                file_info = file_processor.process_file(file)
                                processed_files.append(file_info)
                                st.session_state.processing_log.append(f"✅ Processed {file.name}: {file_info.get('total_rows', 0)} rows")
                            except Exception as e:
                                error_msg = f"❌ Error processing {file.name}: {str(e)}"
                                st.session_state.processing_log.append(error_msg)
                                st.warning(error_msg)
                            
                            progress_bar.progress((idx + 1) / len(uploaded_files))
                        
                        if not processed_files:
                            st.error("❌ No files were successfully processed!")
                            st.stop()
                        
                        st.session_state.processed_files = processed_files
                        status_text.text("🔍 Detecting relationships...")
                        
                        # Detect relationships
                        relationships = data_merger.detect_relationships(processed_files)
                        st.session_state.relationships = relationships
                        
                        if relationships:
                            st.info(f"✅ Found {len(relationships)} potential relationship(s)")
                            with st.expander("View Relationships"):
                                for rel in relationships[:5]:
                                    st.write(f"- {rel['file1']} ↔ {rel['file2']} via '{rel['key_column']}' ({rel['match_rate']:.1f}% match)")
                        
                        # Merge data
                        status_text.text("🔗 Merging data...")
                        merged_data, merge_summary = data_merger.merge_files(processed_files, relationships)
                        
                        if merged_data is None or len(merged_data) == 0:
                            st.error("❌ No data to merge!")
                            st.stop()
                        
                        st.session_state.merged_data = merged_data
                        st.session_state.merge_summary = merge_summary
                        
                        st.success(f"✅ Data merged: {len(merged_data):,} records")
                        
                        # AI Analysis with ACTUAL DATA
                        status_text.text("🤖 Running AI analysis...")
                        try:
                            insights = ai_insights_engine.analyze_data(merged_data, merge_summary)
                            st.session_state.ai_insights = insights
                            
                            # Show quick summary
                            if insights.get('problems'):
                                problems = insights['problems']
                                critical = len([p for p in problems if p.get('severity') == 'critical'])
                                if critical > 0:
                                    st.error(f"🚨 Found {critical} critical issue(s)!")
                                else:
                                    st.success(f"✅ Analysis complete: {len(problems)} issue(s) identified")
                            else:
                                st.success("✅ AI analysis complete!")
                                
                        except Exception as e:
                            error_msg = f"{str(e)}\n\nFull traceback:\n{traceback.format_exc()}"
                            st.session_state.ai_error = error_msg
                            st.error(f"⚠️ AI analysis failed: {str(e)}")
                            with st.expander("🔍 Error Details"):
                                st.code(error_msg)
                            st.info("💡 You can still use other features (Data Explorer, Visualizations, Export)")
                        
                        progress_bar.progress(1.0)
                        status_text.text("✅ Processing complete!")
                        st.success("✅ Processing complete!")
                        st.balloons()
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        with st.expander("🔍 Full Error Details"):
                            st.code(traceback.format_exc())
        
        with col2:
            if st.button("Clear All", type="secondary"):
                st.session_state.processed_files = []
                st.session_state.merged_data = None
                st.session_state.ai_insights = None
                st.session_state.ai_error = None
                st.session_state.relationships = []
                st.rerun()
        
        # Show processing log
        if st.session_state.processing_log:
            with st.expander("📋 Processing Log"):
                for log_entry in st.session_state.processing_log:
                    st.text(log_entry)
    
    # Show results
    if st.session_state.merged_data is not None:
        st.markdown("---")
        st.subheader("✅ Processing Summary")
        
        summary = st.session_state.merge_summary
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Files Processed", summary.get('files_processed', 0))
        col2.metric("Total Records", f"{summary.get('total_records', 0):,}")
        col3.metric("Columns", summary.get('columns', 0))
        col4.metric("Method", summary.get('method', 'unknown').title())
        
        # Merge details
        with st.expander("📋 Merge Details", expanded=True):
            if summary.get('method') == 'merge':
                st.success(f"✅ Files connected via: **{summary.get('merge_key')}**")
                st.info(f"Match rate: {summary.get('match_rate', 'N/A')}")
            else:
                st.info(summary.get('note', 'Data concatenated'))
        
        # Data preview
        with st.expander("👀 Data Preview"):
            st.dataframe(st.session_state.merged_data.head(20), use_container_width=True)

# ============================================================================
# AI INSIGHTS PAGE
# ============================================================================
elif page == "🤖 AI Insights":
    st.header("🤖 AI-Powered Insights")
    
    if st.session_state.merged_data is None:
        st.warning("⚠️ Please upload and process files first")
    elif st.session_state.ai_insights is None:
        if st.session_state.ai_error:
            st.error("❌ AI analysis failed during processing")
            with st.expander("🔍 View Error Details"):
                st.code(st.session_state.ai_error)
            st.info("💡 Try processing files again or check your data format")
        else:
            st.info("⏳ AI analysis not yet run. Go to Upload & Process page and click 'Process & Analyze'")
    else:
        insights = st.session_state.ai_insights
        
        # Executive Summary
        st.subheader("📝 Executive Summary")
        summary = insights.get("executive_summary", "No summary available")
        st.markdown(f'<div class="success-box">{summary}</div>', unsafe_allow_html=True)
        
        # Problems Overview
        problems = insights.get('problems', [])
        if problems:
            st.markdown("---")
            st.subheader("🚨 Issues Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            critical = len([p for p in problems if p.get('severity') == 'critical'])
            high = len([p for p in problems if p.get('severity') == 'high'])
            medium = len([p for p in problems if p.get('severity') == 'medium'])
            
            col1.metric("🔴 Critical", critical)
            col2.metric("🟠 High", high)
            col3.metric("🟡 Medium", medium)
            col4.metric("Total Issues", len(problems))
            
            # Show top problems
            with st.expander("📋 All Issues", expanded=False):
                for idx, problem in enumerate(problems, 1):
                    severity = problem.get('severity', 'medium')
                    icon = "🔴" if severity == 'critical' else "🟠" if severity == 'high' else "🟡"
                    st.markdown(f"**{idx}. {icon} {problem.get('circle', 'Unknown')}** - {problem.get('metric', 'N/A')}: {problem.get('value', 'N/A')}")
        
        # Key Insights
        st.markdown("---")
        st.subheader("💡 Key Insights")
        
        key_insights = insights.get('key_insights', [])
        if key_insights:
            for idx, insight in enumerate(key_insights, 1):
                impact = insight.get('impact', 'medium')
                icon = "🔴" if impact == 'critical' else "🟠" if impact == 'high' else "🟡" if impact == 'medium' else "🟢"
                
                with st.expander(f"{icon} Insight {idx}: {insight.get('title', 'N/A')}", expanded=(idx <= 2)):
                    st.write(insight.get('description', ''))
                    
                    # Show metrics if available
                    if 'metrics' in insight:
                        metrics = insight['metrics']
                        col1, col2, col3 = st.columns(3)
                        if 'key_number' in metrics:
                            col1.metric("Key Metric", metrics['key_number'])
                        if 'percentage' in metrics:
                            col2.metric("Percentage", metrics['percentage'])
                        if 'comparison' in metrics:
                            col3.metric("Comparison", metrics['comparison'])
                    
                    # Show action if available
                    if 'action' in insight and insight['action']:
                        st.markdown(f"**🎯 Recommended Action:**")
                        st.info(insight['action'])
                    
                    st.caption(f"Impact: {impact.upper()} | Category: {insight.get('category', 'general')}")
        else:
            st.info("No specific insights generated")
        
        # Recommendations
        st.markdown("---")
        st.subheader("🎯 Action Plan")
        
        recommendations = insights.get('recommendations', [])
        if recommendations:
            for idx, rec in enumerate(recommendations, 1):
                priority = rec.get('priority', 'medium')
                color = 'critical-box' if priority == 'critical' else 'warning-box' if priority == 'high' else 'success-box'
                
                st.markdown(f'<div class="{color}">', unsafe_allow_html=True)
                st.markdown(f"**{idx}. {rec.get('category', 'General')}** (Priority: {priority.upper()})")
                st.write(rec.get('action', ''))
                
                if 'rationale' in rec:
                    st.caption(f"Rationale: {rec['rationale']}")
                
                if isinstance(rec.get('details'), list):
                    for detail in rec['details']:
                        st.write(f"- {detail}")
                elif rec.get('details'):
                    st.write(rec['details'])
                
                if 'expected_impact' in rec:
                    st.caption(f"💰 Expected Impact: {rec['expected_impact']}")
                
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("No recommendations generated")

# ============================================================================
# ALERTS & ANOMALIES PAGE  
# ============================================================================
elif page == "🚨 Alerts & Anomalies":
    st.header("🚨 Alerts & Anomalies")
    
    if st.session_state.merged_data is None:
        st.warning("⚠️ Please upload and process files first")
    elif st.session_state.ai_insights is None:
        st.info("⏳ Run AI analysis first")
    else:
        problems = st.session_state.ai_insights.get('problems', [])
        
        if not problems:
            st.success("✅ No critical anomalies detected! Data quality looks good.")
        else:
            # Summary
            critical = len([a for a in problems if a.get('severity') == 'critical'])
            high = len([a for a in problems if a.get('severity') == 'high'])
            medium = len([a for a in problems if a.get('severity') == 'medium'])
            
            col1, col2, col3 = st.columns(3)
            col1.metric("🔴 Critical", critical)
            col2.metric("🟠 High", high)
            col3.metric("🟡 Medium", medium)
            
            st.markdown("---")
            
            # Display anomalies grouped by severity
            for severity in ['critical', 'high', 'medium']:
                severity_problems = [p for p in problems if p.get('severity') == severity]
                if severity_problems:
                    st.subheader(f"{'🔴' if severity == 'critical' else '🟠' if severity == 'high' else '🟡'} {severity.upper()} Issues")
                    
                    for problem in severity_problems:
                        box_class = 'critical-box' if severity == 'critical' else 'warning-box' if severity == 'medium' else 'warning-box'
                        
                        st.markdown(f'<div class="{box_class}">', unsafe_allow_html=True)
                        st.markdown(f"### {problem.get('circle', 'Unknown Circle')}")
                        st.write(f"**Type:** {problem.get('type', 'Unknown').title()}")
                        st.write(f"**Metric:** {problem.get('metric', 'N/A')}")
                        st.write(f"**Value:** {problem.get('value', 'N/A')}")
                        
                        if problem.get('gap'):
                            st.write(f"**Gap:** {problem['gap']:.2f} points")
                        if problem.get('calls_affected'):
                            st.write(f"**Calls Affected:** {problem['calls_affected']:,} daily")
                        if problem.get('revenue_loss'):
                            st.write(f"**Revenue Risk:** ₹{problem['revenue_loss']:.1f}L monthly")
                        if problem.get('overload_pct'):
                            st.write(f"**Overload:** {problem['overload_pct']:.1f}% above average")
                        
                        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# DATA EXPLORER PAGE
# ============================================================================
elif page == "📊 Data Explorer":
    st.header("📊 Data Explorer")
    
    if st.session_state.merged_data is None:
        st.warning("⚠️ Please upload and process files first")
    else:
        df = st.session_state.merged_data
        
        # Quick stats
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Records", f"{len(df):,}")
        col2.metric("Columns", len(df.columns))
        col3.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        col4.metric("Completeness", f"{(df.notna().sum().sum() / (len(df) * len(df.columns)) * 100):.1f}%")
        
        st.markdown("---")
        
        # Column selector
        all_cols = [col for col in df.columns if not col.startswith('_')]
        
        col1, col2 = st.columns([3, 1])
        with col1:
            selected_cols = st.multiselect(
                "Select columns to display",
                all_cols,
                default=all_cols[:10] if len(all_cols) > 10 else all_cols
            )
        
        with col2:
            rows_to_show = st.number_input("Rows to display", 10, 1000, 100)
        
        # Display data
        if selected_cols:
            st.dataframe(df[selected_cols].head(rows_to_show), use_container_width=True)
        
        # Column statistics
        with st.expander("📊 Column Statistics"):
            col_info = []
            for col in all_cols:
                col_info.append({
                    'Column': col,
                    'Type': str(df[col].dtype),
                    'Non-Null': df[col].notna().sum(),
                    'Null': df[col].isna().sum(),
                    'Unique': df[col].nunique()
                })
            st.dataframe(pd.DataFrame(col_info), use_container_width=True)

# ============================================================================
# VISUALIZATIONS PAGE
# ============================================================================
elif page == "📈 Visualizations":
    st.header("📈 Visualizations")
    
    if st.session_state.merged_data is None:
        st.warning("⚠️ Please upload and process files first")
    else:
        df = st.session_state.merged_data
        
        # Get numeric and categorical columns
        numeric_cols = [c for c in df.select_dtypes(include=['number']).columns if not c.startswith('_')]
        categorical_cols = [c for c in df.select_dtypes(include=['object']).columns if not c.startswith('_')]
        
        if not numeric_cols and not categorical_cols:
            st.info("No suitable columns for visualization. Upload data with numeric or categorical columns.")
        else:
            # Chart type selector
            chart_type = st.selectbox(
                "Select Chart Type",
                ["Bar Chart", "Line Chart", "Pie Chart", "Histogram", "Scatter Plot"]
            )
            
            try:
                if chart_type == "Bar Chart" and categorical_cols and numeric_cols:
                    col1, col2 = st.columns(2)
                    with col1:
                        x_col = st.selectbox("X-axis (Category)", categorical_cols)
                    with col2:
                        y_col = st.selectbox("Y-axis (Metric)", numeric_cols)
                    
                    if x_col and y_col:
                        chart_data = df.groupby(x_col)[y_col].sum().reset_index()
                        chart_data = chart_data.nlargest(15, y_col)
                        fig = px.bar(chart_data, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
                        st.plotly_chart(fig, use_container_width=True)
                
                elif chart_type == "Pie Chart" and categorical_cols:
                    cat_col = st.selectbox("Category", categorical_cols)
                    if cat_col:
                        chart_data = df[cat_col].value_counts().head(10)
                        fig = px.pie(values=chart_data.values, names=chart_data.index, 
                                    title=f"Distribution of {cat_col}")
                        st.plotly_chart(fig, use_container_width=True)
                
                elif chart_type == "Histogram" and numeric_cols:
                    num_col = st.selectbox("Column", numeric_cols)
                    if num_col:
                        fig = px.histogram(df, x=num_col, title=f"Distribution of {num_col}")
                        st.plotly_chart(fig, use_container_width=True)
                
                elif chart_type == "Line Chart":
                    col1, col2 = st.columns(2)
                    all_cols = [c for c in df.columns if not c.startswith('_')]
                    with col1:
                        x_col = st.selectbox("X-axis", all_cols)
                    with col2:
                        y_col = st.selectbox("Y-axis", numeric_cols) if numeric_cols else None
                    
                    if x_col and y_col:
                        fig = px.line(df.head(100), x=x_col, y=y_col, title=f"{y_col} over {x_col}")
                        st.plotly_chart(fig, use_container_width=True)
                
                elif chart_type == "Scatter Plot" and numeric_cols:
                    col1, col2 = st.columns(2)
                    with col1:
                        x_col = st.selectbox("X-axis", numeric_cols)
                    with col2:
                        y_col = st.selectbox("Y-axis", numeric_cols)
                    
                    if x_col and y_col:
                        fig = px.scatter(df.head(500), x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
                        st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error creating visualization: {str(e)}")
                st.info("Try selecting different columns or chart type")

# ============================================================================
# EXPORT PAGE
# ============================================================================
elif page == "💾 Export":
    st.header("💾 Export Data")
    
    if st.session_state.merged_data is None:
        st.warning("⚠️ Please upload and process files first")
    else:
        df = st.session_state.merged_data
        
        st.subheader("📥 Download Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # CSV Export
            csv = df.to_csv(index=False)
            st.download_button(
                label="📄 Download as CSV",
                data=csv,
                file_name=f"telecom_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # Excel Export
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Data')
            
            st.download_button(
                label="📊 Download as Excel",
                data=buffer.getvalue(),
                file_name=f"telecom_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        
        # Export insights
        if st.session_state.ai_insights:
            st.markdown("---")
            st.subheader("📋 Export AI Insights")
            
            insights = st.session_state.ai_insights
            insights_text = f"""
# AI Insights Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
{insights.get('executive_summary', 'N/A')}

## Key Insights
"""
            for idx, insight in enumerate(insights.get('key_insights', []), 1):
                insights_text += f"\n### {idx}. {insight.get('title', 'N/A')}\n"
                insights_text += f"{insight.get('description', '')}\n"
                if 'action' in insight:
                    insights_text += f"**Action:** {insight['action']}\n"
                insights_text += f"**Impact:** {insight.get('impact', 'N/A')}\n"
            
            insights_text += "\n## Recommendations\n"
            for idx, rec in enumerate(insights.get('recommendations', []), 1):
                insights_text += f"\n### {idx}. {rec.get('category', 'General')}\n"
                insights_text += f"**Priority:** {rec.get('priority', 'medium')}\n"
                insights_text += f"**Action:** {rec.get('action', '')}\n"
                if isinstance(rec.get('details'), list):
                    for detail in rec['details']:
                        insights_text += f"- {detail}\n"
            
            st.download_button(
                label="📝 Download Insights Report",
                data=insights_text,
                file_name=f"ai_insights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

