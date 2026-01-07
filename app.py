import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io

# Import our custom modules
from file_processor import FileProcessor
from data_merger import DataMerger
from ai_insights_engine import AIInsightsEngine

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

# Initialize processors
file_processor = FileProcessor()
data_merger = DataMerger()

# Check for API keys
has_groq = 'GROQ_API_KEY' in st.secrets
has_gemini = 'GEMINI_API_KEY' in st.secrets

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
    else:
        st.info("⏳ No data loaded")
    
    if has_groq:
        st.success("✅ AI Enabled (Groq)")
    else:
        st.warning("⚠️ AI Disabled")
    
    st.markdown("---")
    st.caption("Powered by Groq Llama 3.3 🚀")
    st.caption("v2.0 - Enhanced Edition")

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
        - **🤖 AI Analysis** - Natural language insights powered by Groq Llama 3.3
        - **🚨 Anomaly Detection** - Automatic identification of data quality issues
        - **📊 Visual Analytics** - Interactive charts and dashboards
        - **💡 Smart Recommendations** - Actionable business insights
        
        ### 🚀 Quick Start
        1. Go to **📤 Upload & Process**
        2. Upload your Excel/CSV files (3-4 files recommended)
        3. Click **Process & Analyze**
        4. Review **AI Insights** and **Alerts**
        5. Explore visualizations and export results
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
                st.metric("AI Insights", len(st.session_state.ai_insights.get('key_insights', [])))
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            if st.button("🔄 Process New Files", type="primary"):
                st.session_state.processed_files = []
                st.session_state.merged_data = None
                st.session_state.merge_summary = None
                st.session_state.ai_insights = None
                st.rerun()
        else:
            st.info("👈 Upload files to get started")
            st.markdown("""
            **Supported Formats:**
            - Excel (.xlsx, .xls)
            - CSV files
            - Multiple sheets per file
            
            **Recommended:**
            - 3-4 related files
            - Files with common IDs
            - Telecom datasets
            """)

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
        st.info(f"📁 **{len(uploaded_files)} file(s) selected:** {', '.join([f.name for f in uploaded_files])}")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if st.button("🚀 Process & Analyze", type="primary", use_container_width=True):
                with st.spinner("🔄 Processing files..."):
                    try:
                        # Process each file
                        processed_files = []
                        progress_bar = st.progress(0)
                        
                        for idx, file in enumerate(uploaded_files):
                            st.text(f"Processing: {file.name}")
                            file_info = file_processor.process_file(file)
                            processed_files.append(file_info)
                            progress_bar.progress((idx + 1) / len(uploaded_files))
                        
                        st.session_state.processed_files = processed_files
                        
                        # Detect relationships
                        st.text("🔍 Detecting relationships...")
                        relationships = data_merger.detect_relationships(processed_files)
                        st.session_state.relationships = relationships
                        
                        # Merge data
                        st.text("🔗 Merging data...")
                        merged_data, merge_summary = data_merger.merge_files(processed_files, relationships)
                        st.session_state.merged_data = merged_data
                        st.session_state.merge_summary = merge_summary
                        
                        # AI Analysis (if enabled)
                        if has_groq and merged_data is not None:
                            st.text("🤖 Running AI analysis...")
                            try:
                                ai_engine = AIInsightsEngine(st.secrets['GROQ_API_KEY'])
                                insights = ai_engine.analyze_data(merged_data, merge_summary)
                                st.session_state.ai_insights = insights
                            except Exception as e:
                                st.warning(f"AI analysis skipped: {str(e)}")
                        
                        progress_bar.progress(1.0)
                        st.success("✅ Processing complete!")
                        st.balloons()
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        st.exception(e)
        
        with col2:
            if st.button("Clear All", type="secondary"):
                st.session_state.processed_files = []
                st.session_state.merged_data = None
                st.rerun()
    
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
            
            # Show relationships
            if st.session_state.relationships:
                st.write("**Detected Relationships:**")
                for rel in st.session_state.relationships[:3]:
                    st.write(f"- `{rel['key_column']}`: {rel['file1']} ↔️ {rel['file2']} ({rel['match_rate']:.1f}% match)")
        
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
    elif not has_groq:
        st.error("❌ AI features require Groq API key in Streamlit secrets")
    elif st.session_state.ai_insights is None:
        st.info("⏳ AI analysis not yet run. Go to Upload & Process page.")
    else:
        insights = st.session_state.ai_insights
        
        # Executive Summary
        st.subheader("📝 Executive Summary")
        st.markdown(f'<div class="success-box">{insights.get("executive_summary", "No summary available")}</div>', unsafe_allow_html=True)
        
        # Key Insights
        st.subheader("💡 Key Insights")
        
        key_insights = insights.get('key_insights', [])
        if key_insights:
            for idx, insight in enumerate(key_insights, 1):
                impact = insight.get('impact', 'medium')
                icon = "🔴" if impact == 'high' else "🟡" if impact == 'medium' else "🟢"
                
                with st.expander(f"{icon} Insight {idx}: {insight.get('title', 'N/A')}", expanded=(idx <= 2)):
                    st.write(insight.get('description', ''))
                    st.caption(f"Impact: {impact.upper()}")
        else:
            st.info("No specific insights generated")
        
        # Recommendations
        st.subheader("🎯 Recommendations")
        
        recommendations = insights.get('recommendations', [])
        if recommendations:
            for rec in recommendations:
                priority = rec.get('priority', 'medium')
                color = 'critical-box' if priority == 'high' else 'warning-box' if priority == 'medium' else 'success-box'
                
                st.markdown(f'<div class="{color}">', unsafe_allow_html=True)
                st.markdown(f"**{rec.get('category', 'General')}** (Priority: {priority.upper()})")
                st.write(rec.get('action', ''))
                if isinstance(rec.get('details'), list):
                    for detail in rec['details']:
                        st.write(f"- {detail}")
                else:
                    st.write(rec.get('details', ''))
                st.markdown('</div>', unsafe_allow_html=True)

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
        anomalies = st.session_state.ai_insights.get('anomalies', [])
        
        if not anomalies:
            st.success("✅ No anomalies detected! Data quality looks good.")
        else:
            # Summary
            critical = len([a for a in anomalies if a['severity'] == 'critical'])
            warnings = len([a for a in anomalies if a['severity'] == 'warning'])
            info = len([a for a in anomalies if a['severity'] == 'info'])
            
            col1, col2, col3 = st.columns(3)
            col1.metric("🔴 Critical", critical)
            col2.metric("🟡 Warnings", warnings)
            col3.metric("🔵 Info", info)
            
            st.markdown("---")
            
            # Filter
            severity_filter = st.multiselect(
                "Filter by Severity",
                ['critical', 'warning', 'info'],
                default=['critical', 'warning']
            )
            
            filtered_anomalies = [a for a in anomalies if a['severity'] in severity_filter]
            
            # Display anomalies
            for anomaly in filtered_anomalies:
                severity = anomaly['severity']
                icon = "🔴" if severity == 'critical' else "🟡" if severity == 'warning' else "🔵"
                box_class = 'critical-box' if severity == 'critical' else 'warning-box' if severity == 'warning' else 'success-box'
                
                st.markdown(f'<div class="{box_class}">', unsafe_allow_html=True)
                st.markdown(f"### {icon} {anomaly.get('type', 'Unknown').replace('_', ' ').title()}")
                st.write(anomaly.get('description', ''))
                
                if 'column' in anomaly:
                    st.caption(f"Column: **{anomaly['column']}**")
                if 'count' in anomaly:
                    st.caption(f"Affected records: **{anomaly['count']:,}**")
                if 'percentage' in anomaly:
                    st.caption(f"Percentage: **{anomaly['percentage']}**")
                
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
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove metadata columns
        numeric_cols = [c for c in numeric_cols if not c.startswith('_')]
        categorical_cols = [c for c in categorical_cols if not c.startswith('_')]
        
        if not numeric_cols and not categorical_cols:
            st.info("No suitable columns for visualization")
        else:
            # Chart type selector
            chart_type = st.selectbox(
                "Select Chart Type",
                ["Bar Chart", "Line Chart", "Pie Chart", "Scatter Plot", "Histogram", "Box Plot"]
            )
            
            col1, col2 = st.columns(2)
            
            if chart_type == "Bar Chart":
                with col1:
                    x_col = st.selectbox("X-axis (Category)", categorical_cols)
                with col2:
                    y_col = st.selectbox("Y-axis (Metric)", numeric_cols) if numeric_cols else None
                
                if x_col and y_col:
                    # Aggregate data
                    chart_data = df.groupby(x_col)[y_col].sum().reset_index()
                    chart_data = chart_data.nlargest(15, y_col)  # Top 15
                    
                    fig = px.bar(chart_data, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
                    st.plotly_chart(fig, use_container_width=True)
            
            elif chart_type == "Line Chart":
                with col1:
                    x_col = st.selectbox("X-axis", df.columns.tolist())
                with col2:
                    y_col = st.selectbox("Y-axis", numeric_cols)
                
                if x_col and y_col:
                    fig = px.line(df.head(100), x=x_col, y=y_col, title=f"{y_col} over {x_col}")
                    st.plotly_chart(fig, use_container_width=True)
            
            elif chart_type == "Pie Chart":
                cat_col = st.selectbox("Category", categorical_cols)
                
                if cat_col:
                    chart_data = df[cat_col].value_counts().head(10)
                    fig = px.pie(values=chart_data.values, names=chart_data.index, title=f"Distribution of {cat_col}")
                    st.plotly_chart(fig, use_container_width=True)
            
            elif chart_type == "Histogram":
                num_col = st.selectbox("Column", numeric_cols)
                
                if num_col:
                    fig = px.histogram(df, x=num_col, title=f"Distribution of {num_col}")
                    st.plotly_chart(fig, use_container_width=True)

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
                insights_text += f"**Impact:** {insight.get('impact', 'N/A')}\n"
            
            st.download_button(
                label="📝 Download Insights Report",
                data=insights_text,
                file_name=f"ai_insights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
