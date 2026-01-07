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

# Enhanced Custom CSS for better UX
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        border-radius: 5px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        border-radius: 5px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .critical-box {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        border-radius: 5px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border-left: 4px solid #17a2b8;
        border-radius: 5px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .step-box {
        background-color: #f8f9fa;
        border: 2px solid #dee2e6;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .big-button {
        padding: 1rem 2rem;
        font-size: 1.2rem;
        border-radius: 10px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .insight-card {
        background: white;
        border-left: 5px solid #1f77b4;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    h2 {
        color: #1f77b4;
        border-bottom: 3px solid #1f77b4;
        padding-bottom: 0.5rem;
    }
    h3 {
        color: #495057;
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
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1

# Initialize processors
file_processor = FileProcessor()
data_merger = DataMerger()

# Title
st.markdown('<h1 class="main-header">📡 Telecom Analytics Platform</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Simple, Powerful Analytics for Telecom Data | Upload → Analyze → Get Insights</p>', unsafe_allow_html=True)

# Sidebar - Simplified Navigation
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=80)
    
    # Progress Indicator
    if st.session_state.merged_data is not None:
        st.success("✅ **Step 2/3 Complete**")
        if st.session_state.ai_insights is not None:
            st.success("✅ **Step 3/3 Complete**")
        st.progress(1.0 if st.session_state.ai_insights else 0.67)
    else:
        st.info("📤 **Step 1/3: Upload Files**")
        st.progress(0.33)
    
    st.markdown("---")
    
    # Simple Navigation
    nav_options = {
        "🏠 Start Here": "home",
        "📤 Upload Data": "upload",
        "📊 View Insights": "insights",
        "📈 Charts": "charts",
        "💾 Download Report": "export"
    }
    
    selected = st.radio("**Navigate**", list(nav_options.keys()), label_visibility="collapsed")
    page = nav_options[selected]
    
    st.markdown("---")
    
    # Quick Status
    st.subheader("📊 Quick Status")
    if st.session_state.merged_data is not None:
        st.metric("Records", f"{len(st.session_state.merged_data):,}")
        if st.session_state.ai_insights:
            insights_count = len(st.session_state.ai_insights.get('key_insights', []))
            st.metric("Insights", insights_count)
    else:
        st.info("👈 Upload files to begin")
    
    st.markdown("---")
    st.caption("💡 **Tip**: Start with 'Upload Data' to analyze your files")

# ============================================================================
# HOME PAGE - Simplified Welcome
# ============================================================================
if page == "home":
    st.header("🎯 Welcome! Let's Get Started")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 📋 How It Works (3 Simple Steps)
        
        <div class="step-box">
        <h4>📤 Step 1: Upload Your Files</h4>
        <p>Upload your Excel or CSV files (3-4 files work best). The system automatically detects relationships between files.</p>
        </div>
        
        <div class="step-box">
        <h4>🤖 Step 2: AI Analysis</h4>
        <p>Our AI engine analyzes your data, identifies patterns, and finds business opportunities. No configuration needed!</p>
        </div>
        
        <div class="step-box">
        <h4>📊 Step 3: Get Insights</h4>
        <p>View board-ready insights, download reports, and share with management. All insights are ready for executive presentation.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Quick Actions
        st.subheader("🚀 Quick Actions")
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            if st.button("📤 Upload Files", type="primary", use_container_width=True):
                st.session_state.current_step = 1
                st.rerun()
        
        with col_b:
            if st.session_state.merged_data:
                if st.button("📊 View Insights", use_container_width=True):
                    page = "insights"
                    st.rerun()
            else:
                st.button("📊 View Insights", disabled=True, use_container_width=True)
        
        with col_c:
            if st.session_state.ai_insights:
                if st.button("💾 Download Report", use_container_width=True):
                    page = "export"
                    st.rerun()
            else:
                st.button("💾 Download Report", disabled=True, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Current Status")
        
        if st.session_state.merged_data is not None:
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.success("✅ **Data Ready**")
            st.metric("Total Records", f"{len(st.session_state.merged_data):,}")
            st.metric("Columns", len(st.session_state.merged_data.columns))
            
            if st.session_state.ai_insights:
                st.success("✅ **Analysis Complete**")
                insights = st.session_state.ai_insights
                st.metric("Key Insights", len(insights.get('key_insights', [])))
                problems = insights.get('problems', [])
                if problems:
                    critical = len([p for p in problems if p.get('severity') == 'critical'])
                    if critical > 0:
                        st.error(f"🔴 {critical} Critical Issues")
            elif st.session_state.ai_error:
                st.error("❌ Analysis Failed")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            if st.button("🔄 Start Fresh", type="secondary"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        else:
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.info("**Ready to Start**")
            st.write("Upload your telecom data files to begin analysis.")
            st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# UPLOAD PAGE - Simplified
# ============================================================================
elif page == "upload":
    st.header("📤 Upload Your Data Files")
    
    st.info("💡 **Tip**: Upload 3-4 Excel or CSV files. The system will automatically merge and analyze them.")
    
    uploaded_files = st.file_uploader(
        "**Choose your files**",
        type=['xlsx', 'xls', 'csv'],
        accept_multiple_files=True,
        help="Select multiple files. Supported formats: Excel (.xlsx, .xls) and CSV (.csv)"
    )
    
    if uploaded_files:
        st.success(f"✅ **{len(uploaded_files)} file(s) selected**")
        
        # Show file list
        with st.expander("📋 View Selected Files"):
            for f in uploaded_files:
                st.write(f"• {f.name} ({f.size:,} bytes)")
        
        # Single Big Button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 **ANALYZE MY DATA**", type="primary", use_container_width=True):
                # Clear old state
                st.session_state.merged_data = None
                st.session_state.ai_insights = None
                st.session_state.ai_error = None
                st.session_state.processed_files = []
                st.session_state.relationships = []
                st.session_state.processing_log = []
                
                with st.spinner("🔄 Processing your files..."):
                    try:
                        # Process files
                        processed_files = []
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for idx, file in enumerate(uploaded_files):
                            status_text.text(f"📄 Processing: {file.name}")
                            try:
                                file_info = file_processor.process_file(file)
                                processed_files.append(file_info)
                                st.session_state.processing_log.append(f"✅ {file.name}: {file_info.get('total_rows', 0)} rows")
                            except Exception as e:
                                st.session_state.processing_log.append(f"❌ {file.name}: Error - {str(e)}")
                                st.warning(f"⚠️ Error processing {file.name}: {str(e)}")
                            
                            progress_bar.progress((idx + 1) / len(uploaded_files))
                        
                        if not processed_files:
                            st.error("❌ No files were successfully processed!")
                            st.stop()
                        
                        st.session_state.processed_files = processed_files
                        status_text.text("🔍 Finding relationships...")
                        
                        # Detect relationships
                        relationships = data_merger.detect_relationships(processed_files)
                        st.session_state.relationships = relationships
                        
                        # Merge data
                        status_text.text("🔗 Merging data...")
                        merged_data, merge_summary = data_merger.merge_files(processed_files, relationships)
                        
                        if merged_data is None or len(merged_data) == 0:
                            st.error("❌ No data to merge!")
                            st.stop()
                        
                        st.session_state.merged_data = merged_data
                        st.session_state.merge_summary = merge_summary
                        
                        st.success(f"✅ **Data merged successfully!** {len(merged_data):,} records")
                        
                        # AI Analysis
                        status_text.text("🤖 Running AI analysis...")
                        try:
                            insights = ai_insights_engine.analyze_data(merged_data, merge_summary)
                            st.session_state.ai_insights = insights
                            
                            if insights.get('problems'):
                                problems = insights['problems']
                                critical = len([p for p in problems if p.get('severity') == 'critical'])
                                if critical > 0:
                                    st.error(f"🚨 **{critical} critical issue(s) found!**")
                                else:
                                    st.success(f"✅ **Analysis complete!** {len(problems)} issue(s) identified")
                            else:
                                st.success("✅ **Analysis complete!**")
                                
                        except Exception as e:
                            error_msg = f"{str(e)}\n\n{traceback.format_exc()}"
                            st.session_state.ai_error = error_msg
                            st.error(f"⚠️ AI analysis failed: {str(e)}")
                            with st.expander("🔍 Error Details"):
                                st.code(error_msg)
                        
                        progress_bar.progress(1.0)
                        status_text.text("✅ Complete!")
                        st.success("🎉 **Processing complete!**")
                        st.balloons()
                        
                        # Auto-navigate to insights
                        st.info("👉 **Go to 'View Insights' to see your analysis results**")
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        with st.expander("🔍 Full Error Details"):
                            st.code(traceback.format_exc())
    
    # Show summary if data exists
    if st.session_state.merged_data is not None:
        st.markdown("---")
        st.subheader("✅ Processing Summary")
        
        summary = st.session_state.merge_summary
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Files", summary.get('files_processed', 0))
        col2.metric("Records", f"{summary.get('total_records', 0):,}")
        col3.metric("Columns", summary.get('columns', 0))
        col4.metric("Method", summary.get('method', 'N/A').title())

# ============================================================================
# INSIGHTS PAGE - Board-Ready Format
# ============================================================================
elif page == "insights":
    st.header("📊 AI-Powered Insights")
    
    if st.session_state.merged_data is None:
        st.warning("⚠️ **No data loaded.** Please upload files first.")
        if st.button("📤 Go to Upload Page"):
            page = "upload"
            st.rerun()
    elif st.session_state.ai_insights is None:
        if st.session_state.ai_error:
            st.error("❌ **Analysis failed.** Please try processing again.")
            with st.expander("🔍 Error Details"):
                st.code(st.session_state.ai_error)
        else:
            st.info("⏳ **Analysis not run yet.** Go to Upload page and click 'Analyze My Data'.")
    else:
        insights = st.session_state.ai_insights
        
        # Executive Summary - Prominent
        st.markdown("---")
        st.subheader("📝 Executive Summary")
        summary = insights.get("executive_summary", "No summary available")
        st.markdown(f'<div class="success-box">{summary}</div>', unsafe_allow_html=True)
        
        # Problems Overview - Visual
        problems = insights.get('problems', [])
        if problems:
            st.markdown("---")
            st.subheader("🚨 Critical Issues Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            critical = len([p for p in problems if p.get('severity') == 'critical'])
            high = len([p for p in problems if p.get('severity') == 'high'])
            medium = len([p for p in problems if p.get('severity') == 'medium'])
            
            col1.metric("🔴 Critical", critical, delta=None)
            col2.metric("🟠 High", high, delta=None)
            col3.metric("🟡 Medium", medium, delta=None)
            col4.metric("Total", len(problems), delta=None)
        
        # Key Insights - Board-Ready Format
        st.markdown("---")
        st.subheader("💡 Key Business Insights")
        st.caption("💼 **Board-Ready Insights** - Ready to share with management")
        
        key_insights = insights.get('key_insights', [])
        if key_insights:
            for idx, insight in enumerate(key_insights, 1):
                impact = insight.get('impact', 'medium')
                icon = "🔴" if impact == 'critical' else "🟠" if impact == 'high' else "🟡" if impact == 'medium' else "🟢"
                
                with st.expander(f"{icon} **Insight {idx}**: {insight.get('title', 'N/A')}", expanded=(idx <= 2)):
                    st.markdown(f'<div class="insight-card">', unsafe_allow_html=True)
                    st.markdown(insight.get('description', ''))
                    
                    if 'action' in insight and insight['action']:
                        st.markdown("---")
                        st.markdown("**🎯 Recommended Action:**")
                        st.info(insight['action'])
                    
                    st.caption(f"**Impact Level**: {impact.upper()}")
                    st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("No specific insights generated. Review the executive summary above.")
        
        # Recommendations
        recommendations = insights.get('recommendations', [])
        if recommendations:
            st.markdown("---")
            st.subheader("🎯 Action Plan & Recommendations")
            
            for idx, rec in enumerate(recommendations, 1):
                priority = rec.get('priority', 'medium')
                color = 'critical-box' if priority == 'critical' else 'warning-box' if priority == 'high' else 'info-box'
                
                st.markdown(f'<div class="{color}">', unsafe_allow_html=True)
                st.markdown(f"**{idx}. {rec.get('category', 'General')}** (Priority: {priority.upper()})")
                st.markdown(f"**Action**: {rec.get('action', '')}")
                
                if isinstance(rec.get('details'), list):
                    st.markdown("**Details:**")
                    for detail in rec['details']:
                        st.write(f"• {detail}")
                elif rec.get('details'):
                    st.write(rec['details'])
                
                st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# CHARTS PAGE - Simplified
# ============================================================================
elif page == "charts":
    st.header("📈 Data Visualizations")
    
    if st.session_state.merged_data is None:
        st.warning("⚠️ **No data loaded.** Please upload files first.")
    else:
        df = st.session_state.merged_data
        
        numeric_cols = [c for c in df.select_dtypes(include=['number']).columns if not c.startswith('_')]
        categorical_cols = [c for c in df.select_dtypes(include=['object']).columns if not c.startswith('_')]
        
        if not numeric_cols and not categorical_cols:
            st.info("No suitable columns for visualization.")
        else:
            chart_type = st.selectbox("**Select Chart Type**", ["Bar Chart", "Pie Chart", "Line Chart", "Histogram"])
            
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
                        fig = px.pie(values=chart_data.values, names=chart_data.index, title=f"Distribution of {cat_col}")
                        st.plotly_chart(fig, use_container_width=True)
                
                elif chart_type == "Histogram" and numeric_cols:
                    num_col = st.selectbox("Column", numeric_cols)
                    if num_col:
                        fig = px.histogram(df, x=num_col, title=f"Distribution of {num_col}")
                        st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error creating chart: {str(e)}")

# ============================================================================
# EXPORT PAGE - Simplified
# ============================================================================
elif page == "export":
    st.header("💾 Download Reports")
    
    if st.session_state.merged_data is None:
        st.warning("⚠️ **No data loaded.** Please upload and analyze files first.")
    else:
        df = st.session_state.merged_data
        
        st.subheader("📥 Download Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv = df.to_csv(index=False)
            st.download_button(
                label="📄 Download Data as CSV",
                data=csv,
                file_name=f"telecom_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Data')
            
            st.download_button(
                label="📊 Download Data as Excel",
                data=buffer.getvalue(),
                file_name=f"telecom_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        
        # Export insights report
        if st.session_state.ai_insights:
            st.markdown("---")
            st.subheader("📋 Download Insights Report")
            
            insights = st.session_state.ai_insights
            insights_text = f"""
# Telecom Analytics - Executive Insights Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
{insights.get('executive_summary', 'N/A')}

## Key Business Insights
"""
            for idx, insight in enumerate(insights.get('key_insights', []), 1):
                insights_text += f"\n### {idx}. {insight.get('title', 'N/A')}\n"
                insights_text += f"{insight.get('description', '')}\n"
                if 'action' in insight:
                    insights_text += f"\n**Recommended Action:** {insight['action']}\n"
                insights_text += f"**Impact:** {insight.get('impact', 'N/A')}\n"
            
            insights_text += "\n## Action Plan & Recommendations\n"
            for idx, rec in enumerate(insights.get('recommendations', []), 1):
                insights_text += f"\n### {idx}. {rec.get('category', 'General')}\n"
                insights_text += f"**Priority:** {rec.get('priority', 'medium')}\n"
                insights_text += f"**Action:** {rec.get('action', '')}\n"
                if isinstance(rec.get('details'), list):
                    for detail in rec['details']:
                        insights_text += f"- {detail}\n"
            
            st.download_button(
                label="📝 Download Insights Report (Board-Ready)",
                data=insights_text,
                file_name=f"telecom_insights_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
            
            st.success("✅ **Report ready for executive presentation!**")
