import streamlit as st

# CRITICAL: Must be FIRST Streamlit command
st.set_page_config(
    page_title="Telecom Analytics Platform",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Lazy imports - only load when needed
import pandas as pd
import plotly.express as px
from datetime import datetime
import io

# Initialize session state BEFORE importing heavy modules
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.uploaded_files = None
    st.session_state.processed_data = None
    st.session_state.relationships = None
    st.session_state.merged_data = None
    st.session_state.insights = None
    st.session_state.anomalies = None

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: bold; color: #1f77b4; text-align: center; padding: 1rem;}
    .success-box {background-color: #e8f5e9; border-left: 5px solid #4caf50; padding: 1.5rem; border-radius: 0.5rem; margin: 1rem 0;}
    .alert-critical {background-color: #ffebee; border-left: 5px solid #f44336; padding: 1rem; margin: 0.5rem 0;}
    .alert-warning {background-color: #fff3e0; border-left: 5px solid #ff9800; padding: 1rem; margin: 0.5rem 0;}
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("📡 Telecom Analytics")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigation", 
    ["🏠 Home", "📤 Upload & Process", "🚨 Alerts & Anomalies", 
     "👥 Subscriber Analytics", "📱 Device Analytics", "📞 Usage Analytics", 
     "🗺️ Regional Performance", "📊 Comparisons", "💾 Export & Reports"])

st.sidebar.markdown("---")
st.sidebar.info("**Tip:** Upload files to get started!")

# HOME PAGE
if page == "🏠 Home":
    st.markdown('<p class="main-header">📡 Telecom Analytics Platform</p>', unsafe_allow_html=True)
    st.markdown("""
    ### Welcome to AI-Powered Telecom Analytics
    
    **Key Features:**
    - 🔄 Smart Data Processing - Upload any Excel/CSV files
    - 🤖 AI Insights - Automatic pattern discovery
    - 📊 Interactive Dashboards - Comprehensive analytics
    - 💾 Easy Export - Download reports instantly
    
    ### Getting Started:
    1. Go to **📤 Upload & Process**
    2. Upload your files (Excel or CSV)
    3. Click "🚀 Process & Analyze"
    4. View **🚨 Alerts & Anomalies** for AI insights
    """)
    
    if st.session_state.merged_data is not None:
        st.markdown("---")
        st.subheader("📊 Quick Stats")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", f"{len(st.session_state.merged_data):,}")
        with col2:
            st.metric("Columns", len(st.session_state.merged_data.columns))
        with col3:
            if st.session_state.anomalies:
                st.metric("⚠️ Alerts", len(st.session_state.anomalies))

# UPLOAD & PROCESS PAGE
elif page == "📤 Upload & Process":
    st.title("📤 Upload & Process Data Files")
    
    uploaded_files = st.file_uploader("Upload Files (Excel/CSV)", 
        type=['xlsx', 'xls', 'csv'], accept_multiple_files=True)
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) uploaded")
        
        if st.button("🚀 Process & Analyze", type="primary"):
            from file_processor import FileProcessor
            from relationship_detector import RelationshipDetector
            from data_merger import DataMerger
            from telecom_metrics import TelecomMetrics
            from anomaly_detector import AnomalyDetector
            from insights_generator import InsightsGenerator
            
            with st.spinner("Processing..."):
                try:
                    processor = FileProcessor()
                    processed_data = processor.process_files(uploaded_files)
                    st.session_state.processed_data = processed_data
                    
                    detector = RelationshipDetector()
                    relationships = detector.detect_relationships(processed_data)
                    st.session_state.relationships = relationships
                    
                    merger = DataMerger()
                    if relationships:
                        for rel in relationships:
                            rel['join_type'] = 'outer'
                    
                    merged_data = merger.merge_data(processed_data, relationships)
                    st.session_state.merged_data = merged_data
                    
                    st.markdown(f"""
                    <div class="success-box">
                        <h3>✅ Data Processed Successfully!</h3>
                        <p><strong>📊 Total Records:</strong> {len(merged_data):,}</p>
                        <p><strong>🔗 Connections:</strong> {len(relationships) if relationships else 0}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.subheader("👀 Data Preview")
                    st.dataframe(merged_data.head(10), width="stretch")
                    
                    st.info("🤖 Running analytics...")
                    
                    metrics_calc = TelecomMetrics()
                    metrics = metrics_calc.calculate_metrics(merged_data)
                    
                    anomaly_detector = AnomalyDetector()
                    anomalies = anomaly_detector.detect_anomalies(merged_data, metrics)
                    st.session_state.anomalies = anomalies
                    
                    insights_gen = InsightsGenerator(use_ai=False)
                    insights = insights_gen.generate_insights(merged_data, metrics, anomalies)
                    st.session_state.insights = insights
                    
                    st.success("✅ Analysis complete! Check 🚨 Alerts page")
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

# ALERTS PAGE
elif page == "🚨 Alerts & Anomalies":
    st.title("🚨 Alerts & Anomalies")
    
    if st.session_state.merged_data is None:
        st.warning("⚠️ Please upload and process files first")
    elif st.session_state.anomalies:
        anomalies = st.session_state.anomalies
        critical = [a for a in anomalies if a['severity'] == 'Critical']
        warnings = [a for a in anomalies if a['severity'] == 'Warning']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🔴 Critical", len(critical))
        with col2:
            st.metric("🟡 Warnings", len(warnings))
        with col3:
            st.metric("Total", len(anomalies))
        
        st.markdown("---")
        
        if critical:
            st.subheader("🔴 Critical Alerts")
            for alert in critical:
                st.markdown(f"""
                <div class="alert-critical">
                    <h4>⚠️ {alert['title']}</h4>
                    <p>{alert['description']}</p>
                    <p><strong>Action:</strong> {alert['recommendation']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        if warnings:
            st.subheader("🟡 Warnings")
            for alert in warnings:
                st.markdown(f"""
                <div class="alert-warning">
                    <h4>⚠️ {alert['title']}</h4>
                    <p>{alert['description']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        if st.session_state.insights:
            st.markdown("---")
            st.subheader("🤖 Insights")
            st.write(st.session_state.insights.get('summary', ''))
    else:
        st.info("No anomalies detected")

# OTHER PAGES
elif page in ["👥 Subscriber Analytics", "📱 Device Analytics", "📞 Usage Analytics", 
              "🗺️ Regional Performance", "📊 Comparisons"]:
    st.title(page)
    if st.session_state.merged_data is None:
        st.warning("⚠️ Please process data first")
    else:
        st.dataframe(st.session_state.merged_data.head(50), width="stretch")

# EXPORT PAGE
elif page == "💾 Export & Reports":
    st.title("💾 Export & Reports")
    
    if st.session_state.merged_data is None:
        st.warning("⚠️ Please process data first")
    else:
        df = st.session_state.merged_data
        col1, col2 = st.columns(2)
        
        with col1:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download CSV", csv, 
                f"data_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv", width="stretch")
        
        with col2:
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df.to_excel(writer, index=False)
            st.download_button("📥 Download Excel", buffer.getvalue(), 
                f"data_{datetime.now().strftime('%Y%m%d')}.xlsx", width="stretch")
        
        st.dataframe(df.head(100), width="stretch")

st.sidebar.caption("v2.0 | Lightweight")
