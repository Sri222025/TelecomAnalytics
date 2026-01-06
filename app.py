import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
import json

# Import custom modules
from file_processor import FileProcessor
from relationship_detector import RelationshipDetector
from telecom_metrics import TelecomMetrics
from anomaly_detector import AnomalyDetector
from insights_generator import InsightsGenerator
from visualizations import Visualizations
from data_merger import DataMerger
from export_manager import ExportManager
from config_manager import ConfigManager

# Page configuration
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
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .alert-critical {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .alert-warning {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .alert-info {
        background-color: #e3f2fd;
        border-left: 5px solid #2196f3;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'relationships' not in st.session_state:
    st.session_state.relationships = None
if 'merged_data' not in st.session_state:
    st.session_state.merged_data = None
if 'insights' not in st.session_state:
    st.session_state.insights = None
if 'anomalies' not in st.session_state:
    st.session_state.anomalies = None

# Sidebar navigation
st.sidebar.title("📡 Telecom Analytics")
st.sidebar.markdown("---")

# Main navigation
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "📤 Upload & Process", "🔗 Data Relationships", "🚨 Alerts & Anomalies", 
     "👥 Subscriber Analytics", "📱 Device Analytics", "📞 Usage Analytics", 
     "🗺️ Regional Performance", "📊 Comparisons", "💾 Export & Reports"]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Quick Tips:**
- Upload multiple Excel/CSV files
- Tool auto-detects relationships
- AI flags critical issues
- Export dashboards & reports
""")

# Helper function for metrics display
def display_metric_card(title, value, delta=None, delta_color="normal"):
    col1, col2 = st.columns([3, 1])
    with col1:
        st.metric(label=title, value=value, delta=delta, delta_color=delta_color)

# ============================================================================
# PAGE: HOME
# ============================================================================
if page == "🏠 Home":
    st.markdown('<p class="main-header">📡 Telecom Analytics Platform</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Welcome to the Dynamic Telecom Analytics Platform
    
    **Key Features:**
    - 🔄 **Dynamic Multi-File Processing** - Handle any telecom data structure
    - 🤖 **AI-Powered Anomaly Detection** - Automatic issue flagging
    - 📊 **Comprehensive Dashboards** - Subscriber, Usage, Device, Regional analytics
    - 🔗 **Smart Data Linking** - Auto-detect relationships across files
    - 📈 **Trend Analysis** - WoW, MoM, QoQ comparisons
    - 💾 **Easy Export** - Download processed data and reports
    
    ---
    
    ### Supported Calling Formats:
    - 📞 **POTS (Landline)** - Traditional fixed-line phones
    - 📱 **JioJoin App** - Mobile application calling
    - 📺 **STB Calling** - Set-Top Box calling
    - 📡 **AirFiber** - JioJoin only
    
    ---
    
    ### Getting Started:
    1. Go to **📤 Upload & Process** to upload your files
    2. Review detected **🔗 Data Relationships**
    3. Check **🚨 Alerts & Anomalies** for critical insights
    4. Explore various analytics dashboards
    5. Export reports from **💾 Export & Reports**
    """)
    
    # Quick stats if data is available
    if st.session_state.merged_data is not None:
        st.markdown("---")
        st.subheader("📊 Quick Stats")
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Calculate basic stats
        merged_df = st.session_state.merged_data
        
        with col1:
            st.metric("Total Records", f"{len(merged_df):,}")
        
        with col2:
            # Count unique identifiers if available
            id_cols = [col for col in merged_df.columns if 'id' in col.lower() or 'number' in col.lower()]
            if id_cols:
                st.metric("Unique Entities", f"{merged_df[id_cols[0]].nunique():,}")
            else:
                st.metric("Columns", len(merged_df.columns))
        
        with col3:
            if st.session_state.anomalies:
                st.metric("⚠️ Alerts", len(st.session_state.anomalies), delta_color="inverse")
            else:
                st.metric("Files Processed", len(st.session_state.processed_data) if st.session_state.processed_data else 0)
        
        with col4:
            st.metric("Last Updated", datetime.now().strftime("%Y-%m-%d %H:%M"))

# ============================================================================
# PAGE: UPLOAD & PROCESS
# ============================================================================
elif page == "📤 Upload & Process":
    st.title("📤 Upload & Process Data Files")
    
    st.markdown("""
    Upload your telecom data files (Excel or CSV). The tool will:
    - Read all worksheets from each file
    - Detect data types and structures
    - Preview your data
    - Identify potential linking columns
    """)
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload Files (Excel/CSV)",
        type=['xlsx', 'xls', 'csv'],
        accept_multiple_files=True,
        help="Upload 3-4 files. Each Excel file can contain multiple worksheets."
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully!")
        
        # Process button
        if st.button("🔄 Process Files", type="primary"):
            with st.spinner("Processing files... This may take a moment for large files."):
                try:
                    # Initialize file processor
                    processor = FileProcessor()
                    
                    # Process all files
                    processed_data = processor.process_files(uploaded_files)
                    st.session_state.processed_data = processed_data
                    st.session_state.uploaded_files = uploaded_files
                    
                    st.success("✅ Files processed successfully!")
                    
                    # Display summary
                    st.subheader("📋 Processing Summary")
                    
                    for file_info in processed_data:
                        with st.expander(f"📁 {file_info['filename']} ({len(file_info['sheets'])} sheet(s))"):
                            for sheet_name, sheet_data in file_info['sheets'].items():
                                st.markdown(f"**Sheet: {sheet_name}**")
                                st.write(f"- Rows: {sheet_data['row_count']:,}")
                                st.write(f"- Columns: {sheet_data['column_count']}")
                                st.write(f"- Column Names: {', '.join(sheet_data['columns'][:10])}" + 
                                       ("..." if len(sheet_data['columns']) > 10 else ""))
                                
                                # Data preview
                                st.dataframe(sheet_data['preview'], use_container_width=True)
                                st.markdown("---")
                    
                    # Auto-detect relationships
                    st.subheader("🔗 Detecting Relationships...")
                    detector = RelationshipDetector()
                    relationships = detector.detect_relationships(processed_data)
                    st.session_state.relationships = relationships
                    
                    if relationships:
                        st.success(f"✅ Found {len(relationships)} potential relationship(s)!")
                        st.info("👉 Go to **🔗 Data Relationships** page to review and confirm.")
                    else:
                        st.warning("⚠️ No automatic relationships detected. You can manually configure them.")
                    
                except Exception as e:
                    st.error(f"❌ Error processing files: {str(e)}")
                    st.exception(e)
    
    # Show processed data if available
    elif st.session_state.processed_data:
        st.info("✅ Files already processed. Upload new files to reprocess.")
        
        # Display current data summary
        st.subheader("📋 Current Data Summary")
        for file_info in st.session_state.processed_data:
            st.write(f"📁 **{file_info['filename']}** - {len(file_info['sheets'])} sheet(s)")

# ============================================================================
# PAGE: DATA RELATIONSHIPS
# ============================================================================
elif page == "🔗 Data Relationships":
    st.title("🔗 Data Relationships")
    
    if not st.session_state.processed_data:
        st.warning("⚠️ Please upload and process files first from the **📤 Upload & Process** page.")
    else:
        st.markdown("""
        Review and confirm the detected relationships between your data files.
        These relationships will be used to merge data for comprehensive analysis.
        """)
        
        if st.session_state.relationships:
            st.subheader("🔍 Detected Relationships")
            
            # Display relationships
            for idx, rel in enumerate(st.session_state.relationships):
                with st.expander(f"Relationship {idx + 1}: {rel['confidence']} confidence", expanded=True):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Source:**")
                        st.write(f"- File: {rel['source_file']}")
                        st.write(f"- Sheet: {rel['source_sheet']}")
                        st.write(f"- Column: `{rel['source_column']}`")
                    
                    with col2:
                        st.write("**Target:**")
                        st.write(f"- File: {rel['target_file']}")
                        st.write(f"- Sheet: {rel['target_sheet']}")
                        st.write(f"- Column: `{rel['target_column']}`")
                    
                    st.write(f"**Match Score:** {rel['match_score']:.2%}")
                    st.write(f"**Common Values:** {rel['common_values']}")
        else:
            st.info("ℹ️ No automatic relationships detected.")
        
        # Manual relationship configuration
        st.subheader("⚙️ Manual Relationship Configuration")
        
        with st.form("manual_relationship"):
            st.write("Add a custom relationship between sheets:")
            
            col1, col2 = st.columns(2)
            
            # Get list of all sheets
            all_sheets = []
            for file_info in st.session_state.processed_data:
                for sheet_name in file_info['sheets'].keys():
                    all_sheets.append(f"{file_info['filename']} → {sheet_name}")
            
            with col1:
                source_sheet = st.selectbox("Source Sheet", all_sheets)
                # Get columns for source
                if source_sheet:
                    file_name, sheet_name = source_sheet.split(" → ")
                    file_data = next(f for f in st.session_state.processed_data if f['filename'] == file_name)
                    source_cols = file_data['sheets'][sheet_name]['columns']
                    source_col = st.selectbox("Source Column", source_cols)
            
            with col2:
                target_sheet = st.selectbox("Target Sheet", all_sheets)
                # Get columns for target
                if target_sheet:
                    file_name, sheet_name = target_sheet.split(" → ")
                    file_data = next(f for f in st.session_state.processed_data if f['filename'] == file_name)
                    target_cols = file_data['sheets'][sheet_name]['columns']
                    target_col = st.selectbox("Target Column", target_cols)
            
            join_type = st.selectbox("Join Type", ["inner", "left", "right", "outer"])
            
            if st.form_submit_button("➕ Add Relationship"):
                # Add manual relationship
                if not st.session_state.relationships:
                    st.session_state.relationships = []
                
                manual_rel = {
                    'source_file': source_sheet.split(" → ")[0],
                    'source_sheet': source_sheet.split(" → ")[1],
                    'source_column': source_col,
                    'target_file': target_sheet.split(" → ")[0],
                    'target_sheet': target_sheet.split(" → ")[1],
                    'target_column': target_col,
                    'join_type': join_type,
                    'confidence': 'Manual',
                    'match_score': 1.0,
                    'common_values': 'User defined'
                }
                
                st.session_state.relationships.append(manual_rel)
                st.success("✅ Relationship added!")
                st.rerun()
        
        # Merge data button
        st.markdown("---")
        if st.button("🔀 Merge Data Based on Relationships", type="primary"):
            if st.session_state.relationships:
                with st.spinner("Merging data... This may take a moment."):
                    try:
                        merger = DataMerger()
                        merged_data = merger.merge_data(
                            st.session_state.processed_data,
                            st.session_state.relationships
                        )
                        st.session_state.merged_data = merged_data
                        
                        st.success(f"✅ Data merged successfully! Total rows: {len(merged_data):,}")
                        
                        # Show preview
                        st.subheader("📊 Merged Data Preview")
                        st.dataframe(merged_data.head(100), use_container_width=True)
                        
                        # Run analytics
                        st.info("🤖 Running AI analytics and anomaly detection...")
                        
                        # Calculate telecom metrics
                        metrics_calc = TelecomMetrics()
                        metrics = metrics_calc.calculate_metrics(merged_data)
                        
                        # Detect anomalies
                        anomaly_detector = AnomalyDetector()
                        anomalies = anomaly_detector.detect_anomalies(merged_data, metrics)
                        st.session_state.anomalies = anomalies
                        
                        # Generate insights
                        insights_gen = InsightsGenerator()
                        insights = insights_gen.generate_insights(merged_data, metrics, anomalies)
                        st.session_state.insights = insights
                        
                        st.success("✅ Analytics complete! Check **🚨 Alerts & Anomalies** page.")
                        
                    except Exception as e:
                        st.error(f"❌ Error merging data: {str(e)}")
                        st.exception(e)
            else:
                st.warning("⚠️ Please define at least one relationship before merging.")

# ============================================================================
# PAGE: ALERTS & ANOMALIES (Main USP)
# ============================================================================
elif page == "🚨 Alerts & Anomalies":
    st.title("🚨 Alerts & Anomalies")
    
    if not st.session_state.merged_data:
        st.warning("⚠️ Please process and merge data first.")
    else:
        st.markdown("### AI-Powered Issue Detection")
        
        # Display anomalies
        if st.session_state.anomalies:
            anomalies = st.session_state.anomalies
            
            # Count by severity
            critical = [a for a in anomalies if a['severity'] == 'Critical']
            warnings = [a for a in anomalies if a['severity'] == 'Warning']
            info = [a for a in anomalies if a['severity'] == 'Info']
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🔴 Critical", len(critical))
            with col2:
                st.metric("🟡 Warnings", len(warnings))
            with col3:
                st.metric("🔵 Info", len(info))
            with col4:
                st.metric("Total Alerts", len(anomalies))
            
            st.markdown("---")
            
            # Display critical alerts
            if critical:
                st.subheader("🔴 Critical Alerts")
                for alert in critical:
                    st.markdown(f"""
                    <div class="alert-critical">
                        <h4>⚠️ {alert['title']}</h4>
                        <p>{alert['description']}</p>
                        <p><strong>Impact:</strong> {alert['impact']}</p>
                        <p><strong>Recommendation:</strong> {alert['recommendation']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Display warnings
            if warnings:
                st.subheader("🟡 Warnings")
                for alert in warnings:
                    st.markdown(f"""
                    <div class="alert-warning">
                        <h4>⚠️ {alert['title']}</h4>
                        <p>{alert['description']}</p>
                        <p><strong>Recommendation:</strong> {alert['recommendation']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Display info
            if info:
                with st.expander("🔵 Informational Insights"):
                    for alert in info:
                        st.markdown(f"""
                        <div class="alert-info">
                            <h4>ℹ️ {alert['title']}</h4>
                            <p>{alert['description']}</p>
                        </div>
                        """, unsafe_allow_html=True)
        
        else:
            st.info("🤖 Running anomaly detection...")
            
            # Run detection
            with st.spinner("Analyzing data for anomalies..."):
                try:
                    metrics_calc = TelecomMetrics()
                    metrics = metrics_calc.calculate_metrics(st.session_state.merged_data)
                    
                    anomaly_detector = AnomalyDetector()
                    anomalies = anomaly_detector.detect_anomalies(st.session_state.merged_data, metrics)
                    st.session_state.anomalies = anomalies
                    
                    st.rerun()
                except Exception as e:
                    st.error(f"Error in anomaly detection: {str(e)}")
        
        # AI Insights
        st.markdown("---")
        st.subheader("🤖 AI-Generated Insights")
        
        if st.session_state.insights:
            insights = st.session_state.insights
            
            st.markdown(f"**Executive Summary:**\n\n{insights['summary']}")
            
            st.markdown("**Key Findings:**")
            for finding in insights['key_findings']:
                st.markdown(f"- {finding}")
            
            st.markdown("**Recommendations:**")
            for rec in insights['recommendations']:
                st.markdown(f"- {rec}")
        else:
            if st.button("Generate AI Insights"):
                with st.spinner("Generating insights..."):
                    try:
                        metrics_calc = TelecomMetrics()
                        metrics = metrics_calc.calculate_metrics(st.session_state.merged_data)
                        
                        insights_gen = InsightsGenerator()
                        insights = insights_gen.generate_insights(
                            st.session_state.merged_data,
                            metrics,
                            st.session_state.anomalies or []
                        )
                        st.session_state.insights = insights
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error generating insights: {str(e)}")

# ============================================================================
# PAGE: SUBSCRIBER ANALYTICS
# ============================================================================
elif page == "👥 Subscriber Analytics":
    st.title("👥 Subscriber Analytics")
    
    if not st.session_state.merged_data:
        st.warning("⚠️ Please process and merge data first.")
    else:
        df = st.session_state.merged_data
        viz = Visualizations()
        
        # Identify subscriber-related columns
        st.subheader("📊 Subscriber Metrics")
        
        # Try to find common columns
        id_cols = [col for col in df.columns if any(x in col.lower() for x in ['customer', 'subscriber', 'user', 'id', 'number'])]
        date_cols = [col for col in df.columns if any(x in col.lower() for x in ['date', 'time', 'activation', 'created'])]
        type_cols = [col for col in df.columns if any(x in col.lower() for x in ['type', 'category', 'plan', 'connection', 'device'])]
        region_cols = [col for col in df.columns if any(x in col.lower() for x in ['region', 'circle', 'state', 'zone', 'area'])]
        
        # Basic metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if id_cols:
                unique_subs = df[id_cols[0]].nunique()
                st.metric("Total Subscribers", f"{unique_subs:,}")
            else:
                st.metric("Total Records", f"{len(df):,}")
        
        with col2:
            if type_cols:
                type_count = df[type_cols[0]].nunique()
                st.metric("Connection Types", type_count)
            else:
                st.metric("Columns", len(df.columns))
        
        with col3:
            if region_cols:
                region_count = df[region_cols[0]].nunique()
                st.metric("Regions", region_count)
            else:
                st.metric("Data Points", len(df))
        
        with col4:
            if date_cols:
                st.metric("Date Range", "Available")
            else:
                st.metric("Status", "Active")
        
        # Visualizations
        st.markdown("---")
        
        # Connection type distribution
        if type_cols:
            st.subheader("📊 Distribution by Type")
            type_col = st.selectbox("Select Type Column", type_cols)
            
            type_dist = df[type_col].value_counts()
            fig = px.pie(
                values=type_dist.values,
                names=type_dist.index,
                title=f"Distribution by {type_col}"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Regional distribution
        if region_cols:
            st.subheader("🗺️ Regional Distribution")
            region_col = st.selectbox("Select Region Column", region_cols)
            
            if id_cols:
                region_dist = df.groupby(region_col)[id_cols[0]].nunique().sort_values(ascending=False)
                fig = px.bar(
                    x=region_dist.index,
                    y=region_dist.values,
                    title=f"Subscribers by {region_col}",
                    labels={'x': region_col, 'y': 'Subscriber Count'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Time-based analysis
        if date_cols:
            st.subheader("📈 Trends Over Time")
            date_col = st.selectbox("Select Date Column", date_cols)
            
            try:
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                df_time = df.dropna(subset=[date_col])
                
                if not df_time.empty:
                    df_time['Date'] = df_time[date_col].dt.date
                    time_series = df_time.groupby('Date').size()
                    
                    fig = px.line(
                        x=time_series.index,
                        y=time_series.values,
                        title=f"Activity Over Time",
                        labels={'x': 'Date', 'y': 'Count'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.info(f"Time-based analysis not available for this dataset.")

# ============================================================================
# PAGE: DEVICE ANALYTICS
# ============================================================================
elif page == "📱 Device Analytics":
    st.title("📱 Device Format Analytics")
    
    if not st.session_state.merged_data:
        st.warning("⚠️ Please process and merge data first.")
    else:
        df = st.session_state.merged_data
        
        st.markdown("""
        ### Calling Format Analysis
        Analyze usage patterns across different device types:
        - 📞 POTS (Landline)
        - 📱 JioJoin App
        - 📺 STB Calling
        - 📡 AirFiber
        """)
        
        # Find device-related columns
        device_cols = [col for col in df.columns if any(x in col.lower() for x in ['device', 'format', 'type', 'channel', 'method'])]
        
        if device_cols:
            st.subheader("📊 Device Distribution")
            
            device_col = st.selectbox("Select Device Column", device_cols)
            
            # Device distribution
            device_dist = df[device_col].value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.pie(
                    values=device_dist.values,
                    names=device_dist.index,
                    title="Device Type Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(
                    x=device_dist.index,
                    y=device_dist.values,
                    title="Device Usage Count",
                    labels={'x': 'Device Type', 'y': 'Count'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Device metrics
            st.subheader("📈 Device Metrics")
            
            # Find usage/volume columns
            usage_cols = [col for col in df.columns if any(x in col.lower() for x in ['duration', 'minutes', 'usage', 'volume', 'count', 'calls'])]
            
            if usage_cols:
                usage_col = st.selectbox("Select Usage Metric", usage_cols)
                
                device_usage = df.groupby(device_col)[usage_col].agg(['sum', 'mean', 'count'])
                
                st.dataframe(device_usage, use_container_width=True)
                
                # Visualization
                fig = px.bar(
                    x=device_usage.index,
                    y=device_usage['sum'],
                    title=f"Total {usage_col} by Device Type",
                    labels={'x': 'Device Type', 'y': f'Total {usage_col}'}
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ℹ️ No device-related columns detected in the data.")
            st.write("Available columns:", ", ".join(df.columns.tolist()[:20]))

# ============================================================================
# PAGE: USAGE ANALYTICS
# ============================================================================
elif page == "📞 Usage Analytics":
    st.title("📞 Usage Analytics")
    
    if not st.session_state.merged_data:
        st.warning("⚠️ Please process and merge data first.")
    else:
        df = st.session_state.merged_data
        
        st.subheader("📊 Call & Usage Metrics")
        
        # Find usage-related columns
        usage_cols = [col for col in df.columns if any(x in col.lower() for x in 
            ['call', 'duration', 'minute', 'usage', 'volume', 'count', 'data', 'traffic'])]
        
        if usage_cols:
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            
            for idx, col in enumerate(usage_cols[:4]):
                with [col1, col2, col3, col4][idx]:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        total = df[col].sum()
                        avg = df[col].mean()
                        st.metric(f"Total {col}", f"{total:,.0f}")
                        st.caption(f"Avg: {avg:,.2f}")
            
            # Usage distribution
            st.markdown("---")
            st.subheader("📈 Usage Distribution")
            
            selected_usage = st.selectbox("Select Metric to Analyze", usage_cols)
            
            if pd.api.types.is_numeric_dtype(df[selected_usage]):
                col1, col2 = st.columns(2)
                
                with col1:
                    # Histogram
                    fig = px.histogram(
                        df,
                        x=selected_usage,
                        title=f"Distribution of {selected_usage}",
                        nbins=50
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Box plot
                    fig = px.box(
                        df,
                        y=selected_usage,
                        title=f"{selected_usage} Box Plot"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Statistics
                st.subheader("📊 Statistical Summary")
                stats = df[selected_usage].describe()
                
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Mean", f"{stats['mean']:,.2f}")
                with col2:
                    st.metric("Median", f"{stats['50%']:,.2f}")
                with col3:
                    st.metric("Std Dev", f"{stats['std']:,.2f}")
                with col4:
                    st.metric("Min", f"{stats['min']:,.2f}")
                with col5:
                    st.metric("Max", f"{stats['max']:,.2f}")
        else:
            st.info("ℹ️ No usage-related columns detected in the data.")

# ============================================================================
# PAGE: REGIONAL PERFORMANCE
# ============================================================================
elif page == "🗺️ Regional Performance":
    st.title("🗺️ Regional Performance")
    
    if not st.session_state.merged_data:
        st.warning("⚠️ Please process and merge data first.")
    else:
        df = st.session_state.merged_data
        
        # Find region columns
        region_cols = [col for col in df.columns if any(x in col.lower() for x in 
            ['region', 'circle', 'state', 'zone', 'area', 'location', 'city'])]
        
        if region_cols:
            st.subheader("🗺️ Geographic Analysis")
            
            region_col = st.selectbox("Select Region Column", region_cols)
            
            # Regional distribution
            region_counts = df[region_col].value_counts()
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = px.bar(
                    x=region_counts.index,
                    y=region_counts.values,
                    title=f"Distribution across {region_col}",
                    labels={'x': region_col, 'y': 'Count'}
                )
                fig.update_xaxis(tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.dataframe(
                    region_counts.head(10).to_frame('Count'),
                    use_container_width=True
                )
            
            # Regional metrics
            st.markdown("---")
            st.subheader("📊 Regional Metrics")
            
            # Find numeric columns for aggregation
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            if numeric_cols:
                metric_col = st.selectbox("Select Metric for Regional Comparison", numeric_cols)
                
                regional_metrics = df.groupby(region_col)[metric_col].agg([
                    ('Total', 'sum'),
                    ('Average', 'mean'),
                    ('Count', 'count')
                ]).round(2)
                
                regional_metrics = regional_metrics.sort_values('Total', ascending=False)
                
                st.dataframe(regional_metrics, use_container_width=True)
                
                # Top regions
                top_n = st.slider("Show Top N Regions", 5, 20, 10)
                
                fig = px.bar(
                    x=regional_metrics.head(top_n).index,
                    y=regional_metrics.head(top_n)['Total'],
                    title=f"Top {top_n} Regions by {metric_col}",
                    labels={'x': region_col, 'y': f'Total {metric_col}'}
                )
                fig.update_xaxis(tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ℹ️ No region-related columns detected in the data.")

# ============================================================================
# PAGE: COMPARISONS
# ============================================================================
elif page == "📊 Comparisons":
    st.title("📊 Period Comparisons")
    
    if not st.session_state.merged_data:
        st.warning("⚠️ Please process and merge data first.")
    else:
        df = st.session_state.merged_data
        
        st.markdown("""
        ### Time-Period Comparisons
        Compare metrics across different time periods:
        - Week-over-Week (WoW)
        - Month-over-Month (MoM)
        - Quarter-over-Quarter (QoQ)
        - Custom date ranges
        """)
        
        # Find date columns
        date_cols = [col for col in df.columns if any(x in col.lower() for x in 
            ['date', 'time', 'day', 'month', 'year', 'period'])]
        
        if date_cols:
            date_col = st.selectbox("Select Date Column", date_cols)
            
            # Try to convert to datetime
            try:
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                df_time = df.dropna(subset=[date_col]).copy()
                
                if not df_time.empty:
                    # Add time periods
                    df_time['Year'] = df_time[date_col].dt.year
                    df_time['Month'] = df_time[date_col].dt.month
                    df_time['Week'] = df_time[date_col].dt.isocalendar().week
                    df_time['Quarter'] = df_time[date_col].dt.quarter
                    
                    # Comparison type
                    comparison_type = st.radio(
                        "Select Comparison Type",
                        ["Weekly", "Monthly", "Quarterly", "Custom Range"]
                    )
                    
                    # Select metric
                    numeric_cols = df_time.select_dtypes(include=['int64', 'float64']).columns.tolist()
                    numeric_cols = [c for c in numeric_cols if c not in ['Year', 'Month', 'Week', 'Quarter']]
                    
                    if numeric_cols:
                        metric_col = st.selectbox("Select Metric to Compare", numeric_cols)
                        
                        if comparison_type == "Weekly":
                            weekly_data = df_time.groupby(['Year', 'Week'])[metric_col].sum().reset_index()
                            weekly_data['Period'] = weekly_data['Year'].astype(str) + '-W' + weekly_data['Week'].astype(str)
                            
                            fig = px.line(
                                weekly_data,
                                x='Period',
                                y=metric_col,
                                title=f"Weekly Trend - {metric_col}",
                                markers=True
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Calculate WoW growth
                            weekly_data['WoW_Growth'] = weekly_data[metric_col].pct_change() * 100
                            st.dataframe(weekly_data[['Period', metric_col, 'WoW_Growth']].tail(10), use_container_width=True)
                        
                        elif comparison_type == "Monthly":
                            monthly_data = df_time.groupby(['Year', 'Month'])[metric_col].sum().reset_index()
                            monthly_data['Period'] = monthly_data['Year'].astype(str) + '-' + monthly_data['Month'].astype(str).str.zfill(2)
                            
                            fig = px.line(
                                monthly_data,
                                x='Period',
                                y=metric_col,
                                title=f"Monthly Trend - {metric_col}",
                                markers=True
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Calculate MoM growth
                            monthly_data['MoM_Growth'] = monthly_data[metric_col].pct_change() * 100
                            st.dataframe(monthly_data[['Period', metric_col, 'MoM_Growth']].tail(12), use_container_width=True)
                        
                        elif comparison_type == "Quarterly":
                            quarterly_data = df_time.groupby(['Year', 'Quarter'])[metric_col].sum().reset_index()
                            quarterly_data['Period'] = quarterly_data['Year'].astype(str) + '-Q' + quarterly_data['Quarter'].astype(str)
                            
                            fig = px.bar(
                                quarterly_data,
                                x='Period',
                                y=metric_col,
                                title=f"Quarterly Comparison - {metric_col}"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Calculate QoQ growth
                            quarterly_data['QoQ_Growth'] = quarterly_data[metric_col].pct_change() * 100
                            st.dataframe(quarterly_data[['Period', metric_col, 'QoQ_Growth']], use_container_width=True)
                        
                        else:  # Custom Range
                            st.subheader("Custom Date Range Comparison")
                            
                            col1, col2 = st.columns(2)
                            
                            min_date = df_time[date_col].min().date()
                            max_date = df_time[date_col].max().date()
                            
                            with col1:
                                start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
                            with col2:
                                end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
                            
                            if start_date < end_date:
                                mask = (df_time[date_col].dt.date >= start_date) & (df_time[date_col].dt.date <= end_date)
                                filtered_df = df_time.loc[mask]
                                
                                daily_data = filtered_df.groupby(df_time[date_col].dt.date)[metric_col].sum()
                                
                                fig = px.line(
                                    x=daily_data.index,
                                    y=daily_data.values,
                                    title=f"{metric_col} from {start_date} to {end_date}",
                                    labels={'x': 'Date', 'y': metric_col}
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Summary stats
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Total", f"{filtered_df[metric_col].sum():,.2f}")
                                with col2:
                                    st.metric("Average", f"{filtered_df[metric_col].mean():,.2f}")
                                with col3:
                                    st.metric("Days", len(daily_data))
                            else:
                                st.error("End date must be after start date")
                    
            except Exception as e:
                st.error(f"Error processing date column: {str(e)}")
                st.info("This column may not contain valid date values.")
        else:
            st.info("ℹ️ No date columns detected in the data for time-based comparisons.")

# ============================================================================
# PAGE: EXPORT & REPORTS
# ============================================================================
elif page == "💾 Export & Reports":
    st.title("💾 Export & Reports")
    
    if not st.session_state.merged_data:
        st.warning("⚠️ Please process and merge data first.")
    else:
        st.markdown("""
        ### Download Options
        Export your processed data, insights, and visualizations.
        """)
        
        df = st.session_state.merged_data
        export_mgr = ExportManager()
        
        # Export merged data
        st.subheader("📊 Export Processed Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # CSV export
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f"telecom_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Excel export
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Merged Data', index=False)
            
            st.download_button(
                label="📥 Download as Excel",
                data=buffer.getvalue(),
                file_name=f"telecom_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        # Export insights
        st.markdown("---")
        st.subheader("📝 Export Insights Report")
        
        if st.session_state.insights:
            insights = st.session_state.insights
            
            # Create text report
            report = f"""
TELECOM ANALYTICS INSIGHTS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'='*80}
EXECUTIVE SUMMARY
{'='*80}

{insights['summary']}

{'='*80}
KEY FINDINGS
{'='*80}

"""
            for idx, finding in enumerate(insights['key_findings'], 1):
                report += f"{idx}. {finding}\n"
            
            report += f"""
{'='*80}
RECOMMENDATIONS
{'='*80}

"""
            for idx, rec in enumerate(insights['recommendations'], 1):
                report += f"{idx}. {rec}\n"
            
            # Add anomalies
            if st.session_state.anomalies:
                report += f"""
{'='*80}
ALERTS & ANOMALIES
{'='*80}

"""
                for anomaly in st.session_state.anomalies:
                    report += f"\n[{anomaly['severity'].upper()}] {anomaly['title']}\n"
                    report += f"{anomaly['description']}\n"
                    report += f"Recommendation: {anomaly['recommendation']}\n"
                    report += "-" * 80 + "\n"
            
            st.download_button(
                label="📥 Download Insights Report (TXT)",
                data=report,
                file_name=f"insights_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
        else:
            st.info("Generate insights first from the Alerts & Anomalies page.")
        
        # Data preview
        st.markdown("---")
        st.subheader("👀 Data Preview")
        st.dataframe(df.head(100), use_container_width=True)
        
        # Data statistics
        with st.expander("📊 Data Statistics"):
            st.write(df.describe())

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8rem;'>
    <p>📡 Telecom Analytics Platform</p>
    <p>v1.0 | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)
