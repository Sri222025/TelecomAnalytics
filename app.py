import streamlit as st
import pandas as pd
import io

# Must be FIRST Streamlit command
st.set_page_config(page_title="Telecom Analytics", page_icon="📡", layout="wide")

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'files_info' not in st.session_state:
    st.session_state.files_info = []

# Title
st.title("📡 Telecom Analytics Platform")

# Sidebar navigation
page = st.sidebar.radio("Navigation", ["🏠 Home", "📤 Upload Files", "📊 View Data", "🔍 Quick Insights"])

# HOME PAGE
if page == "🏠 Home":
    st.header("Welcome to Telecom Analytics")
    
    if st.session_state.data is None:
        st.info("👈 Upload your Excel/CSV files to get started")
    else:
        st.success(f"✅ Data loaded: {len(st.session_state.data):,} records")
        st.metric("Total Records", f"{len(st.session_state.data):,}")
        st.metric("Total Columns", len(st.session_state.data.columns))

# UPLOAD PAGE
elif page == "📤 Upload Files":
    st.header("Upload Your Files")
    
    uploaded_files = st.file_uploader(
        "Upload Excel or CSV files",
        type=['xlsx', 'xls', 'csv'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.info(f"📁 {len(uploaded_files)} file(s) selected")
        
        if st.button("🚀 Process Files", type="primary"):
            with st.spinner("Processing files..."):
                all_data = []
                files_info = []
                
                for file in uploaded_files:
                    try:
                        # Read file
                        if file.name.endswith('.csv'):
                            df = pd.read_csv(file)
                        else:
                            # Read all sheets from Excel
                            excel_file = pd.ExcelFile(file)
                            sheet_dfs = []
                            for sheet_name in excel_file.sheet_names:
                                sheet_df = pd.read_excel(file, sheet_name=sheet_name)
                                sheet_df['_source_file'] = file.name
                                sheet_df['_source_sheet'] = sheet_name
                                sheet_dfs.append(sheet_df)
                            df = pd.concat(sheet_dfs, ignore_index=True)
                        
                        # Add source tracking
                        if '_source_file' not in df.columns:
                            df['_source_file'] = file.name
                        
                        all_data.append(df)
                        files_info.append({
                            'name': file.name,
                            'rows': len(df),
                            'columns': len(df.columns)
                        })
                        
                        st.success(f"✅ {file.name}: {len(df):,} rows, {len(df.columns)} columns")
                        
                    except Exception as e:
                        st.error(f"❌ Error reading {file.name}: {str(e)}")
                
                if all_data:
                    # Combine all data
                    st.session_state.data = pd.concat(all_data, ignore_index=True)
                    st.session_state.files_info = files_info
                    
                    st.success(f"🎉 Successfully processed {len(uploaded_files)} file(s)")
                    st.success(f"📊 Total records: {len(st.session_state.data):,}")
                    st.balloons()

# VIEW DATA PAGE
elif page == "📊 View Data":
    st.header("Data Explorer")
    
    if st.session_state.data is None:
        st.warning("⚠️ No data loaded. Please upload files first.")
    else:
        df = st.session_state.data
        
        # Quick stats
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Records", f"{len(df):,}")
        col2.metric("Total Columns", len(df.columns))
        col3.metric("Files Processed", len(st.session_state.files_info))
        col4.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        # File breakdown
        st.subheader("📁 Files Breakdown")
        for info in st.session_state.files_info:
            st.write(f"**{info['name']}**: {info['rows']:,} rows × {info['columns']} columns")
        
        # Data preview
        st.subheader("📋 Data Preview")
        
        # Column selector
        all_columns = df.columns.tolist()
        selected_columns = st.multiselect(
            "Select columns to display",
            all_columns,
            default=all_columns[:10] if len(all_columns) > 10 else all_columns
        )
        
        if selected_columns:
            st.dataframe(df[selected_columns].head(100), use_container_width=True)
        
        # Column info
        with st.expander("📊 Column Information"):
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.astype(str),
                'Non-Null': df.notna().sum(),
                'Null Count': df.isna().sum(),
                'Unique Values': [df[col].nunique() for col in df.columns]
            })
            st.dataframe(col_info, use_container_width=True)
        
        # Download processed data
        st.subheader("💾 Download")
        
        # Convert to CSV
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name="telecom_data_processed.csv",
            mime="text/csv"
        )

# INSIGHTS PAGE
elif page == "🔍 Quick Insights":
    st.header("Quick Insights")
    
    if st.session_state.data is None:
        st.warning("⚠️ No data loaded. Please upload files first.")
    else:
        df = st.session_state.data
        
        st.subheader("📈 Data Quality Report")
        
        # Missing data analysis
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        
        if missing.sum() > 0:
            st.warning("⚠️ Missing Data Detected")
            missing_df = pd.DataFrame({
                'Column': missing.index,
                'Missing Count': missing.values,
                'Missing %': missing_pct.values
            })
            missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing %', ascending=False)
            st.dataframe(missing_df, use_container_width=True)
        else:
            st.success("✅ No missing data")
        
        # Duplicate check
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            st.warning(f"⚠️ Found {duplicates:,} duplicate rows ({duplicates/len(df)*100:.2f}%)")
        else:
            st.success("✅ No duplicate rows")
        
        # Numeric columns summary
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            st.subheader("📊 Numeric Columns Summary")
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)
        
        # Date columns summary
        date_cols = df.select_dtypes(include=['datetime64']).columns
        if len(date_cols) > 0:
            st.subheader("📅 Date Range Information")
            for col in date_cols:
                min_date = df[col].min()
                max_date = df[col].max()
                st.write(f"**{col}**: {min_date} to {max_date}")

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("Telecom Analytics Platform v2.0")
