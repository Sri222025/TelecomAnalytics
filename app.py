import streamlit as st

st.set_page_config(page_title="Telecom Analytics", page_icon="📡", layout="wide")

# Session state
for key in ['data', 'anomalies']:
    if key not in st.session_state:
        st.session_state[key] = None

# Sidebar
page = st.sidebar.radio("Menu", ["Home", "Upload", "Alerts", "Export"])

# HOME
if page == "Home":
    st.title("📡 Telecom Analytics Platform")
    st.write("Go to Upload to start")
    if st.session_state.data is not None:
        st.success(f"✅ Data loaded: {len(st.session_state.data):,} records")

# UPLOAD
elif page == "Upload":
    st.title("Upload Files")
    files = st.file_uploader("Upload Excel/CSV", type=['xlsx','csv'], accept_multiple_files=True)
    
    if files and st.button("Process"):
        import pandas as pd
        try:
            dfs = []
            for f in files:
                if f.name.endswith('.csv'):
                    dfs.append(pd.read_csv(f))
                else:
                    dfs.append(pd.read_excel(f))
            
            st.session_state.data = pd.concat(dfs, ignore_index=True)
            st.success(f"✅ Loaded {len(st.session_state.data):,} records")
            st.dataframe(st.session_state.data.head())
        except Exception as e:
            st.error(f"Error: {e}")

# ALERTS
elif page == "Alerts":
    st.title("Alerts")
    if st.session_state.data is None:
        st.warning("Upload data first")
    else:
        df = st.session_state.data
        st.write(f"Records: {len(df):,}")
        st.write(f"Columns: {len(df.columns)}")
        
        # Simple checks
        missing = df.isnull().sum().sum()
        if missing > len(df) * 0.1:
            st.error(f"⚠️ High missing data: {missing:,} values")
        
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            st.warning(f"⚠️ Duplicates found: {duplicates:,}")
        
        st.success("✅ Data looks good")

# EXPORT
elif page == "Export":
    st.title("Export")
    if st.session_state.data is None:
        st.warning("Upload data first")
    else:
        csv = st.session_state.data.to_csv(index=False)
        st.download_button("Download CSV", csv, "data.csv")
        st.dataframe(st.session_state.data.head(100))
