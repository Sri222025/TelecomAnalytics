# ADD THIS TO THE NAVIGATION SECTION (after line 81)
# Add "📋 Data Summary" to the navigation list

# ADD THIS PAGE CODE AFTER THE "Upload & Process" PAGE (around line 330)

# ============================================================================
# PAGE: DATA SUMMARY & PIVOT TABLES
# ============================================================================
elif page == "📋 Data Summary":
    st.title("📋 Comprehensive Data Summary")
    
    if st.session_state.merged_data is None:
        st.warning("⚠️ Please process data first from the **📤 Upload & Process** page.")
    else:
        df = st.session_state.merged_data
        
        st.markdown("### 📊 Quick Overview")
        
        # Basic stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Total Columns", len(df.columns))
        with col3:
            st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
        with col4:
            st.metric("Missing Values", f"{df.isnull().sum().sum():,}")
        
        st.markdown("---")
        
        # PIVOT TABLE BUILDER
        st.subheader("🔄 Create Pivot Table Summary")
        
        st.markdown("Build custom summaries by selecting dimensions and metrics:")
        
        # Identify column types
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        date_cols = [col for col in df.columns if any(x in col.lower() for x in ['date', 'time'])]
        
        if categorical_cols and numeric_cols:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📌 Group By (Rows):**")
                row_dimension = st.multiselect(
                    "Select dimensions to group by",
                    categorical_cols,
                    default=[categorical_cols[0]] if categorical_cols else []
                )
            
            with col2:
                st.markdown("**📊 Measure (Values):**")
                value_columns = st.multiselect(
                    "Select metrics to analyze",
                    numeric_cols,
                    default=[numeric_cols[0]] if numeric_cols else []
                )
            
            # Aggregation function
            agg_function = st.selectbox(
                "**Calculation Method:**",
                ["Sum", "Average", "Count", "Min", "Max", "Median"],
                index=0
            )
            
            if st.button("📊 Generate Summary", type="primary", use_container_width=True):
                if row_dimension and value_columns:
                    with st.spinner("Generating summary..."):
                        try:
                            # Map aggregation function
                            agg_map = {
                                "Sum": "sum",
                                "Average": "mean",
                                "Count": "count",
                                "Min": "min",
                                "Max": "max",
                                "Median": "median"
                            }
                            
                            agg_func = agg_map[agg_function]
                            
                            # Create pivot summary
                            if len(row_dimension) == 1:
                                summary = df.groupby(row_dimension[0])[value_columns].agg(agg_func).round(2)
                                
                                st.success(f"✅ Summary by {row_dimension[0]}")
                                st.dataframe(summary, use_container_width=True)
                                
                                # Visualization
                                for col in value_columns:
                                    fig = px.bar(
                                        x=summary.index,
                                        y=summary[col] if len(value_columns) > 1 else summary.values,
                                        title=f"{agg_function} of {col} by {row_dimension[0]}",
                                        labels={'x': row_dimension[0], 'y': col}
                                    )
                                    fig.update_xaxis(tickangle=-45)
                                    st.plotly_chart(fig, use_container_width=True)
                                
                            elif len(row_dimension) == 2:
                                # Two-level grouping
                                summary = df.groupby(row_dimension)[value_columns].agg(agg_func).round(2)
                                
                                st.success(f"✅ Summary by {' & '.join(row_dimension)}")
                                st.dataframe(summary, use_container_width=True)
                                
                                # Pivot table format
                                for col in value_columns:
                                    pivot = df.pivot_table(
                                        values=col,
                                        index=row_dimension[0],
                                        columns=row_dimension[1],
                                        aggfunc=agg_func,
                                        fill_value=0
                                    ).round(2)
                                    
                                    st.markdown(f"**Pivot: {col}**")
                                    st.dataframe(pivot, use_container_width=True)
                                    
                                    # Heatmap
                                    fig = px.imshow(
                                        pivot,
                                        title=f"Heatmap: {agg_function} of {col}",
                                        labels=dict(x=row_dimension[1], y=row_dimension[0], color=col),
                                        aspect="auto"
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                            
                            else:
                                # Multi-level grouping
                                summary = df.groupby(row_dimension)[value_columns].agg(agg_func).round(2)
                                st.success(f"✅ Multi-level Summary")
                                st.dataframe(summary, use_container_width=True)
                            
                            # Export option
                            csv = summary.to_csv()
                            st.download_button(
                                "📥 Download Summary as CSV",
                                csv,
                                f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                "text/csv",
                                use_container_width=True
                            )
                            
                        except Exception as e:
                            st.error(f"Error creating summary: {str(e)}")
                else:
                    st.warning("⚠️ Please select at least one dimension and one metric.")
        
        else:
            st.info("ℹ️ No suitable columns found for pivot analysis.")
        
        st.markdown("---")
        
        # PRE-BUILT SUMMARIES
        st.subheader("📊 Pre-Built Summaries")
        
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "🔢 Statistics", "📋 Column Info", "🔍 Value Counts"])
        
        with tab1:
            st.markdown("**Dataset Overview:**")
            
            # Shape
            st.write(f"**Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns")
            
            # Column types
            col_types = df.dtypes.value_counts()
            st.write("**Column Types:**")
            st.dataframe(col_types.to_frame('Count'), use_container_width=True)
            
            # Missing data summary
            st.write("**Missing Data Summary:**")
            missing = df.isnull().sum()
            missing = missing[missing > 0].sort_values(ascending=False)
            if not missing.empty:
                missing_df = pd.DataFrame({
                    'Column': missing.index,
                    'Missing Count': missing.values,
                    'Percentage': (missing.values / len(df) * 100).round(2)
                })
                st.dataframe(missing_df, use_container_width=True)
            else:
                st.success("✅ No missing values found!")
        
        with tab2:
            st.markdown("**Statistical Summary:**")
            
            if numeric_cols:
                stats = df[numeric_cols].describe().T
                stats['missing'] = df[numeric_cols].isnull().sum()
                stats['unique'] = df[numeric_cols].nunique()
                st.dataframe(stats, use_container_width=True)
            else:
                st.info("No numeric columns available for statistics.")
        
        with tab3:
            st.markdown("**Column Information:**")
            
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.values,
                'Non-Null Count': df.count().values,
                'Null Count': df.isnull().sum().values,
                'Unique Values': df.nunique().values,
                'Sample Value': [df[col].iloc[0] if len(df) > 0 else None for col in df.columns]
            })
            st.dataframe(col_info, use_container_width=True)
        
        with tab4:
            st.markdown("**Value Counts for Categorical Columns:**")
            
            if categorical_cols:
                selected_cat = st.selectbox("Select column to analyze", categorical_cols)
                
                value_counts = df[selected_cat].value_counts().head(20)
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    fig = px.bar(
                        x=value_counts.index,
                        y=value_counts.values,
                        title=f"Top 20 Values in {selected_cat}",
                        labels={'x': selected_cat, 'y': 'Count'}
                    )
                    fig.update_xaxis(tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.dataframe(value_counts.to_frame('Count'), use_container_width=True)
            else:
                st.info("No categorical columns available.")
        
        st.markdown("---")
        
        # QUICK FILTERS
        st.subheader("🔍 Quick Data Filter")
        
        if categorical_cols:
            filter_col = st.selectbox("Filter by column:", categorical_cols)
            filter_values = st.multiselect(
                f"Select values from {filter_col}:",
                df[filter_col].unique().tolist()
            )
            
            if filter_values:
                filtered_df = df[df[filter_col].isin(filter_values)]
                st.success(f"✅ Filtered to {len(filtered_df):,} records")
                st.dataframe(filtered_df.head(50), use_container_width=True)
                
                # Export filtered data
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Filtered Data",
                    csv,
                    f"filtered_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv"
                )
