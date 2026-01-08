"""
COMPLETE FIX FOR APP.PY AI INSIGHTS SECTION
Replace the entire AI Insights page section (around line 350-500)
"""

# AI Insights Page - FIXED VERSION
elif page == "AI Insights":
    st.header("📊 AI-Powered Insights")
    
    if st.session_state.ai_insights:
        ai_insights = st.session_state.ai_insights
        
        # Executive Summary
        st.markdown("### 📝 Executive Summary")
        if 'executive_summary' in ai_insights:
            st.text(ai_insights['executive_summary'])
        
        # Critical issues overview
        st.markdown("### 🚨 Critical Issues Overview")
        
        # Count issues by severity
        problems = ai_insights.get('problems', [])
        critical_count = len([p for p in problems if p.get('severity') == 'critical'])
        high_count = len([p for p in problems if p.get('severity') == 'high'])
        medium_count = len([p for p in problems if p.get('severity') == 'medium'])
        total_count = len(problems)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🔴 Critical", critical_count)
        with col2:
            st.metric("🟠 High", high_count)
        with col3:
            st.metric("🟡 Medium", medium_count)
        with col4:
            st.metric("Total", total_count)
        
        # Key Business Insights
        st.markdown("### 💡 Key Business Insights")
        st.info("💼 Board-Ready Insights - Ready to share with management")
        
        # Display key insights text
        if 'key_insights' in ai_insights and ai_insights['key_insights']:
            st.markdown(ai_insights['key_insights'])
        
        # Recommendations Section - FIXED
        st.markdown("### 🎯 Recommendations")
        
        if 'recommendations' in ai_insights:
            recommendations = ai_insights['recommendations']
            
            # If recommendations is a string, display it
            if isinstance(recommendations, str):
                st.markdown(recommendations)
            
            # If recommendations is a list of dicts (structured)
            elif isinstance(recommendations, list):
                for idx, rec in enumerate(recommendations, 1):
                    priority = rec.get('priority', 'MEDIUM')
                    
                    # Priority color
                    if priority == 'CRITICAL':
                        border_color = 'red'
                        emoji = '🔴'
                    elif priority == 'HIGH':
                        border_color = 'orange'
                        emoji = '🟡'
                    else:
                        border_color = 'gray'
                        emoji = '⚪'
                    
                    with st.expander(f"{emoji} Recommendation {idx}: {rec.get('title', 'N/A')} [{priority}]", expanded=(idx==1)):
                        st.markdown(f"**Problem:** {rec.get('problem', 'N/A')}")
                        
                        if 'affected_circles' in rec:
                            circles = rec['affected_circles']
                            if isinstance(circles, list):
                                st.markdown(f"**Affected Circles:** {', '.join(circles[:5])}")
                        
                        st.markdown(f"**Impact:** {rec.get('impact', 'N/A')}")
                        
                        st.markdown("**Recommended Actions:**")
                        actions = rec.get('actions', [])
                        if isinstance(actions, list):
                            for action in actions:
                                if isinstance(action, dict):
                                    st.markdown(f"- {action.get('action', action)}")
                                else:
                                    st.markdown(f"- {action}")
                        elif isinstance(actions, str):
                            st.markdown(actions)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("💰 Investment", rec.get('investment', 'TBD'))
                        with col2:
                            st.metric("⏱️ ROI Timeline", rec.get('roi', 'TBD'))
                        with col3:
                            result = rec.get('expected_result', 'TBD')
                            st.metric("🎯 Expected Result", "See below")
                        
                        st.success(f"**Expected Result:** {rec.get('expected_result', 'TBD')}")
        
        # Circle Analysis Section - FIXED
        st.markdown("### 📍 Circle-by-Circle Analysis")
        
        if 'circle_analysis' in ai_insights and ai_insights['circle_analysis']:
            circle_data = ai_insights['circle_analysis']
            
            # Priority filter
            all_priorities = list(set([c.get('priority', 'normal') for c in circle_data]))
            priority_filter = st.multiselect(
                "Filter by Priority",
                options=all_priorities,
                default=all_priorities
            )
            
            # Count by priority
            filtered_circles = [c for c in circle_data if c.get('priority', 'normal') in priority_filter]
            st.info(f"Showing {len(filtered_circles)} of {len(circle_data)} circles")
            
            # Display circles
            for circle_insight in filtered_circles[:10]:  # Show top 10
                priority = circle_insight.get('priority', 'normal').upper()
                circle_name = circle_insight.get('circle', 'Unknown')
                
                # Priority styling
                priority_config = {
                    'CRITICAL': {'emoji': '🔴', 'color': 'red'},
                    'HIGH': {'emoji': '🟡', 'color': 'orange'},
                    'NORMAL': {'emoji': '🟢', 'color': 'green'}
                }
                config = priority_config.get(priority, {'emoji': '⚪', 'color': 'gray'})
                
                with st.expander(f"{config['emoji']} **{circle_name}** [{priority}]", expanded=(priority=='CRITICAL')):
                    # Display metrics
                    metrics = circle_insight.get('metrics', {})
                    if metrics:
                        st.markdown("**📊 Key Metrics:**")
                        
                        # Create columns for metrics (3 per row)
                        metric_items = list(metrics.items())
                        for i in range(0, len(metric_items), 3):
                            cols = st.columns(3)
                            for j, (metric_name, metric_value) in enumerate(metric_items[i:i+3]):
                                with cols[j]:
                                    # Format metric value
                                    if isinstance(metric_value, dict):
                                        display_value = metric_value.get('value', 'N/A')
                                    else:
                                        display_value = metric_value
                                    
                                    st.metric(
                                        label=str(metric_name)[:30],  # Truncate long names
                                        value=str(display_value)
                                    )
                    
                    # Display problems
                    problems = circle_insight.get('problems', [])
                    if problems:
                        st.markdown("**⚠️ Issues Detected:**")
                        for prob in problems:
                            severity = prob.get('severity', 'medium')
                            
                            severity_config = {
                                'critical': {'emoji': '🔴', 'type': 'error'},
                                'high': {'emoji': '🟡', 'type': 'warning'},
                                'medium': {'emoji': '🟠', 'type': 'warning'},
                                'low': {'emoji': '🔵', 'type': 'info'}
                            }
                            sev_config = severity_config.get(severity, {'emoji': '⚪', 'type': 'info'})
                            
                            metric = prob.get('metric', 'Unknown')
                            value = prob.get('value', 'N/A')
                            target = prob.get('target', 'N/A')
                            gap = prob.get('gap', 'N/A')
                            
                            problem_text = f"{sev_config['emoji']} **{metric}**: Current {value}, Target {target} (Gap: {gap})"
                            
                            if sev_config['type'] == 'error':
                                st.error(problem_text)
                            elif sev_config['type'] == 'warning':
                                st.warning(problem_text)
                            else:
                                st.info(problem_text)
                    else:
                        st.success("✅ No critical issues detected")
        
        # Problems Summary Table
        if problems:
            st.markdown("### 📋 All Problems Summary")
            
            problems_data = []
            for prob in problems:
                problems_data.append({
                    'Circle': prob.get('circle', 'Unknown'),
                    'Priority': prob.get('circle_priority', 'normal').upper(),
                    'Type': prob.get('type', 'Unknown').title(),
                    'Metric': prob.get('metric', 'Unknown'),
                    'Current': f"{prob.get('value', 'N/A')}",
                    'Target': f"{prob.get('target', 'N/A')}",
                    'Gap': f"{prob.get('gap', 'N/A')}",
                    'Severity': prob.get('severity', 'medium').upper()
                })
            
            if problems_data:
                import pandas as pd
                df_problems = pd.DataFrame(problems_data)
                
                # Color code by severity
                def highlight_severity(row):
                    if row['Severity'] == 'CRITICAL':
                        return ['background-color: #ffcccc'] * len(row)
                    elif row['Severity'] == 'HIGH':
                        return ['background-color: #ffffcc'] * len(row)
                    elif row['Severity'] == 'MEDIUM':
                        return ['background-color: #fff4e6'] * len(row)
                    else:
                        return [''] * len(row)
                
                st.dataframe(
                    df_problems.style.apply(highlight_severity, axis=1),
                    use_container_width=True,
                    height=400
                )
                
                # Download button
                csv = df_problems.to_csv(index=False)
                st.download_button(
                    label="📥 Download Problems Report (CSV)",
                    data=csv,
                    file_name=f"telecom_problems_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
        
        # Network Summary
        if 'network_summary' in ai_insights:
            st.markdown("### 🌐 Network Summary")
            network_summary = ai_insights['network_summary']
            
            summary_cols = st.columns(4)
            col_idx = 0
            for key, value in network_summary.items():
                if isinstance(value, (int, float)) and key != 'total_circles':
                    with summary_cols[col_idx % 4]:
                        # Format key
                        display_key = key.replace('_', ' ').title()
                        # Format value
                        if isinstance(value, float):
                            if value > 1000:
                                display_value = f"{value:,.0f}"
                            else:
                                display_value = f"{value:.1f}"
                        else:
                            display_value = f"{value:,}"
                        
                        st.metric(display_key, display_value)
                        col_idx += 1
        
        # Debug info (collapsible)
        with st.expander("🔍 Debug Information"):
            st.json({
                'metadata': ai_insights.get('metadata', {}),
                'total_circles': len(ai_insights.get('circle_analysis', [])),
                'total_problems': len(problems),
                'total_recommendations': len(ai_insights.get('recommendations', []))
            })
    
    elif st.session_state.ai_error:
        st.error(f"AI Analysis Error: {st.session_state.ai_error}")
        st.info("Please try uploading your files again or contact support.")
    
    else:
        st.info("👆 Please upload and process files first to see AI insights.")
        st.markdown("""
        ### How it works:
        1. Go to **Upload & Process** page
        2. Upload 2-4 Excel/CSV files
        3. Click **Process & Analyze**
        4. AI will generate insights automatically
        5. Return here to view the insights
        """)
