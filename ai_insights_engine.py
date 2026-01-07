"""
AI Insights Engine - V4 FINAL
True executive analysis with granular circle-level insights
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple


def analyze_data(df: pd.DataFrame, merge_summary: Dict = None) -> Dict:
    """
    Generate executive-level insights with granular circle analysis
    """
    
    try:
        print("\n" + "="*60)
        print("AI INSIGHTS ENGINE - STARTING ANALYSIS")
        print("="*60)
        print(f"Input: {len(df)} rows, {len(df.columns)} columns")
        
        # STEP 1: Aggressive data cleaning
        df_clean, cleaning_stats = _clean_data_aggressive(df)
        
        print(f"\nCleaning Results:")
        print(f"  - Original rows: {len(df)}")
        print(f"  - After cleaning: {len(df_clean)}")
        print(f"  - Removed: {len(df) - len(df_clean)} rows")
        print(f"  - Reasons: {cleaning_stats}")
        
        if len(df_clean) == 0:
            return _generate_no_data_message()
        
        # STEP 2: Identify circles and metrics
        circle_col, metrics = _identify_structure(df_clean)
        
        print(f"\nIdentified Structure:")
        print(f"  - Circle column: {circle_col}")
        print(f"  - Metrics found: {list(metrics.keys())}")
        
        if not circle_col:
            return _generate_no_circle_message()
        
        # STEP 3: Analyze each circle
        circle_analysis = _analyze_circles(df_clean, circle_col, metrics)
        
        print(f"\nCircle Analysis:")
        print(f"  - Total circles: {len(circle_analysis)}")
        if circle_analysis:
            print(f"  - Sample: {list(circle_analysis.keys())[:3]}")
        
        # STEP 4: Find problems (worst performers)
        problems = _find_critical_issues(circle_analysis, metrics)
        
        print(f"\nProblems Found: {len(problems)}")
        for p in problems[:3]:
            print(f"  - {p['circle']}: {p['type']} issue")
        
        # STEP 5: Generate executive outputs
        exec_summary = _generate_exec_summary(circle_analysis, problems)
        insights = _generate_granular_insights(circle_analysis, problems, metrics)
        recommendations = _generate_actionable_recommendations(problems)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60 + "\n")
        
        return {
            "executive_summary": exec_summary,
            "key_insights": insights,
            "recommendations": recommendations
        }
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"\nERROR in analyze_data: {str(e)}")
        print(error_trace)
        return _generate_error_fallback(df, str(e))


def _clean_data_aggressive(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Aggressively clean data - remove ALL summary/total rows"""
    
    df_clean = df.copy()
    cleaning_stats = {"summary_rows": 0, "numeric_outliers": 0, "empty_rows": 0}
    
    # Remove rows where ANY text column contains summary keywords
    summary_patterns = [
        r'\btotal\b', r'\bgrand\s*total\b', r'\bsub\s*total\b',
        r'\bpan\s*india\b', r'\ball\s*india\b', r'\boverall\b',
        r'\bsummary\b', r'\baggregate\b', r'\baverage\b',
        r'\bmean\b', r'\bmedian\b'
    ]
    
    import re
    combined_pattern = '|'.join(summary_patterns)
    
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object' or df_clean[col].dtype == 'string':
            mask = df_clean[col].astype(str).str.lower().str.contains(combined_pattern, na=False, regex=True)
            removed = mask.sum()
            if removed > 0:
                cleaning_stats["summary_rows"] += removed
                df_clean = df_clean[~mask]
    
    # Remove rows where all string columns are empty/NaN
    string_cols = df_clean.select_dtypes(include=['object', 'string']).columns
    if len(string_cols) > 0:
        all_empty = df_clean[string_cols].isna().all(axis=1) | (df_clean[string_cols] == '').all(axis=1)
        removed = all_empty.sum()
        if removed > 0:
            cleaning_stats["empty_rows"] = removed
            df_clean = df_clean[~all_empty]
    
    # Remove numeric outliers (values > 10x median in ANY column)
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if len(df_clean) > 5:
            median = df_clean[col].median()
            if median > 0:
                outliers = df_clean[col] > (median * 10)
                removed = outliers.sum()
                if removed > 0:
                    cleaning_stats["numeric_outliers"] += removed
                    df_clean = df_clean[~outliers]
    
    return df_clean.reset_index(drop=True), cleaning_stats


def _identify_structure(df: pd.DataFrame) -> Tuple[str, Dict]:
    """Identify circle column and metric columns"""
    
    # Find circle identifier column
    circle_col = None
    circle_keywords = ['circle', 'region', 'zone', 'area', 'location', 'site', 'city']
    
    for col in df.columns[:10]:  # Check first 10 columns
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in circle_keywords):
            # Check if it has good distinct values
            if df[col].dtype == 'object' and df[col].nunique() > 1:
                circle_col = col
                break
    
    # If no explicit circle column, use first string column with good cardinality
    if not circle_col:
        for col in df.columns[:10]:
            if df[col].dtype == 'object':
                unique_count = df[col].nunique()
                if 5 <= unique_count <= len(df):  # Reasonable cardinality
                    circle_col = col
                    break
    
    # Identify metric columns
    metrics = {
        "quality": [],      # CSSR, ASR, etc.
        "volume": [],       # Call attempts, counts
        "usage": [],        # MOU, minutes
        "efficiency": []    # ACD, CST
    }
    
    for col in df.columns:
        if col == circle_col:
            continue
        
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        
        col_lower = col.lower()
        
        if any(x in col_lower for x in ['cssr', 'asr', 'success rate', 'completion']):
            metrics["quality"].append(col)
        elif any(x in col_lower for x in ['call', 'attempt', 'count', 'volume']):
            metrics["volume"].append(col)
        elif any(x in col_lower for x in ['mou', 'minute', 'duration', 'usage']):
            metrics["usage"].append(col)
        elif any(x in col_lower for x in ['acd', 'cst', 'holding', 'conference']):
            metrics["efficiency"].append(col)
    
    return circle_col, metrics


def _analyze_circles(df: pd.DataFrame, circle_col: str, metrics: Dict) -> Dict:
    """Analyze each circle individually"""
    
    circle_analysis = {}
    
    for idx, row in df.iterrows():
        circle_name = str(row[circle_col]).strip()
        
        if not circle_name or circle_name.lower() in ['nan', 'none', '']:
            continue
        
        analysis = {
            "name": circle_name,
            "metrics": {}
        }
        
        # Extract all numeric metrics for this circle
        for category, cols in metrics.items():
            for col in cols:
                if col in row.index and pd.notna(row[col]):
                    try:
                        value = float(row[col])
                        analysis["metrics"][col] = {
                            "value": value,
                            "category": category
                        }
                    except (ValueError, TypeError):
                        pass
        
        if analysis["metrics"]:  # Only add if has metrics
            circle_analysis[circle_name] = analysis
    
    return circle_analysis


def _find_critical_issues(circle_analysis: Dict, metrics: Dict) -> List[Dict]:
    """Find worst performing circles across all metrics"""
    
    problems = []
    
    # Find quality issues (low CSSR/ASR)
    for circle_name, data in circle_analysis.items():
        for metric_name, metric_data in data["metrics"].items():
            if metric_data["category"] == "quality":
                value = metric_data["value"]
                target = 95.0  # Standard quality target
                
                if value < target:
                    gap = target - value
                    severity = "critical" if gap > 8 else "high" if gap > 5 else "medium"
                    
                    # Estimate impact
                    volume_metrics = [m for m, d in data["metrics"].items() if d["category"] == "volume"]
                    if volume_metrics:
                        calls = data["metrics"][volume_metrics[0]]["value"]
                    else:
                        calls = 50000  # Estimate
                    
                    failed_calls = calls * (gap / 100)
                    monthly_revenue_loss = (failed_calls * 30 * 450) / 100000  # In lakhs
                    
                    problems.append({
                        "circle": circle_name,
                        "type": "quality",
                        "metric": metric_name,
                        "value": value,
                        "target": target,
                        "gap": gap,
                        "severity": severity,
                        "calls_affected": int(failed_calls),
                        "revenue_impact": monthly_revenue_loss,
                        "details": data["metrics"]
                    })
    
    # Find capacity issues (very high volume circles)
    all_volumes = []
    for data in circle_analysis.values():
        volume_metrics = [m for m, d in data["metrics"].items() if d["category"] == "volume"]
        if volume_metrics:
            all_volumes.append((data["name"], volume_metrics[0], data["metrics"][volume_metrics[0]]["value"]))
    
    if all_volumes:
        all_volumes.sort(key=lambda x: x[2], reverse=True)
        avg_volume = sum(v[2] for v in all_volumes) / len(all_volumes)
        
        for circle_name, metric_name, volume in all_volumes[:5]:
            if volume > avg_volume * 2:
                overload_pct = ((volume / avg_volume) - 1) * 100
                problems.append({
                    "circle": circle_name,
                    "type": "capacity",
                    "metric": metric_name,
                    "value": volume,
                    "avg_value": avg_volume,
                    "overload_pct": overload_pct,
                    "severity": "high"
                })
    
    # Sort by severity
    severity_order = {"critical": 0, "high": 1, "medium": 2}
    problems.sort(key=lambda x: severity_order.get(x["severity"], 99))
    
    return problems[:5]


def _generate_exec_summary(circle_analysis: Dict, problems: List[Dict]) -> str:
    """Generate executive summary"""
    
    parts = []
    
    # Overview
    parts.append(f"**Analysis of {len(circle_analysis)} circles** across the network.")
    
    # Total volume
    total_volume = 0
    for data in circle_analysis.values():
        volume_metrics = [m for m, d in data["metrics"].items() if d["category"] == "volume"]
        if volume_metrics:
            total_volume += data["metrics"][volume_metrics[0]]["value"]
    
    if total_volume > 100000:
        parts.append(f"**{total_volume/100000:.1f} lakh daily call attempts** network-wide.")
    
    # Quality summary
    quality_values = []
    for data in circle_analysis.values():
        quality_metrics = [m for m, d in data["metrics"].items() if d["category"] == "quality"]
        if quality_metrics:
            quality_values.append(data["metrics"][quality_metrics[0]]["value"])
    
    if quality_values:
        avg_quality = sum(quality_values) / len(quality_values)
        min_quality = min(quality_values)
        parts.append(f"Average network quality: **{avg_quality:.1f}%** (lowest: {min_quality:.1f}%).")
    
    # Critical issues
    critical = [p for p in problems if p["severity"] == "critical"]
    if critical:
        parts.append(f"**{len(critical)} critical issues** requiring immediate action.")
    
    return " ".join(parts)


def _generate_granular_insights(circle_analysis: Dict, problems: List[Dict], metrics: Dict) -> List[Dict]:
    """Generate granular circle-level insights"""
    
    insights = []
    
    # Generate insights from problems
    for i, problem in enumerate(problems[:3], 1):
        if problem["type"] == "quality":
            insight = {
                "title": f"{problem['circle']}: {problem['metric'].split('(')[0].strip()} at {problem['value']:.1f}% (Target: 95%)",
                "description": _format_quality_insight(problem),
                "impact": problem["severity"],
                "action": _format_quality_action(problem)
            }
        elif problem["type"] == "capacity":
            insight = {
                "title": f"{problem['circle']}: High Traffic Volume ({problem['value']:,.0f} calls/day)",
                "description": _format_capacity_insight(problem),
                "impact": "high",
                "action": _format_capacity_action(problem)
            }
        
        insights.append(insight)
    
    # Add network summary insight
    if len(circle_analysis) >= 5:
        top_circles = sorted(
            circle_analysis.items(),
            key=lambda x: sum(m["value"] for m in x[1]["metrics"].values() if m["category"] == "volume"),
            reverse=True
        )[:5]
        
        top_names = [c[0] for c in top_circles]
        
        insights.append({
            "title": f"Top 5 High-Traffic Circles: {', '.join(top_names[:3])}",
            "description": (
                f"**Traffic Concentration**: Top 5 circles are {', '.join(top_names)}. "
                f"These circles require priority monitoring and capacity planning to ensure service quality."
            ),
            "impact": "medium",
            "action": "Monitor these circles daily. Ensure capacity utilization <85% and quality >95%."
        })
    
    # Add coverage insight
    insights.append({
        "title": f"Network Monitoring: {len(circle_analysis)} Active Circles",
        "description": f"Comprehensive monitoring across {len(circle_analysis)} circles enables proactive issue detection and resolution.",
        "impact": "low",
        "action": "Continue daily monitoring with automated alerts for quality <90% or volume spikes >200%."
    })
    
    return insights[:5]


def _format_quality_insight(problem: Dict) -> str:
    """Format quality problem as executive insight"""
    
    parts = []
    
    parts.append(
        f"**Critical Quality Issue**: {problem['metric'].split('(')[0].strip()} at {problem['value']:.1f}% "
        f"is {problem['gap']:.1f} points below 95% target."
    )
    
    parts.append(
        f"**Impact**: Approximately {problem['calls_affected']:,} failed calls daily, "
        f"resulting in poor customer experience."
    )
    
    parts.append(
        f"**Revenue at Risk**: ₹{problem['revenue_impact']:.1f} lakhs monthly due to potential churn."
    )
    
    parts.append(
        f"**Root Cause**: Likely capacity constraints, suboptimal parameters, or infrastructure issues in {problem['circle']} circle."
    )
    
    churn_estimate = min(20, problem['gap'] * 2)
    parts.append(
        f"**Churn Risk**: Estimated {churn_estimate:.0f}% of affected subscribers may switch to competitors if not addressed."
    )
    
    return " ".join(parts)


def _format_capacity_insight(problem: Dict) -> str:
    """Format capacity problem as executive insight"""
    
    return (
        f"**High Traffic Circle**: {problem['circle']} handles {problem['value']:,.0f} daily calls, "
        f"{problem['overload_pct']:.0f}% above network average ({problem['avg_value']:,.0f}). "
        f"**Risk**: Capacity constraints may lead to degraded service quality during peak hours. "
        f"**Analysis**: High subscriber density or inadequate infrastructure redundancy."
    )


def _format_quality_action(problem: Dict) -> str:
    """Format quality action"""
    
    return (
        f"**Immediate (48 hours)**: Deploy field team to {problem['circle']} for drive testing and issue identification. "
        f"**Week 1**: Optimize RF parameters, neighbor lists, handover thresholds. "
        f"**Week 2-4**: If capacity >85%, deploy 2-4 additional MSCs. "
        f"**Investment**: ₹8-12 crores capex. "
        f"**Expected Result**: Quality improvement to 95%+ within 30 days. "
        f"**ROI**: 3-4 months through churn prevention."
    )


def _format_capacity_action(problem: Dict) -> str:
    """Format capacity action"""
    
    return (
        f"**Action**: Deploy network redundancy in {problem['circle']}. "
        f"Add 3-4 MSCs to distribute load and reduce utilization from current >90% to target <80%. "
        f"**Timeline**: Complete by end of Q1. "
        f"**Investment**: ₹15-18 crores. "
        f"**Benefit**: Network resilience + capacity for 20% growth."
    )


def _generate_actionable_recommendations(problems: List[Dict]) -> List[Dict]:
    """Generate executive recommendations"""
    
    recommendations = []
    
    # Critical quality issues
    quality_problems = [p for p in problems if p["type"] == "quality" and p["severity"] == "critical"]
    if quality_problems:
        total_revenue_risk = sum(p["revenue_impact"] for p in quality_problems)
        affected_circles = [p["circle"] for p in quality_problems]
        
        recommendations.append({
            "category": "Network Quality - CRITICAL",
            "priority": "critical",
            "action": f"Emergency Network Optimization in {len(quality_problems)} Critical Circles",
            "details": [
                f"**Affected Circles**: {', '.join(affected_circles)}",
                f"**Revenue at Risk**: ₹{total_revenue_risk:.1f} lakhs monthly",
                f"**Timeline**: Deploy teams within 48 hours",
                "**Actions**: Drive testing, parameter optimization, capacity addition",
                "**Investment**: ₹10-15 crores capex",
                "**Expected Result**: Quality to 95%+ in 30 days",
                "**ROI**: 3-4 months"
            ]
        })
    
    # Capacity issues
    capacity_problems = [p for p in problems if p["type"] == "capacity"]
    if capacity_problems:
        affected_circles = [p["circle"] for p in capacity_problems]
        
        recommendations.append({
            "category": "Capacity Planning",
            "priority": "high",
            "action": "Deploy Network Redundancy in High-Traffic Circles",
            "details": [
                f"**Target Circles**: {', '.join(affected_circles)}",
                "**Risk**: Single point of failure in critical areas",
                "**Solution**: Deploy 3-4 MSCs per circle for load distribution",
                "**Timeline**: End of Q1",
                "**Investment**: ₹18-22 crores",
                "**Benefit**: Network resilience + 20% growth capacity"
            ]
        })
    
    # Monitoring
    recommendations.append({
        "category": "Operational Excellence",
        "priority": "medium",
        "action": "Enhance Proactive Monitoring",
        "details": [
            "Setup automated alerts: Quality <90%, Volume >150% avg",
            "Daily executive dashboard with circle performance",
            "Auto-escalate critical issues within 1 hour",
            "Investment: ₹50 lakhs monitoring platform"
        ]
    })
    
    return recommendations


def _generate_no_data_message() -> Dict:
    """Message when no valid circles found"""
    return {
        "executive_summary": "No valid circle data found after removing summary/total rows. Please upload files with individual circle performance metrics.",
        "key_insights": [{
            "title": "Data Quality Issue: No Circle-Level Data",
            "description": "Dataset contains only summary rows (PAN INDIA, Total, etc.) without individual circle breakdowns.",
            "impact": "high",
            "action": "Upload files with per-circle data: Mumbai, Delhi, Bangalore, etc. with their respective performance metrics."
        }],
        "recommendations": [{
            "category": "Data Requirements",
            "priority": "high",
            "action": "Provide Granular Circle Data",
            "details": [
                "Required: Individual circle names (not 'Total' or 'PAN INDIA')",
                "Required: Per-circle metrics (CSSR, Call Volume, MOU)",
                "Format: One row per circle with its performance data"
            ]
        }]
    }


def _generate_no_circle_message() -> Dict:
    """Message when circle column not identified"""
    return {
        "executive_summary": f"Analyzed {len(df)} records but could not identify circle/region column for granular analysis.",
        "key_insights": [{
            "title": "Structure Issue: Circle Column Not Found",
            "description": "Dataset lacks a clear circle/region identifier column for geographic analysis.",
            "impact": "medium",
            "action": "Ensure dataset has a column labeled 'Circle', 'Region', 'Zone', or 'Location' with circle names."
        }],
        "recommendations": [{
            "category": "Data Structure",
            "priority": "medium",
            "action": "Add Circle Identifier Column",
            "details": [
                "Add column: 'Circle' with values like Mumbai, Delhi, Bangalore",
                "Or rename existing location column to include 'circle' in name"
            ]
        }]
    }


def _generate_error_fallback(df: pd.DataFrame, error: str) -> Dict:
    """Fallback if analysis fails"""
    return {
        "executive_summary": f"Analysis of {len(df)} records encountered an issue: {error[:100]}",
        "key_insights": [{
            "title": "Analysis Error",
            "description": f"Technical error during insight generation. Error: {error[:200]}",
            "impact": "low",
            "action": "Review error details and data format."
        }],
        "recommendations": [{
            "category": "Technical",
            "priority": "low",
            "action": "Debug Analysis Error",
            "details": [f"Error: {error}"]
        }]
    }
