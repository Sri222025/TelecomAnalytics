"""
AI Insights Engine - V6 ENHANCED
Smart data cleaning that preserves real data while removing summaries
Enhanced structure detection and problem identification
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import re


def analyze_data(df: pd.DataFrame, merge_summary: Dict = None) -> Dict:
    """
    Generate executive-level insights with smart data cleaning
    """
    
    try:
        print("\n" + "="*60)
        print("AI INSIGHTS ENGINE V6 - ENHANCED ANALYSIS")
        print("="*60)
        print(f"Input: {len(df)} rows, {len(df.columns)} columns")
        print(f"Columns: {list(df.columns[:10])}")
        
        # STEP 1: Identify structure FIRST (before cleaning)
        circle_col, metrics = _identify_structure(df)
        
        print(f"\nStructure Identified:")
        print(f"  Circle column: {circle_col}")
        print(f"  Quality metrics: {len(metrics['quality'])}")
        print(f"  Volume metrics: {len(metrics['volume'])}")
        print(f"  Usage metrics: {len(metrics['usage'])}")
        
        # STEP 2: Smart cleaning (only removes summary rows, not data)
        df_clean, removed_rows = _smart_clean_data(df, circle_col)
        
        print(f"\nSmart Cleaning Results:")
        print(f"  Original rows: {len(df)}")
        print(f"  After cleaning: {len(df_clean)}")
        print(f"  Removed rows: {len(removed_rows)}")
        if removed_rows:
            print(f"  Removed values: {removed_rows[:5]}")
        
        if len(df_clean) == 0:
            print("\n⚠️ WARNING: All data removed during cleaning!")
            print("Using original data instead...")
            df_clean = df.copy()
        
        # STEP 3: Analyze circles
        circle_analysis = _analyze_circles(df_clean, circle_col, metrics)
        
        print(f"\nCircle Analysis:")
        print(f"  Total unique circles: {len(circle_analysis)}")
        if circle_analysis:
            print(f"  Circles found: {list(circle_analysis.keys())[:5]}")
        
        if len(circle_analysis) == 0:
            return _generate_no_circles_found(df, circle_col)
        
        # STEP 4: Find problems
        problems = _find_critical_issues(circle_analysis, metrics)
        
        print(f"\nProblems Detected: {len(problems)}")
        for p in problems[:3]:
            print(f"  - {p['circle']}: {p.get('metric', 'N/A')} = {p.get('value', 'N/A')}")
        
        # STEP 5: Generate outputs
        exec_summary = _generate_exec_summary(circle_analysis, problems, metrics)
        insights = _generate_insights(circle_analysis, problems, metrics)
        recommendations = _generate_recommendations(problems)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60 + "\n")
        
        return {
            "executive_summary": exec_summary,
            "key_insights": insights,
            "recommendations": recommendations,
            "circle_analysis": circle_analysis,
            "problems": problems
        }
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"\n❌ ERROR: {str(e)}")
        print(error_trace)
        return _generate_error_fallback(df, str(e))


def _identify_structure(df: pd.DataFrame) -> Tuple[str, Dict]:
    """Identify circle column and metrics BEFORE cleaning - ENHANCED"""
    
    # Find circle column with multiple strategies
    circle_col = None
    
    # Strategy 1: Look for explicit circle/region columns (case-insensitive)
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['circle', 'region', 'zone', 'area', 'location', 'state', 'city']):
            if df[col].dtype in ['object', 'string'] or df[col].dtype.name == 'object':
                unique = df[col].nunique()
                non_null = df[col].count()
                # Check if it has reasonable cardinality (not too many unique values)
                if 2 <= unique <= min(50, non_null * 0.8) and non_null > 0:
                    circle_col = col
                    print(f"  Found circle column by name: {col} ({unique} unique values)")
                    break
    
    # Strategy 2: Find first string column with good cardinality
    if not circle_col:
        for col in df.columns[:20]:  # Check first 20 columns
            if df[col].dtype in ['object', 'string'] or df[col].dtype.name == 'object':
                unique = df[col].nunique()
                non_null = df[col].count()
                # Good cardinality: between 2 and 50 unique values
                if 2 <= unique <= 50 and non_null > len(df) * 0.3:
                    # Check if values look like circle names (not numbers, not too long)
                    sample = df[col].dropna().head(10)
                    if len(sample) > 0:
                        sample_str = sample.astype(str)
                        # Most values should be text (not pure numbers)
                        text_count = sum(1 for v in sample_str if not str(v).replace('.','').replace('-','').isdigit())
                        if text_count >= len(sample) * 0.7:
                            circle_col = col
                            print(f"  Found circle column by cardinality: {col} ({unique} unique values)")
                            break
    
    # Identify metric columns with enhanced detection
    metrics = {
        "quality": [],
        "volume": [],
        "usage": [],
        "efficiency": []
    }
    
    for col in df.columns:
        if col == circle_col or col.startswith('_'):
            continue
        
        # Skip non-numeric columns
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        
        col_lower = col.lower()
        
        # Quality metrics (percentage based, usually 0-100)
        quality_keywords = ['cssr', 'asr', 'success', 'completion', 'rate(%)', 'rate', 
                           'quality', 'efficiency', 'performance', 'score', 'percent', '%']
        if any(x in col_lower for x in quality_keywords):
            # Check if values are in percentage range
            sample = df[col].dropna()
            if len(sample) > 0:
                sample_vals = sample.head(100)
                # If most values are between 0-100, it's likely a percentage
                in_range = sum(1 for v in sample_vals if 0 <= v <= 100)
                if in_range >= len(sample_vals) * 0.7:
                    metrics["quality"].append(col)
                    continue
        
        # Volume metrics (counts, usually large numbers)
        volume_keywords = ['call', 'attempt', 'count', 'volume', 'total', 'number', 
                          'traffic', 'session', 'request']
        if any(x in col_lower for x in volume_keywords):
            metrics["volume"].append(col)
            continue
        
        # Usage metrics (minutes/duration)
        usage_keywords = ['mou', 'minute', 'duration', 'usage', 'sec', 'time', 
                         'hour', 'second', 'min']
        if any(x in col_lower for x in usage_keywords):
            metrics["usage"].append(col)
            continue
        
        # Efficiency metrics
        efficiency_keywords = ['acd', 'cst', 'holding', 'conference', 'throughput', 
                              'latency', 'delay']
        if any(x in col_lower for x in efficiency_keywords):
            metrics["efficiency"].append(col)
            continue
    
    # If no metrics found, try to infer from data patterns
    if not any(metrics.values()):
        print("  ⚠️ No metrics found by name, inferring from data patterns...")
        for col in df.columns:
            if col == circle_col or col.startswith('_') or not pd.api.types.is_numeric_dtype(df[col]):
                continue
            
            sample = df[col].dropna().head(100)
            if len(sample) == 0:
                continue
            
            mean_val = sample.mean()
            max_val = sample.max()
            
            # If values are 0-100, likely quality
            if 0 <= mean_val <= 100 and max_val <= 100:
                metrics["quality"].append(col)
            # If values are large (>1000), likely volume
            elif mean_val > 1000:
                metrics["volume"].append(col)
            # If values are medium (10-1000), could be usage
            elif 10 <= mean_val <= 1000:
                metrics["usage"].append(col)
    
    return circle_col, metrics


def _smart_clean_data(df: pd.DataFrame, circle_col: str) -> Tuple[pd.DataFrame, List]:
    """
    Smart cleaning: Only remove rows where CIRCLE column contains summary keywords
    Preserves real data
    """
    
    if not circle_col or circle_col not in df.columns:
        print("  ⚠️ No circle column found, skipping cleaning")
        return df.copy(), []
    
    df_clean = df.copy()
    removed_rows = []
    
    # Summary keywords to check ONLY in circle column
    summary_patterns = [
        r'\btotal\b',
        r'\bgrand\s*total\b',
        r'\bsub\s*total\b',
        r'\bpan\s*india\b',
        r'\ball\s*india\b',
        r'\boverall\b',
        r'\bsummary\b',
        r'\baggregate\b',
        r'\bconsolidated\b',
        r'\bcombined\b'
    ]
    
    combined_pattern = '|'.join(summary_patterns)
    
    # Check ONLY the circle column
    circle_values = df_clean[circle_col].astype(str).str.lower()
    mask = circle_values.str.contains(combined_pattern, na=False, regex=True, case=False)
    
    # Store removed values for debug
    if mask.any():
        removed_rows = df_clean[mask][circle_col].unique().tolist()
    
    # Remove summary rows
    df_clean = df_clean[~mask]
    
    # Safety check: If we removed >80% of data, DON'T clean
    if len(df_clean) < len(df) * 0.2:
        print(f"  ⚠️ Cleaning removed {len(df) - len(df_clean)} rows ({(1 - len(df_clean)/len(df))*100:.0f}%)")
        print(f"  This seems too aggressive, using original data")
        return df.copy(), []
    
    return df_clean.reset_index(drop=True), removed_rows


def _analyze_circles(df: pd.DataFrame, circle_col: str, metrics: Dict) -> Dict:
    """Analyze each circle - ENHANCED"""
    
    if not circle_col:
        return {}
    
    circle_analysis = {}
    
    for idx, row in df.iterrows():
        circle_name = str(row[circle_col]).strip()
        
        # Skip empty/invalid names
        if not circle_name or circle_name.lower() in ['nan', 'none', '', 'null', 'na']:
            continue
        
        # Skip if looks like a number (probably not a circle name)
        try:
            float(circle_name)
            continue
        except:
            pass
        
        # Skip if too long (probably not a circle name)
        if len(circle_name) > 50:
            continue
        
        # Initialize or get existing analysis
        if circle_name not in circle_analysis:
            circle_analysis[circle_name] = {
                "name": circle_name,
                "metrics": {},
                "row_count": 0
            }
        
        circle_analysis[circle_name]["row_count"] += 1
        
        # Extract all metrics
        for category, cols in metrics.items():
            for col in cols:
                if col in row.index and pd.notna(row[col]):
                    try:
                        value = float(row[col])
                        
                        # Skip obviously invalid values
                        if category == "quality" and (value < 0 or value > 100):
                            continue
                        
                        # If metric already exists, aggregate (average)
                        if col in circle_analysis[circle_name]["metrics"]:
                            existing = circle_analysis[circle_name]["metrics"][col]
                            # Average multiple values
                            existing["value"] = (existing["value"] + value) / 2
                        else:
                            circle_analysis[circle_name]["metrics"][col] = {
                                "value": value,
                                "category": category
                            }
                    except (ValueError, TypeError):
                        pass
    
    # Remove circles with no metrics
    circle_analysis = {k: v for k, v in circle_analysis.items() if v["metrics"]}
    
    return circle_analysis


def _find_critical_issues(circle_analysis: Dict, metrics: Dict) -> List[Dict]:
    """Find worst performers - ENHANCED"""
    
    problems = []
    
    if not circle_analysis:
        return problems
    
    # Quality issues - ENHANCED
    for circle_name, data in circle_analysis.items():
        for metric_name, metric_data in data["metrics"].items():
            if metric_data["category"] == "quality":
                value = metric_data["value"]
                target = 95.0
                
                if value < target:
                    gap = target - value
                    
                    # Get volume for this circle
                    volume_metrics = [m for m, d in data["metrics"].items() if d["category"] == "volume"]
                    calls = data["metrics"][volume_metrics[0]]["value"] if volume_metrics else 50000
                    
                    failed_calls = calls * (gap / 100)
                    revenue_loss = (failed_calls * 30 * 450) / 100000  # Lakhs/month
                    
                    problems.append({
                        "circle": circle_name,
                        "type": "quality",
                        "metric": metric_name,
                        "value": round(value, 2),
                        "target": target,
                        "gap": round(gap, 2),
                        "severity": "critical" if gap > 7 else "high" if gap > 4 else "medium",
                        "calls_affected": int(failed_calls),
                        "revenue_loss": round(revenue_loss, 2)
                    })
    
    # Capacity issues - ENHANCED
    all_volumes = []
    for circle_name, data in circle_analysis.items():
        volume_metrics = [m for m, d in data["metrics"].items() if d["category"] == "volume"]
        if volume_metrics:
            vol = data["metrics"][volume_metrics[0]]["value"]
            all_volumes.append((circle_name, volume_metrics[0], vol))
    
    if len(all_volumes) > 3:
        all_volumes.sort(key=lambda x: x[2], reverse=True)
        avg_vol = sum(v[2] for v in all_volumes) / len(all_volumes)
        
        for circle_name, metric_name, vol in all_volumes[:5]:  # Top 5
            if vol > avg_vol * 1.5:
                problems.append({
                    "circle": circle_name,
                    "type": "capacity",
                    "metric": metric_name,
                    "value": round(vol, 0),
                    "avg_value": round(avg_vol, 0),
                    "overload_pct": round(((vol/avg_vol) - 1) * 100, 1),
                    "severity": "high"
                })
    
    # Usage anomalies - NEW
    all_usage = []
    for circle_name, data in circle_analysis.items():
        usage_metrics = [m for m, d in data["metrics"].items() if d["category"] == "usage"]
        if usage_metrics:
            usage = data["metrics"][usage_metrics[0]]["value"]
            all_usage.append((circle_name, usage_metrics[0], usage))
    
    if len(all_usage) > 3:
        all_usage.sort(key=lambda x: x[2], reverse=True)
        avg_usage = sum(u[2] for u in all_usage) / len(all_usage)
        
        # Find circles with unusually low usage (potential issues)
        for circle_name, metric_name, usage in all_usage:
            if usage < avg_usage * 0.5:
                problems.append({
                    "circle": circle_name,
                    "type": "usage",
                    "metric": metric_name,
                    "value": round(usage, 2),
                    "avg_value": round(avg_usage, 2),
                    "severity": "medium"
                })
    
    # Sort by severity
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    problems.sort(key=lambda x: severity_order.get(x.get("severity", "medium"), 99))
    
    return problems[:10]  # Return top 10 problems


def _generate_exec_summary(circle_analysis: Dict, problems: List[Dict], metrics: Dict) -> str:
    """Generate executive summary - ENHANCED"""
    
    parts = []
    
    parts.append(f"**Analyzed {len(circle_analysis)} circles** across the telecom network.")
    
    # Total volume
    total_volume = 0
    for data in circle_analysis.values():
        vol_metrics = [m for m, d in data["metrics"].items() if d["category"] == "volume"]
        if vol_metrics:
            total_volume += data["metrics"][vol_metrics[0]]["value"]
    
    if total_volume > 100000:
        parts.append(f"**{total_volume/100000:.1f} lakh daily call attempts** network-wide.")
    elif total_volume > 0:
        parts.append(f"**{total_volume:,.0f} daily call attempts** network-wide.")
    
    # Quality metrics
    quality_vals = []
    for data in circle_analysis.values():
        qual_metrics = [m for m, d in data["metrics"].items() if d["category"] == "quality"]
        if qual_metrics:
            quality_vals.append(data["metrics"][qual_metrics[0]]["value"])
    
    if quality_vals:
        avg_q = sum(quality_vals) / len(quality_vals)
        min_q = min(quality_vals)
        max_q = max(quality_vals)
        parts.append(f"Average quality: **{avg_q:.1f}%** (range: {min_q:.1f}% - {max_q:.1f}%).")
    
    # Issues summary
    critical = [p for p in problems if p.get("severity") == "critical"]
    high = [p for p in problems if p.get("severity") == "high"]
    
    if critical:
        parts.append(f"**{len(critical)} critical issues** identified requiring immediate action.")
    elif high:
        parts.append(f"**{len(high)} high-priority issues** identified.")
    elif problems:
        parts.append(f"**{len(problems)} issues** identified for review.")
    else:
        parts.append("**No critical issues** detected. Network performance is within acceptable parameters.")
    
    return " ".join(parts)


def _generate_insights(circle_analysis: Dict, problems: List[Dict], metrics: Dict) -> List[Dict]:
    """Generate insights - ENHANCED"""
    
    insights = []
    
    # Problem-based insights
    for problem in problems[:5]:  # Top 5 problems
        if problem["type"] == "quality":
            insights.append({
                "title": f"{problem['circle']}: {problem['metric'].split('(')[0].strip()} at {problem['value']:.1f}% (Target: 95%)",
                "description": (
                    f"**Quality Issue**: {problem['gap']:.1f} points below target. "
                    f"**Impact**: {problem['calls_affected']:,} failed calls daily. "
                    f"**Revenue Risk**: ₹{problem['revenue_loss']:.1f}L monthly. "
                    f"**Root Cause**: Likely capacity constraints or parameter issues."
                ),
                "impact": problem["severity"],
                "action": (
                    f"Deploy optimization team within 48 hours. "
                    f"Week 1: Parameter tuning. Week 2-4: Add MSCs if needed. "
                    f"Investment: ₹8-12 crores. ROI: 3-4 months."
                )
            })
        elif problem["type"] == "capacity":
            insights.append({
                "title": f"{problem['circle']}: High Traffic ({problem['value']:,.0f} calls/day)",
                "description": (
                    f"**Capacity Alert**: {problem['overload_pct']:.0f}% above network average. "
                    f"**Risk**: Service degradation during peak hours."
                ),
                "impact": "high",
                "action": (
                    f"Add network redundancy. Deploy 3-4 MSCs. "
                    f"Timeline: Q1. Investment: ₹15-18 crores."
                )
            })
        elif problem["type"] == "usage":
            insights.append({
                "title": f"{problem['circle']}: Low Usage Detected",
                "description": (
                    f"**Usage Anomaly**: {problem['value']:.1f} vs network average {problem['avg_value']:.1f}. "
                    f"**Possible Causes**: Network issues, customer migration, or data quality."
                ),
                "impact": "medium",
                "action": (
                    f"Investigate root cause. Check network health and customer feedback. "
                    f"Timeline: 1-2 weeks."
                )
            })
    
    # Network summary
    if len(circle_analysis) >= 3:
        top_circles = sorted(
            circle_analysis.items(),
            key=lambda x: sum(m.get("value", 0) for m in x[1]["metrics"].values() if m.get("category") == "volume"),
            reverse=True
        )[:5]
        
        insights.append({
            "title": f"Network Coverage: {len(circle_analysis)} Active Circles Monitored",
            "description": (
                f"Comprehensive monitoring across {len(circle_analysis)} circles. "
                f"Top circles: {', '.join([c[0] for c in top_circles[:3]])}. "
                f"Enables proactive issue detection and resolution."
            ),
            "impact": "low",
            "action": "Continue daily monitoring with automated alerts."
        })
    
    return insights if insights else [_generate_generic_insight(len(circle_analysis))]


def _generate_generic_insight(circle_count: int) -> Dict:
    """Fallback insight"""
    return {
        "title": f"Network Analysis: {circle_count} Circles",
        "description": f"Dataset contains {circle_count} circles with performance metrics.",
        "impact": "low",
        "action": "Review individual circle performance for optimization opportunities."
    }


def _generate_recommendations(problems: List[Dict]) -> List[Dict]:
    """Generate recommendations - ENHANCED"""
    
    recommendations = []
    
    # Quality issues
    quality_problems = [p for p in problems if p["type"] == "quality" and p.get("severity") in ["critical", "high"]]
    if quality_problems:
        circles = [p["circle"] for p in quality_problems]
        total_revenue = sum(p.get("revenue_loss", 0) for p in quality_problems)
        
        recommendations.append({
            "category": "Network Quality - CRITICAL",
            "priority": "critical",
            "action": f"Emergency Optimization in {len(quality_problems)} Circles",
            "details": [
                f"Affected: {', '.join(circles[:5])}",
                f"Revenue at Risk: ₹{total_revenue:.1f}L monthly",
                "Timeline: 48 hours",
                "Investment: ₹10-15 crores",
                "ROI: 3-4 months"
            ]
        })
    
    # Capacity issues
    capacity_problems = [p for p in problems if p["type"] == "capacity"]
    if capacity_problems:
        circles = [p["circle"] for p in capacity_problems]
        
        recommendations.append({
            "category": "Capacity Planning",
            "priority": "high",
            "action": "Deploy Redundancy",
            "details": [
                f"Target: {', '.join(circles[:5])}",
                "Solution: 3-4 MSCs per circle",
                "Timeline: Q1",
                "Investment: ₹18-22 crores"
            ]
        })
    
    # Usage issues
    usage_problems = [p for p in problems if p["type"] == "usage"]
    if usage_problems:
        circles = [p["circle"] for p in usage_problems]
        
        recommendations.append({
            "category": "Network Investigation",
            "priority": "medium",
            "action": "Investigate Low Usage",
            "details": [
                f"Circles: {', '.join(circles[:5])}",
                "Action: Root cause analysis",
                "Timeline: 1-2 weeks"
            ]
        })
    
    # Default monitoring
    if not recommendations:
        recommendations.append({
            "category": "Monitoring",
            "priority": "medium",
            "action": "Enhance Network Monitoring",
            "details": [
                "Setup automated alerts",
                "Daily performance dashboard",
                "Investment: ₹50 lakhs"
            ]
        })
    
    return recommendations


def _generate_no_circles_found(df: pd.DataFrame, circle_col: str) -> Dict:
    """When no circles identified"""
    
    sample_values = []
    if circle_col and circle_col in df.columns:
        sample_values = df[circle_col].dropna().unique()[:5].tolist()
    
    return {
        "executive_summary": f"Analyzed {len(df)} records but could not identify individual circles for detailed analysis.",
        "key_insights": [{
            "title": "Data Structure Issue: No Valid Circles Found",
            "description": (
                f"Circle column '{circle_col}' identified but no valid circle data after cleaning. "
                f"Sample values: {sample_values}"
            ),
            "impact": "high",
            "action": "Verify data contains individual circle names (Mumbai, Delhi, etc.) not just summary rows."
        }],
        "recommendations": [{
            "category": "Data Quality",
            "priority": "high",
            "action": "Provide Circle-Level Data",
            "details": [
                "Required: Individual circle names per row",
                "Avoid: Only 'Total' or 'PAN INDIA' rows",
                f"Current column: {circle_col}",
                f"Sample values: {sample_values}"
            ]
        }]
    }


def _generate_error_fallback(df: pd.DataFrame, error: str) -> Dict:
    """Error fallback"""
    return {
        "executive_summary": f"Analysis error on {len(df)} records: {error[:100]}",
        "key_insights": [{
            "title": "Technical Error During Analysis",
            "description": f"Error: {error[:200]}",
            "impact": "low",
            "action": "Review error and data format."
        }],
        "recommendations": [{
            "category": "Technical",
            "priority": "low",
            "action": "Debug Analysis",
            "details": [f"Error: {error}"]
        }]
    }

