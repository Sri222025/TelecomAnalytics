"""
AI Insights Engine - V7 DOMAIN EXPERT
Telecom-specific analysis with real business understanding
Focus: Fixed Line + JioJoin App Product Analysis
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import re


def analyze_data(df: pd.DataFrame, merge_summary: Dict = None) -> Dict:
    """
    Domain-expert telecom analysis
    """
    
    try:
        print("\n" + "="*70)
        print("TELECOM DOMAIN EXPERT ANALYSIS ENGINE V7")
        print("="*70)
        print(f"Input: {len(df)} rows × {len(df.columns)} columns")
        
        # STEP 1: Understand the data structure
        print("\n[1] ANALYZING DATA STRUCTURE...")
        structure = _analyze_structure(df)
        
        print(f"  → Circle column: {structure['circle_col']}")
        print(f"  → Found {len(structure['metrics'])} metric categories")
        print(f"  → Quality metrics: {len(structure['metrics']['quality'])}")
        print(f"  → Volume metrics: {len(structure['metrics']['volume'])}")
        print(f"  → Customer metrics: {len(structure['metrics']['customer'])}")
        
        # STEP 2: Clean only summary rows
        print("\n[2] CLEANING SUMMARY ROWS...")
        df_clean = _remove_summary_rows(df, structure['circle_col'])
        print(f"  → Kept {len(df_clean)}/{len(df)} rows")
        
        if len(df_clean) == 0:
            return _no_data_response()
        
        # STEP 3: Extract circle-level data
        print("\n[3] EXTRACTING CIRCLE DATA...")
        circles = _extract_circles(df_clean, structure)
        print(f"  → Found {len(circles)} circles")
        if circles:
            print(f"  → Samples: {list(circles.keys())[:5]}")
        
        if not circles:
            return _no_circles_response(df, structure)
        
        # STEP 4: Perform telecom-specific analysis
        print("\n[4] TELECOM BUSINESS ANALYSIS...")
        analysis = _telecom_analysis(circles, structure, df_clean)
        
        # STEP 5: Generate executive insights
        print("\n[5] GENERATING INSIGHTS...")
        insights = _generate_telecom_insights(analysis, circles, structure)
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70 + "\n")
        
        return insights
        
    except Exception as e:
        import traceback
        print(f"\n❌ ERROR: {str(e)}")
        print(traceback.format_exc())
        return _error_response(df, str(e))


def _analyze_structure(df: pd.DataFrame) -> Dict:
    """Understand data structure with telecom domain knowledge"""
    
    structure = {
        "circle_col": None,
        "metrics": {
            "quality": [],     # CSSR, ASR, success rates
            "volume": [],      # Call attempts, counts
            "usage": [],       # MOU, minutes, duration
            "customer": [],    # Customer counts, penetration
            "segmentation": [] # User segments
        },
        "all_columns": list(df.columns)
    }
    
    # Find circle column
    for col in df.columns:
        col_lower = col.lower()
        if any(kw in col_lower for kw in ['circle', 'region', 'zone']):
            if df[col].dtype == 'object':
                structure["circle_col"] = col
                break
    
    # If not found, use first text column with 3-50 unique values
    if not structure["circle_col"]:
        for col in df.columns[:10]:
            if df[col].dtype == 'object':
                uniq = df[col].nunique()
                if 3 <= uniq <= 50:
                    structure["circle_col"] = col
                    break
    
    # Categorize metrics with telecom domain knowledge
    for col in df.columns:
        if col == structure["circle_col"] or col.startswith('_'):
            continue
        
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        
        col_lower = col.lower()
        
        # Quality metrics (percentages)
        if any(kw in col_lower for kw in ['cssr', 'asr', 'success', 'completion', '%', 'rate']):
            # Verify it's actually a percentage (0-100 range)
            sample = df[col].dropna().head(100)
            if len(sample) > 0 and sample.max() <= 100:
                structure["metrics"]["quality"].append(col)
                continue
        
        # Customer/activation metrics
        if any(kw in col_lower for kw in ['customer', 'user', 'subscriber', 'active', 'penetration', 'activation']):
            structure["metrics"]["customer"].append(col)
            continue
        
        # Segmentation
        if any(kw in col_lower for kw in ['non user', 'low user', 'moderate', 'heavy', 'segment']):
            structure["metrics"]["segmentation"].append(col)
            continue
        
        # Volume metrics (large numbers)
        if any(kw in col_lower for kw in ['call', 'attempt', 'count', 'volume', 'traffic']):
            structure["metrics"]["volume"].append(col)
            continue
        
        # Usage metrics
        if any(kw in col_lower for kw in ['mou', 'minute', 'duration', 'usage', 'time']):
            structure["metrics"]["usage"].append(col)
            continue
    
    # If no categorization worked, infer from data
    if not any(structure["metrics"].values()):
        for col in df.columns:
            if col == structure["circle_col"] or col.startswith('_'):
                continue
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            
            sample = df[col].dropna().head(100)
            if len(sample) == 0:
                continue
            
            mean_val = sample.mean()
            max_val = sample.max()
            
            if 0 <= mean_val <= 100 and max_val <= 100:
                structure["metrics"]["quality"].append(col)
            elif mean_val > 10000:
                structure["metrics"]["volume"].append(col)
            elif 100 < mean_val <= 10000:
                structure["metrics"]["customer"].append(col)
            else:
                structure["metrics"]["usage"].append(col)
    
    return structure


def _remove_summary_rows(df: pd.DataFrame, circle_col: str) -> pd.DataFrame:
    """Remove ONLY summary rows, keep real data"""
    
    if not circle_col or circle_col not in df.columns:
        return df
    
    df_clean = df.copy()
    
    # Remove rows with summary keywords in circle column
    summary_keywords = ['total', 'pan india', 'all india', 'grand total', 'sub total', 
                       'overall', 'summary', 'aggregate', 'combined']
    
    pattern = '|'.join([f'\\b{kw}\\b' for kw in summary_keywords])
    mask = df_clean[circle_col].astype(str).str.lower().str.contains(pattern, na=False, regex=True)
    
    removed = mask.sum()
    df_clean = df_clean[~mask]
    
    print(f"  → Removed {removed} summary rows")
    
    # Safety: if removed >80%, use original
    if len(df_clean) < len(df) * 0.2:
        print(f"  ⚠ Too aggressive, using original")
        return df
    
    return df_clean.reset_index(drop=True)


def _extract_circles(df: pd.DataFrame, structure: Dict) -> Dict:
    """Extract circle-level data"""
    
    circle_col = structure["circle_col"]
    if not circle_col:
        return {}
    
    circles = {}
    
    for idx, row in df.iterrows():
        circle_name = str(row[circle_col]).strip()
        
        # Skip invalid
        if not circle_name or circle_name.lower() in ['nan', 'none', '', 'null']:
            continue
        
        # Skip if looks like number
        try:
            float(circle_name)
            continue
        except:
            pass
        
        # Initialize
        if circle_name not in circles:
            circles[circle_name] = {"metrics": {}}
        
        # Extract all metrics
        for category, cols in structure["metrics"].items():
            for col in cols:
                if col in row.index and pd.notna(row[col]):
                    try:
                        value = float(row[col])
                        circles[circle_name]["metrics"][col] = {
                            "value": value,
                            "category": category
                        }
                    except:
                        pass
    
    return circles


def _telecom_analysis(circles: Dict, structure: Dict, df: pd.DataFrame) -> Dict:
    """Perform telecom-specific business analysis"""
    
    analysis = {
        "network_stats": {},
        "problems": [],
        "opportunities": [],
        "segments": {}
    }
    
    # Network-wide statistics
    for category, cols in structure["metrics"].items():
        for col in cols:
            if col in df.columns:
                values = df[col].dropna()
                if len(values) > 0:
                    analysis["network_stats"][col] = {
                        "mean": float(values.mean()),
                        "min": float(values.min()),
                        "max": float(values.max()),
                        "std": float(values.std()),
                        "total": float(values.sum())
                    }
    
    # Find problems (circle-level)
    for circle_name, data in circles.items():
        for metric_name, metric_data in data["metrics"].items():
            value = metric_data["value"]
            category = metric_data["category"]
            
            # Quality issues
            if category == "quality" and value < 95:
                gap = 95 - value
                severity = "critical" if gap > 7 else "high" if gap > 4 else "medium"
                
                analysis["problems"].append({
                    "circle": circle_name,
                    "type": "quality",
                    "metric": metric_name,
                    "value": round(value, 2),
                    "gap": round(gap, 2),
                    "severity": severity
                })
    
    # Find opportunities (high-performing circles)
    for circle_name, data in circles.items():
        for metric_name, metric_data in data["metrics"].items():
            if metric_data["category"] == "customer":
                value = metric_data["value"]
                # High customer count = opportunity to replicate
                if metric_name in analysis["network_stats"]:
                    avg = analysis["network_stats"][metric_name]["mean"]
                    if value > avg * 1.5:
                        analysis["opportunities"].append({
                            "circle": circle_name,
                            "metric": metric_name,
                            "value": round(value, 0),
                            "vs_avg": round(((value/avg) - 1) * 100, 1)
                        })
    
    # Segment analysis
    for col in structure["metrics"]["segmentation"]:
        if col in df.columns:
            analysis["segments"][col] = {
                "total": float(df[col].sum()),
                "avg": float(df[col].mean())
            }
    
    return analysis


def _generate_telecom_insights(analysis: Dict, circles: Dict, structure: Dict) -> Dict:
    """Generate board-ready telecom insights"""
    
    insights = []
    recommendations = []
    
    # 1. CUSTOMER BASE INSIGHTS
    customer_metrics = structure["metrics"]["customer"]
    if customer_metrics:
        for col in customer_metrics:
            if col in analysis["network_stats"]:
                stats = analysis["network_stats"][col]
                total = stats["total"]
                avg = stats["mean"]
                
                # Determine metric type
                is_count = 'customer' in col.lower() or 'user' in col.lower()
                is_penetration = '%' in col or 'penetration' in col.lower() or 'rate' in col.lower()
                
                if is_count:
                    insights.append({
                        "title": f"Customer Base: {col} = {total:,.0f} Total Customers",
                        "description": (
                            f"**Total Active Customers**: {total:,.0f} across all circles. "
                            f"**Average per Circle**: {avg:,.0f}. "
                            f"**Range**: {stats['min']:,.0f} to {stats['max']:,.0f}. "
                            f"**Analysis**: This represents your fixed line + JioJoin mobile app customer base. "
                            f"Top circles show {stats['max']:,.0f} customers vs average {avg:,.0f} - indicating strong market penetration opportunities."
                        ),
                        "impact": "high",
                        "action": (
                            f"**Growth Strategy**: Target circles below {avg:,.0f} customers with aggressive acquisition campaigns. "
                            f"**Best Practice Replication**: Study top 3 circles ({stats['max']:,.0f} customers) and replicate their go-to-market approach. "
                            f"**Target**: Achieve 15% customer base growth in bottom 50% circles within 6 months."
                        )
                    })
                    
                    recommendations.append({
                        "category": "Customer Growth",
                        "priority": "high",
                        "action": "Customer Acquisition Campaign in Underperforming Circles",
                        "details": [
                            f"Current Base: {total:,.0f} customers",
                            f"Target Circles: Those below {avg:,.0f} customers",
                            f"Strategy: Deploy sales teams, promotional offers, JioJoin app marketing",
                            f"Investment: ₹5-8 crores marketing budget",
                            f"Expected ROI: +{avg*0.15:,.0f} customers/circle = +{len(circles)*avg*0.15:,.0f} network-wide",
                            "Timeline: 6 months"
                        ]
                    })
                
                elif is_penetration:
                    insights.append({
                        "title": f"Market Penetration: {col} at {avg:.1f}% Average",
                        "description": (
                            f"**Penetration Rate**: {avg:.1f}% average across circles (range: {stats['min']:.1f}% - {stats['max']:.1f}%). "
                            f"**Market Opportunity**: Gap of {100 - avg:.1f} points to full market penetration. "
                            f"**Best Performer**: {stats['max']:.1f}% shows what's achievable. "
                            f"**Product Context**: This reflects fixed line + JioJoin app activation success rate."
                        ),
                        "impact": "high",
                        "action": (
                            f"**Activation Drive**: Launch 90-day activation campaign in circles below {avg:.1f}%. "
                            f"**Target**: Increase penetration by 5 points (from {avg:.1f}% to {avg+5:.1f}%). "
                            f"**Tactics**: JioJoin app onboarding incentives, fixed line bundled offers, customer referral programs."
                        )
                    })
    
    # 2. USAGE INSIGHTS
    volume_metrics = structure["metrics"]["volume"]
    if volume_metrics:
        for col in volume_metrics:
            if col in analysis["network_stats"]:
                stats = analysis["network_stats"][col]
                
                is_audio = 'audio' in col.lower()
                is_video = 'video' in col.lower()
                call_type = "Audio" if is_audio else "Video" if is_video else "Total"
                
                insights.append({
                    "title": f"{call_type} Call Usage: {stats['total']:,.0f} Calls Network-Wide",
                    "description": (
                        f"**{call_type} Calling**: Total {stats['total']:,.0f} calls across network. "
                        f"**Per Circle**: Average {stats['mean']:,.0f} calls (range: {stats['min']:,.0f} - {stats['max']:,.0f}). "
                        f"**Product Impact**: {'High ' if is_video else ''}usage of JioJoin mobile app {call_type.lower()} calling features. "
                        f"**Engagement**: {'Premium' if is_video else 'Core'} feature driving customer stickiness."
                    ),
                    "impact": "medium" if not is_video else "high",
                    "action": (
                        f"**{call_type} Feature Promotion**: Target circles with below-average usage (<{stats['mean']:,.0f}). "
                        f"{'**Video Upsell**: Promote video calling as premium feature to drive ARPU.' if is_video else '**Audio Optimization**: Ensure quality to maintain baseline engagement.'} "
                        f"**Expected Impact**: +{stats['mean']*0.2:,.0f} calls/circle = {stats['mean']*0.2*len(circles):,.0f} network increase."
                    )
                })
    
    # 3. SEGMENTATION INSIGHTS
    for segment_col, segment_data in analysis["segments"].items():
        segment_name = segment_col.replace('(count)', '').strip()
        
        is_heavy = 'heavy' in segment_col.lower()
        is_moderate = 'moderate' in segment_col.lower()
        is_low = 'low' in segment_col.lower()
        is_non = 'non' in segment_col.lower()
        
        segment_label = "Premium" if is_heavy else "Active" if is_moderate else "Light" if is_low else "Inactive" if is_non else "General"
        
        insights.append({
            "title": f"Customer Segmentation: {segment_name} = {segment_data['total']:,.0f} Customers",
            "description": (
                f"**{segment_label} Segment Analysis**: {segment_data['total']:,.0f} customers classified as {segment_name.lower()}. "
                f"**Per Circle**: Average {segment_data['avg']:,.0f} customers. "
                f"**Strategic Value**: {'High-value segment for retention and upsell' if is_heavy else 'Growth segment for tier migration' if is_moderate or is_low else 'Activation opportunity' if is_non else 'Core user base'}. "
                f"**Revenue Impact**: {'Primary revenue driver' if is_heavy else 'Secondary revenue with growth potential' if is_moderate else 'Low current revenue, high activation potential'}."
            ),
            "impact": "high" if is_heavy else "medium",
            "action": (
                f"**Segment Strategy**: {'Retention program - prevent churn with exclusive features, ₹799+ premium plans' if is_heavy else 'Migration program - upsell to higher tiers with usage-based incentives' if is_moderate or is_low else 'Activation campaign - convert to active users with onboarding support' if is_non else 'General engagement tactics'}. "
                f"**Target**: {'Retain 98%+ of segment' if is_heavy else 'Migrate 20% to higher tier' if is_moderate or is_low else 'Activate 30% within 90 days'}. "
                f"**Timeline**: {'Continuous' if is_heavy else '90 days'}."
            )
        })
    
    # 4. QUALITY ISSUES
    problems = analysis["problems"]
    if problems:
        # Group by severity
        critical = [p for p in problems if p["severity"] == "critical"]
        
        if critical:
            affected_circles = [p["circle"] for p in critical]
            avg_gap = sum(p["gap"] for p in critical) / len(critical)
            
            insights.append({
                "title": f"Quality Crisis: {len(critical)} Circles Below 88% Performance",
                "description": (
                    f"**Critical Quality Issues**: {len(critical)} circles with quality metrics below 88%. "
                    f"**Affected Circles**: {', '.join(affected_circles[:5])}{'...' if len(affected_circles) > 5 else ''}. "
                    f"**Average Gap**: {avg_gap:.1f} points from 95% target. "
                    f"**Customer Impact**: Poor call quality = high churn risk. "
                    f"**Business Risk**: Fixed line + JioJoin app reputation at stake."
                ),
                "impact": "critical",
                "action": (
                    f"**URGENT (48 hours)**: Deploy network teams to all {len(critical)} circles. "
                    f"**Week 1**: Emergency optimization - parameter tuning, capacity check. "
                    f"**Weeks 2-4**: Infrastructure fix if needed (MSC deployment). "
                    f"**Investment**: ₹{len(critical)*1:.0f}-{len(critical)*1.5:.0f} crores. "
                    f"**Success Metric**: All circles >92% within 30 days."
                )
            })
            
            recommendations.append({
                "category": "Network Quality - EMERGENCY",
                "priority": "critical",
                "action": f"Emergency Quality Fix in {len(critical)} Circles",
                "details": [
                    f"Affected: {', '.join(affected_circles[:10])}",
                    f"Average Gap: {avg_gap:.1f} points from target",
                    "Timeline: URGENT - 48 hours to start",
                    f"Investment: ₹{len(critical)*1:.0f}-{len(critical)*1.5:.0f} crores",
                    "Expected Result: >92% quality in 30 days",
                    "ROI: Churn prevention = ₹5-10 crores revenue protection"
                ]
            })
    
    # 5. OPPORTUNITIES
    opportunities = analysis["opportunities"]
    if opportunities:
        top_opportunities = sorted(opportunities, key=lambda x: x["vs_avg"], reverse=True)[:3]
        
        if top_opportunities:
            insights.append({
                "title": f"Best Practice Opportunity: {top_opportunities[0]['circle']} Outperforms by {top_opportunities[0]['vs_avg']:.0f}%",
                "description": (
                    f"**Top Performer**: {top_opportunities[0]['circle']} shows {top_opportunities[0]['vs_avg']:.0f}% higher {top_opportunities[0]['metric']} than network average. "
                    f"**Success Formula**: This circle has cracked the customer acquisition/activation code. "
                    f"**Replication Value**: If all circles matched this performance, network would gain significant customer base. "
                    f"**Other Leaders**: {', '.join([o['circle'] for o in top_opportunities[1:]])}."
                ),
                "impact": "high",
                "action": (
                    f"**Best Practice Study**: Send team to {top_opportunities[0]['circle']} to document: sales tactics, channel partnerships, app onboarding process, promotional offers. "
                    f"**Rollout**: Create playbook and deploy to bottom 30% circles within 60 days. "
                    f"**Expected Impact**: 30-50% improvement in underperformers."
                )
            })
    
    # Executive Summary
    exec_summary = f"**Network-wide analysis of {len(circles)} circles** reveals"
    
    if customer_metrics:
        total_customers = sum(analysis["network_stats"][col]["total"] for col in customer_metrics if col in analysis["network_stats"])
        exec_summary += f" **{total_customers:,.0f} total customers** (fixed line + JioJoin app)."
    
    if problems:
        critical_count = len([p for p in problems if p["severity"] == "critical"])
        if critical_count > 0:
            exec_summary += f" **{critical_count} critical quality issues** requiring immediate action."
        else:
            exec_summary += f" **{len(problems)} quality improvement opportunities** identified."
    else:
        exec_summary += " **Strong network performance** with no critical issues."
    
    if opportunities:
        exec_summary += f" **{len(opportunities)} high-performing circles** provide replication opportunities for network-wide growth."
    
    # Fill in generic insight if none generated
    if not insights:
        insights.append({
            "title": f"Network Analysis: {len(circles)} Circles Monitored",
            "description": f"Analyzed {len(circles)} circles with available performance metrics. Ready for deeper analysis once metric patterns are identified.",
            "impact": "medium",
            "action": "Review data structure and ensure key telecom metrics (CSSR, customer counts, usage) are present for actionable insights."
        })
    
    # Default recommendation if none
    if not recommendations:
        recommendations.append({
            "category": "Network Monitoring",
            "priority": "medium",
            "action": "Establish Baseline Monitoring",
            "details": [
                "Setup daily performance tracking",
                "Define KPI thresholds",
                "Create executive dashboard"
            ]
        })
    
    return {
        "executive_summary": exec_summary,
        "key_insights": insights[:5],  # Top 5
        "recommendations": recommendations[:5]  # Top 5
    }


def _no_data_response() -> Dict:
    return {
        "executive_summary": "No valid data remaining after cleaning summary rows. Please ensure uploaded files contain circle-level data (not just totals).",
        "key_insights": [{
            "title": "Data Issue: No Circle Data Found",
            "description": "All rows appear to be summary rows (PAN INDIA, Total, etc.). Upload files with individual circle performance data.",
            "impact": "high",
            "action": "Ensure each row represents one circle (Mumbai, Delhi, etc.) with its metrics."
        }],
        "recommendations": [{
            "category": "Data Quality",
            "priority": "high",
            "action": "Upload Circle-Level Data",
            "details": ["Required: Individual circle rows", "Avoid: Only summary rows"]
        }]
    }


def _no_circles_response(df: pd.DataFrame, structure: Dict) -> Dict:
    """When circles found but no metrics"""
    
    circle_col = structure["circle_col"]
    sample_circles = df[circle_col].dropna().unique()[:5].tolist() if circle_col else []
    
    # Try to generate SOMETHING useful from available data
    insights = []
    
    # Analyze any numeric columns
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not c.startswith('_')]
    
    if numeric_cols:
        for col in numeric_cols[:3]:
            values = df[col].dropna()
            if len(values) > 0:
                total = values.sum()
                avg = values.mean()
                
                insights.append({
                    "title": f"Data Overview: {col} = {total:,.0f} Total",
                    "description": (
                        f"**Metric Found**: {col} shows {total:,.0f} total across {len(values)} records. "
                        f"**Average**: {avg:,.0f} per record. "
                        f"**Range**: {values.min():,.0f} to {values.max():,.0f}. "
                        f"**Circle Column**: '{circle_col}' with values: {', '.join(str(c) for c in sample_circles)}."
                    ),
                    "impact": "medium",
                    "action": "Review data to identify key telecom metrics (CSSR, customer counts, usage) for deeper analysis."
                })
    
    if not insights:
        insights.append({
            "title": "Data Structure Detected",
            "description": f"Found circle column '{circle_col}' with {len(sample_circles)} unique values: {', '.join(str(c) for c in sample_circles)}. However, no numeric metrics were successfully extracted for analysis.",
            "impact": "medium",
            "action": "Verify data contains performance metrics (call volumes, quality %, customer counts) for each circle."
        })
    
    return {
        "executive_summary": f"Detected {len(sample_circles)} circles ({', '.join(str(c) for c in sample_circles)}) but could not extract detailed metrics for business analysis. Data structure partially understood - needs metric refinement.",
        "key_insights": insights,
        "recommendations": [{
            "category": "Data Analysis",
            "priority": "medium",
            "action": "Enhance Data Structure",
            "details": [
                f"Circle column found: '{circle_col}'",
                f"Circles detected: {', '.join(str(c) for c in sample_circles)}",
                "Add clear metric columns: CSSR%, Customer Count, Call Volume",
                "Ensure numeric data types for metrics"
            ]
        }]
    }


def _error_response(df: pd.DataFrame, error: str) -> Dict:
    return {
        "executive_summary": f"Analysis encountered technical error on {len(df)} records. Error: {error[:100]}",
        "key_insights": [{
            "title": "Technical Issue",
            "description": f"Error during analysis: {error[:200]}",
            "impact": "low",
            "action": "Review error details and data format."
        }],
        "recommendations": [{
            "category": "Technical",
            "priority": "low",
            "action": "Debug Analysis Error",
            "details": [error]
        }]
    }
