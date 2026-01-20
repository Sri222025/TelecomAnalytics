"""
AI Insights Engine - V6 ENHANCED
Smart data cleaning that preserves real data while removing summaries
Enhanced structure detection and problem identification
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import re
import json
import os
import urllib.request
import urllib.error


def analyze_workbooks(processed_files: List[Dict], merged_df: pd.DataFrame = None,
                      merge_summary: Dict = None, llm_config: Dict = None) -> Dict:
    """
    Analyze multiple Excel workbooks/sheets and optionally enrich insights with Groq Llama 3.3.
    """
    dataset_summaries = []
    for file_info in processed_files:
        for sheet in file_info.get("sheets", []):
            dataset_name = f"{file_info.get('file_name', 'Unknown')} | {sheet.get('sheet_name', 'Sheet')}"
            dataset_summaries.append(_summarize_dataset(sheet.get("data"), dataset_name))

    merged_summary = None
    if merged_df is not None and len(merged_df) > 0:
        merged_summary = _summarize_dataset(merged_df, "Merged Dataset")

    executive_summary, key_insights, recommendations = _generate_rule_based_insights(
        dataset_summaries, merged_summary, merge_summary
    )

    llm_payload = None
    if llm_config and llm_config.get("api_key"):
        llm_payload = _generate_llm_insights(dataset_summaries, merged_summary, merge_summary, llm_config)
        if llm_payload:
            if llm_payload.get("executive_summary"):
                executive_summary = llm_payload["executive_summary"]
            if llm_payload.get("key_insights"):
                key_insights = llm_payload["key_insights"]
            if llm_payload.get("recommendations"):
                recommendations = llm_payload["recommendations"]

    return {
        "executive_summary": executive_summary,
        "key_insights": key_insights,
        "recommendations": recommendations,
        "dataset_summaries": dataset_summaries,
        "merged_summary": merged_summary,
        "llm_used": bool(llm_payload),
        "llm_raw": llm_payload
    }


def _summarize_dataset(df: pd.DataFrame, name: str) -> Dict:
    """Create a compact, analytics-friendly summary for a dataset."""
    if df is None or len(df) == 0:
        return {
            "name": name,
            "rows": 0,
            "columns": 0,
            "numeric_columns": [],
            "categorical_columns": [],
            "datetime_columns": [],
            "missing_columns": [],
            "duplicates": 0,
            "stats": {},
            "top_categories": {},
            "top_correlations": [],
            "outliers": {}
        }

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not str(c).startswith("_")]
    datetime_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    categorical_cols = [
        c for c in df.columns
        if not pd.api.types.is_numeric_dtype(df[c])
        and not pd.api.types.is_datetime64_any_dtype(df[c])
        and not str(c).startswith("_")
    ]

    missing_pct = (df.isna().mean() * 100).round(2)
    missing_columns = [
        {"column": col, "missing_pct": float(pct)}
        for col, pct in missing_pct.sort_values(ascending=False).items()
        if pct > 0 and not str(col).startswith("_")
    ][:8]

    stats = {}
    for col in numeric_cols[:8]:
        series = df[col].dropna()
        if len(series) == 0:
            continue
        stats[col] = {
            "mean": float(series.mean()),
            "median": float(series.median()),
            "min": float(series.min()),
            "max": float(series.max()),
            "std": float(series.std()) if len(series) > 1 else 0.0
        }

    top_categories = {}
    for col in categorical_cols[:5]:
        counts = df[col].astype(str).value_counts().head(5)
        top_categories[col] = [{"value": str(idx), "count": int(val)} for idx, val in counts.items()]

    top_correlations = []
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        pairs = upper.stack().sort_values(ascending=False).head(5)
        top_correlations = [
            {"column_1": idx[0], "column_2": idx[1], "correlation": float(val)}
            for idx, val in pairs.items()
            if not np.isnan(val)
        ]

    outliers = {}
    for col in numeric_cols[:8]:
        series = df[col].dropna()
        if len(series) < 10:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_count = int(((series < lower) | (series > upper)).sum())
        if outlier_count > 0:
            outliers[col] = {
                "count": outlier_count,
                "pct": round(outlier_count / len(series) * 100, 2)
            }

    return {
        "name": name,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "datetime_columns": datetime_cols,
        "missing_columns": missing_columns,
        "duplicates": int(df.duplicated().sum()),
        "stats": stats,
        "top_categories": top_categories,
        "top_correlations": top_correlations,
        "outliers": outliers
    }


def _generate_rule_based_insights(dataset_summaries: List[Dict], merged_summary: Dict, merge_summary: Dict) -> Tuple[str, List[Dict], List[Dict]]:
    """Generate business-focused insights for executive decision-making."""
    total_rows = sum(s.get("rows", 0) for s in dataset_summaries)
    total_sheets = len(dataset_summaries)
    
    # Use merged dataset if available, otherwise use first dataset
    primary_data = merged_summary if merged_summary else (dataset_summaries[0] if dataset_summaries else None)
    
    if not primary_data:
        return "No data available for analysis.", [], []
    
    # Build executive summary with key findings
    exec_parts = []
    key_insights = []
    recommendations = []
    
    # Analyze business metrics and patterns
    stats = primary_data.get("stats", {})
    top_categories = primary_data.get("top_categories", {})
    correlations = primary_data.get("top_correlations", [])
    
    # 1. TOP PERFORMERS & OPPORTUNITIES
    if stats:
        # Find highest value metrics
        sorted_metrics = sorted(
            [(col, s) for col, s in stats.items()],
            key=lambda x: abs(x[1].get("mean", 0)),
            reverse=True
        )
        
        if sorted_metrics:
            top_metric = sorted_metrics[0]
            col_name, stat_data = top_metric
            mean_val = stat_data.get("mean", 0)
            max_val = stat_data.get("max", 0)
            min_val = stat_data.get("min", 0)
            
            # Performance spread analysis
            spread = max_val - min_val
            spread_pct = (spread / abs(mean_val) * 100) if mean_val != 0 else 0
            
            if spread_pct > 50:  # Significant variation
                key_insights.append({
                    "title": f"Performance Opportunity: {col_name} Shows {spread_pct:.0f}% Variation",
                    "description": (
                        f"**Current State**: {col_name} ranges from {min_val:,.2f} to {max_val:,.2f} "
                        f"(average: {mean_val:,.2f}). **Gap Analysis**: {spread:,.2f} point spread indicates "
                        f"significant performance differences. **Opportunity**: Top performers achieve {max_val:,.2f} "
                        f"while bottom performers are at {min_val:,.2f} - closing this gap could drive "
                        f"{((max_val - mean_val) / mean_val * 100):.1f}% improvement in average performance."
                    ),
                    "impact": "high",
                    "action": (
                        f"**Strategic Action**: (1) Identify top 20% performers by {col_name} and analyze success factors. "
                        f"(2) Develop best practice playbook. (3) Deploy to bottom 20% performers. "
                        f"**Expected Impact**: Potential {((max_val - mean_val) / mean_val * 50):.1f}% improvement in "
                        f"underperformers within 90 days."
                    )
                })
            
            # High-value metric identification
            if len(sorted_metrics) >= 2:
                top_3_metrics = [m[0] for m in sorted_metrics[:3]]
                key_insights.append({
                    "title": f"Key Performance Indicators: {', '.join(top_3_metrics)}",
                    "description": (
                        f"**Primary Metrics**: {top_3_metrics[0]} averages {sorted_metrics[0][1].get('mean', 0):,.2f}, "
                        f"representing the highest-value metric in your dataset. "
                        f"**Secondary Metrics**: {', '.join(top_3_metrics[1:])} provide additional performance dimensions. "
                        f"**Strategic Focus**: Prioritize initiatives that directly impact these top 3 metrics."
                    ),
                    "impact": "high",
                    "action": (
                        f"**Dashboard Priority**: Feature {top_3_metrics[0]} as primary KPI. "
                        f"**Monitoring**: Track weekly trends. **Target Setting**: Set targets at "
                        f"{sorted_metrics[0][1].get('mean', 0) * 1.15:,.2f} (15% above current average)."
                    )
                })
    
    # 2. CATEGORY PERFORMANCE ANALYSIS
    for col, categories in list(top_categories.items())[:3]:
        if categories and len(categories) >= 2:
            top_cat = categories[0]
            second_cat = categories[1]
            total_count = sum(c["count"] for c in categories)
            
            top_pct = (top_cat["count"] / total_count * 100) if total_count > 0 else 0
            second_pct = (second_cat["count"] / total_count * 100) if total_count > 0 else 0
            
            # Concentration risk
            if top_pct > 60:
                key_insights.append({
                    "title": f"Market Concentration: {col} Dominated by '{top_cat['value']}'",
                    "description": (
                        f"**Distribution**: '{top_cat['value']}' represents {top_pct:.1f}% of records "
                        f"({top_cat['count']:,} out of {total_count:,}), followed by '{second_cat['value']}' "
                        f"at {second_pct:.1f}%. **Risk**: High concentration creates dependency risk. "
                        f"**Opportunity**: Diversification could reduce risk and unlock growth in underperforming segments."
                    ),
                    "impact": "high" if top_pct > 80 else "medium",
                    "action": (
                        f"**Diversification Strategy**: (1) Analyze why '{top_cat['value']}' dominates. "
                        f"(2) Develop growth plan for '{second_cat['value']}' and other segments. "
                        f"(3) Target: Reduce top category to <60% within 6 months. "
                        f"**Risk Mitigation**: Build resilience by reducing over-reliance on single segment."
                    )
                })
            # Balanced distribution opportunity
            elif top_pct < 40 and len(categories) >= 3:
                key_insights.append({
                    "title": f"Balanced Portfolio: {col} Shows Healthy Distribution",
                    "description": (
                        f"**Distribution**: Top category '{top_cat['value']}' represents {top_pct:.1f}%, "
                        f"indicating balanced spread across {len(categories)} segments. "
                        f"**Strength**: Lower concentration risk and diversified portfolio. "
                        f"**Opportunity**: Focus on optimizing each segment rather than consolidation."
                    ),
                    "impact": "medium",
                    "action": (
                        f"**Optimization**: Develop segment-specific strategies for each of the top {min(5, len(categories))} categories. "
                        f"**Growth**: Target 10-15% improvement in each segment rather than focusing on one."
                    )
                })
    
    # 3. CORRELATION & RELATIONSHIP INSIGHTS
    strong_corrs = [c for c in correlations if c.get("correlation", 0) >= 0.7]
    if strong_corrs:
        top_corr = strong_corrs[0]
        key_insights.append({
            "title": f"Key Relationship: {top_corr['column_1']} Drives {top_corr['column_2']}",
            "description": (
                f"**Correlation**: {top_corr['column_1']} and {top_corr['column_2']} show strong relationship "
                f"(r={top_corr['correlation']:.3f}). **Implication**: Changes in {top_corr['column_1']} likely "
                f"influence {top_corr['column_2']}. **Strategic Value**: Optimizing {top_corr['column_1']} "
                f"could have cascading positive effects on {top_corr['column_2']}."
            ),
            "impact": "high",
            "action": (
                f"**Leverage Point**: Focus improvement efforts on {top_corr['column_1']} as primary driver. "
                f"**Measurement**: Track both metrics together to validate relationship. "
                f"**Target**: Improve {top_corr['column_1']} by 10% to drive corresponding improvement in {top_corr['column_2']}."
            )
        })
    
    # 4. OUTLIER OPPORTUNITIES (High performers or anomalies)
    outliers = primary_data.get("outliers", {})
    for col, out_data in list(outliers.items())[:2]:
        if out_data["pct"] >= 3:  # At least 3% outliers
            key_insights.append({
                "title": f"High-Value Segment: {col} Contains {out_data['pct']:.1f}% Exceptional Cases",
                "description": (
                    f"**Finding**: {out_data['count']} records ({out_data['pct']:.1f}%) in {col} are outliers, "
                    f"representing exceptional performance or high-value cases. **Opportunity**: These outliers may "
                    f"represent best practices, premium segments, or untapped potential. **Action Required**: "
                    f"Deep-dive analysis to understand what makes these cases exceptional."
                ),
                "impact": "high",
                "action": (
                    f"**Investigation**: (1) Identify characteristics of outlier records. "
                    f"(2) Determine if they represent replicable best practices. "
                    f"(3) If yes, scale these practices. If no, understand why they're exceptional. "
                    f"**Potential**: Could unlock {out_data['pct']:.1f}% of portfolio to similar performance levels."
                )
            })
            break
    
    # 5. STATISTICAL PERFORMANCE GAPS
    if stats and len(stats) >= 2:
        # Find metrics with highest variance (opportunity for improvement)
        high_variance_metrics = []
        for col, stat_data in stats.items():
            mean_val = stat_data.get("mean", 0)
            std_val = stat_data.get("std", 0)
            cv = (std_val / abs(mean_val) * 100) if mean_val != 0 else 0
            if cv > 40 and mean_val > 0:  # High coefficient of variation
                high_variance_metrics.append((col, cv, mean_val, std_val))
        
        if high_variance_metrics:
            top_var = sorted(high_variance_metrics, key=lambda x: x[1], reverse=True)[0]
            col_name, cv, mean_val, std_val = top_var
            key_insights.append({
                "title": f"Improvement Opportunity: {col_name} Has High Performance Variance",
                "description": (
                    f"**Variance Analysis**: {col_name} shows {cv:.1f}% coefficient of variation "
                    f"(mean: {mean_val:,.2f}, std: {std_val:,.2f}). **Interpretation**: Significant performance "
                    f"differences exist. **Opportunity**: Standardizing performance to the top quartile could "
                    f"drive substantial overall improvement."
                ),
                "impact": "medium",
                "action": (
                    f"**Standardization**: (1) Benchmark top quartile performance in {col_name}. "
                    f"(2) Identify success factors. (3) Deploy to lower performers. "
                    f"**Expected**: Reduce variance by 30% within 120 days."
                )
            })
    
    # 6. EXECUTIVE SUMMARY
    if key_insights:
        high_impact = [i for i in key_insights if i.get("impact") in ["critical", "high"]]
        exec_parts.append(
            f"Analysis of {total_rows:,} records reveals {len(high_impact)} high-priority opportunities "
            f"and {len(key_insights)} strategic insights for decision-making."
        )
    else:
        # Fallback: Generate basic business insights
        if stats:
            top_metric = list(stats.items())[0]
            exec_parts.append(
                f"Analysis of {total_rows:,} records identifies {top_metric[0]} as primary KPI "
                f"(average: {top_metric[1].get('mean', 0):,.2f})."
            )
        else:
            exec_parts.append(f"Analyzed {total_rows:,} records across {total_sheets} dataset(s).")
    
    executive_summary = " ".join(exec_parts) if exec_parts else f"Analyzed {total_rows:,} records."
    
    # 7. STRATEGIC RECOMMENDATIONS
    if key_insights:
        recommendations.append({
            "category": "Strategic Priorities",
            "priority": "high",
            "action": "Focus on Top Performance Opportunities",
            "details": [
                f"Prioritize the {len([i for i in key_insights if i.get('impact') == 'high'])} high-impact opportunities identified",
                "Develop action plans for top 3 insights within 30 days",
                "Establish KPI tracking and weekly review cadence",
                "Set performance targets based on top quartile benchmarks"
            ]
        })
    
    if correlations:
        recommendations.append({
            "category": "Performance Optimization",
            "priority": "medium",
            "action": "Leverage Key Relationships",
            "details": [
                f"Focus improvement efforts on metrics that drive other KPIs (identified {len(strong_corrs)} strong relationships)",
                "Create cross-functional initiatives to optimize correlated metrics together",
                "Measure cascading impact of improvements"
            ]
        })
    
    recommendations.append({
        "category": "Dashboard & Monitoring",
        "priority": "high",
        "action": "Build Executive Dashboard",
        "details": [
            f"Feature top {min(5, len(stats))} KPIs with trend analysis",
            "Include performance distribution charts to show variance",
            "Add category breakdowns for key dimensions",
            "Set up alerts for performance gaps and opportunities"
        ]
    })
    
    # Fallback if no insights
    if not key_insights and stats:
        top_metric = list(stats.items())[0]
        key_insights.append({
            "title": f"Primary KPI: {top_metric[0]}",
            "description": (
                f"**Current Performance**: {top_metric[0]} averages {top_metric[1].get('mean', 0):,.2f} "
                f"across {total_rows:,} records (range: {top_metric[1].get('min', 0):,.2f} to "
                f"{top_metric[1].get('max', 0):,.2f}). **Next Steps**: Establish baseline, set targets, "
                f"and track trends over time."
            ),
            "impact": "medium",
            "action": "Set performance targets and implement tracking dashboard."
        })

    return executive_summary, key_insights, recommendations


def _generate_llm_insights(dataset_summaries: List[Dict], merged_summary: Dict,
                           merge_summary: Dict, llm_config: Dict) -> Dict:
    """Call Groq Llama 3.3 to generate high-quality narrative insights. Returns JSON if successful."""
    api_key = llm_config.get("api_key") or os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return None

    model = llm_config.get("model", "llama-3.3-70b-versatile")
    temperature = llm_config.get("temperature", 0.3)

    # Format data summary for LLM in a more readable way
    data_context = _format_data_for_llm(dataset_summaries, merged_summary, merge_summary)

    system_prompt = """You are a strategic business analyst and executive advisor. Your role is to analyze Excel data and generate BOARD-LEVEL BUSINESS INSIGHTS that help executives make strategic decisions.

CRITICAL FOCUS - BUSINESS INSIGHTS, NOT DATA QUALITY:
1. Identify PERFORMANCE OPPORTUNITIES (top performers, improvement gaps, optimization potential)
2. Highlight STRATEGIC RISKS (concentration, dependencies, underperformance)
3. Reveal BUSINESS PATTERNS (correlations, trends, market dynamics)
4. Provide ACTIONABLE RECOMMENDATIONS for decision-making
5. Focus on WHAT THE DATA TELLS US ABOUT THE BUSINESS, not about data quality

DO NOT focus on:
- Missing values, duplicates, or data quality issues (unless critical)
- Technical data problems
- Generic data quality assessments

DO focus on:
- Performance gaps and opportunities
- Top/bottom performers and why
- Market concentration and diversification needs
- Strategic relationships between metrics
- Growth opportunities and risks
- Actionable business recommendations

OUTPUT FORMAT (JSON only, no markdown):
{
  "executive_summary": "2-3 sentence strategic summary of key business findings and opportunities",
  "key_insights": [
    {
      "title": "Business insight title (e.g., 'Top 20% of Segments Drive 60% of Revenue - Diversification Opportunity')",
      "description": "Detailed business explanation with specific numbers, performance gaps, and strategic implications",
      "impact": "critical|high|medium|low",
      "action": "Specific strategic action executives can take"
    }
  ],
  "recommendations": [
    {
      "category": "Business category (e.g., 'Growth Strategy', 'Performance Optimization', 'Risk Management', 'Market Expansion')",
      "priority": "critical|high|medium|low",
      "action": "Specific strategic action item",
      "details": ["Strategic detail 1", "Strategic detail 2"]
    }
  ]
}

INSIGHT QUALITY STANDARDS:
- Reference specific metrics, percentages, and numbers from the data
- Identify business opportunities, risks, and performance gaps
- Connect data patterns to strategic business implications
- Prioritize insights by business impact and decision-making value
- Provide concrete, actionable recommendations executives can implement"""

    user_prompt = f"""Analyze the following Excel workbook data and generate STRATEGIC BUSINESS INSIGHTS for executive decision-making:

{data_context}

Generate BOARD-LEVEL INSIGHTS focusing on:
1. PERFORMANCE ANALYSIS: Top/bottom performers, performance gaps, improvement opportunities
2. STRATEGIC PATTERNS: Market concentration, diversification needs, key relationships
3. BUSINESS OPPORTUNITIES: Growth potential, optimization areas, competitive advantages
4. RISK ASSESSMENT: Dependencies, concentration risks, underperformance areas
5. ACTIONABLE RECOMMENDATIONS: Strategic actions executives can take

Generate:
1. Executive summary: High-level strategic findings (2-3 sentences)
2. 3-5 key business insights: Specific, data-driven insights with strategic implications
3. 2-4 strategic recommendations: Actionable recommendations with priorities

IMPORTANT: Focus on BUSINESS INSIGHTS that help executives make decisions. Do NOT focus on data quality issues unless they critically impact business decisions.

Return ONLY valid JSON matching the format specified above."""

    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    }

    try:
        request = urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            },
            method="POST"
        )
        with urllib.request.urlopen(request, timeout=90) as response:
            content = response.read().decode("utf-8")
        data = json.loads(content)
        message = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # Robust JSON extraction
        result = None
        
        # Method 1: Try direct JSON parse
        try:
            result = json.loads(message.strip())
        except json.JSONDecodeError:
            # Method 2: Extract JSON from markdown code blocks (```json ... ```)
            import re
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', message, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass
        
        # Method 3: Find first { and last } and try to parse
        if result is None:
            json_start = message.find("{")
            json_end = message.rfind("}")
            if json_start != -1 and json_end != -1 and json_end > json_start:
                try:
                    result = json.loads(message[json_start:json_end + 1])
                except json.JSONDecodeError:
                    pass
        
        # Validate and return
        if result and isinstance(result, dict):
            if "executive_summary" in result and "key_insights" in result:
                # Ensure key_insights is a list
                if not isinstance(result["key_insights"], list):
                    result["key_insights"] = []
                # Ensure recommendations exists
                if "recommendations" not in result:
                    result["recommendations"] = []
                return result
        
        return None
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        print(f"LLM API HTTP error: {str(e)}")
        return None
    except json.JSONDecodeError as e:
        print(f"LLM API JSON decode error: {str(e)}")
        return None
    except Exception as e:
        print(f"LLM API unexpected error: {str(e)}")
        return None


def _format_data_for_llm(dataset_summaries: List[Dict], merged_summary: Dict, merge_summary: Dict) -> str:
    """Format data summaries into a readable text format for LLM analysis."""
    lines = []
    lines.append("=" * 60)
    lines.append("EXCEL WORKBOOK DATA ANALYSIS SUMMARY")
    lines.append("=" * 60)
    lines.append("")
    
    # Individual datasets
    if dataset_summaries:
        lines.append(f"INDIVIDUAL DATASETS ({len(dataset_summaries)} sheets):")
        lines.append("-" * 60)
        for idx, ds in enumerate(dataset_summaries[:6], 1):
            lines.append(f"\n{idx}. {ds.get('name', 'Unknown Dataset')}")
            lines.append(f"   Rows: {ds.get('rows', 0):,} | Columns: {ds.get('columns', 0)}")
            
            # Numeric columns and stats
            numeric = ds.get('numeric_columns', [])
            if numeric:
                lines.append(f"   Numeric Columns ({len(numeric)}): {', '.join(numeric[:5])}")
                stats = ds.get('stats', {})
                for col in list(stats.keys())[:3]:
                    s = stats[col]
                    lines.append(f"     - {col}: mean={s.get('mean', 0):.2f}, min={s.get('min', 0):.2f}, max={s.get('max', 0):.2f}, std={s.get('std', 0):.2f}")
            
            # Categorical columns
            categorical = ds.get('categorical_columns', [])
            if categorical:
                lines.append(f"   Categorical Columns ({len(categorical)}): {', '.join(categorical[:5])}")
                top_cats = ds.get('top_categories', {})
                for col, cats in list(top_cats.items())[:2]:
                    if cats:
                        top_val = cats[0]
                        lines.append(f"     - {col}: Top value '{top_val['value']}' appears {top_val['count']} times")
            
            # Missing data
            missing = ds.get('missing_columns', [])
            if missing:
                lines.append(f"   Data Quality Issues:")
                for m in missing[:3]:
                    lines.append(f"     - {m['column']}: {m['missing_pct']:.1f}% missing values")
            
            # Outliers
            outliers = ds.get('outliers', {})
            if outliers:
                for col, out in list(outliers.items())[:2]:
                    lines.append(f"     - {col}: {out['count']} outliers ({out['pct']:.1f}% of data)")
            
            # Correlations
            corr = ds.get('top_correlations', [])
            if corr:
                lines.append(f"   Key Correlations:")
                for c in corr[:3]:
                    lines.append(f"     - {c['column_1']} ↔ {c['column_2']}: {c['correlation']:.3f}")
            
            if ds.get('duplicates', 0) > 0:
                lines.append(f"   ⚠️ {ds['duplicates']:,} duplicate rows detected")
    
    # Merged dataset
    if merged_summary:
        lines.append("")
        lines.append("=" * 60)
        lines.append("MERGED DATASET (Combined Analysis)")
        lines.append("=" * 60)
        lines.append(f"Total Rows: {merged_summary.get('rows', 0):,}")
        lines.append(f"Total Columns: {merged_summary.get('columns', 0)}")
        
        numeric = merged_summary.get('numeric_columns', [])
        if numeric:
            lines.append(f"Numeric Columns: {', '.join(numeric[:8])}")
        
        stats = merged_summary.get('stats', {})
        if stats:
            lines.append("\nKey Statistics:")
            for col, s in list(stats.items())[:5]:
                lines.append(f"  {col}: mean={s.get('mean', 0):.2f}, range=[{s.get('min', 0):.2f}, {s.get('max', 0):.2f}], std={s.get('std', 0):.2f}")
    
    # Merge summary
    if merge_summary:
        lines.append("")
        lines.append("=" * 60)
        lines.append("DATA INTEGRATION SUMMARY")
        lines.append("=" * 60)
        lines.append(f"Merge Method: {merge_summary.get('method', 'N/A')}")
        if merge_summary.get('merge_key'):
            lines.append(f"Merge Key: {merge_summary.get('merge_key')}")
        lines.append(f"Files Processed: {merge_summary.get('files_processed', 0)}")
        lines.append(f"Total Records: {merge_summary.get('total_records', 0):,}")
    
    return "\n".join(lines)


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
            # Check metrics per circle
            for circle_name, data in list(circle_analysis.items())[:3]:
                print(f"    {circle_name}: {len(data.get('metrics', {}))} metrics")
        
        # STEP 3.5: If no circles found OR very few circles, try alternative analysis
        if len(circle_analysis) == 0:
            # Try to analyze data without circle grouping
            alternative_analysis = _analyze_without_circles(df_clean, metrics)
            if alternative_analysis:
                return alternative_analysis
            return _generate_no_circles_found(df, circle_col)
        
        # STEP 3.6: Ensure we have data to analyze - if circles exist but no metrics, use all numeric columns
        total_metrics = sum(len(data["metrics"]) for data in circle_analysis.values())
        if total_metrics == 0 and len(circle_analysis) > 0:
            print("  ⚠️ Circles found but no metrics - analyzing all numeric columns...")
            # Find all numeric columns
            numeric_cols = [c for c in df_clean.columns if pd.api.types.is_numeric_dtype(df_clean[c]) and not c.startswith('_') and c != circle_col]
            if numeric_cols:
                # Re-analyze with all numeric columns
                for circle_name in circle_analysis.keys():
                    circle_rows = df_clean[df_clean[circle_col] == circle_name]
                    if len(circle_rows) > 0:
                        row = circle_rows.iloc[0]
                        for col in numeric_cols:
                            if col in row.index and pd.notna(row[col]):
                                try:
                                    value = float(row[col])
                                    # Infer category
                                    if 0 <= value <= 100:
                                        category = "quality"
                                    elif value > 1000:
                                        category = "volume"
                                    else:
                                        category = "usage"
                                    
                                    circle_analysis[circle_name]["metrics"][col] = {
                                        "value": value,
                                        "category": category
                                    }
                                except:
                                    pass
        
        # STEP 4: Deep statistical analysis
        stats = _calculate_deep_statistics(df_clean, circle_analysis, metrics)
        
        # STEP 5: Find problems with enhanced detection
        problems = _find_critical_issues(circle_analysis, metrics, stats)
        
        print(f"\nProblems Detected: {len(problems)}")
        for p in problems[:3]:
            print(f"  - {p['circle']}: {p.get('metric', 'N/A')} = {p.get('value', 'N/A')}")
        
        # STEP 6: Generate board-level business insights
        business_insights = _generate_telecom_business_insights(df_clean, circle_analysis, metrics)
        
        # STEP 7: Generate problem-based insights
        problem_insights = _generate_problem_insights(circle_analysis, problems, metrics, stats)
        
        # STEP 8: Combine all insights (business insights first, then problems)
        insights = business_insights + problem_insights
        
        # If no insights, generate from available data
        if not insights:
            # Try to generate from circle analysis data
            if circle_analysis:
                for circle_name, data in list(circle_analysis.items())[:1]:
                    metrics_found = list(data.get("metrics", {}).keys())
                    if metrics_found:
                        insights.append({
                            "title": f"Product Performance Analysis: {len(circle_analysis)} Circles Monitored",
                            "description": (
                                f"**Network Coverage**: Comprehensive analysis across {len(circle_analysis)} operational circles. "
                                f"**Metrics Tracked**: {len(metrics_found)} key performance indicators including {', '.join(metrics_found[:3])}. "
                                f"**Data Quality**: All circles have complete metric data for analysis. "
                                f"**Strategic Value**: Multi-circle presence enables market diversification and performance benchmarking."
                            ),
                            "impact": "medium",
                            "action": (
                                f"**Next Steps**: 1) Conduct circle-wise performance benchmarking. "
                                f"2) Identify top 3 and bottom 3 performers by key metrics. "
                                f"3) Develop best-practice replication strategy. "
                                f"4) Allocate resources based on growth potential analysis."
                            )
                        })
                        break
            
            # If still no insights, use strategic overview
            if not insights:
                insights = _generate_strategic_overview(circle_analysis, metrics)
        
        # STEP 9: Generate outputs
        exec_summary = _generate_exec_summary(circle_analysis, problems, metrics, stats)
        recommendations = _generate_recommendations(problems, stats)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print(f"Generated {len(insights)} insights")
        print(f"Executive summary length: {len(exec_summary)}")
        print("="*60 + "\n")
        
        # Ensure we always have insights
        if not insights:
            print("⚠️ WARNING: No insights generated! Creating fallback...")
            insights = [{
                "title": "Data Analysis Complete",
                "description": f"Analyzed {len(df_clean)} records across {len(circle_analysis)} circles. Review the data explorer for detailed metrics.",
                "impact": "medium",
                "action": "Conduct detailed analysis of circle-wise performance metrics."
            }]
        
        return {
            "executive_summary": exec_summary,
            "key_insights": insights,
            "recommendations": recommendations,
            "circle_analysis": circle_analysis,
            "problems": problems,
            "statistics": stats
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
    """Analyze each circle - ENHANCED to work with any available data"""
    
    if not circle_col:
        return {}
    
    circle_analysis = {}
    
    # If no metrics found, use ALL numeric columns as potential metrics
    if not any(metrics.values()):
        print("  ⚠️ No metrics found by category, using all numeric columns...")
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not c.startswith('_') and c != circle_col]
        if numeric_cols:
            # Categorize by value ranges
            for col in numeric_cols:
                sample = df[col].dropna().head(100)
                if len(sample) > 0:
                    mean_val = sample.mean()
                    max_val = sample.max()
                    if 0 <= mean_val <= 100 and max_val <= 100:
                        metrics["quality"].append(col)
                    elif mean_val > 1000:
                        metrics["volume"].append(col)
                    else:
                        metrics["usage"].append(col)
    
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
        
        # Extract all metrics from defined categories
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
        
        # ALSO extract ANY numeric columns not in metrics (fallback) - ALWAYS try this
        # This ensures we get metrics even if category detection failed
        for col in df.columns:
            if col == circle_col or col.startswith('_'):
                continue
            if pd.api.types.is_numeric_dtype(df[col]) and col in row.index and pd.notna(row[col]):
                # Skip if already in metrics
                if col in circle_analysis[circle_name]["metrics"]:
                    continue
                try:
                    value = float(row[col])
                    # Infer category
                    if 0 <= value <= 100:
                        category = "quality"
                    elif value > 1000:
                        category = "volume"
                    else:
                        category = "usage"
                    
                    circle_analysis[circle_name]["metrics"][col] = {
                        "value": value,
                        "category": category
                    }
                except (ValueError, TypeError):
                    pass
    
    # DON'T remove circles with no metrics - keep them for analysis
    # Just mark them as having limited data
    circles_with_metrics = 0
    for circle_name, data in circle_analysis.items():
        if not data["metrics"]:
            data["limited_data"] = True
        else:
            circles_with_metrics += 1
    
    print(f"  Circles with metrics: {circles_with_metrics}/{len(circle_analysis)}")
    
    return circle_analysis


def _calculate_deep_statistics(df: pd.DataFrame, circle_analysis: Dict, metrics: Dict) -> Dict:
    """Calculate deep statistical insights"""
    stats = {
        "network_wide": {},
        "by_circle": {},
        "percentiles": {},
        "outliers": []
    }
    
    # Network-wide statistics
    for category, cols in metrics.items():
        for col in cols:
            if col in df.columns:
                values = df[col].dropna()
                if len(values) > 0:
                    stats["network_wide"][col] = {
                        "mean": float(values.mean()),
                        "median": float(values.median()),
                        "std": float(values.std()),
                        "min": float(values.min()),
                        "max": float(values.max()),
                        "q25": float(values.quantile(0.25)),
                        "q75": float(values.quantile(0.75)),
                        "count": len(values)
                    }
    
    # Circle-level statistics
    for circle_name, data in circle_analysis.items():
        stats["by_circle"][circle_name] = {}
        for metric_name, metric_data in data["metrics"].items():
            stats["by_circle"][circle_name][metric_name] = metric_data["value"]
    
    # Find outliers (values beyond 2 standard deviations)
    for col, col_stats in stats["network_wide"].items():
        mean = col_stats["mean"]
        std = col_stats["std"]
        if std > 0:
            for circle_name, data in circle_analysis.items():
                if col in data["metrics"]:
                    value = data["metrics"][col]["value"]
                    z_score = abs((value - mean) / std) if std > 0 else 0
                    if z_score > 2:
                        stats["outliers"].append({
                            "circle": circle_name,
                            "metric": col,
                            "value": value,
                            "mean": mean,
                            "z_score": round(z_score, 2)
                        })
    
    return stats


def _analyze_without_circles(df: pd.DataFrame, metrics: Dict) -> Dict:
    """Analyze data even when circle structure isn't perfect - ENHANCED"""
    
    insights = []
    problems = []
    
    # Analyze all numeric columns with deep statistics
    for category, cols in metrics.items():
        for col in cols:
            if col in df.columns:
                values = df[col].dropna()
                if len(values) > 0:
                    mean_val = values.mean()
                    min_val = values.min()
                    max_val = values.max()
                    std_val = values.std()
                    median_val = values.median()
                    q25 = values.quantile(0.25)
                    q75 = values.quantile(0.75)
                    
                    if category == "quality":
                        # Quality analysis with detailed statistics
                        if mean_val < 95:
                            gap = 95 - mean_val
                            iqr = q75 - q25
                            cv = (std_val / mean_val * 100) if mean_val > 0 else 0  # Coefficient of variation
                            
                            insights.append({
                                "title": f"Network Quality Analysis: {col} at {mean_val:.2f}% (Target: 95%)",
                                "description": (
                                    f"**Statistical Summary**: Mean = {mean_val:.2f}%, Median = {median_val:.2f}%, "
                                    f"Range = {min_val:.2f}% - {max_val:.2f}%, Std Dev = {std_val:.2f}%. "
                                    f"**Gap Analysis**: {gap:.2f} points below target. "
                                    f"**Distribution**: IQR = {iqr:.2f}%, CV = {cv:.1f}% ({'High' if cv > 15 else 'Moderate' if cv > 10 else 'Low'} variability). "
                                    f"**Data Points**: {len(values)} records analyzed. "
                                    f"**Bottom Quartile**: {q25:.2f}% indicates {len(values[values < q25])} records need urgent attention."
                                ),
                                "impact": "critical" if gap > 5 else "high",
                                "action": (
                                    f"**Week 1**: Focus on {len(values[values < mean_val - std_val])} records below {mean_val - std_val:.1f}%. "
                                    f"**Week 2-4**: Systematic optimization targeting bottom quartile. "
                                    f"**Target**: Achieve 95%+ mean within 30 days. "
                                    f"**Expected Impact**: {gap*len(values)*0.01:.0f} additional successful calls daily."
                                )
                            })
                            problems.append({
                                "circle": "Network-Wide",
                                "type": "quality",
                                "metric": col,
                                "value": round(mean_val, 2),
                                "target": 95.0,
                                "gap": round(gap, 2),
                                "severity": "critical" if gap > 5 else "high",
                                "std_dev": round(std_val, 2),
                                "records_analyzed": len(values)
                            })
                    elif category == "volume":
                        # Volume analysis
                        total_vol = values.sum()
                        insights.append({
                            "title": f"Network Traffic Analysis: {col} Shows {total_vol:,.0f} Total Volume",
                            "description": (
                                f"**Traffic Metrics**: Total = {total_vol:,.0f}, Mean = {mean_val:,.0f}, "
                                f"Range = {min_val:,.0f} - {max_val:,.0f}. "
                                f"**Distribution**: Median = {median_val:,.0f}, IQR = {q75 - q25:,.0f}. "
                                f"**Peak Analysis**: Top 10% of records handle {values.nlargest(int(len(values)*0.1)).sum():,.0f} volume."
                            ),
                            "impact": "medium",
                            "action": "Monitor for capacity planning and peak hour optimization."
                        })
    
    if not insights:
        # Try to find ANY numeric columns and analyze them
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not c.startswith('_')]
        if numeric_cols:
            for col in numeric_cols[:3]:  # Analyze first 3 numeric columns
                values = df[col].dropna()
                if len(values) > 0:
                    mean_val = values.mean()
                    insights.append({
                        "title": f"Data Analysis: {col} Average = {mean_val:.2f}",
                        "description": f"Analyzed {len(values)} records. Range: {values.min():.2f} - {values.max():.2f}.",
                        "impact": "low",
                        "action": "Review data structure and metrics for deeper analysis."
                    })
    
    if not insights:
        return None
    
    return {
        "executive_summary": f"**Network-wide analysis** of {len(df)} records reveals {len(problems)} critical performance issues requiring immediate attention. Deep statistical analysis performed on all available metrics.",
        "key_insights": insights,
        "recommendations": _generate_recommendations(problems, {}),
        "problems": problems
    }


def _find_critical_issues(circle_analysis: Dict, metrics: Dict, stats: Dict = None) -> List[Dict]:
    """Find worst performers - ENHANCED with statistical analysis"""
    
    problems = []
    
    if not circle_analysis:
        return problems
    
    # Use statistics if available
    network_stats = stats.get("network_wide", {}) if stats else {}
    
    # Quality issues - ENHANCED with statistical comparison
    for circle_name, data in circle_analysis.items():
        for metric_name, metric_data in data["metrics"].items():
            if metric_data["category"] == "quality":
                value = metric_data["value"]
                target = 95.0
                
                # Compare with network average if available
                network_avg = None
                if metric_name in network_stats:
                    network_avg = network_stats[metric_name]["mean"]
                
                if value < target:
                    gap = target - value
                    
                    # Get volume for this circle
                    volume_metrics = [m for m, d in data["metrics"].items() if d["category"] == "volume"]
                    calls = data["metrics"][volume_metrics[0]]["value"] if volume_metrics else 50000
                    
                    failed_calls = calls * (gap / 100)
                    revenue_loss = (failed_calls * 30 * 450) / 100000  # Lakhs/month
                    
                    # Calculate percentile if network stats available
                    percentile = None
                    if network_avg and metric_name in network_stats:
                        network_min = network_stats[metric_name]["min"]
                        network_max = network_stats[metric_name]["max"]
                        if network_max > network_min:
                            percentile = ((value - network_min) / (network_max - network_min)) * 100
                    
                    problem = {
                        "circle": circle_name,
                        "type": "quality",
                        "metric": metric_name,
                        "value": round(value, 2),
                        "target": target,
                        "gap": round(gap, 2),
                        "severity": "critical" if gap > 7 else "high" if gap > 4 else "medium",
                        "calls_affected": int(failed_calls),
                        "revenue_loss": round(revenue_loss, 2)
                    }
                    
                    if network_avg:
                        problem["network_avg"] = round(network_avg, 2)
                        problem["vs_network"] = round(value - network_avg, 2)
                    
                    if percentile is not None:
                        problem["percentile"] = round(percentile, 1)
                    
                    problems.append(problem)
    
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


def _generate_strategic_overview(circle_analysis: Dict, metrics: Dict) -> List[Dict]:
    """Generate insights from actual data - no assumptions"""
    return []  # Don't generate generic insights - let data-driven insights handle it


def _generate_exec_summary(circle_analysis: Dict, problems: List[Dict], metrics: Dict, stats: Dict = None) -> str:
    """Generate executive summary - ENHANCED with statistics"""
    
    parts = []
    
    parts.append(f"**Analyzed {len(circle_analysis)} circles** across the telecom network.")
    
    # Total volume with precision
    total_volume = 0
    for data in circle_analysis.values():
        vol_metrics = [m for m, d in data["metrics"].items() if d["category"] == "volume"]
        if vol_metrics:
            total_volume += data["metrics"][vol_metrics[0]]["value"]
    
    if total_volume > 100000:
        parts.append(f"**{total_volume/100000:.2f} lakh daily call attempts** network-wide.")
    elif total_volume > 0:
        parts.append(f"**{total_volume:,.0f} daily call attempts** network-wide.")
    
    # Quality metrics with statistical depth
    quality_vals = []
    for data in circle_analysis.values():
        qual_metrics = [m for m, d in data["metrics"].items() if d["category"] == "quality"]
        if qual_metrics:
            quality_vals.append(data["metrics"][qual_metrics[0]]["value"])
    
    if quality_vals:
        avg_q = sum(quality_vals) / len(quality_vals)
        min_q = min(quality_vals)
        max_q = max(quality_vals)
        std_q = np.std(quality_vals) if len(quality_vals) > 1 else 0
        parts.append(f"Quality metrics: **{avg_q:.2f}%** average (range: {min_q:.2f}% - {max_q:.2f}%, std: {std_q:.2f}%).")
        
        # Add variance insight
        if std_q > 5:
            parts.append(f"**High variance ({std_q:.1f}%)** indicates inconsistent performance across circles.")
    
    # Issues summary with impact
    critical = [p for p in problems if p.get("severity") == "critical"]
    high = [p for p in problems if p.get("severity") == "high"]
    
    if critical:
        total_revenue_risk = sum(p.get("revenue_loss", 0) for p in critical)
        parts.append(f"**{len(critical)} critical issues** identified. **Revenue at risk: ₹{total_revenue_risk:.1f}L/month**.")
    elif high:
        parts.append(f"**{len(high)} high-priority issues** identified requiring attention.")
    elif problems:
        parts.append(f"**{len(problems)} issues** identified for review.")
    else:
        parts.append("**No critical issues** detected. Network performance is within acceptable parameters.")
    
    return " ".join(parts)


def _generate_telecom_business_insights(df: pd.DataFrame, circle_analysis: Dict, metrics: Dict) -> List[Dict]:
    """Generate board-level product insights from ACTUAL data - Fixed Line + App Product"""
    
    insights = []
    
    # Get all column names for analysis
    all_cols = [c for c in df.columns if not c.startswith('_')]
    numeric_cols = [c for c in all_cols if pd.api.types.is_numeric_dtype(df[c])]
    
    # Get circle column name
    circle_col = None
    for col in df.columns:
        if col.lower() in ['circle', 'region', 'zone', 'area']:
            circle_col = col
            break
    
    print(f"  Analyzing {len(numeric_cols)} numeric columns for insights...")
    print(f"  Columns: {numeric_cols[:5]}")
    
    # FIRST: Analyze using circle_analysis data (more reliable)
    if circle_analysis:
        # Analyze penetration/activation metrics
        for circle_name, data in circle_analysis.items():
            for metric_name, metric_data in data.get("metrics", {}).items():
                value = metric_data.get("value", 0)
                category = metric_data.get("category", "")
                
                # Skip if already analyzed
                if any(metric_name in insight.get('title', '') for insight in insights):
                    continue
                
                # Get all values for this metric across circles
                all_values = [d["metrics"].get(metric_name, {}).get("value", 0) 
                            for d in circle_analysis.values() 
                            if metric_name in d.get("metrics", {})]
                
                if len(all_values) > 0:
                    mean_val = sum(all_values) / len(all_values)
                    min_val = min(all_values)
                    max_val = max(all_values)
                    std_val = np.std(all_values) if len(all_values) > 1 else 0
                    
                    metric_lower = metric_name.lower()
                    
                    # Penetration/Analysis
                    if 'penetration' in metric_lower or ('rate' in metric_lower and '%' in metric_name):
                        insights.append({
                            "title": f"Market Penetration Analysis: {metric_name} Shows {mean_val:.1f}% Network Average",
                            "description": (
                                f"**Strategic Finding**: {metric_name} averages **{mean_val:.1f}%** across {len(circle_analysis)} circles "
                                f"(range: {min_val:.1f}% - {max_val:.1f}%). "
                                f"**Variance**: {std_val:.1f}% standard deviation. "
                                f"**Top Performer**: Achieves {max_val:.1f}% ({max_val - mean_val:.1f} points above average). "
                                f"**Growth Opportunity**: Closing the {max_val - min_val:.1f}% gap could unlock significant market expansion."
                            ),
                            "impact": "high" if (max_val - min_val) > 8 else "medium",
                            "action": (
                                f"**Strategy**: Deploy top performer's approach to underperforming circles. "
                                f"**Target**: Achieve {mean_val + 2:.1f}% network average within 90 days. "
                                f"**Expected Impact**: +{len(circle_analysis) * 50000 * 0.02:,.0f} additional activations."
                            )
                        })
                        break  # Only analyze first penetration metric
    
    # SECOND: Analyze ACTUAL dataframe columns - more comprehensive
    for col in all_cols:
        if col.lower() in ['circle', 'region', 'zone', 'area']:
            continue
        
        # Skip if already analyzed
        if any(col in insight.get('title', '') for insight in insights):
            continue
        
        # Analyze based on column name and actual values
        if pd.api.types.is_numeric_dtype(df[col]):
            values = df[col].dropna()
            if len(values) > 0:
                mean_val = values.mean()
                min_val = values.min()
                max_val = values.max()
                std_val = values.std()
                
                # Product-specific insights based on column names
                col_lower = col.lower()
                
                # Customer/Activation Analysis
                if any(x in col_lower for x in ['customer', 'user', 'active', 'activated']):
                    if 'penetration' in col_lower or 'rate' in col_lower or '%' in col:
                        # This is a percentage metric
                        insights.append({
                            "title": f"Customer Activation Analysis: {col} Shows {mean_val:.1f}% Average",
                            "description": (
                                f"**Product Performance**: {col} averages **{mean_val:.1f}%** across all circles "
                                f"(range: {min_val:.1f}% - {max_val:.1f}%). "
                                f"**Variance**: {std_val:.1f}% standard deviation indicates {'high' if std_val > 5 else 'moderate' if std_val > 2 else 'low'} variability. "
                                f"**Growth Opportunity**: Circles below {mean_val:.1f}% represent untapped market potential. "
                                f"**Best Performer**: Top circle achieves {max_val:.1f}% - replication opportunity exists."
                            ),
                            "impact": "high" if std_val > 5 else "medium",
                            "action": (
                                f"**Strategy**: Analyze top-performing circle's activation strategy. "
                                f"**Action**: Deploy similar approach to circles below {mean_val - std_val:.1f}%. "
                                f"**Target**: Achieve {mean_val + 2:.1f}% average within 90 days. "
                                f"**Expected Impact**: +{len(df) * (mean_val + 2 - mean_val) / 100:.0f} additional activations."
                            )
                        })
                    else:
                        # This is a count metric
                        total = values.sum()
                        insights.append({
                            "title": f"Customer Base Analysis: {col} Totals {total:,.0f} Across Network",
                            "description": (
                                f"**Customer Metrics**: Total {col.lower()} = **{total:,.0f}** across all circles. "
                                f"**Average per Circle**: {mean_val:,.0f} (range: {min_val:,.0f} - {max_val:,.0f}). "
                                f"**Distribution**: Top 3 circles likely represent significant portion of customer base. "
                                f"**Strategic Insight**: Focus growth efforts on underperforming circles to balance distribution."
                            ),
                            "impact": "medium",
                            "action": (
                                f"**Analysis**: Identify top 3 and bottom 3 circles by {col.lower()}. "
                                f"**Strategy**: Replicate top performers' customer acquisition model. "
                                f"**Target**: Increase bottom performers by 20% within 6 months."
                            )
                        })
                
                # Usage/Calling Analysis
                elif any(x in col_lower for x in ['call', 'attempt', 'usage', 'mou', 'minute', 'duration']):
                    if 'audio' in col_lower or 'video' in col_lower:
                        # Call type analysis
                        call_type = 'Video' if 'video' in col_lower else 'Audio'
                        insights.append({
                            "title": f"{call_type} Call Usage: {col} Shows {mean_val:,.0f} Average",
                            "description": (
                                f"**{call_type} Calling Pattern**: Average {col.lower()} = **{mean_val:,.0f}** per circle "
                                f"(range: {min_val:,.0f} - {max_val:,.0f}). "
                                f"**Product Usage**: This reflects mobile app usage of fixed line service. "
                                f"**Engagement Level**: {'High' if mean_val > values.median() * 1.2 else 'Moderate' if mean_val > values.median() * 0.8 else 'Low'} engagement across network. "
                                f"**Opportunity**: Circles below average represent upsell potential for {call_type.lower()} calling features."
                            ),
                            "impact": "medium",
                            "action": (
                                f"**Engagement Strategy**: Promote {call_type.lower()} calling features in low-usage circles. "
                                f"**Marketing**: Target circles below {mean_val:,.0f} with app feature campaigns. "
                                f"**Expected**: +{mean_val * 0.15:,.0f} usage increase in 60 days."
                            )
                        })
                    else:
                        # General usage
                        total = values.sum()
                        insights.append({
                            "title": f"Product Usage Analysis: {col} Totals {total:,.0f}",
                            "description": (
                                f"**Usage Metrics**: Total {col.lower()} = **{total:,.0f}** across network. "
                                f"**Per Circle Average**: {mean_val:,.0f} (range: {min_val:,.0f} - {max_val:,.0f}). "
                                f"**Product Engagement**: This reflects customer usage of fixed line + mobile app product. "
                                f"**Variance Analysis**: {std_val:,.0f} standard deviation shows {'significant' if std_val > mean_val * 0.3 else 'moderate'} circle-to-circle variation."
                            ),
                            "impact": "medium",
                            "action": (
                                f"**Optimization**: Analyze high-usage circles to identify success factors. "
                                f"**Replication**: Apply learnings to low-usage circles. "
                                f"**Target**: Increase network average by 15% within 90 days."
                            )
                        })
                
                # Segmentation Analysis
                elif any(x in col_lower for x in ['non user', 'low', 'moderate', 'heavy', 'segment']):
                    total = values.sum()
                    segment_name = col.split('(')[1].split(')')[0] if '(' in col else col
                    insights.append({
                        "title": f"Customer Segmentation: {segment_name} Segment = {total:,.0f} Customers",
                        "description": (
                            f"**Segment Analysis**: {segment_name} segment represents **{total:,.0f} customers** network-wide. "
                            f"**Per Circle**: Average {mean_val:,.0f} customers per circle (range: {min_val:,.0f} - {max_val:,.0f}). "
                            f"**Strategic Value**: {'High-value' if 'heavy' in col_lower else 'Medium-value' if 'moderate' in col_lower else 'Low-value'} segment for revenue optimization. "
                            f"**Upsell Opportunity**: {'Focus on retention and premium features' if 'heavy' in col_lower else 'Upsell to higher usage tiers' if 'moderate' in col_lower or 'low' in col_lower else 'Activation campaigns needed'}."
                        ),
                        "impact": "high" if 'heavy' in col_lower else "medium",
                        "action": (
                            f"**Segment Strategy**: {'Retain and upsell premium features' if 'heavy' in col_lower else 'Migrate to higher usage plans' if 'moderate' in col_lower or 'low' in col_lower else 'Activation campaigns'}. "
                            f"**Target Circles**: Focus on circles with above-average {segment_name.lower()} concentration. "
                            f"**Timeline**: 90-day campaign. **Expected**: {'Revenue retention' if 'heavy' in col_lower else '15-20% migration to higher tiers'}."
                        )
                    })
                
                # Percentage metrics (any column with % or rate)
                elif '%' in col or 'rate' in col_lower or (0 <= mean_val <= 100 and max_val <= 100):
                    insights.append({
                        "title": f"Performance Metric: {col} at {mean_val:.1f}% Network Average",
                        "description": (
                            f"**Metric Analysis**: {col} averages **{mean_val:.1f}%** across all circles "
                            f"(range: {min_val:.1f}% - {max_val:.1f}%). "
                            f"**Performance Variance**: {std_val:.1f}% standard deviation. "
                            f"**Top Performer**: Achieves {max_val:.1f}% - {max_val - mean_val:.1f} points above average. "
                            f"**Improvement Opportunity**: Closing gap in underperformers could drive significant business impact."
                        ),
                        "impact": "high" if (max_val - min_val) > 10 else "medium",
                        "action": (
                            f"**Benchmarking**: Study top-performing circle's approach. "
                            f"**Replication**: Deploy best practices to circles below {mean_val:.1f}%. "
                            f"**Target**: Achieve {mean_val + 2:.1f}% network average within 120 days."
                        )
                    })
    
    # THIRD: ALWAYS analyze top numeric columns to ensure insights are generated
    # This is the fallback that should always run
    if numeric_cols:
        analyzed_cols = set()
        for insight in insights:
            # Extract column names from existing insights
            title = insight.get('title', '')
            for col in numeric_cols:
                if col in title:
                    analyzed_cols.add(col)
        
        # Analyze remaining important columns - be more aggressive
        cols_to_analyze = [col for col in numeric_cols if col not in analyzed_cols and col.lower() not in ['circle', 'region', 'zone', 'area']]
        
        for col in cols_to_analyze[:5]:  # Top 5 unanalyzed numeric columns
                
            values = df[col].dropna()
            if len(values) > 0:
                mean_val = values.mean()
                min_val = values.min()
                max_val = values.max()
                std_val = values.std()
                total = values.sum()
                
                # Calculate circle-level stats if circle column exists
                if circle_col and circle_col in df.columns:
                    circle_stats = df.groupby(circle_col)[col].agg(['mean', 'min', 'max']).reset_index()
                    top_circle = circle_stats.loc[circle_stats['mean'].idxmax()]
                    bottom_circle = circle_stats.loc[circle_stats['mean'].idxmin()]
                    
                    insights.append({
                        "title": f"Performance Analysis: {col} Network Average = {mean_val:,.0f}",
                        "description": (
                            f"**Metric Overview**: {col} averages **{mean_val:,.0f}** across all circles "
                            f"(total: {total:,.0f} network-wide). "
                            f"**Range**: {min_val:,.0f} - {max_val:,.0f} (std: {std_val:,.0f}). "
                            f"**Top Circle**: {top_circle[circle_col]} achieves {top_circle['mean']:,.0f} average. "
                            f"**Bottom Circle**: {bottom_circle[circle_col]} at {bottom_circle['mean']:,.0f} average. "
                            f"**Gap**: {top_circle['mean'] - bottom_circle['mean']:,.0f} difference indicates optimization opportunity."
                        ),
                        "impact": "high" if (max_val - min_val) > mean_val * 0.5 else "medium",
                        "action": (
                            f"**Benchmarking**: Study {top_circle[circle_col]}'s approach. "
                            f"**Replication**: Deploy similar strategy to {bottom_circle[circle_col]} and underperformers. "
                            f"**Target**: Increase network average by 15% within 90 days. "
                            f"**Expected**: +{total * 0.15:,.0f} improvement network-wide."
                        )
                    })
                else:
                    insights.append({
                        "title": f"Data Analysis: {col} Shows {mean_val:,.0f} Average",
                        "description": (
                            f"**Metric Overview**: {col} averages **{mean_val:,.0f}** per record "
                            f"(total: {total:,.0f} network-wide). "
                            f"**Data Range**: {min_val:,.0f} - {max_val:,.0f} (std: {std_val:,.0f}). "
                            f"**Analysis**: Significant variance ({std_val:,.0f}) indicates optimization opportunities."
                        ),
                        "impact": "medium",
                        "action": (
                            f"**Next Steps**: 1) Identify top and bottom performers. "
                            f"2) Analyze factors driving performance differences. "
                            f"3) Develop action plan for underperformers."
                        )
                    })
                
                # Limit to prevent too many insights
                if len(insights) >= 5:
                    break
    
    # FOURTH: Generate circle comparison insights
    if circle_analysis and len(circle_analysis) >= 3:
        # Find top and bottom circles by any metric
        circle_totals = {}
        for circle_name, data in circle_analysis.items():
            total_metric = sum(m.get("value", 0) for m in data.get("metrics", {}).values())
            circle_totals[circle_name] = total_metric
        
        if circle_totals:
            sorted_circles = sorted(circle_totals.items(), key=lambda x: x[1], reverse=True)
            top_circle = sorted_circles[0]
            bottom_circle = sorted_circles[-1]
            gap = top_circle[1] - bottom_circle[1]
            
            if gap > 0:
                insights.append({
                    "title": f"Circle Performance Variance: {top_circle[0]} vs {bottom_circle[0]}",
                    "description": (
                        f"**Performance Comparison**: {top_circle[0]} leads with total metric value of **{top_circle[1]:,.0f}**, "
                        f"while {bottom_circle[0]} shows **{bottom_circle[1]:,.0f}**. "
                        f"**Gap Analysis**: {gap:,.0f} point difference ({((gap/top_circle[1])*100):.1f}% variance) indicates significant optimization opportunity. "
                        f"**Strategic Value**: Replicating {top_circle[0]}'s success factors could drive network-wide improvement."
                    ),
                    "impact": "high" if gap > top_circle[1] * 0.3 else "medium",
                    "action": (
                        f"**Best Practice Study**: Conduct deep-dive analysis of {top_circle[0]}'s operations. "
                        f"**Replication Plan**: Deploy learnings to {bottom_circle[0]} and similar underperformers. "
                        f"**Timeline**: 90-day implementation. **Expected**: +{gap * 0.5:,.0f} improvement in bottom performers."
                    )
                })
    
    # FINAL FALLBACK: If still no insights, generate from ANY available data
    if not insights:
        if numeric_cols:
            # Just analyze the first numeric column
            col = numeric_cols[0]
            values = df[col].dropna()
            if len(values) > 0:
                mean_val = values.mean()
                total = values.sum()
                min_val = values.min()
                max_val = values.max()
                
                insights.append({
                    "title": f"Product Performance: {col} Analysis",
                    "description": (
                        f"**Data Analysis**: {col} shows network-wide average of **{mean_val:,.0f}** "
                        f"(total: {total:,.0f}, range: {min_val:,.0f} - {max_val:,.0f}). "
                        f"**Circles Analyzed**: {len(circle_analysis) if circle_analysis else len(df)} circles. "
                        f"**Strategic Insight**: Review circle-wise performance to identify top and bottom performers for optimization."
                    ),
                    "impact": "medium",
                    "action": (
                        f"**Analysis Required**: 1) Identify top 3 and bottom 3 circles by {col}. "
                        f"2) Analyze factors driving performance differences. "
                        f"3) Develop action plan to improve underperformers."
                    )
                })
        elif circle_analysis:
            # Generate from circle count
            insights.append({
                "title": f"Network Coverage: {len(circle_analysis)} Circles Under Analysis",
                "description": (
                    f"**Coverage Analysis**: Comprehensive monitoring across {len(circle_analysis)} operational circles. "
                    f"**Data Quality**: All circles have performance metrics available. "
                    f"**Strategic Value**: Multi-circle presence enables market diversification and performance benchmarking."
                ),
                "impact": "medium",
                "action": (
                    f"**Next Steps**: Conduct detailed circle-wise performance analysis to identify optimization opportunities."
                )
            })
    
    print(f"  Generated {len(insights)} business insights")
    return insights if insights else []  # Return empty list, not None


def _generate_problem_insights(circle_analysis: Dict, problems: List[Dict], metrics: Dict, stats: Dict = None) -> List[Dict]:
    """Generate insights from identified problems"""
    
    insights = []
    network_stats = stats.get("network_wide", {}) if stats else {}
    
    # Problem-based insights with statistical context
    for problem in problems[:5]:  # Top 5 problems
        if problem["type"] == "quality":
            # Build detailed description
            desc_parts = [
                f"**Critical Quality Gap**: {problem['circle']} shows {problem['metric']} at **{problem['value']:.2f}%**, "
                f"{problem['gap']:.2f} points below 95% target."
            ]
            
            if problem.get("network_avg"):
                desc_parts.append(
                    f"**Network Comparison**: {problem['vs_network']:.2f} points below network average "
                    f"({problem['network_avg']:.2f}%)."
                )
            
            if problem.get("percentile") is not None:
                percentile = problem["percentile"]
                if percentile < 25:
                    desc_parts.append(f"**Performance Rank**: Bottom quartile ({percentile:.1f}th percentile).")
                elif percentile < 50:
                    desc_parts.append(f"**Performance Rank**: Below median ({percentile:.1f}th percentile).")
            
            desc_parts.append(
                f"**Business Impact**: {problem['calls_affected']:,} failed calls daily = "
                f"**₹{problem['revenue_loss']:.2f}L monthly revenue risk**."
            )
            
            # Root cause analysis
            if problem['gap'] > 7:
                desc_parts.append("**Root Cause**: Severe capacity constraints or critical parameter misconfiguration.")
            elif problem['gap'] > 4:
                desc_parts.append("**Root Cause**: Capacity pressure or suboptimal parameter settings.")
            else:
                desc_parts.append("**Root Cause**: Minor optimization needed.")
            
            insights.append({
                "title": f"{problem['circle']}: {problem['metric'].split('(')[0].strip()} Critical Gap - {problem['value']:.2f}% vs 95% Target",
                "description": " ".join(desc_parts),
                "impact": problem["severity"],
                "action": (
                    f"**Immediate Action (48hrs)**: Deploy field optimization team. "
                    f"**Week 1**: Parameter tuning and capacity audit. "
                    f"**Weeks 2-4**: Infrastructure upgrade if needed (3-4 MSCs). "
                    f"**Investment**: ₹8-12 crores. **ROI**: 3-4 months. "
                    f"**Expected Improvement**: +{problem['gap']:.1f} points to reach target."
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
    
    # Top/Bottom performers analysis
    if len(circle_analysis) >= 3:
        # Find top and bottom performers by quality
        quality_rankings = []
        for circle_name, data in circle_analysis.items():
            qual_metrics = [m for m, d in data["metrics"].items() if d["category"] == "quality"]
            if qual_metrics:
                quality_val = data["metrics"][qual_metrics[0]]["value"]
                quality_rankings.append((circle_name, quality_val, qual_metrics[0]))
        
        if quality_rankings:
            quality_rankings.sort(key=lambda x: x[1], reverse=True)
            top_performer = quality_rankings[0]
            bottom_performer = quality_rankings[-1]
            gap = top_performer[1] - bottom_performer[1]
            
            insights.append({
                "title": f"Performance Variance: {gap:.2f} Points Between Best and Worst Circles",
                "description": (
                    f"**Top Performer**: {top_performer[0]} at {top_performer[1]:.2f}% ({top_performer[2]}). "
                    f"**Bottom Performer**: {bottom_performer[0]} at {bottom_performer[1]:.2f}% ({bottom_performer[2]}). "
                    f"**Gap Analysis**: {gap:.2f} point variance indicates significant optimization opportunity. "
                    f"**Best Practice**: Replicate {top_performer[0]}'s configuration to underperformers."
                ),
                "impact": "high" if gap > 10 else "medium",
                "action": (
                    f"Conduct best practice analysis of {top_performer[0]}. "
                    f"Deploy configuration template to {bottom_performer[0]} and similar underperformers. "
                    f"Expected improvement: +{gap*0.7:.1f} points within 60 days."
                )
            })
        
        # Volume leaders
        volume_rankings = []
        for circle_name, data in circle_analysis.items():
            vol_metrics = [m for m, d in data["metrics"].items() if d["category"] == "volume"]
            if vol_metrics:
                vol_val = data["metrics"][vol_metrics[0]]["value"]
                volume_rankings.append((circle_name, vol_val, vol_metrics[0]))
        
        if volume_rankings:
            volume_rankings.sort(key=lambda x: x[1], reverse=True)
            top_3_vol = volume_rankings[:3]
            total_vol = sum(v[1] for v in volume_rankings)
            top_3_pct = (sum(v[1] for v in top_3_vol) / total_vol * 100) if total_vol > 0 else 0
            
            insights.append({
                "title": f"Traffic Concentration: Top 3 Circles Handle {top_3_pct:.1f}% of Network Volume",
                "description": (
                    f"**Traffic Leaders**: {', '.join([c[0] for c in top_3_vol])} collectively handle "
                    f"{top_3_pct:.1f}% of network traffic. "
                    f"**Implication**: These circles require highest redundancy and capacity planning. "
                    f"**Risk**: Single point of failure in top circles could impact {top_3_pct:.0f}% of customers."
                ),
                "impact": "high" if top_3_pct > 50 else "medium",
                "action": (
                    f"Prioritize capacity expansion in {', '.join([c[0] for c in top_3_vol])}. "
                    f"Deploy redundant infrastructure. "
                    f"Target: Reduce top-3 concentration to <40% within 6 months."
                )
            })
    
    # Statistical insights if available
    if stats and stats.get("outliers"):
        outlier_count = len(stats["outliers"])
        if outlier_count > 0:
            top_outlier = max(stats["outliers"], key=lambda x: x["z_score"])
            insights.append({
                "title": f"Statistical Anomaly Detected: {top_outlier['circle']} Shows Extreme Deviation",
                "description": (
                    f"**Outlier Analysis**: {top_outlier['circle']} has {top_outlier['metric']} = "
                    f"{top_outlier['value']:.2f} (Z-score: {top_outlier['z_score']:.2f}). "
                    f"**Interpretation**: This is {top_outlier['z_score']:.1f} standard deviations from network mean "
                    f"({top_outlier['mean']:.2f}). **Total Outliers**: {outlier_count} circles require investigation."
                ),
                "impact": "high",
                "action": (
                    f"Immediate investigation of {top_outlier['circle']}. "
                    f"Review data quality, network configuration, and operational parameters. "
                    f"Timeline: 1 week for root cause analysis."
                )
            })
    
    return insights


def _generate_generic_insight(circle_count: int) -> Dict:
    """Fallback insight"""
    return {
        "title": f"Network Analysis: {circle_count} Circles",
        "description": f"Dataset contains {circle_count} circles with performance metrics.",
        "impact": "low",
        "action": "Review individual circle performance for optimization opportunities."
    }


def _generate_recommendations(problems: List[Dict], stats: Dict = None) -> List[Dict]:
    """Generate sharp, actionable recommendations - ENHANCED"""
    
    recommendations = []
    
    # Quality issues with detailed breakdown
    quality_problems = [p for p in problems if p["type"] == "quality" and p.get("severity") in ["critical", "high"]]
    if quality_problems:
        circles = [p["circle"] for p in quality_problems]
        total_revenue = sum(p.get("revenue_loss", 0) for p in quality_problems)
        avg_gap = sum(p.get("gap", 0) for p in quality_problems) / len(quality_problems)
        
        recommendations.append({
            "category": "Network Quality - CRITICAL",
            "priority": "critical",
            "action": f"Emergency Optimization in {len(quality_problems)} Circles",
            "details": [
                f"**Affected Circles**: {', '.join(circles[:5])}{' +' + str(len(circles)-5) if len(circles) > 5 else ''}",
                f"**Average Gap**: {avg_gap:.2f} points below target",
                f"**Revenue at Risk**: ₹{total_revenue:.2f}L monthly",
                f"**Total Failed Calls**: {sum(p.get('calls_affected', 0) for p in quality_problems):,} daily",
                "**Timeline**: 48 hours (urgent), 30 days (full resolution)",
                "**Investment**: ₹10-15 crores",
                "**ROI**: 3-4 months",
                "**Success Metric**: Achieve 95%+ quality across all affected circles"
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
    """When no circles identified - try to analyze anyway"""
    
    sample_values = []
    if circle_col and circle_col in df.columns:
        sample_values = df[circle_col].dropna().unique()[:10].tolist()
    
    # Try one more time - analyze without strict filtering
    insights = []
    problems = []
    
    # Analyze all numeric columns
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not c.startswith('_')]
    
    if numeric_cols:
        for col in numeric_cols[:5]:  # Analyze top 5 numeric columns
            values = df[col].dropna()
            if len(values) > 0:
                mean_val = values.mean()
                min_val = values.min()
                max_val = values.max()
                
                insights.append({
                    "title": f"Network Analysis: {col} = {mean_val:.2f} (Range: {min_val:.2f} - {max_val:.2f})",
                    "description": (
                        f"**Data Summary**: Analyzed {len(values)} records. "
                        f"**Statistics**: Mean = {mean_val:.2f}, Range = {min_val:.2f} to {max_val:.2f}. "
                        f"**Circle Column**: '{circle_col}' found with values: {', '.join([str(v) for v in sample_values[:5]])}. "
                        f"**Note**: Circle structure detected but requires data format adjustment for granular analysis."
                    ),
                    "impact": "medium",
                    "action": (
                        f"Review data format. Ensure each row represents one circle with consistent naming. "
                        f"Remove summary rows (PAN INDIA, Total) before analysis."
                    )
                })
    
    if not insights:
        insights.append({
            "title": "Data Structure Issue: Circle Detection Challenge",
            "description": (
                f"Circle column '{circle_col}' identified with sample values: {sample_values}. "
                f"After cleaning, no individual circles could be extracted for detailed analysis. "
                f"**Possible Causes**: All rows are summary rows, or circle names need standardization."
            ),
            "impact": "high",
            "action": "Verify data contains individual circle names (Mumbai, Delhi, etc.) not just summary rows."
        })
    
    return {
        "executive_summary": f"Analyzed {len(df)} records. Circle column '{circle_col}' detected with values: {', '.join([str(v) for v in sample_values[:5]])}. Performing network-wide analysis on available data.",
        "key_insights": insights,
        "recommendations": [{
            "category": "Data Quality",
            "priority": "high",
            "action": "Optimize Data Structure for Circle Analysis",
            "details": [
                "Required: Individual circle names per row (e.g., 'DELHI', 'MUMBAI')",
                "Remove: Summary rows like 'PAN INDIA', 'Total', 'All India'",
                f"Current column: {circle_col}",
                f"Sample values found: {sample_values}",
                "Action: Clean data to have one circle per row before upload"
            ]
        }],
        "problems": problems
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

