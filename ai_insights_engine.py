"""
AI Insights Engine v2 - Business-Focused Analysis
Prioritizes business insights over data quality issues
"""
import pandas as pd
import numpy as np
from groq import Groq
import json

class AIInsightsEngine:
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.3-70b-versatile"
    
    def analyze_data(self, df, file_summary):
        """
        Main analysis function - generates business-focused insights
        """
        # Prepare business context (not just data quality)
        business_context = self._prepare_business_context(df, file_summary)
        
        # Generate business insights using AI
        insights = self._generate_business_insights(business_context)
        
        # Detect anomalies (only severe ones)
        anomalies = self._detect_critical_anomalies(df)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(insights, anomalies)
        
        return {
            'executive_summary': insights.get('summary', ''),
            'key_insights': insights.get('insights', []),
            'anomalies': anomalies,
            'recommendations': recommendations,
            'business_context': business_context
        }
    
    def _prepare_business_context(self, df, file_summary):
        """Prepare business-oriented context for AI"""
        
        # Basic metrics
        total_records = len(df)
        total_columns = len([col for col in df.columns if not col.startswith('_')])
        
        # Identify business dimensions
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove metadata columns
        numeric_cols = [c for c in numeric_cols if not c.startswith('_')]
        date_cols = [c for c in date_cols if not c.startswith('_')]
        categorical_cols = [c for c in categorical_cols if not c.startswith('_')]
        
        # Get top-level aggregations for business insights
        business_metrics = {}
        
        # Numeric summaries (focus on business KPIs)
        for col in numeric_cols[:15]:
            valid_data = df[col].dropna()
            if len(valid_data) > 0:
                business_metrics[col] = {
                    'total': float(valid_data.sum()),
                    'average': float(valid_data.mean()),
                    'median': float(valid_data.median()),
                    'min': float(valid_data.min()),
                    'max': float(valid_data.max()),
                    'records': len(valid_data)
                }
        
        # Categorical breakdowns (for segmentation)
        categorical_insights = {}
        for col in categorical_cols[:10]:
            value_counts = df[col].value_counts().head(10)
            if len(value_counts) > 0:
                categorical_insights[col] = {
                    'top_categories': value_counts.to_dict(),
                    'unique_count': int(df[col].nunique()),
                    'distribution': 'diverse' if df[col].nunique() > len(df) * 0.5 else 'concentrated'
                }
        
        # Time-based trends (if date columns exist)
        time_insights = {}
        for date_col in date_cols:
            if df[date_col].notna().sum() > 0:
                time_insights[date_col] = {
                    'start_date': str(df[date_col].min()),
                    'end_date': str(df[date_col].max()),
                    'date_range_days': (df[date_col].max() - df[date_col].min()).days if pd.notna(df[date_col].min()) else 0
                }
        
        # Cross-dimensional analysis (e.g., metrics by category)
        cross_analysis = {}
        if len(categorical_cols) > 0 and len(numeric_cols) > 0:
            # Take first categorical and first numeric for sample analysis
            cat_col = categorical_cols[0]
            num_col = numeric_cols[0]
            
            grouped = df.groupby(cat_col)[num_col].agg(['sum', 'mean', 'count']).round(2)
            top_performers = grouped.nlargest(5, 'sum')
            
            cross_analysis[f'{num_col}_by_{cat_col}'] = {
                'top_5': top_performers.to_dict('index')
            }
        
        # Data completeness (only mention if severe)
        completeness = (df.notna().sum().sum() / (len(df) * len(df.columns))) * 100
        
        return {
            'total_records': total_records,
            'total_columns': total_columns,
            'files_processed': file_summary.get('files_processed', 1),
            'merge_method': file_summary.get('method', 'unknown'),
            'business_metrics': business_metrics,
            'categorical_insights': categorical_insights,
            'time_insights': time_insights,
            'cross_analysis': cross_analysis,
            'completeness_score': float(completeness)
        }
    
    def _generate_business_insights(self, context):
        """Use AI to generate BUSINESS insights (not data quality issues)"""
        
        prompt = f"""You are a senior telecom business analyst. Your job is to find BUSINESS INSIGHTS, not data quality issues.

Dataset Context:
- Records: {context['total_records']:,}
- Dimensions: {context['total_columns']}
- Files: {context['files_processed']}
- Data Completeness: {context['completeness_score']:.1f}%

Business Metrics:
{json.dumps(context['business_metrics'], indent=2)}

Categorical Breakdowns:
{json.dumps(context['categorical_insights'], indent=2)}

Cross-Dimensional Analysis:
{json.dumps(context['cross_analysis'], indent=2)}

Time Period:
{json.dumps(context['time_insights'], indent=2)}

CRITICAL INSTRUCTIONS:
1. IGNORE minor data quality issues (unless completeness < 70%)
2. FOCUS ON business patterns, trends, and opportunities
3. Look for:
   - High/low performing segments
   - Usage patterns and trends
   - Revenue opportunities
   - Customer behavior insights
   - Operational efficiency findings
   - Regional/device/plan variations
   - Growth areas or risks

4. DO NOT mention:
   - Missing data (unless > 30%)
   - Duplicate records
   - Data types or column issues

Provide:
1. Executive Summary (2-3 sentences about KEY BUSINESS FINDINGS)
2. Top 5 BUSINESS Insights (specific, actionable, numbers-focused)

Example Good Insight:
"High Impact: Delhi region shows 45% higher usage than national average with 12K active users, indicating strong market penetration"

Example BAD Insight (DON'T DO THIS):
"Region column has 94.7% missing values"

Format as JSON:
{{
  "summary": "Business-focused executive summary highlighting key patterns and opportunities...",
  "insights": [
    {{
      "title": "Business insight title (e.g., 'Top Region Performance')",
      "description": "Detailed business finding with specific numbers and implications...",
      "impact": "high/medium/low",
      "category": "revenue/usage/growth/efficiency/risk"
    }}
  ]
}}
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a business analyst focused on actionable insights, NOT a data quality checker. Always respond with valid JSON."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=2500
            )
            
            content = response.choices[0].message.content
            
            # Extract JSON
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0]
            elif '```' in content:
                content = content.split('```')[1].split('```')[0]
            
            result = json.loads(content.strip())
            return result
            
        except Exception as e:
            # Fallback with generic business insights
            return self._generate_fallback_insights(context)
    
    def _generate_fallback_insights(self, context):
        """Generate rule-based business insights if AI fails"""
        insights = []
        
        # Analyze numeric metrics
        for metric, stats in list(context['business_metrics'].items())[:5]:
            if stats['records'] > 100:
                insights.append({
                    'title': f"{metric} Analysis",
                    'description': f"Total {metric}: {stats['total']:,.0f} across {stats['records']:,} records. Average: {stats['average']:.2f}, Range: {stats['min']:.2f} to {stats['max']:.2f}",
                    'impact': 'medium',
                    'category': 'usage'
                })
        
        # Analyze categorical distributions
        for dimension, info in list(context['categorical_insights'].items())[:3]:
            top_cat = max(info['top_categories'].items(), key=lambda x: x[1])
            insights.append({
                'title': f"{dimension} Distribution",
                'description': f"Top category '{top_cat[0]}' accounts for {top_cat[1]:,} records ({top_cat[1]/context['total_records']*100:.1f}% of total). {info['unique_count']} unique values identified.",
                'impact': 'medium',
                'category': 'segmentation'
            })
        
        summary = f"Analysis completed on {context['total_records']:,} records across {context['total_columns']} dimensions from {context['files_processed']} file(s)."
        
        return {
            'summary': summary,
            'insights': insights[:5]
        }
    
    def _detect_critical_anomalies(self, df):
        """Detect ONLY critical anomalies (not minor data quality issues)"""
        anomalies = []
        
        # Only flag missing data if > 30%
        for col in df.columns:
            if col.startswith('_'):
                continue
            
            null_pct = (df[col].isna().sum() / len(df)) * 100
            
            if null_pct > 30:
                severity = 'critical' if null_pct > 60 else 'warning'
                anomalies.append({
                    'type': 'missing_data',
                    'column': col,
                    'severity': severity,
                    'percentage': f"{null_pct:.1f}%",
                    'description': f"{col} has {null_pct:.1f}% missing values - may impact analysis"
                })
        
        # Detect extreme outliers in numeric columns (only if very extreme)
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        for col in numeric_cols:
            if col.startswith('_'):
                continue
            
            valid_data = df[col].dropna()
            if len(valid_data) < 10:
                continue
            
            Q1 = valid_data.quantile(0.25)
            Q3 = valid_data.quantile(0.75)
            IQR = Q3 - Q1
            
            # Only flag EXTREME outliers (3x IQR, not 1.5x)
            extreme_outliers = valid_data[(valid_data < Q1 - 3*IQR) | (valid_data > Q3 + 3*IQR)]
            
            if len(extreme_outliers) > 0 and len(extreme_outliers) < len(valid_data) * 0.01:  # Less than 1%
                anomalies.append({
                    'type': 'extreme_outlier',
                    'column': col,
                    'severity': 'info',
                    'count': len(extreme_outliers),
                    'description': f"Found {len(extreme_outliers)} extreme outliers in {col} (may indicate data entry errors)"
                })
        
        # Sort by severity
        severity_order = {'critical': 0, 'warning': 1, 'info': 2}
        anomalies.sort(key=lambda x: severity_order.get(x['severity'], 3))
        
        return anomalies[:5]  # Limit to top 5 anomalies only
    
    def _generate_recommendations(self, insights, anomalies):
        """Generate business-focused recommendations"""
        recommendations = []
        
        # Based on insights
        high_impact = [i for i in insights.get('insights', []) if i.get('impact') == 'high']
        if high_impact:
            for insight in high_impact[:2]:
                recommendations.append({
                    'priority': 'high',
                    'category': insight.get('category', 'Business').title(),
                    'action': f"Investigate: {insight['title']}",
                    'details': insight['description']
                })
        
        # Revenue/growth opportunities
        revenue_insights = [i for i in insights.get('insights', []) if i.get('category') == 'revenue']
        if revenue_insights:
            recommendations.append({
                'priority': 'high',
                'category': 'Revenue',
                'action': 'Capitalize on revenue opportunities identified',
                'details': [i['title'] for i in revenue_insights[:2]]
            })
        
        # Only mention data quality if critical
        critical_data = [a for a in anomalies if a['severity'] == 'critical']
        if critical_data:
            recommendations.append({
                'priority': 'medium',
                'category': 'Data Quality',
                'action': f'Address {len(critical_data)} critical data issue(s)',
                'details': [a['description'] for a in critical_data[:2]]
            })
        
        return recommendations[:5]  # Limit to top 5
