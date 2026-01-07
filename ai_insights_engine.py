"""
AI Insights Engine using Groq Llama 3.3
Generates natural language insights from telecom data
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
        Main analysis function - generates comprehensive insights
        """
        # Prepare data summary for AI
        data_context = self._prepare_data_context(df, file_summary)
        
        # Generate insights using AI
        insights = self._generate_insights(data_context)
        
        # Detect anomalies
        anomalies = self._detect_anomalies(df)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(insights, anomalies)
        
        return {
            'executive_summary': insights.get('summary', ''),
            'key_insights': insights.get('insights', []),
            'anomalies': anomalies,
            'recommendations': recommendations,
            'data_context': data_context
        }
    
    def _prepare_data_context(self, df, file_summary):
        """Prepare structured context about the data for AI"""
        # Basic stats
        total_records = len(df)
        total_columns = len([col for col in df.columns if not col.startswith('_')])
        
        # Column types
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        text_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Numeric summaries
        numeric_summary = {}
        for col in numeric_cols[:10]:  # Limit to top 10
            if not col.startswith('_'):
                numeric_summary[col] = {
                    'mean': float(df[col].mean()) if df[col].notna().sum() > 0 else 0,
                    'median': float(df[col].median()) if df[col].notna().sum() > 0 else 0,
                    'min': float(df[col].min()) if df[col].notna().sum() > 0 else 0,
                    'max': float(df[col].max()) if df[col].notna().sum() > 0 else 0,
                    'std': float(df[col].std()) if df[col].notna().sum() > 1 else 0
                }
        
        # Categorical summaries
        categorical_summary = {}
        for col in text_cols[:10]:  # Limit to top 10
            if not col.startswith('_'):
                value_counts = df[col].value_counts().head(5)
                categorical_summary[col] = {
                    'unique_values': int(df[col].nunique()),
                    'top_values': value_counts.to_dict()
                }
        
        # Date range
        date_ranges = {}
        for col in date_cols:
            if not col.startswith('_'):
                date_ranges[col] = {
                    'min': str(df[col].min()),
                    'max': str(df[col].max())
                }
        
        # Data quality
        missing_data = {}
        for col in df.columns:
            if not col.startswith('_'):
                null_count = df[col].isna().sum()
                if null_count > 0:
                    missing_data[col] = {
                        'count': int(null_count),
                        'percentage': float((null_count / len(df)) * 100)
                    }
        
        return {
            'total_records': total_records,
            'total_columns': total_columns,
            'files_processed': file_summary.get('files_processed', 1),
            'merge_method': file_summary.get('method', 'unknown'),
            'numeric_columns': numeric_summary,
            'categorical_columns': categorical_summary,
            'date_ranges': date_ranges,
            'missing_data': missing_data
        }
    
    def _generate_insights(self, data_context):
        """Use Groq AI to generate natural language insights"""
        prompt = f"""You are a senior telecom business analyst. Analyze this dataset and provide actionable insights.

Dataset Overview:
- Total Records: {data_context['total_records']:,}
- Total Columns: {data_context['total_columns']}
- Files Processed: {data_context['files_processed']}

Numeric Metrics:
{json.dumps(data_context['numeric_columns'], indent=2)}

Categorical Dimensions:
{json.dumps(data_context['categorical_columns'], indent=2)}

Data Quality Issues:
{json.dumps(data_context['missing_data'], indent=2)}

Please provide:
1. Executive Summary (2-3 sentences about overall data health and key patterns)
2. Top 5 Business Insights (specific, actionable findings with numbers)

Focus on:
- Subscriber trends and patterns
- Usage anomalies
- Data quality issues
- Regional/device/plan variations
- Business risks or opportunities

Format as JSON:
{{
  "summary": "Executive summary here...",
  "insights": [
    {{"title": "Insight 1", "description": "Detailed finding...", "impact": "high/medium/low"}},
    ...
  ]
}}
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a telecom business analyst expert. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content
            
            # Extract JSON from response
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0]
            elif '```' in content:
                content = content.split('```')[1].split('```')[0]
            
            result = json.loads(content.strip())
            return result
            
        except Exception as e:
            # Fallback if AI fails
            return {
                'summary': f"Analysis completed on {data_context['total_records']:,} records across {data_context['total_columns']} dimensions.",
                'insights': [
                    {
                        'title': 'Data Processing Complete',
                        'description': f"Successfully processed {data_context['files_processed']} file(s) with {data_context['total_records']:,} total records.",
                        'impact': 'high'
                    }
                ]
            }
    
    def _detect_anomalies(self, df):
        """Detect statistical anomalies in the data"""
        anomalies = []
        
        # Check numeric columns for outliers
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        for col in numeric_cols:
            if col.startswith('_'):
                continue
            
            # Skip if too many nulls
            if df[col].isna().sum() / len(df) > 0.5:
                continue
            
            # Z-score method for outliers
            mean = df[col].mean()
            std = df[col].std()
            
            if std > 0:
                outliers = df[np.abs((df[col] - mean) / std) > 3]
                
                if len(outliers) > 0:
                    anomalies.append({
                        'type': 'outlier',
                        'column': col,
                        'severity': 'warning',
                        'count': len(outliers),
                        'description': f"Found {len(outliers)} outlier(s) in {col} (values > 3 std deviations)",
                        'sample_values': outliers[col].head(3).tolist()
                    })
        
        # Check for missing data
        for col in df.columns:
            if col.startswith('_'):
                continue
            
            null_count = df[col].isna().sum()
            null_pct = (null_count / len(df)) * 100
            
            if null_pct > 20:
                severity = 'critical' if null_pct > 50 else 'warning'
                anomalies.append({
                    'type': 'missing_data',
                    'column': col,
                    'severity': severity,
                    'count': null_count,
                    'percentage': f"{null_pct:.1f}%",
                    'description': f"{col} has {null_pct:.1f}% missing values ({null_count:,} records)"
                })
        
        # Check for duplicates
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            anomalies.append({
                'type': 'duplicate',
                'severity': 'info',
                'count': duplicate_count,
                'percentage': f"{(duplicate_count/len(df)*100):.1f}%",
                'description': f"Found {duplicate_count:,} duplicate records ({(duplicate_count/len(df)*100):.1f}% of data)"
            })
        
        # Sort by severity
        severity_order = {'critical': 0, 'warning': 1, 'info': 2}
        anomalies.sort(key=lambda x: severity_order.get(x['severity'], 3))
        
        return anomalies
    
    def _generate_recommendations(self, insights, anomalies):
        """Generate actionable recommendations"""
        recommendations = []
        
        # Based on anomalies
        critical_anomalies = [a for a in anomalies if a['severity'] == 'critical']
        if critical_anomalies:
            recommendations.append({
                'priority': 'high',
                'category': 'Data Quality',
                'action': f"Address {len(critical_anomalies)} critical data quality issues immediately",
                'details': [a['description'] for a in critical_anomalies[:3]]
            })
        
        # Based on missing data
        missing_issues = [a for a in anomalies if a['type'] == 'missing_data']
        if len(missing_issues) > 3:
            recommendations.append({
                'priority': 'medium',
                'category': 'Data Completeness',
                'action': 'Improve data collection processes',
                'details': f"{len(missing_issues)} columns have significant missing data"
            })
        
        # Based on insights
        high_impact_insights = [i for i in insights.get('insights', []) if i.get('impact') == 'high']
        if high_impact_insights:
            recommendations.append({
                'priority': 'high',
                'category': 'Business Action',
                'action': 'Review high-impact findings',
                'details': [i['title'] for i in high_impact_insights[:3]]
            })
        
        return recommendations
