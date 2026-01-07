"""
AI Insights Engine - DEBUG VERSION
Shows what data AI is actually seeing
"""
import pandas as pd
import numpy as np
from groq import Groq
import json
from datetime import datetime
import streamlit as st

def convert_to_serializable(obj):
    """Convert pandas/numpy types to JSON-serializable types"""
    if isinstance(obj, (pd.Timestamp, datetime)):
        return str(obj)
    elif isinstance(obj, (np.int64, np.int32, np.int16)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {convert_to_serializable(k): convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif pd.isna(obj):
        return None
    else:
        return obj

class AIInsightsEngine:
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.3-70b-versatile"
    
    def analyze_data(self, df, file_summary):
        """Generate deep, actionable telecom insights"""
        
        # DEBUG: Show what we're working with
        st.write("### 🔍 DEBUG: Data Being Analyzed")
        st.write(f"**Total Rows:** {len(df):,}")
        st.write(f"**Total Columns:** {len(df.columns)}")
        
        # Show column names
        st.write("**Available Columns:**")
        st.write(df.columns.tolist())
        
        # Show data types
        st.write("**Column Data Types:**")
        st.dataframe(pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes,
            'Non-Null': df.notna().sum(),
            'Unique': [df[col].nunique() for col in df.columns]
        }))
        
        # Show sample data
        st.write("**Sample Data (First 5 Rows):**")
        st.dataframe(df.head())
        
        # Prepare business context
        business_context = self._prepare_deep_context(df)
        
        # DEBUG: Show what context was prepared
        st.write("### 🔍 DEBUG: Context Prepared for AI")
        st.json(convert_to_serializable(business_context))
        
        # Convert to serializable
        business_context = convert_to_serializable(business_context)
        
        # Generate insights
        insights = self._generate_telecom_insights(business_context, df)
        
        # Detect anomalies
        anomalies = self._detect_business_anomalies(df, business_context)
        
        # Generate recommendations
        recommendations = self._generate_action_plans(insights, anomalies, business_context)
        
        return {
            'executive_summary': insights.get('summary', ''),
            'key_insights': insights.get('insights', []),
            'anomalies': anomalies,
            'recommendations': recommendations,
            'business_context': business_context
        }
    
    def _prepare_deep_context(self, df):
        """Prepare detailed business context"""
        
        context = {
            'total_records': int(len(df)),
            'total_columns': int(len(df.columns)),
            'columns_available': df.columns.tolist(),
            'dimensions': {},
            'kpis': {},
            'comparisons': {}
        }
        
        # Show ALL columns and their basic stats
        column_analysis = {}
        for col in df.columns:
            if col.startswith('_'):
                continue
            
            col_data = {
                'name': col,
                'type': str(df[col].dtype),
                'non_null': int(df[col].notna().sum()),
                'unique': int(df[col].nunique())
            }
            
            # If numeric, get aggregates
            if pd.api.types.is_numeric_dtype(df[col]):
                valid = df[col].dropna()
                if len(valid) > 0:
                    col_data['stats'] = {
                        'sum': float(valid.sum()),
                        'mean': float(valid.mean()),
                        'min': float(valid.min()),
                        'max': float(valid.max()),
                        'median': float(valid.median())
                    }
            
            # If categorical, get top values
            elif df[col].dtype == 'object':
                value_counts = df[col].value_counts().head(10)
                col_data['top_values'] = {str(k): int(v) for k, v in value_counts.items()}
            
            column_analysis[col] = col_data
        
        context['column_analysis'] = column_analysis
        
        # Identify telecom patterns (be more flexible)
        telecom_keywords = {
            'subscribers': ['customer', 'subscriber', 'user', 'msisdn', 'mobile', 'account', 'cli', 'dn'],
            'usage': ['call', 'duration', 'minute', 'usage', 'data', 'mb', 'gb', 'session', 'mou', 'traffic'],
            'revenue': ['revenue', 'arpu', 'price', 'charge', 'amount', 'value', 'billing', 'recharge'],
            'devices': ['device', 'phone', 'handset', 'model', 'pots', 'jiojoin', 'stb', 'airfiber', 'type'],
            'regions': ['region', 'circle', 'state', 'city', 'area', 'zone', 'lsa', 'location'],
            'plans': ['plan', 'package', 'tariff', 'subscription', 'scheme'],
            'status': ['active', 'inactive', 'churn', 'disconnect', 'status', 'state']
        }
        
        # Map columns to categories
        mapped_columns = {}
        for col in df.columns:
            if col.startswith('_'):
                continue
            
            col_lower = col.lower()
            for category, keywords in telecom_keywords.items():
                if any(kw in col_lower for kw in keywords):
                    if category not in mapped_columns:
                        mapped_columns[category] = []
                    mapped_columns[category].append({
                        'column': col,
                        'data': column_analysis.get(col, {})
                    })
        
        context['mapped_columns'] = mapped_columns
        
        # Calculate simple aggregates
        summary_stats = {
            'total_records': int(len(df)),
            'total_columns': int(len(df.columns)),
            'numeric_columns': len(df.select_dtypes(include=['number']).columns),
            'categorical_columns': len(df.select_dtypes(include=['object']).columns)
        }
        
        context['summary'] = summary_stats
        
        return context
    
    def _generate_telecom_insights(self, context, df):
        """Generate insights with better data understanding"""
        
        # Build a detailed data summary for the AI
        data_summary = f"""
TELECOM DATASET ANALYSIS
========================

Total Records: {context['total_records']:,}
Total Columns: {context['total_columns']}

AVAILABLE COLUMNS AND DATA:
"""
        
        # Add column details
        for col, data in context.get('column_analysis', {}).items():
            data_summary += f"\n{col}:"
            data_summary += f"\n  - Type: {data['type']}"
            data_summary += f"\n  - Non-null values: {data['non_null']:,}"
            data_summary += f"\n  - Unique values: {data['unique']:,}"
            
            if 'stats' in data:
                data_summary += f"\n  - Sum: {data['stats']['sum']:,.2f}"
                data_summary += f"\n  - Average: {data['stats']['mean']:,.2f}"
                data_summary += f"\n  - Range: {data['stats']['min']:,.2f} to {data['stats']['max']:,.2f}"
            
            if 'top_values' in data:
                data_summary += f"\n  - Top values: {list(data['top_values'].items())[:3]}"
        
        # Add mapped columns
        data_summary += f"\n\nIDENTIFIED TELECOM DIMENSIONS:"
        for category, cols in context.get('mapped_columns', {}).items():
            data_summary += f"\n{category.upper()}: {[c['column'] for c in cols]}"
        
        prompt = f"""{data_summary}

You are a SENIOR TELECOM ANALYST. Analyze this REAL DATA and provide SPECIFIC insights.

CRITICAL INSTRUCTIONS:
1. Look at the ACTUAL NUMBERS in the data above
2. If you see "Total Records: 75" that means 75 CUSTOMERS/SUBSCRIBERS
3. Use the SUM values for totals (e.g., if "Call_Duration" sum is 125,000 minutes, say "125K total minutes")
4. Use the COUNT for customer counts
5. BE SPECIFIC with the numbers you see

Generate 3-5 insights based on the ACTUAL DATA SHOWN ABOVE.

BAD Example (making up numbers):
"The dataset shows 1 subscriber"

GOOD Example (using real data):
"Dataset contains {context['total_records']:,} records. Top region accounts for 45% with specific performance metrics..."

JSON format:
{{
  "summary": "Brief summary using REAL numbers from data",
  "insights": [
    {{
      "title": "Specific finding from data",
      "description": "Analysis based on actual values shown above",
      "impact": "high/medium/low",
      "category": "relevant category",
      "action": "Recommended next step"
    }}
  ]
}}
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are analyzing REAL telecom data. Use the ACTUAL NUMBERS provided. Never make up data."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=2500
            )
            
            content = response.choices[0].message.content
            
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0]
            elif '```' in content:
                content = content.split('```')[1].split('```')[0]
            
            result = json.loads(content.strip())
            return result
            
        except Exception as e:
            st.error(f"AI generation failed: {str(e)}")
            return self._generate_simple_insights(context, df)
    
    def _generate_simple_insights(self, context, df):
        """Generate simple rule-based insights from actual data"""
        insights = []
        
        # Overall summary
        total_records = len(df)
        
        # Find numeric columns and get their totals
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols[:5]:
            if not col.startswith('_'):
                total = df[col].sum()
                avg = df[col].mean()
                insights.append({
                    'title': f'{col}: {total:,.0f} Total Across {total_records:,} Records',
                    'description': f'The dataset contains {total_records:,} records with {col} totaling {total:,.0f} (average: {avg:,.2f} per record). This represents the actual data volume being analyzed.',
                    'impact': 'high',
                    'category': 'data_overview',
                    'action': 'Review this metric for business decision-making'
                })
        
        # Find categorical columns
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols[:3]:
            if not col.startswith('_'):
                top_value = df[col].value_counts().iloc[0]
                top_name = df[col].value_counts().index[0]
                pct = (top_value / len(df)) * 100
                
                insights.append({
                    'title': f'{col}: {top_name} Leads with {top_value:,} Records ({pct:.1f}%)',
                    'description': f'In the {col} dimension, {top_name} accounts for {top_value:,} out of {len(df):,} total records ({pct:.1f}% share). This shows actual distribution in your data.',
                    'impact': 'medium',
                    'category': 'segmentation',
                    'action': f'Analyze why {top_name} is dominant in {col}'
                })
        
        summary = f"Analysis of {total_records:,} actual records from your telecom data. The insights below are based on real numbers from your dataset."
        
        return {
            'summary': summary,
            'insights': insights[:5]
        }
    
    def _detect_business_anomalies(self, df, context):
        """Detect anomalies"""
        anomalies = []
        
        for col in df.columns:
            if col.startswith('_'):
                continue
            null_pct = (df[col].isna().sum() / len(df)) * 100
            if null_pct > 40:
                anomalies.append({
                    'type': 'data_gap',
                    'severity': 'warning',
                    'description': f'{col}: {null_pct:.0f}% incomplete',
                    'business_impact': 'Limited analysis possible'
                })
        
        return anomalies[:3]
    
    def _generate_action_plans(self, insights, anomalies, context):
        """Generate action plans"""
        actions = []
        
        for insight in insights.get('insights', [])[:3]:
            if 'action' in insight:
                actions.append({
                    'priority': insight.get('impact', 'medium'),
                    'category': insight.get('category', 'General'),
                    'action': insight['action'],
                    'rationale': insight['title']
                })
        
        return actions
