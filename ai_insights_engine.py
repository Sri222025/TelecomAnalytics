"""
AI Insights Engine v3 - Telecom-Specific Deep Analysis (FIXED)
Generates actionable, quantified insights with telecom domain expertise
"""
import pandas as pd
import numpy as np
from groq import Groq
import json
from datetime import datetime

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
        
        # Prepare rich business context
        business_context = self._prepare_deep_context(df)
        
        # Convert to JSON-serializable format
        business_context = convert_to_serializable(business_context)
        
        # Generate insights with telecom domain knowledge
        insights = self._generate_telecom_insights(business_context)
        
        # Detect business anomalies (not data quality)
        anomalies = self._detect_business_anomalies(df, business_context)
        
        # Generate specific action plans
        recommendations = self._generate_action_plans(insights, anomalies, business_context)
        
        return {
            'executive_summary': insights.get('summary', ''),
            'key_insights': insights.get('insights', []),
            'anomalies': anomalies,
            'recommendations': recommendations,
            'business_context': business_context
        }
    
    def _prepare_deep_context(self, df):
        """Prepare detailed business context with telecom metrics"""
        
        context = {
            'total_records': int(len(df)),
            'dimensions': {},
            'trends': {},
            'comparisons': {},
            'calculations': {}
        }
        
        # Identify telecom-specific columns
        telecom_patterns = {
            'subscribers': ['customer', 'subscriber', 'user', 'msisdn', 'account'],
            'usage': ['call', 'duration', 'minute', 'usage', 'data', 'mb', 'gb', 'session'],
            'revenue': ['revenue', 'arpu', 'price', 'charge', 'amount', 'value'],
            'devices': ['device', 'phone', 'handset', 'model', 'pots', 'jiojoin', 'stb', 'airfiber'],
            'regions': ['region', 'circle', 'state', 'city', 'area', 'zone', 'lsa'],
            'plans': ['plan', 'package', 'tariff', 'subscription'],
            'churn': ['churn', 'disconnect', 'inactive', 'active', 'status'],
            'quality': ['complaint', 'ticket', 'issue', 'quality', 'drop']
        }
        
        # Map actual columns to categories
        col_mapping = {}
        for col in df.columns:
            if col.startswith('_'):
                continue
            col_lower = col.lower()
            for category, patterns in telecom_patterns.items():
                if any(p in col_lower for p in patterns):
                    if category not in col_mapping:
                        col_mapping[category] = []
                    col_mapping[category].append(col)
        
        context['column_mapping'] = col_mapping
        
        # Analyze each dimension
        for category, columns in col_mapping.items():
            context['dimensions'][category] = self._analyze_dimension(df, columns, category)
        
        # Calculate telecom KPIs if possible
        context['kpis'] = self._calculate_kpis(df, col_mapping)
        
        # Identify trends over time
        date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        if len(date_cols) > 0:
            context['trends'] = self._analyze_trends(df, date_cols[0], col_mapping)
        
        # Cross-dimensional comparisons
        context['comparisons'] = self._cross_compare(df, col_mapping)
        
        return context
    
    def _analyze_dimension(self, df, columns, category):
        """Deep analysis of a specific dimension"""
        analysis = {}
        
        for col in columns[:3]:  # Top 3 columns per category
            if pd.api.types.is_numeric_dtype(df[col]):
                valid_data = df[col].dropna()
                if len(valid_data) > 0:
                    analysis[col] = {
                        'type': 'numeric',
                        'total': float(valid_data.sum()),
                        'average': float(valid_data.mean()),
                        'median': float(valid_data.median()),
                        'std': float(valid_data.std()),
                        'min': float(valid_data.min()),
                        'max': float(valid_data.max()),
                        'q25': float(valid_data.quantile(0.25)),
                        'q75': float(valid_data.quantile(0.75)),
                        'count': int(len(valid_data))
                    }
            else:
                # Categorical
                value_counts = df[col].value_counts()
                if len(value_counts) > 0:
                    top_5 = value_counts.head(5)
                    analysis[col] = {
                        'type': 'categorical',
                        'unique': int(df[col].nunique()),
                        'top_5': {str(k): int(v) for k, v in top_5.items()},
                        'top_5_pct': {str(k): f"{(v/len(df)*100):.1f}%" for k, v in top_5.items()},
                        'concentration': float(top_5.sum() / len(df) * 100)
                    }
        
        return analysis
    
    def _calculate_kpis(self, df, col_mapping):
        """Calculate telecom KPIs"""
        kpis = {}
        
        # ARPU
        if 'revenue' in col_mapping and 'subscribers' in col_mapping:
            rev_col = col_mapping['revenue'][0]
            sub_col = col_mapping['subscribers'][0]
            
            if pd.api.types.is_numeric_dtype(df[rev_col]):
                total_rev = df[rev_col].sum()
                total_subs = df[sub_col].nunique() if df[sub_col].dtype == 'object' else len(df)
                
                kpis['ARPU'] = {
                    'value': float(total_rev / total_subs) if total_subs > 0 else 0,
                    'total_revenue': float(total_rev),
                    'total_subscribers': int(total_subs)
                }
        
        # MOU
        if 'usage' in col_mapping:
            for col in col_mapping['usage']:
                if 'minute' in col.lower() or 'duration' in col.lower():
                    if pd.api.types.is_numeric_dtype(df[col]):
                        kpis['MOU'] = {
                            'average': float(df[col].mean()),
                            'total': float(df[col].sum()),
                            'median': float(df[col].median())
                        }
                        break
        
        # Device adoption
        if 'devices' in col_mapping:
            device_col = col_mapping['devices'][0]
            device_dist = df[device_col].value_counts()
            if len(device_dist) > 0:
                kpis['device_adoption'] = {
                    'distribution': {str(k): int(v) for k, v in device_dist.items()},
                    'percentages': {str(k): float(v/len(df)*100) for k, v in device_dist.items()}
                }
        
        # Regional distribution
        if 'regions' in col_mapping:
            region_col = col_mapping['regions'][0]
            region_counts = df[region_col].value_counts()
            if len(region_counts) > 0:
                kpis['regional_distribution'] = {
                    'by_count': {str(k): int(v) for k, v in region_counts.items()},
                    'top_region': str(region_counts.index[0]),
                    'top_region_share': float(region_counts.iloc[0] / len(df) * 100)
                }
        
        return kpis
    
    def _analyze_trends(self, df, date_col, col_mapping):
        """Analyze trends over time"""
        trends = {}
        
        try:
            df_sorted = df.sort_values(date_col).copy()
            df_sorted['period'] = pd.to_datetime(df_sorted[date_col]).dt.to_period('M')
            
            for category, columns in col_mapping.items():
                for col in columns[:2]:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        monthly = df_sorted.groupby('period')[col].sum()
                        if len(monthly) > 1:
                            growth = ((monthly.iloc[-1] - monthly.iloc[0]) / monthly.iloc[0] * 100) if monthly.iloc[0] != 0 else 0
                            trends[col] = {
                                'first_period': float(monthly.iloc[0]),
                                'last_period': float(monthly.iloc[-1]),
                                'growth_rate': float(growth),
                                'trend': 'growing' if growth > 5 else 'declining' if growth < -5 else 'stable'
                            }
        except Exception:
            pass  # Skip trends if date processing fails
        
        return trends
    
    def _cross_compare(self, df, col_mapping):
        """Cross-dimensional comparisons"""
        comparisons = {}
        
        # Region vs Usage
        if 'regions' in col_mapping and 'usage' in col_mapping:
            region_col = col_mapping['regions'][0]
            usage_col = col_mapping['usage'][0]
            
            if pd.api.types.is_numeric_dtype(df[usage_col]):
                regional_usage = df.groupby(region_col)[usage_col].agg(['sum', 'mean', 'count'])
                if len(regional_usage) > 0:
                    top_region = regional_usage['sum'].idxmax()
                    bottom_region = regional_usage['sum'].idxmin()
                    
                    comparisons['region_usage'] = {
                        'top_region': str(top_region),
                        'top_region_total': float(regional_usage.loc[top_region, 'sum']),
                        'top_region_avg': float(regional_usage.loc[top_region, 'mean']),
                        'bottom_region': str(bottom_region),
                        'bottom_region_total': float(regional_usage.loc[bottom_region, 'sum']),
                        'variance': float((regional_usage['sum'].std() / regional_usage['sum'].mean() * 100)) if regional_usage['sum'].mean() > 0 else 0
                    }
        
        # Device vs Usage
        if 'devices' in col_mapping and 'usage' in col_mapping:
            device_col = col_mapping['devices'][0]
            usage_col = col_mapping['usage'][0]
            
            if pd.api.types.is_numeric_dtype(df[usage_col]):
                device_usage = df.groupby(device_col)[usage_col].mean()
                if len(device_usage) > 1:
                    comparisons['device_usage'] = {
                        'by_device': {str(k): float(v) for k, v in device_usage.items()},
                        'highest': str(device_usage.idxmax()),
                        'highest_value': float(device_usage.max()),
                        'lowest': str(device_usage.idxmin()),
                        'lowest_value': float(device_usage.min()),
                        'difference_pct': float((device_usage.max() - device_usage.min()) / device_usage.min() * 100) if device_usage.min() > 0 else 0
                    }
        
        return comparisons
    
    def _generate_telecom_insights(self, context):
        """Generate deep, actionable telecom insights"""
        
        prompt = f"""You are a SENIOR TELECOM OPERATIONS ANALYST with 15+ years experience.

Dataset: {context['total_records']:,} records

DIMENSIONS:
{json.dumps(context['dimensions'], indent=2)}

KPIs:
{json.dumps(context['kpis'], indent=2)}

COMPARISONS:
{json.dumps(context['comparisons'], indent=2)}

Generate 5 SPECIFIC, QUANTIFIED insights with ACTION PLANS.

REQUIREMENTS:
1. Use ACTUAL NUMBERS from data (e.g., "23.4K subscribers", "₹423 ARPU")
2. COMPARE segments (e.g., "Delhi 45% higher than NCR")
3. STATE business impact (e.g., "₹18.2L monthly revenue opportunity")
4. PROVIDE specific action (e.g., "Add 3 FTEs in North by Jan 15")
5. QUANTIFY expected result (e.g., "+5K subs in 90 days = ₹21.5L MRR")

BAD: "Region shows concentrated distribution"
GOOD: "North: 32.4K subs (45% of base) with ₹456 ARPU - Launch 3 sales teams to capture growth → +5K subs in 90 days"

JSON format:
{{
  "summary": "1-2 sentences with KEY NUMBERS and MAIN ACTION",
  "insights": [
    {{
      "title": "Quantified Finding (e.g., 'North ARPU ₹456 vs South ₹298 - 53% Gap')",
      "description": "Detailed analysis with: (1) Specific numbers, (2) Comparison, (3) Root cause, (4) Business impact, (5) Specific action with timeline",
      "impact": "high/medium/low",
      "category": "revenue/usage/device_adoption/regional/churn",
      "metrics": {{
        "key_number": "32.4K",
        "percentage": "+45%",
        "comparison": "vs 22.1K South"
      }},
      "action": "Specific next step with owner and date (e.g., 'Deploy 3 sales FTEs in North by Jan 15')"
    }}
  ]
}}
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a telecom analyst. Provide SPECIFIC, QUANTIFIED insights. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=3000
            )
            
            content = response.choices[0].message.content
            
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0]
            elif '```' in content:
                content = content.split('```')[1].split('```')[0]
            
            result = json.loads(content.strip())
            return result
            
        except Exception as e:
            return self._generate_fallback_insights(context)
    
    def _generate_fallback_insights(self, context):
        """Rule-based insights if AI fails"""
        insights = []
        
        # Regional analysis
        if 'regional_distribution' in context.get('kpis', {}):
            kpi = context['kpis']['regional_distribution']
            top_region = kpi.get('top_region', 'Unknown')
            top_share = kpi.get('top_region_share', 0)
            
            insights.append({
                'title': f'{top_region} Leads with {top_share:.1f}% Market Share',
                'description': f'{top_region} accounts for {top_share:.1f}% of total records. Analyze success factors and replicate in other regions to unlock growth potential.',
                'impact': 'high',
                'category': 'regional',
                'metrics': {'key_number': f'{top_share:.1f}%', 'comparison': 'vs other regions'},
                'action': f'Conduct {top_region} success study - identify 3 key factors by next week'
            })
        
        # Device comparison
        if 'device_usage' in context.get('comparisons', {}):
            dev = context['comparisons']['device_usage']
            diff = dev.get('difference_pct', 0)
            
            insights.append({
                'title': f'{dev.get("highest")} Usage {diff:.0f}% Higher Than {dev.get("lowest")}',
                'description': f'Users on {dev.get("highest")} average {dev.get("highest_value", 0):.1f} vs {dev.get("lowest")} at {dev.get("lowest_value", 0):.1f}. This {diff:.0f}% gap indicates migration opportunity.',
                'impact': 'medium',
                'category': 'device_adoption',
                'metrics': {'key_number': f'{diff:.0f}%', 'comparison': f'{dev.get("highest")} vs {dev.get("lowest")}'},
                'action': f'Launch device migration campaign: {dev.get("lowest")} → {dev.get("highest")}'
            })
        
        summary = f"Analysis of {context['total_records']:,} records reveals actionable patterns in regional performance and device usage."
        
        return {'summary': summary, 'insights': insights}
    
    def _detect_business_anomalies(self, df, context):
        """Detect business anomalies"""
        anomalies = []
        
        for col in df.columns:
            if col.startswith('_'):
                continue
            null_pct = (df[col].isna().sum() / len(df)) * 100
            if null_pct > 40:
                anomalies.append({
                    'type': 'data_gap',
                    'severity': 'warning',
                    'description': f'{col}: {null_pct:.0f}% incomplete - limits analysis depth',
                    'business_impact': 'May miss insights in this dimension'
                })
        
        return anomalies[:3]
    
    def _generate_action_plans(self, insights, anomalies, context):
        """Generate specific action plans"""
        actions = []
        
        for insight in insights.get('insights', [])[:3]:
            if 'action' in insight and insight['action']:
                actions.append({
                    'priority': insight['impact'],
                    'category': insight.get('category', 'General').replace('_', ' ').title(),
                    'action': insight['action'],
                    'rationale': insight['title'],
                    'expected_impact': f"Based on {insight.get('metrics', {}).get('key_number', 'analysis')}"
                })
        
        return actions
