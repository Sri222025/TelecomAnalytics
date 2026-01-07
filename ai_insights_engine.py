"""
AI Insights Engine v3 - Telecom-Specific Deep Analysis
Generates actionable, quantified insights with telecom domain expertise
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
        """Generate deep, actionable telecom insights"""
        
        # Prepare rich business context
        business_context = self._prepare_deep_context(df)
        
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
            'total_records': len(df),
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
        date_cols = df.select_dtypes(include=['datetime64']).columns
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
                        'count': len(valid_data)
                    }
            else:
                # Categorical
                value_counts = df[col].value_counts()
                if len(value_counts) > 0:
                    top_5 = value_counts.head(5)
                    analysis[col] = {
                        'type': 'categorical',
                        'unique': int(df[col].nunique()),
                        'top_5': top_5.to_dict(),
                        'top_5_pct': {k: f"{(v/len(df)*100):.1f}%" for k, v in top_5.items()},
                        'concentration': float(top_5.sum() / len(df) * 100)  # % in top 5
                    }
        
        return analysis
    
    def _calculate_kpis(self, df, col_mapping):
        """Calculate telecom KPIs"""
        kpis = {}
        
        # ARPU (if revenue and subscribers exist)
        if 'revenue' in col_mapping and 'subscribers' in col_mapping:
            rev_col = col_mapping['revenue'][0]
            sub_col = col_mapping['subscribers'][0]
            
            if pd.api.types.is_numeric_dtype(df[rev_col]):
                total_rev = df[rev_col].sum()
                total_subs = df[sub_col].nunique() if df[sub_col].dtype == 'object' else len(df)
                
                kpis['ARPU'] = {
                    'value': float(total_rev / total_subs) if total_subs > 0 else 0,
                    'total_revenue': float(total_rev),
                    'total_subscribers': total_subs
                }
        
        # MOU (Minutes of Usage)
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
        
        # Device adoption rates
        if 'devices' in col_mapping:
            device_col = col_mapping['devices'][0]
            device_dist = df[device_col].value_counts()
            if len(device_dist) > 0:
                kpis['device_adoption'] = {
                    'distribution': device_dist.to_dict(),
                    'percentages': {k: float(v/len(df)*100) for k, v in device_dist.items()}
                }
        
        # Regional penetration
        if 'regions' in col_mapping:
            region_col = col_mapping['regions'][0]
            region_counts = df[region_col].value_counts()
            kpis['regional_distribution'] = {
                'by_count': region_counts.to_dict(),
                'top_region': region_counts.index[0] if len(region_counts) > 0 else None,
                'top_region_share': float(region_counts.iloc[0] / len(df) * 100) if len(region_counts) > 0 else 0
            }
        
        return kpis
    
    def _analyze_trends(self, df, date_col, col_mapping):
        """Analyze trends over time"""
        trends = {}
        
        df_sorted = df.sort_values(date_col)
        
        # Split into periods
        df_sorted['period'] = pd.to_datetime(df_sorted[date_col]).dt.to_period('M')
        
        # Trend analysis for numeric columns
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
                        'by_device': device_usage.to_dict(),
                        'highest': str(device_usage.idxmax()),
                        'highest_value': float(device_usage.max()),
                        'lowest': str(device_usage.idxmin()),
                        'lowest_value': float(device_usage.min()),
                        'difference_pct': float((device_usage.max() - device_usage.min()) / device_usage.min() * 100) if device_usage.min() > 0 else 0
                    }
        
        return comparisons
    
    def _generate_telecom_insights(self, context):
        """Generate deep, actionable telecom insights"""
        
        prompt = f"""You are a SENIOR TELECOM OPERATIONS ANALYST with 15+ years experience in:
- Subscriber analytics & churn prediction
- Network optimization & capacity planning  
- Revenue assurance & ARPU optimization
- Device adoption strategies (POTS, VoIP, Mobile apps)
- Regional performance management

Dataset Analysis:
Records: {context['total_records']:,}

IDENTIFIED DIMENSIONS:
{json.dumps(context['dimensions'], indent=2)}

KEY PERFORMANCE INDICATORS:
{json.dumps(context['kpis'], indent=2)}

TRENDS:
{json.dumps(context['trends'], indent=2)}

CROSS-DIMENSIONAL COMPARISONS:
{json.dumps(context['comparisons'], indent=2)}

YOUR TASK:
Generate 5 DEEP, ACTIONABLE insights that a VP of Operations can ACT ON immediately.

REQUIREMENTS:
1. BE SPECIFIC with numbers: "23.4K subscribers in Delhi" not "subscribers in Delhi"
2. QUANTIFY impact: "12% revenue increase possible" not "revenue opportunity"
3. COMPARE segments: "X is 45% higher than Y" not "X performs well"
4. STATE business implication: "Focus sales team on..." not "interesting pattern"
5. PROVIDE time context: "Last month vs previous" not vague references

TELECOM FOCUS AREAS:
- Subscriber growth/churn in specific regions
- Device adoption gaps (JioJoin vs POTS vs STB)
- Usage patterns (peak hours, weekday vs weekend)
- Revenue concentration & ARPU by segment
- Network capacity issues (high usage areas)
- Service quality gaps (complaints, tickets)
- Competitive pressure (market share shifts)

BAD EXAMPLE (too vague):
"North region shows concentrated distribution indicating market leader potential"

GOOD EXAMPLE (specific & actionable):
"North region: 45.2K active subscribers (32% of total base) with ARPU of ₹423, which is 18% higher than national average of ₹358. Recommend: (1) Increase sales headcount by 3 in North to capture growth, (2) Launch premium plan for high-ARPU segment targeting ₹500+ ARPU"

Format as JSON:
{{
  "summary": "1-2 sentence executive summary with KEY NUMBERS and MAIN ACTION",
  "insights": [
    {{
      "title": "Specific, Quantified Finding (e.g., 'Delhi ARPU 18% Above National Average')",
      "description": "DETAILED analysis with: (1) Specific numbers, (2) Comparison/benchmark, (3) Business implication, (4) Root cause if identifiable, (5) Recommended action",
      "impact": "high/medium/low",
      "category": "subscriber_growth/revenue/usage/churn/device_adoption/regional/quality",
      "metrics": {{
        "key_number": "45.2K",
        "percentage": "32%",
        "comparison": "+18% vs avg"
      }},
      "action": "Specific next step: 'Increase sales team by 3 FTEs in North region'"
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
                        "content": "You are a senior telecom analyst. Provide SPECIFIC, QUANTIFIED, ACTIONABLE insights. Always respond with valid JSON."
                    },
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
        if 'regional_distribution' in context['kpis']:
            top_region = context['kpis']['regional_distribution'].get('top_region')
            top_share = context['kpis']['regional_distribution'].get('top_region_share', 0)
            
            insights.append({
                'title': f'{top_region} Dominates Regional Distribution',
                'description': f'{top_region} accounts for {top_share:.1f}% of total records, indicating strong market concentration. Recommended action: Analyze why {top_region} performs better and replicate success factors in other regions.',
                'impact': 'high',
                'category': 'regional',
                'action': f'Conduct deep-dive study on {top_region} success factors'
            })
        
        # Usage comparison
        if 'region_usage' in context['comparisons']:
            comp = context['comparisons']['region_usage']
            insights.append({
                'title': f'{comp["top_region"]} Usage {comp["variance"]:.0f}% Higher',
                'description': f'{comp["top_region"]} shows {comp["top_region_total"]:.0f} total usage vs {comp["bottom_region"]} at {comp["bottom_region_total"]:.0f}. This {comp["variance"]:.0f}% variance suggests significant regional performance gaps. Action: Investigate infrastructure or marketing differences.',
                'impact': 'high',
                'category': 'usage',
                'action': f'Launch network optimization project in {comp["bottom_region"]}'
            })
        
        # Device adoption
        if 'device_usage' in context['comparisons']:
            dev = context['comparisons']['device_usage']
            insights.append({
                'title': f'{dev["highest"]} Shows {dev["difference_pct"]:.0f}% Higher Usage',
                'description': f'Users on {dev["highest"]} average {dev["highest_value"]:.1f} usage vs {dev["lowest"]} at {dev["lowest_value"]:.1f}. This {dev["difference_pct"]:.0f}% gap indicates device type significantly impacts engagement. Action: Push users to migrate from {dev["lowest"]} to {dev["highest"]}.',
                'impact': 'medium',
                'category': 'device_adoption',
                'action': f'Launch device migration campaign: {dev["lowest"]} → {dev["highest"]}'
            })
        
        summary = f"Analysis of {context['total_records']:,} records identifies significant regional and device-based performance variations requiring immediate action."
        
        return {
            'summary': summary,
            'insights': insights
        }
    
    def _detect_business_anomalies(self, df, context):
        """Detect business anomalies (not data quality)"""
        anomalies = []
        
        # Severe missing data that impacts business analysis
        for col in df.columns:
            if col.startswith('_'):
                continue
            null_pct = (df[col].isna().sum() / len(df)) * 100
            if null_pct > 40:  # Only if > 40%
                anomalies.append({
                    'type': 'data_gap',
                    'severity': 'warning',
                    'description': f'{col}: {null_pct:.0f}% incomplete - limits analysis depth',
                    'business_impact': 'May miss insights in this dimension'
                })
        
        return anomalies[:3]  # Max 3
    
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
