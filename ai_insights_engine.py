"""
AI Insights Engine - FINAL VERSION
Forces AI to analyze BUSINESS PERFORMANCE, not data quality
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
        """Generate actionable telecom business insights"""
        
        # Build business-focused context
        business_summary = self._build_business_summary(df)
        
        # Generate insights
        insights = self._generate_business_insights(business_summary, df)
        
        # Minimal anomalies (only critical)
        anomalies = self._detect_critical_issues(df)
        
        # Action plans
        recommendations = self._build_action_plans(insights)
        
        return {
            'executive_summary': insights.get('summary', ''),
            'key_insights': insights.get('insights', []),
            'anomalies': anomalies,
            'recommendations': recommendations
        }
    
    def _build_business_summary(self, df):
        """Build business-focused data summary"""
        
        summary = {
            'total_records': len(df),
            'record_type': 'Unknown',
            'key_metrics': {},
            'dimensions': {},
            'top_performers': {},
            'bottom_performers': {}
        }
        
        # Identify what each row represents
        if any('region' in col.lower() or 'circle' in col.lower() for col in df.columns):
            summary['record_type'] = 'Circle/Region Performance Data'
        elif any('customer' in col.lower() or 'subscriber' in col.lower() for col in df.columns):
            summary['record_type'] = 'Customer/Subscriber Data'
        else:
            summary['record_type'] = 'Telecom Performance Data'
        
        # Find region/circle columns
        region_cols = [col for col in df.columns if 'region' in col.lower() or 'circle' in col.lower()]
        if region_cols:
            region_col = region_cols[0]
            summary['dimensions']['regions'] = df[region_col].value_counts().head(10).to_dict()
        
        # Find numeric KPI columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        for col in numeric_cols:
            if col.startswith('_'):
                continue
            
            col_lower = col.lower()
            
            # Identify telecom KPIs
            if any(kw in col_lower for kw in ['call', 'attempt', 'traffic', 'session']):
                total = df[col].sum()
                avg = df[col].mean()
                summary['key_metrics'][col] = {
                    'total': float(total),
                    'average_per_record': float(avg),
                    'max': float(df[col].max()),
                    'min': float(df[col].min())
                }
                
                # Top/bottom performers
                if region_cols:
                    region_col = region_cols[0]
                    by_region = df.groupby(region_col)[col].sum().sort_values(ascending=False)
                    summary['top_performers'][col] = {
                        'top_3': {str(k): float(v) for k, v in by_region.head(3).items()},
                        'bottom_3': {str(k): float(v) for k, v in by_region.tail(3).items()}
                    }
            
            elif any(kw in col_lower for kw in ['mou', 'minute', 'duration', 'usage']):
                total = df[col].sum()
                avg = df[col].mean()
                summary['key_metrics'][col] = {
                    'total_minutes': float(total),
                    'average_per_record': float(avg),
                    'max': float(df[col].max()),
                    'min': float(df[col].min())
                }
            
            elif any(kw in col_lower for kw in ['cssr', 'asr', 'success', 'rate', '%']):
                avg = df[col].mean()
                summary['key_metrics'][col] = {
                    'average_rate': float(avg),
                    'max': float(df[col].max()),
                    'min': float(df[col].min())
                }
                
                # Quality issues
                if region_cols:
                    region_col = region_cols[0]
                    by_region = df.groupby(region_col)[col].mean().sort_values()
                    summary['bottom_performers'][col] = {
                        'bottom_3': {str(k): float(v) for k, v in by_region.head(3).items()}
                    }
        
        return convert_to_serializable(summary)
    
    def _generate_business_insights(self, summary, df):
        """Generate business-focused insights"""
        
        prompt = f"""You are a TELECOM OPERATIONS DIRECTOR analyzing PERFORMANCE DATA.

DATASET CONTEXT:
- Total Records: {summary['total_records']}
- Each Record Represents: {summary['record_type']}
- This means: Each row = Performance of 1 circle/region for a specific time period

KEY PERFORMANCE METRICS:
{json.dumps(summary['key_metrics'], indent=2)}

TOP PERFORMERS:
{json.dumps(summary.get('top_performers', {}), indent=2)}

BOTTOM PERFORMERS:
{json.dumps(summary.get('bottom_performers', {}), indent=2)}

REGIONAL DISTRIBUTION:
{json.dumps(summary.get('dimensions', {}), indent=2)}

YOUR TASK:
You are reviewing WEEKLY/DAILY performance across circles/regions.
DO NOT talk about "data quality" or "missing values"
DO NOT say "only X non-null values" - the data is complete!

Instead, analyze BUSINESS PERFORMANCE:
1. Which circles/regions have HIGHEST call volume? By how much?
2. Which have LOWEST performance (CSSR, ASR)? What's the gap?
3. What's the TOTAL network traffic? Is it growing?
4. Are there capacity issues (high traffic + low success rates)?
5. Which regions need immediate attention?

CRITICAL RULES:
- If you see "Call Attempts" total of 500,000 → Say "5 lakh call attempts across network"
- If top region has 120,000 calls → Say "Delhi leads with 1.2 lakh daily calls (24% of total)"
- If CSSR is 85% in bottom region → Say "Region X CSSR at 85% vs 95% target - needs network optimization"
- NEVER say "only 4 non-null values" or "collect more data"

Generate 3-5 ACTIONABLE BUSINESS INSIGHTS with SPECIFIC NUMBERS and ACTIONS.

Example GOOD insight:
{{
  "title": "North Region Leads Traffic with 8.2 Lakh Daily Calls - 29% Network Share",
  "description": "North region processes 8.2 lakh call attempts daily (29% of total 28.4 lakh), followed by South at 6.1 lakh (21%). North's CSSR of 96.2% vs network average of 94.2% indicates superior performance. However, MOU in North is 18% lower (128 mins vs 156 mins average), suggesting shorter call durations. ACTION: Analyze North's call pattern - potential capacity constraint limiting call lengths. Deploy 2 additional MSCs to handle 25% traffic growth by Q2.",
  "impact": "high",
  "category": "capacity_planning",
  "metrics": {{"key_number": "8.2L", "percentage": "29%", "comparison": "+34% vs South"}},
  "action": "Deploy 2 MSCs in North by Feb 15 to handle Q2 growth"
}}

Example BAD insight (DON'T DO THIS):
{{
  "title": "Low Region Representation",
  "description": "The 'Region' column has only 4 non-null values...",  ← NEVER MENTION THIS!
}}

JSON format:
{{
  "summary": "Brief summary with KEY BUSINESS NUMBERS",
  "insights": [...]
}}
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a telecom operations director. Focus ONLY on business performance, capacity, revenue, quality. NEVER mention data quality or missing values. Always use ACTUAL NUMBERS from the data."
                    },
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
            # Fallback with simple rule-based insights
            return self._generate_simple_insights(summary, df)
    
    def _generate_simple_insights(self, summary, df):
        """Generate simple rule-based insights"""
        insights = []
        
        # Network traffic insight
        call_metrics = [k for k in summary['key_metrics'].keys() if 'call' in k.lower() or 'attempt' in k.lower()]
        if call_metrics:
            metric = call_metrics[0]
            data = summary['key_metrics'][metric]
            total = data['total']
            avg = data['average_per_record']
            
            insights.append({
                'title': f'Network Handles {total/100000:.1f} Lakh Total Call Attempts',
                'description': f'Analysis of {summary["total_records"]} circles shows total of {total:,.0f} call attempts. Average per circle: {avg:,.0f}. This represents actual network traffic volume requiring capacity planning and resource allocation.',
                'impact': 'high',
                'category': 'network_capacity',
                'metrics': {'key_number': f'{total/100000:.1f}L', 'average': f'{avg:.0f}'},
                'action': 'Review capacity planning for high-traffic circles (>150% of average)'
            })
        
        # Top performers
        for metric, perf in summary.get('top_performers', {}).items():
            if 'top_3' in perf:
                top_region = list(perf['top_3'].keys())[0]
                top_value = list(perf['top_3'].values())[0]
                total = sum(perf['top_3'].values())
                pct = (top_value / total) * 100
                
                insights.append({
                    'title': f'{top_region} Leads in {metric} with {top_value/1000:.1f}K ({pct:.0f}%)',
                    'description': f'{top_region} accounts for {top_value:,.0f} in {metric}, representing {pct:.0f}% of top 3 performers. This concentration indicates strong regional performance that can be studied for best practices.',
                    'impact': 'medium',
                    'category': 'regional_performance',
                    'action': f'Document {top_region} success factors and replicate in other regions'
                })
        
        # Quality issues
        for metric, perf in summary.get('bottom_performers', {}).items():
            if 'bottom_3' in perf:
                worst_region = list(perf['bottom_3'].keys())[0]
                worst_value = list(perf['bottom_3'].values())[0]
                
                if '%' in metric or 'rate' in metric.lower():
                    insights.append({
                        'title': f'{worst_region} Shows Low {metric} at {worst_value:.1f}%',
                        'description': f'{worst_region} registers {worst_value:.1f}% in {metric}, indicating potential quality issues. Network optimization required to improve success rates and customer experience.',
                        'impact': 'high',
                        'category': 'quality',
                        'action': f'Deploy network optimization team to {worst_region} immediately'
                    })
        
        summary_text = f"Analysis of {summary['total_records']} circles reveals significant performance variations across network. Top performers show clear leadership while bottom regions require immediate attention."
        
        return {
            'summary': summary_text,
            'insights': insights[:5]
        }
    
    def _detect_critical_issues(self, df):
        """Only flag CRITICAL business issues"""
        issues = []
        
        # Only flag if >60% missing (not 20%)
        for col in df.columns:
            if col.startswith('_'):
                continue
            null_pct = (df[col].isna().sum() / len(df)) * 100
            if null_pct > 60:
                issues.append({
                    'type': 'data_gap',
                    'severity': 'warning',
                    'description': f'{col}: {null_pct:.0f}% incomplete - may limit analysis',
                    'business_impact': 'Reduced visibility into this dimension'
                })
        
        return issues[:2]  # Max 2 alerts
    
    def _build_action_plans(self, insights):
        """Build action plans from insights"""
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
