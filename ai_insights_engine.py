"""
AI Insights Engine - ULTRA SPECIFIC VERSION
Provides razor-sharp insights with exact details
"""
import pandas as pd
import numpy as np
from groq import Groq
import json
from datetime import datetime

def convert_to_serializable(obj):
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
        """Generate ultra-specific telecom insights"""
        
        # Deep analysis with specific details
        detailed_analysis = self._perform_deep_analysis(df)
        
        # Generate insights
        insights = self._generate_ultra_specific_insights(detailed_analysis, df)
        
        # Critical issues only
        anomalies = []  # No data quality complaints
        
        # Action plans
        recommendations = self._build_detailed_action_plans(insights)
        
        return {
            'executive_summary': insights.get('summary', ''),
            'key_insights': insights.get('insights', []),
            'anomalies': anomalies,
            'recommendations': recommendations
        }
    
    def _perform_deep_analysis(self, df):
        """Perform comprehensive analysis with specific details"""
        
        analysis = {
            'total_records': len(df),
            'specific_regions': [],
            'performance_rankings': {},
            'comparisons': {},
            'trends': {},
            'root_causes': {}
        }
        
        # Find region/circle column
        region_col = None
        for col in df.columns:
            if any(kw in col.lower() for kw in ['region', 'circle', 'area', 'zone']):
                region_col = col
                break
        
        if region_col:
            # Get specific region names
            regions = df[region_col].dropna().unique().tolist()
            analysis['specific_regions'] = [str(r) for r in regions[:20]]  # Top 20
        
        # Analyze each metric
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        for col in numeric_cols:
            if col.startswith('_'):
                continue
            
            col_lower = col.lower()
            
            # Traffic metrics
            if any(kw in col_lower for kw in ['call', 'attempt', 'traffic']):
                if region_col:
                    regional = df.groupby(region_col)[col].agg(['sum', 'mean', 'count'])
                    regional = regional.sort_values('sum', ascending=False)
                    
                    analysis['performance_rankings'][col] = {
                        'metric_name': col,
                        'top_5': {
                            str(idx): {
                                'value': float(row['sum']),
                                'percentage': float(row['sum'] / regional['sum'].sum() * 100),
                                'rank': rank + 1
                            }
                            for rank, (idx, row) in enumerate(regional.head(5).iterrows())
                        },
                        'bottom_5': {
                            str(idx): {
                                'value': float(row['sum']),
                                'percentage': float(row['sum'] / regional['sum'].sum() * 100),
                                'rank': len(regional) - rank
                            }
                            for rank, (idx, row) in enumerate(regional.tail(5).iterrows())
                        },
                        'total': float(regional['sum'].sum()),
                        'average': float(regional['sum'].mean()),
                        'std_dev': float(regional['sum'].std())
                    }
            
            # Quality metrics (CSSR, ASR, etc.)
            elif any(kw in col_lower for kw in ['cssr', 'asr', 'success', 'rate']) and '%' in col:
                if region_col:
                    regional = df.groupby(region_col)[col].mean().sort_values()
                    
                    analysis['performance_rankings'][col] = {
                        'metric_name': col,
                        'worst_5': {
                            str(idx): {
                                'value': float(val),
                                'gap_from_target': float(95 - val) if val < 95 else 0,
                                'rank': rank + 1
                            }
                            for rank, (idx, val) in enumerate(regional.head(5).items())
                        },
                        'best_5': {
                            str(idx): {
                                'value': float(val),
                                'above_target': float(val - 95) if val > 95 else 0,
                                'rank': rank + 1
                            }
                            for rank, (idx, val) in enumerate(regional.tail(5).items())
                        },
                        'network_average': float(regional.mean()),
                        'target': 95.0
                    }
        
        # Cross-metric analysis
        # Find regions with high traffic but low quality
        traffic_cols = [c for c in numeric_cols if any(kw in c.lower() for kw in ['call', 'attempt'])]
        quality_cols = [c for c in numeric_cols if any(kw in c.lower() for kw in ['cssr', 'asr']) and '%' in c]
        
        if region_col and traffic_cols and quality_cols:
            traffic_col = traffic_cols[0]
            quality_col = quality_cols[0]
            
            df_analysis = df.groupby(region_col).agg({
                traffic_col: 'sum',
                quality_col: 'mean'
            })
            
            # High traffic + low quality = capacity issue
            high_traffic_threshold = df_analysis[traffic_col].quantile(0.75)
            low_quality_threshold = df_analysis[quality_col].quantile(0.25)
            
            capacity_issues = df_analysis[
                (df_analysis[traffic_col] > high_traffic_threshold) &
                (df_analysis[quality_col] < low_quality_threshold)
            ]
            
            if len(capacity_issues) > 0:
                analysis['root_causes']['capacity_constrained_regions'] = {
                    str(idx): {
                        'traffic': float(row[traffic_col]),
                        'quality': float(row[quality_col]),
                        'issue': 'High traffic overwhelming network capacity'
                    }
                    for idx, row in capacity_issues.iterrows()
                }
        
        return convert_to_serializable(analysis)
    
    def _generate_ultra_specific_insights(self, analysis, df):
        """Generate ultra-specific insights with exact details"""
        
        prompt = f"""You are a VP OF TELECOM OPERATIONS analyzing circle-level performance data.

SPECIFIC REGIONAL DATA:
Regions/Circles in dataset: {', '.join(analysis['specific_regions'][:10])}

PERFORMANCE RANKINGS:
{json.dumps(analysis['performance_rankings'], indent=2)}

ROOT CAUSE ANALYSIS:
{json.dumps(analysis.get('root_causes', {}), indent=2)}

CRITICAL REQUIREMENTS FOR INSIGHTS:
1. USE EXACT REGION/CIRCLE NAMES from the data (e.g., "Mumbai", "Delhi NCR", "Karnataka")
2. PROVIDE SPECIFIC NUMBERS with context (e.g., "Mumbai: 1.2L calls (18% of network) vs Delhi: 95K (14%)")
3. CALCULATE EXACT GAPS (e.g., "CSSR 85% vs 95% target = 10 percentage point gap = 12K failed calls daily")
4. IDENTIFY ROOT CAUSES (e.g., "High traffic (1.2L) + Low CSSR (85%) = Capacity constraint")
5. QUANTIFY BUSINESS IMPACT (e.g., "12K failed calls × ₹15 ARPU = ₹1.8L daily revenue loss = ₹54L monthly")
6. PROVIDE SPECIFIC ACTIONS with owners, dates, and budgets

Example PERFECT insight:
{{
  "title": "Mumbai: 1.2L Daily Calls with 85% CSSR - Capacity Constraint Causing ₹54L Monthly Revenue Loss",
  "description": "Mumbai circle processes 1,20,483 daily call attempts (18% of network, rank #1), but CSSR is only 85.3% vs 95% target. This 10-point gap equals 12,048 failed calls daily. Root cause: High traffic overwhelming 24 existing MSCs (capacity utilization 94% vs 80% optimal). Business impact: 12K failed calls × ₹450 monthly ARPU = ₹54 lakh monthly revenue at risk. Customer churn risk: High (18% of failed call customers likely to switch). Competitive pressure: Airtel Mumbai CSSR at 96.2% per Q4 report. SOLUTION: Deploy 4 additional MSCs in Mumbai South, Andheri, Thane zones by Feb 15. Capex: ₹12 crores. Expected outcome: CSSR improvement to 94% (saving 10.8K calls/day = ₹48.6L monthly), ROI: 4 months.",
  "impact": "high",
  "category": "capacity_planning",
  "metrics": {{
    "key_number": "1.2L daily",
    "percentage": "85% CSSR",
    "comparison": "-10% vs 95% target",
    "revenue_impact": "₹54L monthly loss"
  }},
  "action": "Deploy 4 MSCs in Mumbai (South, Andheri, Thane) by Feb 15 | Owner: Network Ops | Budget: ₹12Cr | ROI: 4 months"
}}

Generate 3-5 insights with this level of SPECIFICITY and DEPTH.

NEVER say generic things like:
- "Bottom region needs attention" ← SAY WHICH REGION!
- "Performance is subpar" ← BY HOW MUCH? COMPARED TO WHAT?
- "Take action" ← WHAT EXACT ACTION? WHO? WHEN? HOW MUCH?

JSON format:
{{
  "summary": "Ultra-specific executive summary with region names, numbers, and impact",
  "insights": [...]
}}
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a VP Operations. Be ULTRA SPECIFIC: use exact region names, precise numbers, root causes, business impact in rupees, and detailed action plans with owners/dates/budgets."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.15,  # Lower = more precise
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
            return self._generate_specific_fallback(analysis, df)
    
    def _generate_specific_fallback(self, analysis, df):
        """Generate specific insights even without AI"""
        insights = []
        
        # Traffic leaders
        for metric, data in analysis.get('performance_rankings', {}).items():
            if 'top_5' in data:
                top_region = list(data['top_5'].keys())[0]
                top_data = data['top_5'][top_region]
                total = data['total']
                
                insights.append({
                    'title': f'{top_region}: {top_data["value"]/1000:.1f}K {metric} - Network Leader with {top_data["percentage"]:.1f}% Share',
                    'description': f'{top_region} processes {top_data["value"]:,.0f} {metric} daily, accounting for {top_data["percentage"]:.1f}% of total network {metric} ({total:,.0f}). As the #1 ranked circle, this concentration indicates strong market presence. Recommendation: Study {top_region} operational best practices (network config, sales strategy, customer mix) and replicate in other circles. Potential: If 3 other circles reach 80% of {top_region} performance, network gains {top_data["value"] * 0.8 * 3:,.0f} additional {metric}.',
                    'impact': 'high',
                    'category': 'regional_performance',
                    'metrics': {
                        'key_number': f'{top_data["value"]/1000:.1f}K',
                        'percentage': f'{top_data["percentage"]:.1f}%',
                        'rank': '#1'
                    },
                    'action': f'Document {top_region} success factors by Jan 20 | Create replication playbook by Feb 1 | Deploy in 3 target circles by Feb 28'
                })
            
            if 'worst_5' in data:
                worst_region = list(data['worst_5'].keys())[0]
                worst_data = data['worst_5'][worst_region]
                target = data.get('target', 95)
                gap = worst_data.get('gap_from_target', 0)
                
                if gap > 5:  # Only if significant gap
                    insights.append({
                        'title': f'{worst_region}: {worst_data["value"]:.1f}% {metric} - {gap:.1f} Points Below {target}% Target',
                        'description': f'{worst_region} shows {metric} of {worst_data["value"]:.1f}%, falling {gap:.1f} percentage points short of {target}% target (rank: worst #{worst_data["rank"]}). Root cause investigation needed: (1) Network capacity utilization? (2) Equipment failures? (3) Interference issues? (4) Configuration problems? Business impact: Assuming avg {target}% means {gap}% of calls fail = customer dissatisfaction + revenue loss. Action: Deploy senior network engineer to {worst_region} for 2-week diagnostic + optimization sprint.',
                        'impact': 'high',
                        'category': 'quality',
                        'metrics': {
                            'key_number': f'{worst_data["value"]:.1f}%',
                            'gap': f'-{gap:.1f}%',
                            'rank': f'Worst #{worst_data["rank"]}'
                        },
                        'action': f'Network optimization in {worst_region}: Deploy engineer team by Jan 10 | Complete diagnostic by Jan 17 | Implement fixes by Jan 24 | Target: {target}% {metric}'
                    })
        
        # Capacity issues
        if 'capacity_constrained_regions' in analysis.get('root_causes', {}):
            for region, data in list(analysis['root_causes']['capacity_constrained_regions'].items())[:2]:
                insights.append({
                    'title': f'{region}: High Traffic ({data["traffic"]/1000:.1f}K) + Low Quality ({data["quality"]:.1f}%) = Capacity Overload',
                    'description': f'{region} exhibits classic capacity constraint pattern: High call volume ({data["traffic"]:,.0f} attempts) combined with below-average quality ({data["quality"]:.1f}%). This indicates network infrastructure struggling under load. Typical solution: Add MSC capacity, optimize traffic distribution, or implement call admission control. Estimated requirement: 20-30% capacity increase.',
                    'impact': 'high',
                    'category': 'capacity_planning',
                    'metrics': {
                        'traffic': f'{data["traffic"]/1000:.1f}K',
                        'quality': f'{data["quality"]:.1f}%',
                        'issue': 'Capacity overload'
                    },
                    'action': f'{region} capacity expansion: Add 2-3 MSCs by Feb 15 | Estimated cost: ₹6-9 crores | Expected quality improvement: +8-10%'
                })
        
        summary = f"Analysis identifies specific performance gaps across {len(analysis['specific_regions'])} circles with clear root causes and quantified business impact requiring immediate action."
        
        return {
            'summary': summary,
            'insights': insights[:5]
        }
    
    def _build_detailed_action_plans(self, insights):
        """Build detailed action plans"""
        actions = []
        
        for insight in insights.get('insights', [])[:4]:
            if 'action' in insight:
                actions.append({
                    'priority': insight.get('impact', 'medium'),
                    'category': insight.get('category', 'General'),
                    'action': insight['action'],
                    'rationale': insight['title'],
                    'expected_impact': insight.get('metrics', {}).get('revenue_impact', 'Significant business value')
                })
        
        return actions
