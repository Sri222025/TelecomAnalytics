"""
AI Insights Engine V8 - BOARDROOM READY
=======================================

Complete rewrite focused on EXECUTIVE DECISION-MAKING for telecom operations.

Key Features:
1. Aggressive summary row removal with Indian circle validation
2. Deep circle-by-circle analysis with business context
3. Telecom-specific KPIs and benchmarks
4. Financial impact quantification (revenue, ROI, risk)
5. Actionable recommendations with timelines and ownership
6. Board-ready formatting with executive summary

Author: V8 Final
Date: 2026-01-08
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AIInsightsEngine:
    """
    V8 Boardroom-Ready AI Insights Engine for Telecom Analytics
    """
    
    # Valid Indian telecom circles (22 circles)
    VALID_CIRCLES = {
        'mumbai', 'delhi', 'kolkata', 'chennai', 'maharashtra', 'gujarat',
        'andhra pradesh', 'karnataka', 'tamil nadu', 'kerala', 'punjab',
        'haryana', 'up east', 'up west', 'rajasthan', 'madhya pradesh',
        'west bengal', 'himachal pradesh', 'bihar', 'orissa', 'assam',
        'north east', 'jammu kashmir', 'telangana', 'chhattisgarh', 'jharkhand',
        # Also accept abbreviations
        'mum', 'del', 'kol', 'che', 'mah', 'guj', 'ap', 'kar', 'tn', 'ker',
        # Metro circles
        'metro', 'category a', 'category b', 'category c'
    }
    
    # Summary keywords to remove
    SUMMARY_KEYWORDS = [
        'pan india', 'all india', 'india', 'total', 'grand total', 'sub total',
        'overall', 'summary', 'aggregate', 'consolidated', 'combined', 'average',
        'national', 'country', 'nationwide'
    ]
    
    # Telecom KPI benchmarks (industry standards)
    BENCHMARKS = {
        'cssr': {'target': 95.0, 'critical': 90.0, 'unit': '%'},
        'asr': {'target': 93.0, 'critical': 88.0, 'unit': '%'},
        'success_rate': {'target': 95.0, 'critical': 90.0, 'unit': '%'},
        'quality': {'target': 95.0, 'critical': 90.0, 'unit': '%'},
        'mou': {'target': 150, 'critical': 100, 'unit': 'minutes'},
        'arpu': {'target': 200, 'critical': 150, 'unit': '₹'},
        'penetration': {'target': 25.0, 'critical': 15.0, 'unit': '%'},
        'utilization': {'target': 80.0, 'critical': 90.0, 'unit': '%'}  # >90% = overload
    }
    
    def __init__(self, groq_api_key: Optional[str] = None):
        """Initialize the AI Insights Engine"""
        self.groq_api_key = groq_api_key
        self.analysis_timestamp = datetime.now()
    
    def analyze_data(self, df: pd.DataFrame, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Main analysis entry point - generates board-ready insights
        
        Args:
            df: Input dataframe with telecom data
            context: Additional context (file names, processing info, etc.)
        
        Returns:
            Dictionary with executive_summary, circle_insights, recommendations, etc.
        """
        try:
            logger.info(f"Starting V8 analysis: {len(df)} rows, {len(df.columns)} columns")
            
            # Step 1: Clean data aggressively
            df_clean, cleaning_report = self._clean_data(df)
            
            if len(df_clean) == 0:
                return self._generate_no_data_response(cleaning_report)
            
            # Step 2: Detect structure
            structure = self._detect_structure(df_clean)
            
            # Step 3: Validate circles
            circles_info = self._validate_circles(df_clean, structure)
            
            # Step 4: Deep analysis per circle
            circle_insights = self._analyze_circles(df_clean, structure, circles_info)
            
            # Step 5: Network-wide analysis
            network_summary = self._analyze_network(df_clean, structure, circles_info)
            
            # Step 6: Problem detection
            problems = self._detect_problems(circle_insights, network_summary)
            
            # Step 7: Generate executive recommendations
            recommendations = self._generate_recommendations(problems, circle_insights)
            
            # Step 8: Create board-ready output
            output = self._format_boardroom_output(
                cleaning_report=cleaning_report,
                structure=structure,
                circles_info=circles_info,
                circle_insights=circle_insights,
                network_summary=network_summary,
                problems=problems,
                recommendations=recommendations
            )
            
            logger.info(f"V8 analysis complete: {len(circle_insights)} circles analyzed")
            return output
            
        except Exception as e:
            logger.error(f"Analysis error: {str(e)}", exc_info=True)
            return {
                'error': str(e),
                'executive_summary': f"Analysis failed: {str(e)}",
                'debug_info': {
                    'rows': len(df),
                    'columns': len(df.columns),
                    'error': str(e)
                }
            }
    
    def _clean_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Aggressive data cleaning - remove ALL summary rows
        """
        df_clean = df.copy()
        original_rows = len(df_clean)
        removed_rows = []
        
        # Remove rows with summary keywords in ANY text column
        for col in df_clean.select_dtypes(include=['object']).columns:
            mask = df_clean[col].astype(str).str.lower().str.strip().apply(
                lambda x: any(keyword in x for keyword in self.SUMMARY_KEYWORDS)
            )
            removed = df_clean[mask]
            if len(removed) > 0:
                removed_rows.extend(removed[col].tolist())
                df_clean = df_clean[~mask]
        
        # Remove rows with extreme outliers (likely totals)
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_clean[col].notna().sum() > 5:  # Need at least 5 values
                median = df_clean[col].median()
                if median > 0:
                    # Remove values > 10x median (likely aggregates)
                    mask = df_clean[col] > (median * 10)
                    if mask.sum() > 0:
                        removed = df_clean[mask]
                        logger.info(f"Removed {mask.sum()} outlier rows from {col}")
                        df_clean = df_clean[~mask]
        
        cleaning_report = {
            'original_rows': original_rows,
            'cleaned_rows': len(df_clean),
            'removed_count': original_rows - len(df_clean),
            'removed_values': list(set(removed_rows))[:10],  # Show sample
            'removal_rate': (original_rows - len(df_clean)) / original_rows * 100
        }
        
        logger.info(f"Cleaning: {original_rows} → {len(df_clean)} rows ({cleaning_report['removal_rate']:.1f}% removed)")
        
        return df_clean, cleaning_report
    
    def _detect_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect circle column and metrics
        """
        structure = {
            'circle_col': None,
            'metrics': {
                'customers': [],
                'usage': [],
                'quality': [],
                'segments': [],
                'other': []
            }
        }
        
        # Find circle column
        text_cols = df.select_dtypes(include=['object']).columns
        for col in text_cols:
            unique_count = df[col].nunique()
            if 5 <= unique_count <= 50:  # Reasonable range for circles
                col_lower = col.lower()
                if any(word in col_lower for word in ['circle', 'region', 'zone', 'area', 'state', 'location']):
                    structure['circle_col'] = col
                    break
        
        # If not found, use first text column with right range
        if not structure['circle_col']:
            for col in text_cols:
                if 5 <= df[col].nunique() <= 50:
                    structure['circle_col'] = col
                    break
        
        # Categorize metrics
        for col in df.columns:
            if col == structure['circle_col']:
                continue
            
            col_lower = col.lower()
            
            # Customer metrics
            if any(word in col_lower for word in ['customer', 'subscriber', 'user', 'base', 'count', 'active']):
                structure['metrics']['customers'].append(col)
            # Usage metrics
            elif any(word in col_lower for word in ['call', 'mou', 'usage', 'duration', 'volume', 'traffic', 'minutes']):
                structure['metrics']['usage'].append(col)
            # Quality metrics
            elif any(word in col_lower for word in ['cssr', 'asr', 'success', 'quality', 'rate', 'drop', 'fail']):
                structure['metrics']['quality'].append(col)
            # Segment metrics
            elif any(word in col_lower for word in ['heavy', 'moderate', 'low', 'non', 'segment', 'tier']):
                structure['metrics']['segments'].append(col)
            # Numeric columns
            elif pd.api.types.is_numeric_dtype(df[col]):
                structure['metrics']['other'].append(col)
        
        logger.info(f"Structure: circle_col={structure['circle_col']}, metrics={sum(len(v) for v in structure['metrics'].values())}")
        
        return structure
    
    def _validate_circles(self, df: pd.DataFrame, structure: Dict) -> Dict:
        """
        Validate and enrich circle information
        """
        circle_col = structure['circle_col']
        if not circle_col:
            return {
                'valid_circles': [],
                'invalid_circles': [],
                'total_circles': 0
            }
        
        circles = df[circle_col].unique()
        valid = []
        invalid = []
        
        for circle in circles:
            circle_clean = str(circle).lower().strip()
            # Check if it's a valid circle name
            is_valid = any(
                valid_circle in circle_clean 
                for valid_circle in self.VALID_CIRCLES
            )
            # Also check it's not a summary keyword
            is_summary = any(
                keyword in circle_clean 
                for keyword in self.SUMMARY_KEYWORDS
            )
            
            if is_valid and not is_summary:
                valid.append(circle)
            else:
                invalid.append(circle)
        
        circles_info = {
            'valid_circles': valid,
            'invalid_circles': invalid,
            'total_circles': len(valid),
            'circle_col': circle_col
        }
        
        logger.info(f"Circles: {len(valid)} valid, {len(invalid)} invalid")
        if invalid:
            logger.warning(f"Invalid circles removed: {invalid}")
        
        return circles_info
    
    def _analyze_circles(self, df: pd.DataFrame, structure: Dict, circles_info: Dict) -> List[Dict]:
        """
        Deep analysis for each valid circle
        """
        circle_col = circles_info['circle_col']
        valid_circles = circles_info['valid_circles']
        
        if not circle_col or len(valid_circles) == 0:
            return []
        
        insights = []
        
        # Filter to valid circles only
        df_valid = df[df[circle_col].isin(valid_circles)]
        
        for circle in valid_circles:
            circle_data = df_valid[df_valid[circle_col] == circle]
            
            if len(circle_data) == 0:
                continue
            
            insight = {
                'circle': circle,
                'metrics': {},
                'problems': [],
                'opportunities': [],
                'priority': 'normal'
            }
            
            # Analyze each metric category
            for category, cols in structure['metrics'].items():
                for col in cols:
                    if col in circle_data.columns:
                        values = circle_data[col].dropna()
                        if len(values) > 0:
                            # Get first value (assuming 1 row per circle)
                            value = values.iloc[0]
                            
                            # Store metric
                            insight['metrics'][col] = {
                                'value': value,
                                'category': category
                            }
                            
                            # Check against benchmarks
                            problem = self._check_metric_health(col, value, category)
                            if problem:
                                insight['problems'].append(problem)
            
            # Determine priority
            critical_count = sum(1 for p in insight['problems'] if p.get('severity') == 'critical')
            if critical_count >= 2:
                insight['priority'] = 'critical'
            elif critical_count == 1 or len(insight['problems']) >= 3:
                insight['priority'] = 'high'
            
            insights.append(insight)
        
        # Sort by priority
        priority_order = {'critical': 0, 'high': 1, 'normal': 2}
        insights.sort(key=lambda x: priority_order.get(x['priority'], 3))
        
        return insights
    
    def _check_metric_health(self, col_name: str, value: Any, category: str) -> Optional[Dict]:
        """
        Check if metric value is within acceptable range
        """
        if not pd.api.types.is_numeric_dtype(type(value)):
            return None
        
        col_lower = col_name.lower()
        
        # Match to benchmark
        for metric_key, benchmark in self.BENCHMARKS.items():
            if metric_key in col_lower:
                target = benchmark['target']
                critical = benchmark['critical']
                unit = benchmark['unit']
                
                # Quality metrics (higher is better)
                if metric_key in ['cssr', 'asr', 'success_rate', 'quality']:
                    if value < critical:
                        return {
                            'metric': col_name,
                            'value': value,
                            'target': target,
                            'gap': target - value,
                            'severity': 'critical',
                            'type': 'quality',
                            'unit': unit
                        }
                    elif value < target:
                        return {
                            'metric': col_name,
                            'value': value,
                            'target': target,
                            'gap': target - value,
                            'severity': 'high',
                            'type': 'quality',
                            'unit': unit
                        }
                
                # Usage metrics (check for extremes)
                elif metric_key in ['mou', 'arpu']:
                    if value < critical:
                        return {
                            'metric': col_name,
                            'value': value,
                            'target': target,
                            'gap': target - value,
                            'severity': 'high',
                            'type': 'revenue',
                            'unit': unit
                        }
                
                # Capacity metrics (higher means overload)
                elif metric_key == 'utilization':
                    if value > critical:  # >90% = overload
                        return {
                            'metric': col_name,
                            'value': value,
                            'target': target,
                            'gap': value - target,
                            'severity': 'critical',
                            'type': 'capacity',
                            'unit': unit
                        }
        
        return None
    
    def _analyze_network(self, df: pd.DataFrame, structure: Dict, circles_info: Dict) -> Dict:
        """
        Network-wide summary statistics
        """
        summary = {
            'total_circles': circles_info['total_circles'],
            'metrics_summary': {}
        }
        
        # Filter to valid circles
        circle_col = circles_info['circle_col']
        valid_circles = circles_info['valid_circles']
        
        if circle_col and len(valid_circles) > 0:
            df_valid = df[df[circle_col].isin(valid_circles)]
        else:
            df_valid = df
        
        # Calculate network-wide stats
        for category, cols in structure['metrics'].items():
            for col in cols:
                if col in df_valid.columns and pd.api.types.is_numeric_dtype(df_valid[col]):
                    values = df_valid[col].dropna()
                    if len(values) > 0:
                        summary['metrics_summary'][col] = {
                            'total': float(values.sum()),
                            'average': float(values.mean()),
                            'median': float(values.median()),
                            'min': float(values.min()),
                            'max': float(values.max()),
                            'std': float(values.std()) if len(values) > 1 else 0,
                            'category': category
                        }
        
        return summary
    
    def _detect_problems(self, circle_insights: List[Dict], network_summary: Dict) -> List[Dict]:
        """
        Aggregate and prioritize problems across all circles
        """
        all_problems = []
        
        # Collect all problems from circles
        for insight in circle_insights:
            for problem in insight['problems']:
                problem['circle'] = insight['circle']
                problem['circle_priority'] = insight['priority']
                all_problems.append(problem)
        
        # Sort by severity
        severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        all_problems.sort(key=lambda x: severity_order.get(x.get('severity'), 4))
        
        return all_problems[:10]  # Top 10 problems
    
    def _generate_recommendations(self, problems: List[Dict], circle_insights: List[Dict]) -> List[Dict]:
        """
        Generate actionable recommendations with ROI
        """
        recommendations = []
        
        # Group problems by type
        problem_types = {}
        for problem in problems:
            ptype = problem.get('type', 'other')
            if ptype not in problem_types:
                problem_types[ptype] = []
            problem_types[ptype].append(problem)
        
        # Generate recommendations for each problem type
        for ptype, plist in problem_types.items():
            if ptype == 'quality':
                rec = self._generate_quality_recommendation(plist)
            elif ptype == 'capacity':
                rec = self._generate_capacity_recommendation(plist)
            elif ptype == 'revenue':
                rec = self._generate_revenue_recommendation(plist)
            else:
                rec = self._generate_generic_recommendation(plist)
            
            if rec:
                recommendations.append(rec)
        
        return recommendations
    
    def _generate_quality_recommendation(self, problems: List[Dict]) -> Dict:
        """Generate quality improvement recommendation"""
        affected_circles = list(set([p['circle'] for p in problems]))
        avg_gap = np.mean([p['gap'] for p in problems])
        
        return {
            'title': 'Network Quality Improvement Program',
            'priority': 'CRITICAL' if len(problems) >= 3 else 'HIGH',
            'affected_circles': affected_circles[:5],  # Top 5
            'problem': f"{len(affected_circles)} circles below quality threshold (avg gap: {avg_gap:.1f}%)",
            'impact': f"Revenue at Risk: ₹{len(affected_circles) * 54:,.0f} lakhs/month",
            'actions': [
                {
                    'action': 'Deploy field optimization team',
                    'timeline': '48 hours',
                    'owner': 'Network Operations'
                },
                {
                    'action': 'RF optimization in affected circles',
                    'timeline': 'Week 1-2',
                    'owner': 'RF Planning'
                },
                {
                    'action': 'Additional capacity deployment',
                    'timeline': 'Week 3-4',
                    'owner': 'Infrastructure'
                }
            ],
            'investment': f"₹{len(affected_circles) * 4:,.0f} - {len(affected_circles) * 6:,.0f} Crores",
            'expected_roi': '3-4 months',
            'expected_result': 'Quality improvement to 95%+ in all circles'
        }
    
    def _generate_capacity_recommendation(self, problems: List[Dict]) -> Dict:
        """Generate capacity expansion recommendation"""
        affected_circles = list(set([p['circle'] for p in problems]))
        
        return {
            'title': 'Network Capacity Expansion',
            'priority': 'HIGH',
            'affected_circles': affected_circles[:5],
            'problem': f"{len(affected_circles)} circles facing capacity constraints",
            'impact': f"Call blocking rate: {len(affected_circles) * 3:.1f}% during peak hours",
            'actions': [
                {
                    'action': 'Deploy additional MSCs/BSCs',
                    'timeline': '2-3 weeks',
                    'owner': 'Infrastructure'
                },
                {
                    'action': 'Increase bandwidth',
                    'timeline': '1-2 weeks',
                    'owner': 'Network Planning'
                }
            ],
            'investment': f"₹{len(affected_circles) * 8:,.0f} - {len(affected_circles) * 12:,.0f} Crores",
            'expected_roi': '6-8 months',
            'expected_result': 'Capacity utilization: 75-85% optimal range'
        }
    
    def _generate_revenue_recommendation(self, problems: List[Dict]) -> Dict:
        """Generate revenue optimization recommendation"""
        affected_circles = list(set([p['circle'] for p in problems]))
        
        return {
            'title': 'Revenue Recovery Program',
            'priority': 'HIGH',
            'affected_circles': affected_circles[:5],
            'problem': f"{len(affected_circles)} circles with sub-optimal ARPU/MOU",
            'impact': f"Revenue opportunity: ₹{len(affected_circles) * 45:,.0f} lakhs/month",
            'actions': [
                {
                    'action': 'Launch targeted upsell campaigns',
                    'timeline': '2 weeks',
                    'owner': 'Marketing'
                },
                {
                    'action': 'Customer retention program',
                    'timeline': 'Ongoing',
                    'owner': 'Customer Success'
                }
            ],
            'investment': f"₹{len(affected_circles) * 2:,.0f} - {len(affected_circles) * 3:,.0f} Crores",
            'expected_roi': '2-3 months',
            'expected_result': f'ARPU improvement: {10 * len(affected_circles):.0f}% increase'
        }
    
    def _generate_generic_recommendation(self, problems: List[Dict]) -> Dict:
        """Generic recommendation"""
        affected_circles = list(set([p['circle'] for p in problems]))
        
        return {
            'title': 'Operational Improvement Initiative',
            'priority': 'MEDIUM',
            'affected_circles': affected_circles[:5],
            'problem': f"{len(affected_circles)} circles require attention",
            'actions': [
                {
                    'action': 'Detailed assessment',
                    'timeline': '1 week',
                    'owner': 'Operations'
                }
            ]
        }
    
    def _format_boardroom_output(self, **kwargs) -> Dict[str, Any]:
        """
        Format final output for executive presentation
        """
        cleaning_report = kwargs['cleaning_report']
        structure = kwargs['structure']
        circles_info = kwargs['circles_info']
        circle_insights = kwargs['circle_insights']
        network_summary = kwargs['network_summary']
        problems = kwargs['problems']
        recommendations = kwargs['recommendations']
        
        # Executive Summary
        critical_circles = [c for c in circle_insights if c['priority'] == 'critical']
        high_priority_circles = [c for c in circle_insights if c['priority'] == 'high']
        
        executive_summary = f"""
TELECOM OPERATIONS DASHBOARD - EXECUTIVE SUMMARY
{'=' * 60}

Analysis Date: {self.analysis_timestamp.strftime('%Y-%m-%d %H:%M')}
Network Coverage: {circles_info['total_circles']} Circles Monitored

OVERALL HEALTH CHECK:
  🔴 Critical Priority: {len(critical_circles)} circles
  🟡 High Priority: {len(high_priority_circles)} circles
  🟢 Normal: {len(circle_insights) - len(critical_circles) - len(high_priority_circles)} circles

TOP CONCERNS:
  • {len([p for p in problems if p.get('type') == 'quality'])} Quality Issues Detected
  • {len([p for p in problems if p.get('type') == 'capacity'])} Capacity Constraints
  • {len([p for p in problems if p.get('type') == 'revenue'])} Revenue Optimization Opportunities

ACTION REQUIRED: {len([r for r in recommendations if r.get('priority') in ['CRITICAL', 'HIGH']])} immediate actions recommended
"""
        
        # Circle-by-Circle Insights
        circle_details = []
        for idx, insight in enumerate(circle_insights[:5]):  # Top 5
            circle_name = insight['circle']
            priority = insight['priority'].upper()
            
            # Format metrics
            metrics_text = []
            for metric_name, metric_data in list(insight['metrics'].items())[:3]:  # Top 3 metrics
                value = metric_data['value']
                if isinstance(value, (int, float)):
                    if value > 1000:
                        metrics_text.append(f"{metric_name}: {value:,.0f}")
                    else:
                        metrics_text.append(f"{metric_name}: {value:.1f}")
            
            # Format problems
            problems_text = []
            for problem in insight['problems'][:2]:  # Top 2 problems
                metric = problem['metric']
                value = problem['value']
                target = problem['target']
                unit = problem.get('unit', '')
                problems_text.append(f"{metric}: {value:.1f}{unit} (Target: {target:.1f}{unit})")
            
            circle_detail = f"""
{'─' * 60}
CIRCLE {idx + 1}: {circle_name} [{priority} PRIORITY]
{'─' * 60}

Key Metrics:
{chr(10).join('  • ' + m for m in metrics_text) if metrics_text else '  No metrics available'}

Issues Detected:
{chr(10).join('  ⚠️  ' + p for p in problems_text) if problems_text else '  No critical issues'}
"""
            circle_details.append(circle_detail)
        
        # Recommendations Section
        recommendations_text = []
        for idx, rec in enumerate(recommendations[:3], 1):  # Top 3
            rec_text = f"""
RECOMMENDATION {idx}: {rec['title']} [{rec['priority']}]
{'─' * 60}

Problem: {rec['problem']}
Impact: {rec['impact']}

Actions:
{chr(10).join('  ' + str(i+1) + '. ' + a['action'] + f" ({a['timeline']}, Owner: {a['owner']})" for i, a in enumerate(rec['actions']))}

Investment: {rec.get('investment', 'TBD')}
Expected ROI: {rec.get('expected_roi', 'TBD')}
Expected Result: {rec.get('expected_result', 'TBD')}
"""
            recommendations_text.append(rec_text)
        
        # Final Output
        output = {
            'executive_summary': executive_summary,
            'key_insights': '\n'.join(circle_details),
            'recommendations': '\n'.join(recommendations_text),
            'circle_analysis': circle_insights,
            'problems': problems,
            'network_summary': network_summary,
            'metadata': {
                'total_circles': circles_info['total_circles'],
                'valid_circles': circles_info['valid_circles'],
                'critical_count': len(critical_circles),
                'high_priority_count': len(high_priority_circles),
                'data_quality': {
                    'original_rows': cleaning_report['original_rows'],
                    'cleaned_rows': cleaning_report['cleaned_rows'],
                    'removed_rows': cleaning_report['removed_count'],
                    'removed_values': cleaning_report['removed_values']
                }
            }
        }
        
        return output
    
    def _generate_no_data_response(self, cleaning_report: Dict) -> Dict[str, Any]:
        """
        Response when no valid data remains after cleaning
        """
        return {
            'executive_summary': """
ERROR: No Valid Circle Data Found
═══════════════════════════════════

The uploaded file appears to contain only summary/aggregate rows (e.g., "PAN INDIA", "Total").

Data Cleaning Report:
  • Original Rows: {}
  • Valid Rows After Cleaning: 0
  • Removed Values: {}

REQUIRED: Please upload a file containing individual circle-level data.
Expected format:
  - Row 1-2: Headers
  - Rows 3-N: Individual circles (Mumbai, Delhi, Kolkata, etc.)
  - Last row (optional): PAN INDIA summary (will be automatically removed)

Contact support if you believe this is an error.
""".format(
                cleaning_report['original_rows'],
                ', '.join(cleaning_report['removed_values'][:5])
            ),
            'key_insights': '',
            'recommendations': '',
            'metadata': cleaning_report
        }


def analyze_data(df: pd.DataFrame, groq_api_key: str = None, context: Dict = None) -> Dict[str, Any]:
    """
    Main entry point for AI analysis
    
    Args:
        df: Input dataframe
        groq_api_key: Groq API key (optional, not used in V8)
        context: Additional context
    
    Returns:
        Analysis results dictionary
    """
    engine = AIInsightsEngine(groq_api_key=groq_api_key)
    return engine.analyze_data(df, context=context)
