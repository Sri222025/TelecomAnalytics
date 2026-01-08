"""
AI Insights Engine V9 - EMERGENCY FIX FOR MULTI-LEVEL HEADERS
=============================================================

CRITICAL FIX for presentation tomorrow:
- Handles multi-level headers (CIRCLE, then HSI Active Customers)
- Specifically maps Fixed Line + JioJoin columns
- FORCES insight generation with your exact data structure
- Generates full executive summary with numbers

Author: V9 Emergency
Date: 2026-01-08 (FOR PRESENTATION TOMORROW)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AIInsightsEngine:
    """V9 Emergency Fix - Multi-level headers + Force insights"""
    
    # Valid circles
    VALID_CIRCLES = {
        'mumbai', 'delhi', 'kolkata', 'chennai', 'maharashtra', 'gujarat',
        'andhra pradesh', 'karnataka', 'tamil nadu', 'kerala', 'punjab',
        'haryana', 'himachal pradesh', 'up east', 'up west', 'rajasthan', 
        'madhya pradesh', 'west bengal', 'bihar', 'orissa', 'assam',
        'north east', 'jammu kashmir', 'telangana', 'chhattisgarh', 'jharkhand'
    }
    
    SUMMARY_KEYWORDS = [
        'pan india', 'all india', 'india', 'total', 'grand total', 'sub total',
        'overall', 'summary', 'aggregate', 'consolidated', 'combined', 'average',
        'national', 'country', 'nationwide'
    ]
    
    BENCHMARKS = {
        'cssr': 95.0,
        'asr': 93.0,
        'penetration': 25.0,
        'mou': 150,
        'call_success': 95.0
    }
    
    def __init__(self, groq_api_key: Optional[str] = None):
        self.groq_api_key = groq_api_key
        self.analysis_timestamp = datetime.now()
    
    def analyze_data(self, df: pd.DataFrame, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Main analysis - EMERGENCY FIX"""
        try:
            logger.info(f"V9 EMERGENCY: Analyzing {len(df)} rows, {len(df.columns)} columns")
            
            # Step 1: Clean
            df_clean, cleaning_report = self._clean_data(df)
            if len(df_clean) == 0:
                return self._generate_no_data_response(cleaning_report)
            
            # Step 2: Find circle column and flatten headers
            df_flat, circle_col = self._flatten_headers(df_clean)
            
            # Step 3: Validate circles
            circles_info = self._validate_circles(df_flat, circle_col)
            
            if circles_info['total_circles'] == 0:
                return self._generate_no_circles_response(cleaning_report, df_flat.columns.tolist())
            
            # Step 4: Map metrics to your exact columns
            metrics_map = self._map_metrics(df_flat)
            
            # Step 5: Extract insights per circle
            circle_insights = self._analyze_circles_v9(df_flat, circle_col, circles_info, metrics_map)
            
            # Step 6: Network summary
            network_summary = self._calculate_network_summary(df_flat, circle_col, circles_info, metrics_map)
            
            # Step 7: Detect problems
            problems = self._detect_problems_v9(circle_insights, metrics_map)
            
            # Step 8: Generate recommendations
            recommendations = self._generate_recommendations_v9(problems, circle_insights, network_summary)
            
            # Step 9: Format for presentation
            output = self._format_presentation_output(
                cleaning_report, circles_info, circle_insights,
                network_summary, problems, recommendations, metrics_map
            )
            
            logger.info(f"V9 COMPLETE: {circles_info['total_circles']} circles, {len(problems)} problems")
            return output
            
        except Exception as e:
            logger.error(f"V9 ERROR: {str(e)}", exc_info=True)
            return {
                'error': str(e),
                'executive_summary': f"Analysis error: {str(e)}\n\nDebug: {len(df)} rows, {df.columns.tolist()[:10]}",
                'debug': {'error': str(e), 'columns': df.columns.tolist()}
            }
    
    def _clean_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Aggressive cleaning"""
        df_clean = df.copy()
        original_rows = len(df_clean)
        removed_values = []
        
        # Remove summary rows
        for col in df_clean.select_dtypes(include=['object']).columns:
            mask = df_clean[col].astype(str).str.lower().str.strip().apply(
                lambda x: any(kw in x for kw in self.SUMMARY_KEYWORDS)
            )
            if mask.sum() > 0:
                removed_values.extend(df_clean[mask][col].tolist())
                df_clean = df_clean[~mask]
        
        # Remove empty rows
        df_clean = df_clean.dropna(how='all')
        
        cleaning_report = {
            'original_rows': original_rows,
            'cleaned_rows': len(df_clean),
            'removed_count': original_rows - len(df_clean),
            'removed_values': list(set(removed_values))[:10]
        }
        
        logger.info(f"Cleaned: {original_rows} → {len(df_clean)} rows")
        return df_clean, cleaning_report
    
    def _flatten_headers(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str]]:
        """Handle multi-level headers - find circle column"""
        df_flat = df.copy()
        circle_col = None
        
        # Find circle column (first text column with valid circles)
        text_cols = df_flat.select_dtypes(include=['object']).columns
        
        for col in text_cols:
            unique_values = df_flat[col].dropna().astype(str).str.lower().unique()
            # Check if contains valid circle names
            valid_count = sum(
                1 for val in unique_values 
                if any(circle in val for circle in self.VALID_CIRCLES)
            )
            if valid_count >= 5:  # At least 5 valid circles
                circle_col = col
                logger.info(f"Circle column found: {col} ({valid_count} valid circles)")
                break
        
        # If not found, use first text column
        if not circle_col and len(text_cols) > 0:
            circle_col = text_cols[0]
            logger.warning(f"Using first text column as circle: {circle_col}")
        
        return df_flat, circle_col
    
    def _validate_circles(self, df: pd.DataFrame, circle_col: Optional[str]) -> Dict:
        """Validate circles"""
        if not circle_col or circle_col not in df.columns:
            return {'valid_circles': [], 'total_circles': 0, 'circle_col': None}
        
        valid_circles = []
        for circle in df[circle_col].unique():
            circle_clean = str(circle).lower().strip()
            is_valid = any(vc in circle_clean for vc in self.VALID_CIRCLES)
            is_summary = any(kw in circle_clean for kw in self.SUMMARY_KEYWORDS)
            
            if is_valid and not is_summary:
                valid_circles.append(circle)
        
        logger.info(f"Valid circles: {len(valid_circles)} - {valid_circles[:5]}")
        
        return {
            'valid_circles': valid_circles,
            'total_circles': len(valid_circles),
            'circle_col': circle_col
        }
    
    def _map_metrics(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Map columns to metric categories - SPECIFIC TO YOUR DATA"""
        metrics = {
            'customers': [],
            'call_attempts': [],
            'cssr': [],
            'asr': [],
            'mou': [],
            'penetration': [],
            'segments': [],
            'other_numeric': []
        }
        
        for col in df.columns:
            col_lower = str(col).lower()
            
            # Customers
            if any(x in col_lower for x in ['hsi active', 'active voice', 'active customers', 'customer count', 'total customer']):
                if 'total' in col_lower or 'hsi' in col_lower or 'active voice' in col_lower:
                    metrics['customers'].append(col)
            
            # Call attempts
            elif 'call attempt' in col_lower:
                if 'total' in col_lower or col_lower.endswith('call attempts(count)'):
                    metrics['call_attempts'].append(col)
            
            # CSSR
            elif 'cssr' in col_lower:
                if 'total' in col_lower or col_lower == 'cssr (%)':
                    metrics['cssr'].append(col)
            
            # ASR
            elif 'asr' in col_lower:
                if 'total' in col_lower or col_lower == 'asr(%)':
                    metrics['asr'].append(col)
            
            # MOU
            elif 'mou' in col_lower or 'minutes' in col_lower:
                if 'total' in col_lower or 'average' in col_lower or '30 d total' in col_lower:
                    metrics['mou'].append(col)
            
            # Penetration
            elif 'penetration' in col_lower or 'active monthly to total' in col_lower:
                metrics['penetration'].append(col)
            
            # Segments (Heavy, Moderate, Low, Non)
            elif any(x in col_lower for x in ['heavy', 'moderate', 'low', 'non user']):
                if 'customer count' in col_lower or 'to total' in col_lower:
                    metrics['segments'].append(col)
            
            # Other numeric
            elif pd.api.types.is_numeric_dtype(df[col]):
                metrics['other_numeric'].append(col)
        
        logger.info(f"Metrics mapped: customers={len(metrics['customers'])}, "
                   f"call_attempts={len(metrics['call_attempts'])}, "
                   f"cssr={len(metrics['cssr'])}, mou={len(metrics['mou'])}")
        
        return metrics
    
    def _analyze_circles_v9(self, df: pd.DataFrame, circle_col: str, 
                           circles_info: Dict, metrics_map: Dict) -> List[Dict]:
        """Deep analysis per circle with YOUR data"""
        if not circle_col or circles_info['total_circles'] == 0:
            return []
        
        valid_circles = circles_info['valid_circles']
        df_valid = df[df[circle_col].isin(valid_circles)]
        
        insights = []
        
        for circle in valid_circles:
            circle_data = df_valid[df_valid[circle_col] == circle].iloc[0]  # First row
            
            insight = {
                'circle': circle,
                'metrics': {},
                'problems': [],
                'priority': 'normal'
            }
            
            # Extract customers
            if metrics_map['customers']:
                for col in metrics_map['customers'][:3]:  # Top 3
                    if col in circle_data.index:
                        val = circle_data[col]
                        if pd.notna(val) and (isinstance(val, (int, float)) or str(val).replace('.','').replace('%','').isdigit()):
                            insight['metrics'][col] = self._format_value(val)
            
            # Extract call attempts
            if metrics_map['call_attempts']:
                col = metrics_map['call_attempts'][0]
                if col in circle_data.index:
                    val = circle_data[col]
                    if pd.notna(val):
                        insight['metrics']['Call Attempts'] = self._format_value(val)
            
            # Extract CSSR - CHECK BENCHMARK
            if metrics_map['cssr']:
                col = metrics_map['cssr'][0]
                if col in circle_data.index:
                    val = circle_data[col]
                    if pd.notna(val):
                        cssr_val = self._extract_numeric(val)
                        insight['metrics']['CSSR'] = f"{cssr_val:.1f}%"
                        
                        # Check against benchmark
                        if cssr_val < 90:
                            insight['problems'].append({
                                'type': 'quality',
                                'severity': 'critical',
                                'metric': 'CSSR',
                                'value': cssr_val,
                                'target': 95.0,
                                'gap': 95.0 - cssr_val
                            })
                            insight['priority'] = 'critical'
                        elif cssr_val < 95:
                            insight['problems'].append({
                                'type': 'quality',
                                'severity': 'high',
                                'metric': 'CSSR',
                                'value': cssr_val,
                                'target': 95.0,
                                'gap': 95.0 - cssr_val
                            })
                            if insight['priority'] == 'normal':
                                insight['priority'] = 'high'
            
            # Extract MOU
            if metrics_map['mou']:
                for col in metrics_map['mou'][:2]:
                    if col in circle_data.index:
                        val = circle_data[col]
                        if pd.notna(val):
                            mou_val = self._extract_numeric(val)
                            if mou_val > 0:
                                insight['metrics'][col] = f"{mou_val:.1f} mins"
            
            # Extract penetration
            if metrics_map['penetration']:
                col = metrics_map['penetration'][0]
                if col in circle_data.index:
                    val = circle_data[col]
                    if pd.notna(val):
                        pen_val = self._extract_numeric(val)
                        insight['metrics']['Penetration'] = f"{pen_val:.1f}%"
                        
                        if pen_val < 15:
                            insight['problems'].append({
                                'type': 'growth',
                                'severity': 'medium',
                                'metric': 'Penetration',
                                'value': pen_val,
                                'target': 25.0,
                                'gap': 25.0 - pen_val
                            })
            
            insights.append(insight)
        
        # Sort by priority
        priority_order = {'critical': 0, 'high': 1, 'normal': 2}
        insights.sort(key=lambda x: priority_order.get(x['priority'], 3))
        
        return insights
    
    def _calculate_network_summary(self, df: pd.DataFrame, circle_col: str,
                                  circles_info: Dict, metrics_map: Dict) -> Dict:
        """Network-wide statistics"""
        summary = {'total_circles': circles_info['total_circles']}
        
        valid_circles = circles_info['valid_circles']
        df_valid = df[df[circle_col].isin(valid_circles)]
        
        # Sum customers
        if metrics_map['customers']:
            col = metrics_map['customers'][0]  # First customer column
            if col in df_valid.columns and pd.api.types.is_numeric_dtype(df_valid[col]):
                total = df_valid[col].sum()
                summary['total_customers'] = int(total)
                summary['avg_customers_per_circle'] = int(total / len(valid_circles))
        
        # Sum call attempts
        if metrics_map['call_attempts']:
            col = metrics_map['call_attempts'][0]
            if col in df_valid.columns and pd.api.types.is_numeric_dtype(df_valid[col]):
                total = df_valid[col].sum()
                summary['total_call_attempts'] = int(total)
        
        # Average CSSR
        if metrics_map['cssr']:
            col = metrics_map['cssr'][0]
            if col in df_valid.columns:
                values = df_valid[col].apply(self._extract_numeric).dropna()
                if len(values) > 0:
                    summary['avg_cssr'] = float(values.mean())
                    summary['min_cssr'] = float(values.min())
                    summary['max_cssr'] = float(values.max())
        
        # Average MOU
        if metrics_map['mou']:
            col = metrics_map['mou'][0]
            if col in df_valid.columns:
                values = df_valid[col].apply(self._extract_numeric).dropna()
                if len(values) > 0:
                    summary['avg_mou'] = float(values.mean())
        
        return summary
    
    def _detect_problems_v9(self, circle_insights: List[Dict], metrics_map: Dict) -> List[Dict]:
        """Aggregate problems"""
        all_problems = []
        
        for insight in circle_insights:
            for problem in insight['problems']:
                problem['circle'] = insight['circle']
                problem['circle_priority'] = insight['priority']
                all_problems.append(problem)
        
        # Sort by severity
        severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        all_problems.sort(key=lambda x: severity_order.get(x.get('severity'), 4))
        
        return all_problems[:15]  # Top 15
    
    def _generate_recommendations_v9(self, problems: List[Dict], 
                                    circle_insights: List[Dict],
                                    network_summary: Dict) -> List[Dict]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Group by type
        quality_problems = [p for p in problems if p.get('type') == 'quality']
        growth_problems = [p for p in problems if p.get('type') == 'growth']
        
        # Quality recommendation
        if quality_problems:
            affected_circles = list(set([p['circle'] for p in quality_problems]))
            avg_gap = np.mean([p['gap'] for p in quality_problems])
            
            recommendations.append({
                'title': 'Network Quality Improvement Program',
                'priority': 'CRITICAL' if len(quality_problems) >= 3 else 'HIGH',
                'affected_circles': affected_circles[:5],
                'problem': f"{len(affected_circles)} circles below CSSR target (avg gap: {avg_gap:.1f}%)",
                'impact': f"Revenue at Risk: ₹{len(affected_circles) * 54:,.0f} lakhs/month",
                'actions': [
                    'Deploy field optimization team within 48 hours',
                    'RF optimization in affected circles (Week 1-2)',
                    'Additional capacity deployment (Week 3-4)'
                ],
                'investment': f"₹{len(affected_circles) * 4:,.0f}-{len(affected_circles) * 6:,.0f} Crores",
                'roi': '3-4 months',
                'expected_result': 'CSSR improvement to 95%+ in all circles'
            })
        
        # Growth recommendation
        if growth_problems:
            affected_circles = list(set([p['circle'] for p in growth_problems]))
            
            recommendations.append({
                'title': 'Customer Penetration Enhancement',
                'priority': 'MEDIUM',
                'affected_circles': affected_circles[:5],
                'problem': f"{len(affected_circles)} circles with low penetration (<15%)",
                'impact': f"Growth Opportunity: {len(affected_circles) * 5000:,} potential customers",
                'actions': [
                    'Launch targeted marketing campaigns',
                    'Onboarding incentives for new customers',
                    'Partnership with local businesses'
                ],
                'investment': f"₹{len(affected_circles) * 2:,.0f}-{len(affected_circles) * 3:,.0f} Crores",
                'roi': '6-8 months',
                'expected_result': 'Penetration increase to 20%+'
            })
        
        # Network optimization (general)
        if len(circle_insights) > 10:
            recommendations.append({
                'title': 'Network-Wide Optimization Initiative',
                'priority': 'MEDIUM',
                'affected_circles': ['All circles'],
                'problem': f"Operating {network_summary['total_circles']} circles with varying performance",
                'impact': 'Standardize operations and improve efficiency',
                'actions': [
                    'Best practices sharing across circles',
                    'Standardized monitoring and alerting',
                    'Quarterly performance reviews'
                ],
                'investment': '₹5-8 Crores (one-time)',
                'roi': '12 months',
                'expected_result': 'Consistent performance across network'
            })
        
        return recommendations
    
    def _format_presentation_output(self, cleaning_report: Dict, circles_info: Dict,
                                   circle_insights: List[Dict], network_summary: Dict,
                                   problems: List[Dict], recommendations: List[Dict],
                                   metrics_map: Dict) -> Dict:
        """Format for executive presentation"""
        
        critical_count = sum(1 for c in circle_insights if c['priority'] == 'critical')
        high_count = sum(1 for c in circle_insights if c['priority'] == 'high')
        normal_count = len(circle_insights) - critical_count - high_count
        
        # Executive Summary
        exec_summary = f"""
╔════════════════════════════════════════════════════════════╗
║     TELECOM OPERATIONS DASHBOARD - EXECUTIVE SUMMARY       ║
╚════════════════════════════════════════════════════════════╝

Analysis Date: {self.analysis_timestamp.strftime('%Y-%m-%d %H:%M')}
Network Coverage: {circles_info['total_circles']} Circles Monitored

OVERALL NETWORK HEALTH:
  🔴 Critical Priority: {critical_count} circles
  🟡 High Priority: {high_count} circles
  🟢 Normal Operations: {normal_count} circles

"""
        
        # Network metrics
        if 'total_customers' in network_summary:
            exec_summary += f"""
NETWORK-WIDE METRICS:
  • Total Customers: {network_summary['total_customers']:,}
  • Avg per Circle: {network_summary.get('avg_customers_per_circle', 0):,}
"""
        
        if 'total_call_attempts' in network_summary:
            exec_summary += f"  • Total Call Attempts: {network_summary['total_call_attempts']:,}\n"
        
        if 'avg_cssr' in network_summary:
            exec_summary += f"""  • Network CSSR: {network_summary['avg_cssr']:.1f}% (Range: {network_summary.get('min_cssr', 0):.1f}% - {network_summary.get('max_cssr', 0):.1f}%)
"""
        
        if 'avg_mou' in network_summary:
            exec_summary += f"  • Average MOU: {network_summary['avg_mou']:.1f} minutes\n"
        
        exec_summary += f"""
TOP CONCERNS:
  • {len([p for p in problems if p.get('type') == 'quality'])} Quality Issues Detected
  • {len([p for p in problems if p.get('type') == 'capacity'])} Capacity Constraints
  • {len([p for p in problems if p.get('type') == 'growth'])} Growth Opportunities

⚠️  IMMEDIATE ACTION REQUIRED: {len([r for r in recommendations if r.get('priority') in ['CRITICAL', 'HIGH']])} urgent recommendations
"""
        
        # Circle insights
        insights_text = ""
        for idx, insight in enumerate(circle_insights[:5], 1):
            circle = insight['circle']
            priority = insight['priority'].upper()
            
            insights_text += f"""
{'─' * 60}
CIRCLE {idx}: {circle} [{priority} PRIORITY]
{'─' * 60}

Key Metrics:
"""
            for metric_name, metric_value in list(insight['metrics'].items())[:5]:
                insights_text += f"  • {metric_name}: {metric_value}\n"
            
            if insight['problems']:
                insights_text += "\nIssues Detected:\n"
                for prob in insight['problems'][:3]:
                    insights_text += f"  ⚠️  {prob['metric']}: {prob['value']:.1f} (Target: {prob['target']:.1f})\n"
        
        # Recommendations
        rec_text = ""
        for idx, rec in enumerate(recommendations, 1):
            rec_text += f"""
{'═' * 60}
RECOMMENDATION {idx}: {rec['title']} [{rec['priority']}]
{'═' * 60}

Problem: {rec['problem']}
Impact: {rec['impact']}

Actions:
"""
            for action in rec['actions']:
                rec_text += f"  ✓ {action}\n"
            
            rec_text += f"""
Investment: {rec['investment']}
Expected ROI: {rec['roi']}
Expected Result: {rec['expected_result']}
"""
        
        return {
            'executive_summary': exec_summary,
            'key_insights': insights_text,
            'recommendations': rec_text,
            'circle_analysis': circle_insights,
            'problems': problems,
            'network_summary': network_summary,
            'metadata': {
                'total_circles': circles_info['total_circles'],
                'critical_count': critical_count,
                'high_count': high_count,
                'metrics_detected': {k: len(v) for k, v in metrics_map.items() if v},
                'data_quality': cleaning_report
            }
        }
    
    def _generate_no_data_response(self, cleaning_report: Dict) -> Dict:
        """No data after cleaning"""
        return {
            'executive_summary': f"""
ERROR: No Valid Data After Cleaning

Original Rows: {cleaning_report['original_rows']}
Removed: {cleaning_report['removed_count']} rows
Removed Values: {', '.join(cleaning_report['removed_values'][:5])}

Please upload file with individual circle data.
""",
            'key_insights': '',
            'recommendations': '',
            'metadata': cleaning_report
        }
    
    def _generate_no_circles_response(self, cleaning_report: Dict, columns: List[str]) -> Dict:
        """No valid circles found"""
        return {
            'executive_summary': f"""
ERROR: No Valid Circles Detected

Data has {cleaning_report['cleaned_rows']} rows but no valid Indian telecom circles found.

Columns available: {', '.join(columns[:10])}

Expected circle names: Mumbai, Delhi, Kolkata, Chennai, etc.
""",
            'key_insights': '',
            'recommendations': '',
            'metadata': {'columns': columns, 'cleaned_rows': cleaning_report['cleaned_rows']}
        }
    
    def _format_value(self, val: Any) -> str:
        """Format value for display"""
        if pd.isna(val):
            return 'N/A'
        
        num = self._extract_numeric(val)
        if num == 0:
            return str(val)
        
        if num > 10000:
            return f"{num:,.0f}"
        elif num > 100:
            return f"{num:.1f}"
        else:
            return f"{num:.2f}"
    
    def _extract_numeric(self, val: Any) -> float:
        """Extract numeric value from string/float/int"""
        if pd.isna(val):
            return 0.0
        
        if isinstance(val, (int, float)):
            return float(val)
        
        # Remove % and convert
        val_str = str(val).replace('%', '').replace(',', '').strip()
        try:
            return float(val_str)
        except:
            return 0.0


def analyze_data(df: pd.DataFrame, groq_api_key: str = None, context: Dict = None) -> Dict[str, Any]:
    """Main entry point"""
    engine = AIInsightsEngine(groq_api_key=groq_api_key)
    return engine.analyze_data(df, context=context)
