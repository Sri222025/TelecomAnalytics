import pandas as pd
import numpy as np
from typing import Dict, List, Any
import streamlit as st

class InsightsGenerator:
    """Generate natural language insights from data and anomalies - now with real AI"""
    
    def __init__(self, use_ai: bool = True):
        """
        Initialize insights generator
        
        Args:
            use_ai: Whether to use AI (Groq) for insights generation
        """
        self.use_ai = use_ai
        self.ai_engine = None
        
        if use_ai:
            try:
                from ai_insights_engine import AIInsightsEngine
                # Get API key from Streamlit secrets or environment
                api_key = st.secrets.get("GROQ_API_KEY") if hasattr(st, 'secrets') else None
                if api_key:
                    self.ai_engine = AIInsightsEngine(api_key=api_key)
                    st.info("✅ AI Insights Engine (Groq Llama 3.3) activated!")
                else:
                    st.warning("⚠️ Groq API key not found. Using rule-based insights. Add GROQ_API_KEY to Streamlit secrets for AI insights.")
                    self.use_ai = False
            except Exception as e:
                st.warning(f"⚠️ AI engine not available: {str(e)}. Using rule-based insights.")
                self.use_ai = False
    
    def generate_insights(self, df: pd.DataFrame, metrics: Dict, anomalies: List[Dict]) -> Dict[str, Any]:
        """
        Generate comprehensive insights from data analysis
        Uses AI if available, falls back to rule-based
        """
        if self.use_ai and self.ai_engine:
            return self._generate_ai_insights(df, metrics, anomalies)
        else:
            return self._generate_rule_based_insights(df, metrics, anomalies)
    
    def _generate_ai_insights(self, df: pd.DataFrame, metrics: Dict, anomalies: List[Dict]) -> Dict[str, Any]:
        """Generate insights using AI (Groq Llama 3.3)"""
        try:
            # Get comprehensive AI insights
            ai_insights = self.ai_engine.generate_comprehensive_insights(df, anomalies)
            
            # Parse AI response
            full_text = ai_insights.get('full_text', '')
            
            # Extract structured information
            insights = {
                'summary': self._extract_section(full_text, 'Executive Summary', 'Top'),
                'key_findings': self._extract_list_items(full_text, 'Critical Insights', 'Hidden'),
                'recommendations': self._extract_list_items(full_text, 'Recommendations', 'Risk'),
                'trends': self._extract_list_items(full_text, 'Hidden Patterns', 'Recommendations'),
                'risks': self._extract_list_items(full_text, 'Risk Areas', 'Opportunities'),
                'opportunities': self._extract_list_items(full_text, 'Opportunities', None),
                'priorities': self._prioritize_actions(anomalies),
                'full_ai_response': full_text,
                'ai_powered': True
            }
            
            return insights
            
        except Exception as e:
            st.error(f"AI insights generation failed: {str(e)}")
            return self._generate_rule_based_insights(df, metrics, anomalies)
    
    def _generate_rule_based_insights(self, df: pd.DataFrame, metrics: Dict, anomalies: List[Dict]) -> Dict[str, Any]:
        """Fallback: Generate insights using rules (original method)"""
        insights = {
            'summary': self._generate_executive_summary(df, metrics, anomalies),
            'key_findings': self._generate_key_findings(df, metrics),
            'recommendations': self._generate_recommendations(anomalies, metrics),
            'trends': self._identify_trends(df, metrics),
            'priorities': self._prioritize_actions(anomalies),
            'ai_powered': False
        }
        
        return insights
    
    def _extract_section(self, text: str, start_marker: str, end_marker: str = None) -> str:
        """Extract a section from AI response"""
        try:
            start_idx = text.find(start_marker)
            if start_idx == -1:
                return ""
            
            start_idx = text.find('\n', start_idx) + 1
            
            if end_marker:
                end_idx = text.find(end_marker, start_idx)
                if end_idx != -1:
                    section = text[start_idx:end_idx].strip()
                else:
                    section = text[start_idx:start_idx+500].strip()
            else:
                section = text[start_idx:].strip()
            
            # Clean up the section
            lines = section.split('\n')
            cleaned_lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith('#')]
            return ' '.join(cleaned_lines[:3])  # First 3 meaningful lines
            
        except:
            return ""
    
    def _extract_list_items(self, text: str, section_name: str, next_section: str = None) -> List[str]:
        """Extract list items from AI response"""
        try:
            start_idx = text.find(section_name)
            if start_idx == -1:
                return []
            
            # Find the end of this section
            if next_section:
                end_idx = text.find(next_section, start_idx)
                if end_idx == -1:
                    end_idx = len(text)
            else:
                end_idx = len(text)
            
            section_text = text[start_idx:end_idx]
            
            # Extract bullet points or numbered items
            items = []
            for line in section_text.split('\n'):
                line = line.strip()
                # Match numbered items (1., 2., etc.) or bullet points (-, *, •)
                if line and (line[0].isdigit() or line.startswith(('-', '*', '•', '▪'))):
                    # Remove numbering/bullets
                    clean_line = line.lstrip('0123456789.-*•▪ ').strip()
                    if len(clean_line) > 10:  # Meaningful content
                        items.append(clean_line)
            
            return items[:10]  # Max 10 items
            
        except:
            return []
    
    def _generate_executive_summary(self, df: pd.DataFrame, metrics: Dict, anomalies: List[Dict]) -> str:
        """Generate executive summary (rule-based fallback)"""
        
        basic_stats = metrics.get('basic_stats', {})
        subscriber_metrics = metrics.get('subscriber_metrics', {})
        
        total_records = basic_stats.get('total_records', 0)
        unique_subs = subscriber_metrics.get('unique_subscribers', 0)
        
        # Count critical issues
        critical_count = len([a for a in anomalies if a['severity'] == 'Critical'])
        warning_count = len([a for a in anomalies if a['severity'] == 'Warning'])
        
        summary = f"""Analysis of {total_records:,} records"""
        
        if unique_subs > 0:
            summary += f" across {unique_subs:,} unique subscribers"
        
        summary += " reveals "
        
        if critical_count > 0:
            summary += f"{critical_count} critical issue{'s' if critical_count > 1 else ''} "
        
        if warning_count > 0:
            if critical_count > 0:
                summary += "and "
            summary += f"{warning_count} warning{'s' if warning_count > 1 else ''} "
        
        if critical_count == 0 and warning_count == 0:
            summary += "no critical issues. "
        else:
            summary += "requiring attention. "
        
        # Add key metric
        usage_metrics = metrics.get('usage_metrics', {})
        if 'arpu' in usage_metrics:
            summary += f"Average Revenue Per User (ARPU) is ₹{usage_metrics['arpu']:.2f}. "
        
        if 'mou' in usage_metrics:
            summary += f"Minutes of Usage (MOU) per user is {usage_metrics['mou']:.0f} minutes. "
        
        # Device adoption
        device_metrics = metrics.get('device_metrics', {})
        if 'jiojoin_count' in device_metrics and 'pots_count' in device_metrics:
            total = device_metrics['jiojoin_count'] + device_metrics['pots_count']
            if total > 0:
                jiojoin_pct = (device_metrics['jiojoin_count'] / total) * 100
                summary += f"JioJoin app adoption is at {jiojoin_pct:.1f}%."
        
        return summary
    
    def _generate_key_findings(self, df: pd.DataFrame, metrics: Dict) -> List[str]:
        """Generate key findings from the data (rule-based fallback)"""
        findings = []
        
        basic_stats = metrics.get('basic_stats', {})
        subscriber_metrics = metrics.get('subscriber_metrics', {})
        usage_metrics = metrics.get('usage_metrics', {})
        device_metrics = metrics.get('device_metrics', {})
        regional_metrics = metrics.get('regional_metrics', {})
        
        # Data volume findings
        total_records = basic_stats.get('total_records', 0)
        findings.append(f"Dataset contains {total_records:,} records across {basic_stats.get('total_columns', 0)} data fields")
        
        # Subscriber findings
        if 'unique_subscribers' in subscriber_metrics:
            unique_subs = subscriber_metrics['unique_subscribers']
            avg_records = basic_stats.get('total_records', 0) / unique_subs if unique_subs > 0 else 0
            findings.append(f"{unique_subs:,} unique subscribers with an average of {avg_records:.1f} records per subscriber")
        
        # Connection type findings
        if 'connection_type_distribution' in subscriber_metrics:
            dist = subscriber_metrics['connection_type_distribution']
            if dist:
                top_type = max(dist, key=dist.get)
                top_pct = dist[top_type]
                findings.append(f"{top_type} is the dominant connection type at {top_pct:.1f}% of the subscriber base")
        
        # Usage findings
        if usage_metrics:
            usage_keys = [k for k in usage_metrics.keys() if '_total' in k]
            if usage_keys:
                for key in usage_keys[:2]:
                    metric_name = key.replace('_total', '').replace('_', ' ').title()
                    value = usage_metrics[key]
                    findings.append(f"Total {metric_name}: {value:,.0f}")
        
        # Device findings
        if 'device_distribution_pct' in device_metrics:
            dist = device_metrics['device_distribution_pct']
            if dist:
                top_device = max(dist, key=dist.get)
                findings.append(f"{top_device} is the most used device format at {dist[top_device]:.1f}% usage share")
        
        # Regional findings
        if 'total_regions' in regional_metrics:
            findings.append(f"Service spans {regional_metrics['total_regions']} geographic regions")
        
        # Data quality findings
        quality_metrics = metrics.get('quality_metrics', {})
        if 'consistency' in quality_metrics:
            dup_pct = quality_metrics['consistency'].get('duplicate_percentage', 0)
            if dup_pct > 0:
                findings.append(f"Data quality: {dup_pct:.1f}% duplicate records detected")
        
        # Zero usage findings
        for key, value in usage_metrics.items():
            if 'zero_usage_percentage' in key and value > 10:
                metric_name = key.replace('_zero_usage_percentage', '').replace('_', ' ').title()
                findings.append(f"{value:.1f}% of records show zero {metric_name}")
        
        return findings[:10]
    
    def _generate_recommendations(self, anomalies: List[Dict], metrics: Dict) -> List[str]:
        """Generate actionable recommendations (rule-based fallback)"""
        recommendations = []
        
        # From anomalies
        critical_anomalies = [a for a in anomalies if a['severity'] == 'Critical']
        for anomaly in critical_anomalies[:3]:
            if anomaly['recommendation']:
                recommendations.append(f"[URGENT] {anomaly['recommendation']}")
        
        # From metrics analysis
        usage_metrics = metrics.get('usage_metrics', {})
        device_metrics = metrics.get('device_metrics', {})
        
        # ARPU recommendations
        if 'arpu' in usage_metrics:
            arpu = usage_metrics['arpu']
            if arpu < 200:
                recommendations.append("Consider strategies to increase ARPU through value-added services or plan upgrades")
        
        # Device adoption recommendations
        if 'jiojoin_count' in device_metrics and 'pots_count' in device_metrics:
            total = device_metrics['jiojoin_count'] + device_metrics['pots_count']
            if total > 0:
                jiojoin_pct = (device_metrics['jiojoin_count'] / total) * 100
                if jiojoin_pct < 40:
                    recommendations.append("Launch targeted campaigns to drive JioJoin app adoption and reduce POTS dependency")
        
        # Zero usage recommendations
        for key, value in usage_metrics.items():
            if 'zero_usage_percentage' in key and value > 15:
                metric_name = key.replace('_zero_usage_percentage', '').replace('_', ' ')
                recommendations.append(f"Investigate and re-engage the {value:.0f}% inactive users with zero {metric_name}")
                break
        
        # Regional recommendations
        regional_metrics = metrics.get('regional_metrics', {})
        if 'bottom_5_regions' in regional_metrics:
            recommendations.append("Focus network expansion and marketing efforts in underperforming regions")
        
        # Data quality recommendations
        quality_metrics = metrics.get('quality_metrics', {})
        if 'consistency' in quality_metrics:
            dup_pct = quality_metrics['consistency'].get('duplicate_percentage', 0)
            if dup_pct > 5:
                recommendations.append("Implement data deduplication processes to improve data quality and accuracy")
        
        # General recommendations
        if len(recommendations) < 5:
            recommendations.append("Establish regular monitoring dashboards for key metrics and anomalies")
            recommendations.append("Implement automated alerts for critical threshold breaches")
        
        return recommendations[:8]
    
    def _identify_trends(self, df: pd.DataFrame, metrics: Dict) -> List[str]:
        """Identify and describe trends (rule-based fallback)"""
        trends = []
        
        temporal_metrics = metrics.get('temporal_metrics', {})
        
        for key, value in temporal_metrics.items():
            if 'dow_distribution' in key:
                if isinstance(value, dict) and value:
                    max_day = max(value, key=value.get)
                    trends.append(f"Peak activity occurs on {max_day}")
        
        usage_metrics = metrics.get('usage_metrics', {})
        for key in usage_metrics.keys():
            if '_p90' in key and '_p50' in key.replace('_p90', '_p50'):
                p90_val = usage_metrics[key]
                p50_key = key.replace('_p90', '_p50')
                p50_val = usage_metrics.get(p50_key, 0)
                
                if p50_val > 0:
                    ratio = p90_val / p50_val
                    if ratio > 3:
                        metric_name = key.replace('_p90', '').replace('_', ' ').title()
                        trends.append(f"{metric_name} shows high variance with top 10% users significantly outpacing median")
        
        return trends[:5]
    
    def _prioritize_actions(self, anomalies: List[Dict]) -> List[Dict]:
        """Prioritize actions based on severity and impact"""
        priorities = []
        
        critical = [a for a in anomalies if a['severity'] == 'Critical']
        for idx, anomaly in enumerate(critical[:3], 1):
            priorities.append({
                'priority': f'P{idx}',
                'severity': 'Critical',
                'action': anomaly['title'],
                'recommendation': anomaly['recommendation']
            })
        
        warnings = [a for a in anomalies if a['severity'] == 'Warning']
        for idx, anomaly in enumerate(warnings[:3], len(priorities)+1):
            priorities.append({
                'priority': f'P{idx}',
                'severity': 'Warning',
                'action': anomaly['title'],
                'recommendation': anomaly['recommendation']
            })
        
        return priorities
