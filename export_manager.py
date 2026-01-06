import pandas as pd
import io
from typing import Dict, Any
from datetime import datetime

class ExportManager:
    """Handle data export in various formats"""
    
    def export_to_csv(self, df: pd.DataFrame, filename: str = None) -> bytes:
        """Export dataframe to CSV"""
        if filename is None:
            filename = f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        return df.to_csv(index=False).encode('utf-8')
    
    def export_to_excel(self, data_dict: Dict[str, pd.DataFrame], filename: str = None) -> bytes:
        """
        Export multiple dataframes to Excel with different sheets
        data_dict: {'sheet_name': dataframe}
        """
        if filename is None:
            filename = f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            for sheet_name, df in data_dict.items():
                # Truncate sheet name to 31 characters (Excel limit)
                sheet_name = sheet_name[:31]
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        return buffer.getvalue()
    
    def export_insights_to_text(self, insights: Dict, anomalies: list) -> str:
        """Export insights and anomalies as formatted text"""
        report = f"""
TELECOM ANALYTICS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}

EXECUTIVE SUMMARY
{'='*80}
{insights.get('summary', 'No summary available')}

{'='*80}
KEY FINDINGS
{'='*80}
"""
        
        for idx, finding in enumerate(insights.get('key_findings', []), 1):
            report += f"\n{idx}. {finding}"
        
        report += f"""

{'='*80}
RECOMMENDATIONS
{'='*80}
"""
        
        for idx, rec in enumerate(insights.get('recommendations', []), 1):
            report += f"\n{idx}. {rec}"
        
        if anomalies:
            report += f"""

{'='*80}
ALERTS & ANOMALIES
{'='*80}
"""
            
            for anomaly in anomalies:
                report += f"""
[{anomaly['severity'].upper()}] {anomaly['title']}
Description: {anomaly['description']}
Recommendation: {anomaly['recommendation']}
{'-'*80}
"""
        
        return report
    
    def create_summary_dataframe(self, metrics: Dict) -> pd.DataFrame:
        """Create a summary dataframe from metrics"""
        summary_data = []
        
        for category, values in metrics.items():
            if isinstance(values, dict):
                for key, value in values.items():
                    summary_data.append({
                        'Category': category,
                        'Metric': key,
                        'Value': str(value)
                    })
        
        return pd.DataFrame(summary_data)
