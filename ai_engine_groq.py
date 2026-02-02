"""
Telecom & SaaS Data Analysis Engine with Groq Llama 3 AI Integration
Provides pattern detection, correlation analysis, and AI-powered domain-specific insights
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
from groq import Groq
import json
warnings.filterwarnings('ignore')


class TelecomSaaSAnalyzerWithAI:
    """
    Advanced analyzer for Telecom and SaaS data with Groq Llama 3 AI integration
    """
    
    def __init__(self, dataframes_dict, groq_api_key=None):
        """
        Initialize with dictionary of dataframes from Excel worksheets and Groq API key
        
        Args:
            dataframes_dict: Dict with worksheet names as keys and DataFrames as values
            groq_api_key: Groq API key for AI-powered insights
        """
        self.dataframes = dataframes_dict
        self.analysis_results = {}
        self.correlations = {}
        self.patterns = {}
        self.insights = {}
        self.ai_insights = {}
        
        # Initialize Groq client
        self.groq_client = None
        if groq_api_key:
            self.groq_client = Groq(api_key=groq_api_key)
    
    def detect_data_type(self, df):
        """
        Detect if data is Fiber/AirFiber, Voice, or JioJoin related
        """
        columns_lower = [col.lower() for col in df.columns]
        
        is_fiber = any(keyword in ' '.join(columns_lower) for keyword in 
                      ['fiber', 'airfiber', 'sub6', 'ubr', 'bandwidth', 'speed', 'latency', 'hsi'])
        
        is_voice = any(keyword in ' '.join(columns_lower) for keyword in 
                      ['voice', 'call', 'cssr', 'asr', 'mou', 'attempt', 'cst', 'acd', 'onnet', 'offnet'])
        
        is_jiojoin = any(keyword in ' '.join(columns_lower) for keyword in 
                        ['jiojoin', 'android', 'ios', 'stb', 'app', 'platform', 'device', 'video', 'stream'])
        
        is_circle = any(keyword in ' '.join(columns_lower) for keyword in 
                       ['circle', 'region', 'location', 'state', 'city', 'pan india'])
        
        is_penetration = any(keyword in ' '.join(columns_lower) for keyword in 
                           ['penetration', 'non user', 'low', 'moderate', 'heavy', 'active customer'])
        
        return {
            'is_fiber': is_fiber,
            'is_voice': is_voice,
            'is_jiojoin': is_jiojoin,
            'is_circle': is_circle,
            'is_penetration': is_penetration
        }
    
    def analyze_all(self):
        """
        Run comprehensive analysis on all worksheets
        """
        for sheet_name, df in self.dataframes.items():
            if df.empty:
                continue
                
            data_type = self.detect_data_type(df)
            
            # Basic statistics
            self.analysis_results[sheet_name] = {
                'data_type': data_type,
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'dtypes': df.dtypes.to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'numeric_summary': df.describe().to_dict()
            }
            
            # Correlation analysis
            numeric_df = df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) > 1:
                self.correlations[sheet_name] = numeric_df.corr()
            
            # Pattern detection
            self.patterns[sheet_name] = self._detect_patterns(df, data_type)
            
            # Domain-specific insights
            self.insights[sheet_name] = self._generate_insights(df, data_type, sheet_name)
            
            # AI-powered insights using Groq Llama 3
            if self.groq_client:
                self.ai_insights[sheet_name] = self._generate_ai_insights(df, data_type, sheet_name)
    
    def _detect_patterns(self, df, data_type):
        """
        Detect patterns in the data
        """
        patterns = {
            'growth_trends': {},
            'anomalies': {},
            'distributions': {},
            'clusters': {},
            'quality_metrics': {}
        }
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Growth trends
        for col in numeric_cols:
            if len(df) > 1:
                values = df[col].dropna()
                if len(values) > 1:
                    trend = np.polyfit(range(len(values)), values, 1)[0]
                    patterns['growth_trends'][col] = {
                        'slope': float(trend),
                        'direction': 'increasing' if trend > 0 else 'decreasing',
                        'magnitude': abs(float(trend))
                    }
        
        # Anomalies using IQR method
        for col in numeric_cols:
            values = df[col].dropna()
            if len(values) > 3:
                Q1 = values.quantile(0.25)
                Q3 = values.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                anomalies = values[(values < lower_bound) | (values > upper_bound)]
                if len(anomalies) > 0:
                    patterns['anomalies'][col] = {
                        'count': len(anomalies),
                        'percentage': (len(anomalies) / len(values)) * 100,
                        'values': anomalies.tolist()[:5]
                    }
        
        # Distribution analysis
        for col in numeric_cols:
            values = df[col].dropna()
            if len(values) > 3:
                skewness = stats.skew(values)
                kurtosis = stats.kurtosis(values)
                patterns['distributions'][col] = {
                    'skewness': float(skewness),
                    'kurtosis': float(kurtosis),
                    'mean': float(values.mean()),
                    'median': float(values.median()),
                    'std': float(values.std())
                }
        
        # Quality metrics for voice/call data
        if data_type['is_voice']:
            cssr_cols = [col for col in df.columns if 'cssr' in col.lower()]
            asr_cols = [col for col in df.columns if 'asr' in col.lower()]
            
            for col in cssr_cols + asr_cols:
                if col in numeric_cols:
                    values = df[col].dropna()
                    patterns['quality_metrics'][col] = {
                        'avg': float(values.mean()),
                        'min': float(values.min()),
                        'max': float(values.max()),
                        'std': float(values.std())
                    }
        
        return patterns
    
    def _generate_insights(self, df, data_type, sheet_name):
        """
        Generate domain-specific insights based on data type
        """
        insights = []
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return insights
        
        # Penetration insights
        if data_type['is_penetration']:
            insights.extend(self._penetration_insights(df, numeric_df))
        
        # Fiber/AirFiber specific insights
        if data_type['is_fiber']:
            insights.extend(self._fiber_insights(df, numeric_df))
        
        # Voice/Call specific insights
        if data_type['is_voice']:
            insights.extend(self._voice_insights(df, numeric_df))
        
        # JioJoin specific insights
        if data_type['is_jiojoin']:
            insights.extend(self._jiojoin_insights(df, numeric_df))
        
        # Circle-wise insights
        if data_type['is_circle']:
            insights.extend(self._circle_insights(df, numeric_df))
        
        # General insights
        insights.extend(self._general_insights(df, numeric_df))
        
        return insights
    
    def _penetration_insights(self, df, numeric_df):
        """
        Penetration metric specific insights
        """
        insights = []
        
        # Identify penetration columns
        penetration_cols = [col for col in df.columns if any(p in col.lower() for p in ['penetration', 'non user', 'low', 'moderate', 'heavy'])]
        
        if penetration_cols:
            for col in penetration_cols:
                if col in numeric_df.columns:
                    total = numeric_df[col].sum()
                    avg = numeric_df[col].mean()
                    insights.append({
                        'type': 'Penetration Metric',
                        'metric': col,
                        'value': f"{avg:.2f}%",
                        'insight': f"Average {col}: {avg:.2f}%. Total: {total:,.0f}. Monitor penetration trends for market expansion."
                    })
        
        return insights
    
    def _fiber_insights(self, df, numeric_df):
        """
        Fiber/AirFiber specific insights
        """
        insights = []
        
        # HSI/Bandwidth analysis
        hsi_cols = [col for col in df.columns if 'hsi' in col.lower()]
        bandwidth_cols = [col for col in df.columns if 'bandwidth' in col.lower() or 'speed' in col.lower()]
        
        for col in hsi_cols + bandwidth_cols:
            if col in numeric_df.columns:
                avg_val = numeric_df[col].mean()
                insights.append({
                    'type': 'Fiber Performance',
                    'metric': col,
                    'value': f"{avg_val:,.0f}",
                    'insight': f"Average {col}: {avg_val:,.0f}. Optimize broadband delivery and customer experience."
                })
        
        # Latency analysis
        latency_cols = [col for col in df.columns if 'latency' in col.lower()]
        for col in latency_cols:
            if col in numeric_df.columns:
                avg_latency = numeric_df[col].mean()
                insights.append({
                    'type': 'Network Quality',
                    'metric': col,
                    'value': f"{avg_latency:.2f}ms",
                    'insight': f"Average latency: {avg_latency:.2f}ms. {'Excellent' if avg_latency < 50 else 'Needs improvement' if avg_latency > 100 else 'Good'} performance."
                })
        
        return insights
    
    def _voice_insights(self, df, numeric_df):
        """
        Voice/Call quality specific insights
        """
        insights = []
        
        # CSSR (Call Setup Success Rate) analysis
        cssr_cols = [col for col in df.columns if 'cssr' in col.lower()]
        for col in cssr_cols:
            if col in numeric_df.columns:
                avg_cssr = numeric_df[col].mean()
                insights.append({
                    'type': 'Call Quality - CSSR',
                    'metric': col,
                    'value': f"{avg_cssr:.2f}%",
                    'insight': f"CSSR: {avg_cssr:.2f}%. {'Excellent' if avg_cssr > 99 else 'Good' if avg_cssr > 98 else 'Needs improvement'}. Target: >99%"
                })
        
        # ASR (Answer Seizure Ratio) analysis
        asr_cols = [col for col in df.columns if 'asr' in col.lower()]
        for col in asr_cols:
            if col in numeric_df.columns:
                avg_asr = numeric_df[col].mean()
                insights.append({
                    'type': 'Call Quality - ASR',
                    'metric': col,
                    'value': f"{avg_asr:.2f}%",
                    'insight': f"ASR: {avg_asr:.2f}%. Network efficiency indicator. Higher is better."
                })
        
        # MoU (Minutes of Usage) analysis
        mou_cols = [col for col in df.columns if 'mou' in col.lower()]
        for col in mou_cols:
            if col in numeric_df.columns:
                total_mou = numeric_df[col].sum()
                avg_mou = numeric_df[col].mean()
                insights.append({
                    'type': 'Voice Usage',
                    'metric': col,
                    'value': f"{avg_mou:.2f} mins",
                    'insight': f"Average MoU: {avg_mou:.2f} minutes. Total: {total_mou:,.0f}. Track usage patterns."
                })
        
        # Call Attempts analysis
        attempt_cols = [col for col in df.columns if 'attempt' in col.lower()]
        for col in attempt_cols:
            if col in numeric_df.columns:
                avg_attempts = numeric_df[col].mean()
                insights.append({
                    'type': 'Call Volume',
                    'metric': col,
                    'value': f"{avg_attempts:,.0f}",
                    'insight': f"Average daily attempts: {avg_attempts:,.0f}. Monitor network capacity."
                })
        
        return insights
    
    def _jiojoin_insights(self, df, numeric_df):
        """
        JioJoin app specific insights
        """
        insights = []
        
        # Platform analysis
        platform_cols = [col for col in df.columns if any(p in col.lower() for p in ['android', 'ios', 'stb'])]
        if platform_cols:
            for col in platform_cols:
                if col in numeric_df.columns:
                    total_users = numeric_df[col].sum()
                    insights.append({
                        'type': 'Platform Usage',
                        'metric': col,
                        'value': f"{total_users:,.0f}",
                        'insight': f"{col}: {total_users:,.0f} users. Analyze platform-specific trends."
                    })
        
        # Video streaming analysis
        video_cols = [col for col in df.columns if 'video' in col.lower() or 'stream' in col.lower()]
        for col in video_cols:
            if col in numeric_df.columns:
                total_video = numeric_df[col].sum()
                insights.append({
                    'type': 'Video Engagement',
                    'metric': col,
                    'value': f"{total_video:,.0f}",
                    'insight': f"Total {col}: {total_video:,.0f}. Monitor video consumption trends."
                })
        
        # Active users analysis
        active_cols = [col for col in df.columns if 'active' in col.lower() or 'dau' in col.lower() or 'mau' in col.lower()]
        if active_cols:
            for col in active_cols:
                if col in numeric_df.columns:
                    active_count = numeric_df[col].sum()
                    insights.append({
                        'type': 'User Engagement',
                        'metric': col,
                        'value': f"{active_count:,.0f}",
                        'insight': f"Total {col}: {active_count:,.0f}. Track engagement trends."
                    })
        
        return insights
    
    def _circle_insights(self, df, numeric_df):
        """
        Circle/region-wise insights
        """
        insights = []
        
        circle_cols = [col for col in df.columns if any(c in col.lower() for c in ['circle', 'region', 'location', 'state', 'city'])]
        
        if circle_cols:
            for circle_col in circle_cols:
                if circle_col in df.columns:
                    unique_circles = df[circle_col].nunique()
                    insights.append({
                        'type': 'Geographic Distribution',
                        'metric': circle_col,
                        'value': f"{unique_circles} regions",
                        'insight': f"Data spans {unique_circles} different {circle_col.lower()}. Analyze regional variations."
                    })
        
        return insights
    
    def _general_insights(self, df, numeric_df):
        """
        General data insights
        """
        insights = []
        
        # Data completeness
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        completeness = ((total_cells - missing_cells) / total_cells) * 100
        
        insights.append({
            'type': 'Data Quality',
            'metric': 'Completeness',
            'value': f"{completeness:.1f}%",
            'insight': f"Dataset is {completeness:.1f}% complete. {'Good data quality' if completeness > 90 else 'Consider data cleaning'}"
        })
        
        # Top correlations
        if len(numeric_df.columns) > 1:
            corr_matrix = numeric_df.corr()
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_pairs.append({
                        'col1': corr_matrix.columns[i],
                        'col2': corr_matrix.columns[j],
                        'corr': abs(corr_matrix.iloc[i, j])
                    })
            
            if corr_pairs:
                top_corr = sorted(corr_pairs, key=lambda x: x['corr'], reverse=True)[0]
                insights.append({
                    'type': 'Correlation',
                    'metric': f"{top_corr['col1']} vs {top_corr['col2']}",
                    'value': f"{top_corr['corr']:.3f}",
                    'insight': f"Strong correlation detected. Investigate relationship."
                })
        
        return insights
    
    def _generate_ai_insights(self, df, data_type, sheet_name):
        """
        Generate AI-powered insights using Groq Llama 3
        """
        if not self.groq_client:
            return []
        
        try:
            # Prepare data summary for AI analysis
            numeric_df = df.select_dtypes(include=[np.number])
            summary_stats = numeric_df.describe().to_dict()
            
            # Create a concise data summary
            data_summary = f"""
            Sheet: {sheet_name}
            Shape: {df.shape[0]} rows, {df.shape[1]} columns
            Columns: {', '.join(df.columns.tolist()[:10])}
            Data Types: {data_type}
            
            Key Statistics:
            {json.dumps(summary_stats, indent=2, default=str)[:1000]}
            """
            
            # Create prompt for Llama 3
            prompt = f"""You are a telecom and SaaS data analysis expert. Analyze the following telecom data and provide 3-4 key actionable insights.

Data Summary:
{data_summary}

Provide insights in this JSON format:
{{
    "insights": [
        {{
            "title": "Insight Title",
            "description": "Detailed insight",
            "recommendation": "Actionable recommendation",
            "priority": "High/Medium/Low"
        }}
    ]
}}

Focus on:
1. Performance metrics and KPIs
2. Anomalies or concerning trends
3. Opportunities for improvement
4. Comparative analysis if multiple circles/regions exist

Provide concise, actionable insights."""
            
            # Call Groq API
            message = self.groq_client.messages.create(
                model="llama-3.1-70b-versatile",
                max_tokens=1024,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Parse response
            response_text = message.content[0].text
            
            # Try to extract JSON from response
            try:
                # Find JSON in response
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                if start_idx != -1 and end_idx > start_idx:
                    json_str = response_text[start_idx:end_idx]
                    ai_data = json.loads(json_str)
                    return ai_data.get('insights', [])
            except json.JSONDecodeError:
                # If JSON parsing fails, return the raw response as insights
                return [{
                    'title': 'AI Analysis',
                    'description': response_text[:500],
                    'recommendation': 'Review full analysis',
                    'priority': 'Medium'
                }]
        
        except Exception as e:
            return [{
                'title': 'AI Analysis Error',
                'description': f"Error generating AI insights: {str(e)}",
                'recommendation': 'Check API key and connection',
                'priority': 'Low'
            }]
    
    def get_correlation_matrix(self, sheet_name):
        """
        Get correlation matrix for a specific sheet
        """
        return self.correlations.get(sheet_name, pd.DataFrame())
    
    def get_top_correlations(self, sheet_name, top_n=10):
        """
        Get top N correlations for a sheet
        """
        corr_matrix = self.get_correlation_matrix(sheet_name)
        if corr_matrix.empty:
            return pd.DataFrame()
        
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_pairs.append({
                    'Variable 1': corr_matrix.columns[i],
                    'Variable 2': corr_matrix.columns[j],
                    'Correlation': corr_matrix.iloc[i, j]
                })
        
        df_corr = pd.DataFrame(corr_pairs)
        df_corr['Abs_Correlation'] = df_corr['Correlation'].abs()
        return df_corr.nlargest(top_n, 'Abs_Correlation')[['Variable 1', 'Variable 2', 'Correlation']]
    
    def get_summary_report(self):
        """
        Get a summary report of all analysis
        """
        return {
            'analysis_results': self.analysis_results,
            'correlations': self.correlations,
            'patterns': self.patterns,
            'insights': self.insights,
            'ai_insights': self.ai_insights
        }
