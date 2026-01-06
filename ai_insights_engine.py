"""
AI-Powered Insights Engine using Groq Llama 3.3
Generates intelligent data insights, auto-slicing, and recommendations
"""

import pandas as pd
import json
from typing import Dict, List, Any
import os

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

class AIInsightsEngine:
    """Generate AI-powered insights using Groq Llama 3.3"""
    
    def __init__(self, api_key: str = None):
        """
        Initialize AI engine with Groq API key
        
        Args:
            api_key: Groq API key (or set GROQ_API_KEY env variable)
        """
        if not GROQ_AVAILABLE:
            raise ImportError("Groq library not installed. Run: pip install groq")
        
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Groq API key required. Set GROQ_API_KEY environment variable or pass api_key parameter")
        
        self.client = Groq(api_key=self.api_key)
        self.model = "llama-3.3-70b-versatile"  # Groq's fastest Llama 3.3 model
    
    def generate_comprehensive_insights(self, df: pd.DataFrame, anomalies: List[Dict] = None) -> Dict[str, Any]:
        """
        Generate comprehensive AI insights from the dataset
        
        Args:
            df: Input dataframe
            anomalies: Optional list of detected anomalies
            
        Returns:
            Dictionary containing AI-generated insights
        """
        # Prepare data summary for AI
        data_summary = self._prepare_data_summary(df)
        
        # Generate insights using AI
        prompt = self._build_comprehensive_prompt(data_summary, anomalies)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert telecom data analyst specializing in fixed-line services, device analytics, and customer behavior. Provide actionable, business-focused insights."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            insights_text = response.choices[0].message.content
            
            # Parse and structure the insights
            return self._parse_ai_response(insights_text)
            
        except Exception as e:
            return {
                'error': str(e),
                'fallback': self._generate_fallback_insights(df)
            }
    
    def auto_discover_interesting_slices(self, df: pd.DataFrame, top_n: int = 5) -> List[Dict]:
        """
        Use AI to automatically discover interesting data slices
        
        Args:
            df: Input dataframe
            top_n: Number of top insights to return
            
        Returns:
            List of interesting data slices with AI explanations
        """
        # First, compute statistical slices
        statistical_slices = self._compute_statistical_slices(df)
        
        # Then ask AI to interpret and rank them
        prompt = f"""
        Analyze these data patterns and identify the TOP {top_n} most business-critical insights:
        
        {json.dumps(statistical_slices[:20], indent=2)}
        
        For each insight, provide:
        1. Business impact (High/Medium/Low)
        2. Clear explanation in one sentence
        3. Recommended action
        4. Potential root cause
        
        Return as JSON array with keys: rank, dimension, segment, metric, impact, explanation, action, root_cause
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a telecom business analyst. Focus on actionable insights."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=1500
            )
            
            ai_insights = response.choices[0].message.content
            
            # Try to parse JSON from response
            try:
                # Extract JSON if wrapped in markdown
                if "```json" in ai_insights:
                    ai_insights = ai_insights.split("```json")[1].split("```")[0]
                elif "```" in ai_insights:
                    ai_insights = ai_insights.split("```")[1].split("```")[0]
                
                parsed_insights = json.loads(ai_insights)
                return parsed_insights[:top_n]
            except:
                # If JSON parsing fails, return structured format from text
                return self._extract_insights_from_text(ai_insights, statistical_slices[:top_n])
                
        except Exception as e:
            # Fallback to statistical slices with simple descriptions
            return statistical_slices[:top_n]
    
    def explain_anomaly(self, anomaly: Dict, df: pd.DataFrame) -> str:
        """
        Use AI to explain an anomaly in business terms
        
        Args:
            anomaly: Anomaly dictionary
            df: Source dataframe for context
            
        Returns:
            AI-generated explanation
        """
        context = self._get_anomaly_context(anomaly, df)
        
        prompt = f"""
        Explain this data anomaly in simple business terms:
        
        Anomaly: {anomaly['title']}
        Details: {anomaly['description']}
        Context: {context}
        
        Provide:
        1. What this means for the business (2-3 sentences)
        2. Likely root causes (2-3 possibilities)
        3. Recommended immediate action
        4. Long-term prevention strategy
        
        Keep it concise and actionable.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a telecom operations expert. Explain issues clearly to non-technical stakeholders."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.6,
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return anomaly.get('recommendation', 'Unable to generate AI explanation.')
    
    def chat_with_data(self, question: str, df: pd.DataFrame, conversation_history: List = None) -> str:
        """
        Conversational interface - ask questions about your data
        
        Args:
            question: User's question
            df: Dataframe to analyze
            conversation_history: Previous conversation for context
            
        Returns:
            AI response
        """
        data_context = self._prepare_data_summary(df, detailed=False)
        
        messages = [
            {
                "role": "system",
                "content": f"""You are a telecom data analyst assistant. You have access to this dataset:
                
{data_context}

Answer questions about the data with specific numbers and insights. If you need to calculate something, explain your reasoning."""
            }
        ]
        
        # Add conversation history
        if conversation_history:
            messages.extend(conversation_history)
        
        # Add current question
        messages.append({"role": "user", "content": question})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=800
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _prepare_data_summary(self, df: pd.DataFrame, detailed: bool = True) -> str:
        """Prepare concise data summary for AI"""
        summary = []
        
        # Basic stats
        summary.append(f"Dataset: {len(df):,} records, {len(df.columns)} columns")
        
        # Column types
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        summary.append(f"Categorical columns: {', '.join(categorical_cols[:10])}")
        summary.append(f"Numeric columns: {', '.join(numeric_cols[:10])}")
        
        if detailed:
            # Key metrics
            for col in numeric_cols[:5]:
                summary.append(f"{col}: mean={df[col].mean():.2f}, median={df[col].median():.2f}, std={df[col].std():.2f}")
            
            # Top categories
            for col in categorical_cols[:3]:
                top_values = df[col].value_counts().head(5)
                summary.append(f"{col} top values: {dict(top_values)}")
            
            # Missing data
            missing = df.isnull().sum()
            missing = missing[missing > 0]
            if not missing.empty:
                summary.append(f"Missing data: {dict(missing)}")
            
            # Sample data
            summary.append(f"\nSample records:\n{df.head(3).to_dict('records')}")
        
        return "\n".join(summary)
    
    def _build_comprehensive_prompt(self, data_summary: str, anomalies: List[Dict] = None) -> str:
        """Build comprehensive analysis prompt"""
        prompt = f"""
Analyze this telecom dataset and provide comprehensive insights:

{data_summary}

{"Detected Anomalies: " + json.dumps(anomalies, indent=2) if anomalies else ""}

Provide:

1. **Executive Summary** (3-4 sentences): Key takeaways for leadership

2. **Top 5 Critical Insights**: Most important findings with business impact

3. **Hidden Patterns**: 3 non-obvious patterns that could be valuable

4. **Recommendations**: 5 specific, actionable recommendations prioritized by impact

5. **Risk Areas**: Any concerning trends or issues that need immediate attention

6. **Opportunities**: Business opportunities revealed by the data

Format your response clearly with headers and bullet points.
"""
        return prompt
    
    def _compute_statistical_slices(self, df: pd.DataFrame) -> List[Dict]:
        """Compute statistically interesting data slices"""
        insights = []
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        # Limit analysis to prevent timeout
        categorical_cols = categorical_cols[:5]
        numeric_cols = numeric_cols[:10]
        
        for cat in categorical_cols:
            # Skip if too many unique values
            if df[cat].nunique() > 50:
                continue
                
            for num in numeric_cols:
                try:
                    # Group statistics
                    groups = df.groupby(cat)[num].agg(['mean', 'median', 'std', 'count'])
                    overall_mean = df[num].mean()
                    overall_median = df[num].median()
                    
                    for segment, stats in groups.iterrows():
                        if stats['count'] < 10:  # Skip small segments
                            continue
                        
                        # Calculate deviation from overall
                        mean_deviation = abs(stats['mean'] - overall_mean) / overall_mean if overall_mean != 0 else 0
                        
                        # Only keep significant deviations (>20%)
                        if mean_deviation > 0.2:
                            insights.append({
                                'dimension': cat,
                                'segment': str(segment),
                                'metric': num,
                                'segment_value': round(stats['mean'], 2),
                                'overall_value': round(overall_mean, 2),
                                'deviation_pct': round(mean_deviation * 100, 1),
                                'count': int(stats['count']),
                                'direction': 'higher' if stats['mean'] > overall_mean else 'lower'
                            })
                except:
                    continue
        
        # Sort by deviation magnitude
        insights.sort(key=lambda x: x['deviation_pct'], reverse=True)
        
        return insights
    
    def _parse_ai_response(self, response_text: str) -> Dict[str, Any]:
        """Parse AI response into structured format"""
        return {
            'full_text': response_text,
            'generated_at': pd.Timestamp.now().isoformat(),
            'model': self.model
        }
    
    def _extract_insights_from_text(self, text: str, statistical_slices: List[Dict]) -> List[Dict]:
        """Extract structured insights from AI text response"""
        # Fallback: combine AI text with statistical slices
        insights = []
        for i, slice_data in enumerate(statistical_slices, 1):
            insights.append({
                'rank': i,
                'dimension': slice_data['dimension'],
                'segment': slice_data['segment'],
                'metric': slice_data['metric'],
                'impact': 'High' if slice_data['deviation_pct'] > 50 else 'Medium',
                'explanation': f"{slice_data['segment']} shows {slice_data['deviation_pct']:.0f}% {slice_data['direction']} {slice_data['metric']}",
                'action': 'Investigate further',
                'root_cause': 'To be determined'
            })
        return insights
    
    def _get_anomaly_context(self, anomaly: Dict, df: pd.DataFrame) -> str:
        """Get relevant context for an anomaly"""
        context = []
        context.append(f"Dataset size: {len(df):,} records")
        context.append(f"Severity: {anomaly.get('severity', 'Unknown')}")
        return "; ".join(context)
    
    def _generate_fallback_insights(self, df: pd.DataFrame) -> Dict:
        """Generate basic insights if AI fails"""
        return {
            'summary': f"Dataset contains {len(df):,} records with {len(df.columns)} columns.",
            'key_findings': [
                f"Total records: {len(df):,}",
                f"Columns: {len(df.columns)}",
                f"Missing values: {df.isnull().sum().sum():,}"
            ],
            'recommendations': [
                "Review data quality",
                "Analyze key metrics",
                "Investigate anomalies"
            ]
        }
