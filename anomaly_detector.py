import pandas as pd
import numpy as np
from typing import Dict, List, Any
from scipy import stats

class AnomalyDetector:
    """AI-powered anomaly detection for telecom data"""
    
    def __init__(self):
        self.anomalies = []
        self.thresholds = {
            'churn_spike': 0.2,  # 20% increase
            'usage_drop': 0.3,   # 30% decrease
            'zero_usage_threshold': 0.15,  # 15% zero usage
            'outlier_zscore': 3,  # 3 standard deviations
            'missing_data_critical': 20,  # 20% missing data
        }
    
    def detect_anomalies(self, df: pd.DataFrame, metrics: Dict) -> List[Dict[str, Any]]:
        """
        Main anomaly detection function
        Returns list of detected anomalies with severity levels
        """
        self.anomalies = []
        
        # Run various anomaly checks
        self._detect_data_quality_issues(df, metrics)
        self._detect_usage_anomalies(df, metrics)
        self._detect_subscriber_anomalies(df, metrics)
        self._detect_regional_anomalies(df, metrics)
        self._detect_device_anomalies(df, metrics)
        self._detect_statistical_outliers(df)
        self._detect_temporal_anomalies(df)
        
        # Sort by severity
        severity_order = {'Critical': 0, 'Warning': 1, 'Info': 2}
        self.anomalies.sort(key=lambda x: severity_order.get(x['severity'], 3))
        
        return self.anomalies
    
    def _add_anomaly(self, title: str, description: str, severity: str,
                    impact: str = "", recommendation: str = ""):
        """Add an anomaly to the list"""
        self.anomalies.append({
            'title': title,
            'description': description,
            'severity': severity,
            'impact': impact,
            'recommendation': recommendation
        })
    
    def _detect_data_quality_issues(self, df: pd.DataFrame, metrics: Dict):
        """Detect data quality problems"""
        quality_metrics = metrics.get('quality_metrics', {})
        
        # Check for high missing values
        if 'completeness' in quality_metrics:
            for col, info in quality_metrics['completeness'].items():
                if info['status'] == 'critical':
                    self._add_anomaly(
                        f"Critical Data Missing: {col}",
                        f"Column '{col}' has {info['missing_percentage']:.1f}% missing values ({info['missing_count']} records).",
                        'Critical',
                        "Missing data may lead to incorrect analysis and insights.",
                        f"Investigate the source of missing data for '{col}'. Consider data validation at collection point."
                    )
                elif info['status'] == 'warning':
                    self._add_anomaly(
                        f"Data Quality Warning: {col}",
                        f"Column '{col}' has {info['missing_percentage']:.1f}% missing values.",
                        'Warning',
                        "May affect accuracy of specific analyses involving this field.",
                        f"Monitor data collection process for '{col}'."
                    )
        
        # Check for duplicates
        if 'consistency' in quality_metrics:
            dup_pct = quality_metrics['consistency'].get('duplicate_percentage', 0)
            if dup_pct > 5:
                self._add_anomaly(
                    "Duplicate Records Detected",
                    f"Found {dup_pct:.1f}% duplicate records in the dataset.",
                    'Warning',
                    "Duplicates can skew metrics and lead to double-counting.",
                    "Review data ingestion process. Consider deduplication before analysis."
                )
    
    def _detect_usage_anomalies(self, df: pd.DataFrame, metrics: Dict):
        """Detect unusual usage patterns"""
        usage_metrics = metrics.get('usage_metrics', {})
        
        # Check for zero usage
        for key, value in usage_metrics.items():
            if 'zero_usage_percentage' in key:
                if value > self.thresholds['zero_usage_threshold'] * 100:
                    col_name = key.replace('_zero_usage_percentage', '')
                    self._add_anomaly(
                        f"High Zero Usage: {col_name}",
                        f"{value:.1f}% of records show zero usage for {col_name}.",
                        'Warning',
                        "High proportion of inactive users may indicate service issues or data collection problems.",
                        "Investigate reasons for zero usage. Check if users are inactive or if there are technical issues."
                    )
        
        # Check for extreme values
        for key, value in usage_metrics.items():
            if '_max' in key and '_min' not in key:
                avg_key = key.replace('_max', '_average')
                if avg_key in usage_metrics:
                    max_val = value
                    avg_val = usage_metrics[avg_key]
                    if max_val > avg_val * 100:  # Max is 100x the average
                        col_name = key.replace('_max', '')
                        self._add_anomaly(
                            f"Extreme Values Detected: {col_name}",
                            f"Maximum value ({max_val:.0f}) is {max_val/avg_val:.0f}x the average ({avg_val:.2f}).",
                            'Info',
                            "May indicate data entry errors or extraordinary usage patterns.",
                            f"Review records with extreme {col_name} values for accuracy."
                        )
    
    def _detect_subscriber_anomalies(self, df: pd.DataFrame, metrics: Dict):
        """Detect subscriber-related anomalies"""
        subscriber_metrics = metrics.get('subscriber_metrics', {})
        
        # Check connection type imbalance
        if 'connection_type_distribution' in subscriber_metrics:
            distribution = subscriber_metrics['connection_type_distribution']
            if distribution:
                max_pct = max(distribution.values())
                if max_pct > 80:
                    dominant_type = max(distribution, key=distribution.get)
                    self._add_anomaly(
                        "Connection Type Imbalance",
                        f"{dominant_type} accounts for {max_pct:.1f}% of all connections.",
                        'Info',
                        "Highly concentrated connection type may limit service diversity.",
                        "Consider strategies to diversify connection types or ensure this concentration is intentional."
                    )
    
    def _detect_regional_anomalies(self, df: pd.DataFrame, metrics: Dict):
        """Detect regional performance anomalies"""
        regional_metrics = metrics.get('regional_metrics', {})
        
        if 'regional_distribution' in regional_metrics:
            distribution = regional_metrics['regional_distribution']
            
            if len(distribution) > 1:
                values = list(distribution.values())
                mean_val = np.mean(values)
                std_val = np.std(values)
                
                # Find regions significantly below average
                underperforming = []
                for region, count in distribution.items():
                    if count < mean_val - 2 * std_val:
                        underperforming.append((region, count, mean_val))
                
                if underperforming:
                    regions_str = ", ".join([f"{r[0]} ({r[1]})" for r in underperforming[:3]])
                    self._add_anomaly(
                        "Underperforming Regions Detected",
                        f"Regions significantly below average: {regions_str}",
                        'Warning',
                        "Low subscriber counts in specific regions may indicate market penetration issues.",
                        "Focus on marketing and network expansion in underperforming regions."
                    )
    
    def _detect_device_anomalies(self, df: pd.DataFrame, metrics: Dict):
        """Detect device usage anomalies"""
        device_metrics = metrics.get('device_metrics', {})
        
        # Check for low JioJoin adoption
        if 'jiojoin_count' in device_metrics and 'pots_count' in device_metrics:
            jiojoin = device_metrics['jiojoin_count']
            pots = device_metrics['pots_count']
            total = jiojoin + pots
            
            if total > 0:
                jiojoin_pct = (jiojoin / total) * 100
                
                if jiojoin_pct < 30:
                    self._add_anomaly(
                        "Low JioJoin App Adoption",
                        f"Only {jiojoin_pct:.1f}% of users are using JioJoin app. POTS usage: {100-jiojoin_pct:.1f}%.",
                        'Warning',
                        "Low app adoption may indicate user experience issues or lack of awareness.",
                        "Launch user education campaigns. Investigate app usability and feature awareness."
                    )
        
        # Check for device distribution
        if 'device_distribution_pct' in device_metrics:
            distribution = device_metrics['device_distribution_pct']
            if len(distribution) > 0:
                min_pct = min(distribution.values())
                if min_pct < 5:
                    rare_device = min(distribution, key=distribution.get)
                    self._add_anomaly(
                        f"Low Usage Device Format: {rare_device}",
                        f"{rare_device} accounts for only {min_pct:.1f}% of usage.",
                        'Info',
                        "Consider whether maintaining support for rarely-used device types is cost-effective.",
                        f"Evaluate ROI of supporting {rare_device}. Consider user migration strategies if discontinuing."
                    )
    
    def _detect_statistical_outliers(self, df: pd.DataFrame):
        """Detect statistical outliers in numeric columns"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # Skip columns with too many zeros or missing values
            if df[col].isnull().sum() / len(df) > 0.5:
                continue
            
            non_zero = df[col][df[col] != 0]
            if len(non_zero) < 10:
                continue
            
            # Calculate Z-scores
            z_scores = np.abs(stats.zscore(non_zero, nan_policy='omit'))
            outliers = (z_scores > self.thresholds['outlier_zscore']).sum()
            outlier_pct = (outliers / len(non_zero)) * 100
            
            if outlier_pct > 5:
                self._add_anomaly(
                    f"Statistical Outliers: {col}",
                    f"{outliers} records ({outlier_pct:.1f}%) contain statistical outliers in {col}.",
                    'Info',
                    "Outliers may represent data errors or genuine edge cases requiring attention.",
                    f"Review outlier records for {col}. Verify data accuracy and investigate extreme cases."
                )
    
    def _detect_temporal_anomalies(self, df: pd.DataFrame):
        """Detect time-based anomalies"""
        # Find date columns
        date_cols = [col for col in df.columns if any(x in col.lower() for x in 
            ['date', 'time', 'timestamp'])]
        
        for date_col in date_cols:
            try:
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                valid_dates = df[date_col].dropna()
                
                if len(valid_dates) == 0:
                    continue
                
                # Check for data recency
                latest_date = valid_dates.max()
                current_date = pd.Timestamp.now()
                days_old = (current_date - latest_date).days
                
                if days_old > 30:
                    self._add_anomaly(
                        f"Stale Data Warning: {date_col}",
                        f"Latest data is {days_old} days old (last update: {latest_date.strftime('%Y-%m-%d')}).",
                        'Warning',
                        "Analysis based on outdated data may not reflect current situation.",
                        "Ensure regular data updates. Investigate delays in data pipeline."
                    )
                
                # Check for data gaps
                date_range = pd.date_range(start=valid_dates.min(), end=valid_dates.max(), freq='D')
                missing_dates = len(date_range) - valid_dates.dt.date.nunique()
                
                if missing_dates > len(date_range) * 0.1:  # More than 10% dates missing
                    self._add_anomaly(
                        f"Data Gaps Detected: {date_col}",
                        f"{missing_dates} days missing in the date range ({(missing_dates/len(date_range)*100):.1f}%).",
                        'Warning',
                        "Missing dates may indicate data collection issues or irregular reporting.",
                        "Investigate reasons for missing data. Ensure continuous data collection."
                    )
                
            except Exception as e:
                continue
    
    def detect_trend_anomalies(self, df: pd.DataFrame, date_col: str, metric_col: str) -> List[Dict]:
        """
        Detect anomalies in trends over time
        """
        anomalies = []
        
        try:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df_sorted = df.sort_values(date_col)
            
            # Daily aggregation
            daily = df_sorted.groupby(pd.Grouper(key=date_col, freq='D'))[metric_col].sum()
            
            # Calculate rolling statistics
            rolling_mean = daily.rolling(window=7).mean()
            rolling_std = daily.rolling(window=7).std()
            
            # Detect points outside 2 standard deviations
            for date, value in daily.items():
                if pd.notna(rolling_mean[date]) and pd.notna(rolling_std[date]):
                    if abs(value - rolling_mean[date]) > 2 * rolling_std[date]:
                        anomalies.append({
                            'date': date,
                            'value': value,
                            'expected': rolling_mean[date],
                            'deviation': abs(value - rolling_mean[date]) / rolling_std[date]
                        })
        except Exception as e:
            pass
        
        return anomalies
