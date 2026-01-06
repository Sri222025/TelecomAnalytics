import pandas as pd
import numpy as np
from typing import Dict, Any, List

class TelecomMetrics:
    """Calculate telecom-specific KPIs and metrics"""
    
    def calculate_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate comprehensive telecom metrics from the data
        """
        metrics = {
            'basic_stats': self._calculate_basic_stats(df),
            'subscriber_metrics': self._calculate_subscriber_metrics(df),
            'usage_metrics': self._calculate_usage_metrics(df),
            'device_metrics': self._calculate_device_metrics(df),
            'regional_metrics': self._calculate_regional_metrics(df),
            'temporal_metrics': self._calculate_temporal_metrics(df),
            'quality_metrics': self._calculate_quality_metrics(df)
        }
        
        return metrics
    
    def _calculate_basic_stats(self, df: pd.DataFrame) -> Dict:
        """Basic dataset statistics"""
        return {
            'total_records': len(df),
            'total_columns': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'missing_values_total': df.isnull().sum().sum(),
            'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        }
    
    def _calculate_subscriber_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate subscriber-related metrics"""
        metrics = {}
        
        # Find subscriber identifier columns
        id_cols = [col for col in df.columns if any(x in col.lower() for x in 
            ['customer', 'subscriber', 'user', 'account', 'msisdn', 'number'])]
        
        if id_cols:
            primary_id = id_cols[0]
            metrics['unique_subscribers'] = df[primary_id].nunique()
            metrics['total_records_per_subscriber'] = len(df) / df[primary_id].nunique()
        
        # Find connection type columns
        type_cols = [col for col in df.columns if any(x in col.lower() for x in 
            ['type', 'plan', 'connection', 'category', 'segment'])]
        
        if type_cols:
            type_col = type_cols[0]
            metrics['connection_types'] = df[type_col].value_counts().to_dict()
            metrics['connection_type_distribution'] = (df[type_col].value_counts(normalize=True) * 100).to_dict()
        
        # Find status columns
        status_cols = [col for col in df.columns if 'status' in col.lower() or 'state' in col.lower()]
        if status_cols:
            metrics['status_distribution'] = df[status_cols[0]].value_counts().to_dict()
        
        return metrics
    
    def _calculate_usage_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate usage-related metrics (calls, data, etc.)"""
        metrics = {}
        
        # Find usage columns
        usage_cols = [col for col in df.columns if any(x in col.lower() for x in 
            ['call', 'duration', 'minute', 'usage', 'volume', 'data', 'traffic', 'session'])]
        
        for col in usage_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                metrics[f'{col}_total'] = df[col].sum()
                metrics[f'{col}_average'] = df[col].mean()
                metrics[f'{col}_median'] = df[col].median()
                metrics[f'{col}_std'] = df[col].std()
                metrics[f'{col}_min'] = df[col].min()
                metrics[f'{col}_max'] = df[col].max()
                
                # Calculate percentiles
                metrics[f'{col}_p25'] = df[col].quantile(0.25)
                metrics[f'{col}_p75'] = df[col].quantile(0.75)
                metrics[f'{col}_p90'] = df[col].quantile(0.90)
                metrics[f'{col}_p95'] = df[col].quantile(0.95)
                
                # Zero usage count
                metrics[f'{col}_zero_usage_count'] = (df[col] == 0).sum()
                metrics[f'{col}_zero_usage_percentage'] = ((df[col] == 0).sum() / len(df)) * 100
        
        # Calculate ARPU if revenue column exists
        revenue_cols = [col for col in df.columns if any(x in col.lower() for x in 
            ['revenue', 'arpu', 'charge', 'amount', 'price'])]
        
        id_cols = [col for col in df.columns if any(x in col.lower() for x in 
            ['customer', 'subscriber', 'user'])]
        
        if revenue_cols and id_cols:
            revenue_col = revenue_cols[0]
            id_col = id_cols[0]
            total_revenue = df[revenue_col].sum()
            unique_users = df[id_col].nunique()
            metrics['arpu'] = total_revenue / unique_users if unique_users > 0 else 0
        
        # Calculate MOU (Minutes of Usage) if applicable
        duration_cols = [col for col in df.columns if any(x in col.lower() for x in 
            ['duration', 'minute', 'time'])]
        
        if duration_cols and id_cols:
            duration_col = duration_cols[0]
            id_col = id_cols[0]
            total_minutes = df[duration_col].sum()
            unique_users = df[id_col].nunique()
            metrics['mou'] = total_minutes / unique_users if unique_users > 0 else 0
        
        return metrics
    
    def _calculate_device_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate device/format specific metrics"""
        metrics = {}
        
        # Find device columns
        device_cols = [col for col in df.columns if any(x in col.lower() for x in 
            ['device', 'format', 'channel', 'method', 'platform'])]
        
        if device_cols:
            device_col = device_cols[0]
            metrics['device_distribution'] = df[device_col].value_counts().to_dict()
            metrics['device_distribution_pct'] = (df[device_col].value_counts(normalize=True) * 100).to_dict()
            
            # Count specific device types
            device_values = df[device_col].str.lower().fillna('')
            metrics['pots_count'] = device_values.str.contains('pots|landline').sum()
            metrics['jiojoin_count'] = device_values.str.contains('jiojoin|app|mobile').sum()
            metrics['stb_count'] = device_values.str.contains('stb|settop|set-top|set top').sum()
            metrics['airfiber_count'] = device_values.str.contains('airfiber|air fiber').sum()
        
        return metrics
    
    def _calculate_regional_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate region-specific metrics"""
        metrics = {}
        
        # Find region columns
        region_cols = [col for col in df.columns if any(x in col.lower() for x in 
            ['region', 'circle', 'state', 'zone', 'area', 'location', 'city'])]
        
        if region_cols:
            region_col = region_cols[0]
            metrics['total_regions'] = df[region_col].nunique()
            metrics['regional_distribution'] = df[region_col].value_counts().to_dict()
            metrics['regional_distribution_pct'] = (df[region_col].value_counts(normalize=True) * 100).to_dict()
            
            # Top and bottom regions
            region_counts = df[region_col].value_counts()
            metrics['top_5_regions'] = region_counts.head(5).to_dict()
            metrics['bottom_5_regions'] = region_counts.tail(5).to_dict()
        
        return metrics
    
    def _calculate_temporal_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate time-based metrics"""
        metrics = {}
        
        # Find date columns
        date_cols = [col for col in df.columns if any(x in col.lower() for x in 
            ['date', 'time', 'day', 'month', 'year', 'timestamp', 'created', 'activation'])]
        
        for date_col in date_cols:
            try:
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                valid_dates = df[date_col].dropna()
                
                if len(valid_dates) > 0:
                    metrics[f'{date_col}_min'] = valid_dates.min()
                    metrics[f'{date_col}_max'] = valid_dates.max()
                    metrics[f'{date_col}_range_days'] = (valid_dates.max() - valid_dates.min()).days
                    
                    # Day of week distribution
                    metrics[f'{date_col}_dow_distribution'] = valid_dates.dt.day_name().value_counts().to_dict()
                    
                    # Monthly distribution
                    metrics[f'{date_col}_monthly_distribution'] = valid_dates.dt.to_period('M').value_counts().to_dict()
            except:
                continue
        
        return metrics
    
    def _calculate_quality_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate data quality metrics"""
        metrics = {
            'completeness': {},
            'consistency': {},
            'validity': {}
        }
        
        # Completeness - check missing values per column
        for col in df.columns:
            missing_pct = (df[col].isnull().sum() / len(df)) * 100
            metrics['completeness'][col] = {
                'missing_count': df[col].isnull().sum(),
                'missing_percentage': missing_pct,
                'status': 'good' if missing_pct < 5 else 'warning' if missing_pct < 20 else 'critical'
            }
        
        # Consistency - check for duplicates
        duplicate_count = df.duplicated().sum()
        metrics['consistency']['duplicate_rows'] = duplicate_count
        metrics['consistency']['duplicate_percentage'] = (duplicate_count / len(df)) * 100
        
        # Validity - check for outliers in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
            metrics['validity'][col] = {
                'outlier_count': outliers,
                'outlier_percentage': (outliers / len(df)) * 100
            }
        
        return metrics
    
    def calculate_growth_metrics(self, df: pd.DataFrame, date_col: str, metric_col: str) -> Dict:
        """
        Calculate growth metrics over time
        """
        try:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df_sorted = df.sort_values(date_col)
            
            # Monthly aggregation
            monthly = df_sorted.groupby(pd.Grouper(key=date_col, freq='M'))[metric_col].sum()
            
            # Calculate growth rates
            mom_growth = monthly.pct_change() * 100
            
            return {
                'monthly_values': monthly.to_dict(),
                'mom_growth': mom_growth.to_dict(),
                'average_growth_rate': mom_growth.mean(),
                'total_growth': ((monthly.iloc[-1] - monthly.iloc[0]) / monthly.iloc[0]) * 100 if len(monthly) > 0 else 0
            }
        except Exception as e:
            return {'error': str(e)}
