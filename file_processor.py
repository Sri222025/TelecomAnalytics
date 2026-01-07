"""
Smart File Processor with Column Type Detection
Handles multi-sheet Excel files and intelligent data typing
"""
import pandas as pd
import numpy as np
from datetime import datetime
import io

class FileProcessor:
    def __init__(self):
        self.supported_formats = ['.xlsx', '.xls', '.csv']
        
    def process_file(self, uploaded_file):
        """
        Process uploaded file and return structured data with metadata
        """
        file_name = uploaded_file.name
        file_extension = file_name[file_name.rfind('.'):]
        
        if file_extension not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        # Process based on file type
        if file_extension == '.csv':
            return self._process_csv(uploaded_file, file_name)
        else:
            return self._process_excel(uploaded_file, file_name)
    
    def _process_csv(self, file, file_name):
        """Process CSV file"""
        df = pd.read_csv(file)
        df = self._clean_dataframe(df)
        df = self._detect_column_types(df)
        
        # Add metadata
        df['_source_file'] = file_name
        df['_source_sheet'] = 'Sheet1'
        
        return {
            'file_name': file_name,
            'sheets': [{
                'sheet_name': 'Sheet1',
                'data': df,
                'rows': len(df),
                'columns': len(df.columns)
            }],
            'total_rows': len(df),
            'total_columns': len(df.columns)
        }
    
    def _process_excel(self, file, file_name):
        """Process Excel file with multiple sheets"""
        excel_file = pd.ExcelFile(file)
        sheets_data = []
        total_rows = 0
        all_columns = set()
        
        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(file, sheet_name=sheet_name)
            df = self._clean_dataframe(df)
            df = self._detect_column_types(df)
            
            # Add metadata
            df['_source_file'] = file_name
            df['_source_sheet'] = sheet_name
            
            sheets_data.append({
                'sheet_name': sheet_name,
                'data': df,
                'rows': len(df),
                'columns': len(df.columns)
            })
            
            total_rows += len(df)
            all_columns.update(df.columns)
        
        return {
            'file_name': file_name,
            'sheets': sheets_data,
            'total_rows': total_rows,
            'total_columns': len(all_columns),
            'sheet_count': len(sheets_data)
        }
    
    def _clean_dataframe(self, df):
        """Clean and standardize dataframe"""
        # Remove completely empty rows and columns
        df = df.dropna(how='all', axis=0)
        df = df.dropna(how='all', axis=1)
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        
        # Reset index
        df = df.reset_index(drop=True)
        
        return df
    
    def _detect_column_types(self, df):
        """
        Intelligent column type detection
        Identifies: IDs, dates, metrics, categories
        """
        for col in df.columns:
            # Skip metadata columns
            if col.startswith('_'):
                continue
            
            # Try to detect dates
            if self._is_date_column(df[col]):
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce', format='mixed')
                except:
                    pass
            
            # Try to detect numeric columns
            elif self._is_numeric_column(df[col]):
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    pass
            
            # Convert to string for ID-like columns
            elif self._is_id_column(col, df[col]):
                df[col] = df[col].astype(str).str.strip()
        
        return df
    
    def _is_date_column(self, series):
        """Check if column contains dates"""
        if series.dtype == 'datetime64[ns]':
            return True
        
        # Check column name
        date_keywords = ['date', 'time', 'dt', 'timestamp', 'day', 'month', 'year']
        col_lower = str(series.name).lower()
        if any(keyword in col_lower for keyword in date_keywords):
            return True
        
        # Sample check
        sample = series.dropna().head(100)
        if len(sample) == 0:
            return False
        
        try:
            pd.to_datetime(sample, errors='coerce')
            valid_dates = pd.to_datetime(sample, errors='coerce').notna().sum()
            return valid_dates / len(sample) > 0.5
        except:
            return False
    
    def _is_numeric_column(self, series):
        """Check if column should be numeric"""
        if pd.api.types.is_numeric_dtype(series):
            return True
        
        # Check if can be converted to numeric
        sample = series.dropna().head(100)
        if len(sample) == 0:
            return False
        
        try:
            pd.to_numeric(sample, errors='coerce')
            valid_nums = pd.to_numeric(sample, errors='coerce').notna().sum()
            return valid_nums / len(sample) > 0.7
        except:
            return False
    
    def _is_id_column(self, col_name, series):
        """Check if column is an ID field"""
        id_keywords = ['id', 'number', 'code', 'serial', 'customer', 'account', 
                       'msisdn', 'imsi', 'imei', 'mac', 'version']
        col_lower = str(col_name).lower()
        
        return any(keyword in col_lower for keyword in id_keywords)
    
    def get_column_metadata(self, df):
        """Get detailed metadata about columns"""
        metadata = []
        
        for col in df.columns:
            if col.startswith('_'):
                continue
            
            col_type = 'text'
            if pd.api.types.is_numeric_dtype(df[col]):
                col_type = 'numeric'
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                col_type = 'date'
            
            metadata.append({
                'column': col,
                'type': col_type,
                'non_null': df[col].notna().sum(),
                'null_count': df[col].isna().sum(),
                'unique_values': df[col].nunique(),
                'sample_values': df[col].dropna().head(3).tolist()
            })
        
        return metadata
