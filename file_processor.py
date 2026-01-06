import pandas as pd
import io
from typing import List, Dict, Any
import streamlit as st

class FileProcessor:
    """Process uploaded files and extract data from all worksheets"""
    
    def __init__(self):
        self.supported_formats = ['xlsx', 'xls', 'csv']
    
    def process_files(self, uploaded_files: List) -> List[Dict[str, Any]]:
        """
        Process multiple uploaded files
        Returns: List of file information dictionaries
        """
        processed_data = []
        
        for file in uploaded_files:
            try:
                file_info = {
                    'filename': file.name,
                    'file_type': file.name.split('.')[-1],
                    'sheets': {}
                }
                
                # Read based on file type
                if file.name.endswith('.csv'):
                    # CSV files have only one sheet
                    df = pd.read_csv(file)
                    sheet_name = 'Sheet1'
                    file_info['sheets'][sheet_name] = self._process_dataframe(df, sheet_name)
                
                elif file.name.endswith(('.xlsx', '.xls')):
                    # Excel files can have multiple sheets
                    excel_file = pd.ExcelFile(file)
                    
                    for sheet_name in excel_file.sheet_names:
                        df = pd.read_excel(excel_file, sheet_name=sheet_name)
                        file_info['sheets'][sheet_name] = self._process_dataframe(df, sheet_name)
                
                processed_data.append(file_info)
                
            except Exception as e:
                st.error(f"Error processing {file.name}: {str(e)}")
                continue
        
        return processed_data
    
    def _process_dataframe(self, df: pd.DataFrame, sheet_name: str) -> Dict[str, Any]:
        """
        Process individual dataframe and extract metadata
        """
        # Clean column names (remove extra spaces, etc.)
        df.columns = df.columns.str.strip()
        
        # Detect data types
        date_cols = []
        numeric_cols = []
        categorical_cols = []
        
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_cols.append(col)
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                date_cols.append(col)
            else:
                # Try to convert to datetime with format='mixed' to avoid warning
                try:
                    pd.to_datetime(df[col], errors='raise', format='mixed')
                    date_cols.append(col)
                except:
                    categorical_cols.append(col)
        
        sheet_data = {
            'dataframe': df,
            'row_count': len(df),
            'column_count': len(df.columns),
            'columns': df.columns.tolist(),
            'date_columns': date_cols,
            'numeric_columns': numeric_cols,
            'categorical_columns': categorical_cols,
            'preview': df.head(10),
            'data_types': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'unique_counts': {col: df[col].nunique() for col in df.columns}
        }
        
        return sheet_data
    
    def get_column_info(self, processed_data: List[Dict]) -> Dict[str, List]:
        """
        Get aggregated information about all columns across all files
        """
        all_columns = {}
        
        for file_info in processed_data:
            for sheet_name, sheet_data in file_info['sheets'].items():
                for col in sheet_data['columns']:
                    key = f"{file_info['filename']}::{sheet_name}::{col}"
                    all_columns[key] = {
                        'file': file_info['filename'],
                        'sheet': sheet_name,
                        'column': col,
                        'type': str(sheet_data['data_types'].get(col, 'unknown')),
                        'unique_count': sheet_data['unique_counts'].get(col, 0),
                        'missing_count': sheet_data['missing_values'].get(col, 0)
                    }
        
        return all_columns
