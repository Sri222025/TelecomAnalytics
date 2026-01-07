"""
Smart File Processor - Handles Multi-Level Excel Headers
Fixes merged cells and "Unnamed" column issues
"""
import pandas as pd
import numpy as np
from datetime import datetime
import io

class FileProcessor:
    def __init__(self):
        self.supported_formats = ['.xlsx', '.xls', '.csv']
        
    def process_file(self, uploaded_file):
        """Process uploaded file and return structured data with metadata"""
        file_name = uploaded_file.name
        file_extension = file_name[file_name.rfind('.'):]
        
        if file_extension not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        if file_extension == '.csv':
            return self._process_csv(uploaded_file, file_name)
        else:
            return self._process_excel(uploaded_file, file_name)
    
    def _process_csv(self, file, file_name):
        """Process CSV file"""
        df = pd.read_csv(file)
        df = self._clean_and_fix_headers(df)
        df = self._clean_dataframe(df)
        df = self._detect_column_types(df)
        
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
        """Process Excel file with multi-level header detection"""
        excel_file = pd.ExcelFile(file)
        sheets_data = []
        total_rows = 0
        all_columns = set()
        
        for sheet_name in excel_file.sheet_names:
            # Try to detect header rows
            df = self._read_excel_with_smart_headers(file, sheet_name)
            
            df = self._clean_and_fix_headers(df)
            df = self._clean_dataframe(df)
            df = self._detect_column_types(df)
            
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
    
    def _read_excel_with_smart_headers(self, file, sheet_name):
        """Read Excel with multi-level header detection"""
        
        # Try reading first few rows to detect header structure
        preview = pd.read_excel(file, sheet_name=sheet_name, nrows=5, header=None)
        
        # Detect how many header rows (look for rows with mostly text)
        header_rows = []
        for idx in range(min(3, len(preview))):
            row = preview.iloc[idx]
            # If >50% cells are non-numeric text, it's likely a header
            text_cells = row.apply(lambda x: isinstance(x, str) and not str(x).replace('.','').replace('-','').isdigit())
            if text_cells.sum() / len(row) > 0.5:
                header_rows.append(idx)
            else:
                break
        
        # Read with detected headers
        if len(header_rows) > 1:
            # Multi-level headers
            df = pd.read_excel(file, sheet_name=sheet_name, header=header_rows)
            # Flatten multi-level columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [' '.join([str(c) for c in col if not str(c).startswith('Unnamed')]).strip() 
                             for col in df.columns.values]
        else:
            # Single header row
            header_row = header_rows[0] if header_rows else 0
            df = pd.read_excel(file, sheet_name=sheet_name, header=header_row)
        
        return df
    
    def _clean_and_fix_headers(self, df):
        """Clean and fix column headers, especially 'Unnamed' columns"""
        
        new_columns = []
        last_valid_name = ""
        unnamed_counter = {}
        
        for col in df.columns:
            col_str = str(col).strip()
            
            # If it's an "Unnamed" column
            if col_str.startswith('Unnamed'):
                # Try to infer from nearby columns or data
                col_idx = df.columns.get_loc(col)
                
                # Option 1: Check if previous column suggests a parent category
                if last_valid_name and not last_valid_name.startswith('Unnamed'):
                    # Create sub-column name
                    if last_valid_name not in unnamed_counter:
                        unnamed_counter[last_valid_name] = 0
                    unnamed_counter[last_valid_name] += 1
                    new_col = f"{last_valid_name}_{unnamed_counter[last_valid_name]}"
                else:
                    # Option 2: Try to infer from data in first few rows
                    sample_data = df[col].dropna().head(10)
                    if len(sample_data) > 0:
                        first_val = str(sample_data.iloc[0])
                        # If first value looks like a header (text, not number)
                        if not first_val.replace('.','').replace('-','').isdigit():
                            new_col = first_val[:30]  # Use as column name
                        else:
                            new_col = f"Column_{col_idx}"
                    else:
                        new_col = f"Column_{col_idx}"
                
                new_columns.append(new_col)
            else:
                new_columns.append(col_str)
                last_valid_name = col_str
        
        df.columns = new_columns
        
        # Remove duplicate column names by adding suffixes
        seen = {}
        final_columns = []
        for col in new_columns:
            if col in seen:
                seen[col] += 1
                final_columns.append(f"{col}_{seen[col]}")
            else:
                seen[col] = 0
                final_columns.append(col)
        
        df.columns = final_columns
        
        return df
    
    def _clean_dataframe(self, df):
        """Clean and standardize dataframe"""
        
        # Remove rows that are actually headers (common in telecom reports)
        # If first row has column names repeated, remove it
        if len(df) > 0:
            first_row = df.iloc[0].astype(str)
            if any(first_row.str.contains('Total|Audio|Video|FO|FT', case=False, na=False)):
                df = df.iloc[1:]
        
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
        """Intelligent column type detection"""
        for col in df.columns:
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
            
            # Keep ID columns as strings
            elif self._is_id_column(col, df[col]):
                df[col] = df[col].astype(str).str.strip()
        
        return df
    
    def _is_date_column(self, series):
        """Check if column contains dates"""
        if series.dtype == 'datetime64[ns]':
            return True
        
        date_keywords = ['date', 'time', 'dt', 'timestamp', 'day', 'month', 'year']
        col_lower = str(series.name).lower()
        if any(keyword in col_lower for keyword in date_keywords):
            return True
        
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
                       'msisdn', 'imsi', 'imei', 'mac', 'version', 'sl', 'no']
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
