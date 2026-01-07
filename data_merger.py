"""
Intelligent Data Merger
Auto-detects relationships between files and merges smartly
"""
import pandas as pd
import numpy as np

class DataMerger:
    def __init__(self):
        self.common_id_patterns = [
            'customer_id', 'customerid', 'cust_id',
            'serial_number', 'serial', 'serialnumber',
            'msisdn', 'mobile', 'phone',
            'account', 'account_id',
            'id', 'code', 'number',
            'version', 'fw_version', 'firmware',
            'imei', 'imsi', 'mac'
        ]
    
    def detect_relationships(self, processed_files):
        """
        Auto-detect potential relationships between files
        Returns list of potential merge keys
        """
        if len(processed_files) <= 1:
            return []
        
        # Collect all dataframes with their sources
        all_dfs = []
        for file_info in processed_files:
            for sheet in file_info['sheets']:
                all_dfs.append({
                    'file': file_info['file_name'],
                    'sheet': sheet['sheet_name'],
                    'df': sheet['data'],
                    'columns': [col for col in sheet['data'].columns if not col.startswith('_')]
                })
        
        # Find common columns
        relationships = []
        
        for i in range(len(all_dfs)):
            for j in range(i + 1, len(all_dfs)):
                df1 = all_dfs[i]
                df2 = all_dfs[j]
                
                # Find common column names (case-insensitive)
                common_cols = self._find_common_columns(df1['columns'], df2['columns'])
                
                for col in common_cols:
                    # Check if it's a good merge key
                    if self._is_good_merge_key(df1['df'], df2['df'], col):
                        match_rate = self._calculate_match_rate(df1['df'], df2['df'], col)
                        relationships.append({
                            'file1': df1['file'],
                            'sheet1': df1['sheet'],
                            'file2': df2['file'],
                            'sheet2': df2['sheet'],
                            'key_column': col,
                            'match_rate': match_rate
                        })
        
        # Sort by match rate
        relationships.sort(key=lambda x: x['match_rate'], reverse=True)
        
        return relationships
    
    def _find_common_columns(self, cols1, cols2):
        """Find common column names (case-insensitive)"""
        cols1_lower = {col.lower(): col for col in cols1}
        cols2_lower = {col.lower(): col for col in cols2}
        
        common = []
        for col_lower in cols1_lower:
            if col_lower in cols2_lower:
                common.append(cols1_lower[col_lower])
        
        return common
    
    def _is_good_merge_key(self, df1, df2, col):
        """Check if column is suitable for merging"""
        # Must exist in both
        if col not in df1.columns or col not in df2.columns:
            return False
        
        # Should have reasonable number of unique values
        unique1 = df1[col].nunique()
        unique2 = df2[col].nunique()
        
        # At least 2 unique values
        if unique1 < 2 or unique2 < 2:
            return False
        
        # Not too many nulls (< 50%)
        null_pct1 = df1[col].isna().sum() / len(df1) if len(df1) > 0 else 1.0
        null_pct2 = df2[col].isna().sum() / len(df2) if len(df2) > 0 else 1.0
        
        if null_pct1 > 0.5 or null_pct2 > 0.5:
            return False
        
        return True
    
    def _calculate_match_rate(self, df1, df2, col):
        """Calculate percentage of matching values"""
        try:
            values1 = set(df1[col].dropna().astype(str).str.strip().str.lower())
            values2 = set(df2[col].dropna().astype(str).str.strip().str.lower())
            
            if not values1 or not values2:
                return 0.0
            
            intersection = len(values1 & values2)
            union = len(values1 | values2)
            
            return (intersection / union) * 100 if union > 0 else 0.0
        except Exception:
            return 0.0
    
    def merge_files(self, processed_files, relationships=None):
        """
        Merge multiple files based on detected relationships
        If no relationships, concatenate all data
        """
        if len(processed_files) == 0:
            return pd.DataFrame(), {"error": "No files to merge"}
        
        if len(processed_files) == 1:
            # Single file - just concatenate sheets
            return self._merge_single_file(processed_files[0])
        
        # Multiple files
        if relationships and len(relationships) > 0:
            return self._merge_with_relationships(processed_files, relationships)
        else:
            # No relationships - concatenate all
            return self._concatenate_all(processed_files)
    
    def _merge_single_file(self, file_info):
        """Merge all sheets from a single file"""
        all_data = []
        
        for sheet in file_info['sheets']:
            all_data.append(sheet['data'])
        
        if not all_data:
            return pd.DataFrame(), {"error": "No data in file"}
        
        merged = pd.concat(all_data, ignore_index=True)
        
        summary = {
            'method': 'single_file',
            'files_processed': 1,
            'sheets_processed': len(file_info['sheets']),
            'total_records': len(merged),
            'columns': len(merged.columns)
        }
        
        return merged, summary
    
    def _concatenate_all(self, processed_files):
        """Concatenate all data (no relationships)"""
        all_data = []
        total_sheets = 0
        
        for file_info in processed_files:
            for sheet in file_info['sheets']:
                if len(sheet['data']) > 0:
                    all_data.append(sheet['data'])
                    total_sheets += 1
        
        if not all_data:
            return pd.DataFrame(), {"error": "No data to concatenate"}
        
        # Concatenate with outer join to keep all columns
        merged = pd.concat(all_data, ignore_index=True, sort=False)
        
        summary = {
            'method': 'concatenate',
            'files_processed': len(processed_files),
            'sheets_processed': total_sheets,
            'total_records': len(merged),
            'columns': len(merged.columns),
            'note': 'Files concatenated (no relationships detected)'
        }
        
        return merged, summary
    
    def _merge_with_relationships(self, processed_files, relationships):
        """Merge files using detected relationships"""
        # Start with first file's data
        all_dfs = []
        for file_info in processed_files:
            for sheet in file_info['sheets']:
                if len(sheet['data']) > 0:
                    all_dfs.append({
                        'file': file_info['file_name'],
                        'sheet': sheet['sheet_name'],
                        'df': sheet['data']
                    })
        
        if not all_dfs:
            return pd.DataFrame(), {"error": "No data to merge"}
        
        # Use the best relationship
        best_rel = relationships[0]
        merge_key = best_rel['key_column']
        
        # Convert merge key to string and strip whitespace
        for df_info in all_dfs:
            if merge_key in df_info['df'].columns:
                df_info['df'][merge_key] = df_info['df'][merge_key].astype(str).str.strip()
        
        # Merge all dataframes on the common key
        merged = all_dfs[0]['df'].copy()
        
        for i in range(1, len(all_dfs)):
            try:
                # Use outer join to keep all records
                merged = pd.merge(
                    merged,
                    all_dfs[i]['df'],
                    on=merge_key,
                    how='outer',
                    suffixes=('', f'_file{i}')
                )
            except Exception as e:
                # If merge fails, concatenate
                merged = pd.concat([merged, all_dfs[i]['df']], ignore_index=True, sort=False)
        
        summary = {
            'method': 'merge',
            'merge_key': merge_key,
            'files_processed': len(processed_files),
            'total_records': len(merged),
            'columns': len(merged.columns),
            'match_rate': f"{best_rel['match_rate']:.1f}%",
            'note': f"Files merged on '{merge_key}'"
        }
        
        return merged, summary
    
    def get_merge_preview(self, merged_df, summary):
        """Generate a preview of merged data"""
        if merged_df is None or len(merged_df) == 0:
            return {
                'error': 'No data available for preview'
            }
        
        preview = {
            'total_records': len(merged_df),
            'total_columns': len(merged_df.columns),
            'method': summary.get('method', 'unknown'),
            'sample_data': merged_df.head(5).to_dict('records'),
            'column_list': [col for col in merged_df.columns if not col.startswith('_')],
            'data_quality': {
                'complete_records': merged_df.notna().all(axis=1).sum(),
                'records_with_nulls': merged_df.isna().any(axis=1).sum(),
                'total_nulls': merged_df.isna().sum().sum()
            }
        }
        
        return preview

