import pandas as pd
from typing import List, Dict, Any
import streamlit as st

class DataMerger:
    """Merge datasets based on detected relationships"""
    
    def merge_data(self, processed_data: List[Dict], relationships: List[Dict]) -> pd.DataFrame:
        """
        Merge all datasets based on defined relationships
        """
        if not relationships:
            # If no relationships, concatenate all data
            return self._concatenate_all(processed_data)
        
        # Start with the first dataset
        merged_df = None
        merged_sheets = set()
        
        # Group relationships by source
        for rel in relationships:
            source_key = f"{rel['source_file']}::{rel['source_sheet']}"
            target_key = f"{rel['target_file']}::{rel['target_sheet']}"
            
            # Get dataframes
            source_df = self._get_dataframe(processed_data, rel['source_file'], rel['source_sheet'])
            target_df = self._get_dataframe(processed_data, rel['target_file'], rel['target_sheet'])
            
            if source_df is None or target_df is None:
                continue
            
            # Add prefix to columns to avoid conflicts (except join columns)
            source_prefix = f"{rel['source_sheet']}_"
            target_prefix = f"{rel['target_sheet']}_"
            
            source_df_renamed = source_df.copy()
            target_df_renamed = target_df.copy()
            
            # Rename columns except the join columns
            source_df_renamed.columns = [
                col if col == rel['source_column'] else f"{source_prefix}{col}" 
                for col in source_df.columns
            ]
            target_df_renamed.columns = [
                col if col == rel['target_column'] else f"{target_prefix}{col}" 
                for col in target_df.columns
            ]
            
            # Perform merge
            if merged_df is None:
                # First merge
                merged_df = pd.merge(
                    source_df_renamed,
                    target_df_renamed,
                    left_on=rel['source_column'],
                    right_on=rel['target_column'],
                    how=rel.get('join_type', 'inner'),
                    suffixes=('', '_dup')
                )
                merged_sheets.add(source_key)
                merged_sheets.add(target_key)
            else:
                # Subsequent merges
                if source_key in merged_sheets and target_key not in merged_sheets:
                    # Merge target into existing merged_df
                    merged_df = pd.merge(
                        merged_df,
                        target_df_renamed,
                        left_on=rel['source_column'],
                        right_on=rel['target_column'],
                        how='left',
                        suffixes=('', '_dup')
                    )
                    merged_sheets.add(target_key)
                elif target_key in merged_sheets and source_key not in merged_sheets:
                    # Merge source into existing merged_df
                    merged_df = pd.merge(
                        merged_df,
                        source_df_renamed,
                        left_on=rel['target_column'],
                        right_on=rel['source_column'],
                        how='left',
                        suffixes=('', '_dup')
                    )
                    merged_sheets.add(source_key)
        
        # Add any sheets that weren't merged
        for file_info in processed_data:
            for sheet_name, sheet_data in file_info['sheets'].items():
                sheet_key = f"{file_info['filename']}::{sheet_name}"
                if sheet_key not in merged_sheets:
                    df = sheet_data['dataframe']
                    # Add as separate columns with prefix
                    df_prefixed = df.copy()
                    df_prefixed.columns = [f"{sheet_name}_{col}" for col in df.columns]
                    
                    if merged_df is None:
                        merged_df = df_prefixed
                    # else:
                    #     # Concatenate horizontally (this might not make sense without a relationship)
                    #     # Skip for now
                    #     pass
        
        # Remove duplicate columns
        if merged_df is not None:
            merged_df = self._remove_duplicate_columns(merged_df)
        
        return merged_df if merged_df is not None else pd.DataFrame()
    
    def _get_dataframe(self, processed_data: List[Dict], filename: str, sheet_name: str) -> pd.DataFrame:
        """
        Get a specific dataframe from processed data
        """
        for file_info in processed_data:
            if file_info['filename'] == filename:
                if sheet_name in file_info['sheets']:
                    return file_info['sheets'][sheet_name]['dataframe'].copy()
        return None
    
    def _concatenate_all(self, processed_data: List[Dict]) -> pd.DataFrame:
        """
        Concatenate all sheets when no relationships are defined
        """
        all_dfs = []
        
        for file_info in processed_data:
            for sheet_name, sheet_data in file_info['sheets'].items():
                df = sheet_data['dataframe'].copy()
                # Add metadata columns
                df['_source_file'] = file_info['filename']
                df['_source_sheet'] = sheet_name
                all_dfs.append(df)
        
        if all_dfs:
            # Concatenate vertically
            return pd.concat(all_dfs, ignore_index=True, sort=False)
        else:
            return pd.DataFrame()
    
    def _remove_duplicate_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate columns that end with _dup
        """
        # Remove columns ending with _dup
        cols_to_keep = [col for col in df.columns if not col.endswith('_dup')]
        return df[cols_to_keep]
    
    def merge_two_dataframes(self, df1: pd.DataFrame, df2: pd.DataFrame,
                           left_on: str, right_on: str, 
                           how: str = 'inner') -> pd.DataFrame:
        """
        Merge two dataframes on specified columns
        """
        try:
            merged = pd.merge(
                df1, df2,
                left_on=left_on,
                right_on=right_on,
                how=how,
                suffixes=('', '_right')
            )
            return merged
        except Exception as e:
            st.error(f"Error merging dataframes: {str(e)}")
            return pd.DataFrame()
