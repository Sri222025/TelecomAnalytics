import pandas as pd
from typing import List, Dict, Any
import re

class RelationshipDetector:
    """Automatically detect relationships between datasets"""
    
    def __init__(self):
        # Common linking column patterns
        self.linking_patterns = [
            r'.*customer.*id.*',
            r'.*user.*id.*',
            r'.*subscriber.*id.*',
            r'.*account.*id.*',
            r'.*serial.*number.*',
            r'.*serial.*no.*',
            r'.*msisdn.*',
            r'.*mobile.*number.*',
            r'.*phone.*number.*',
            r'.*fixed.*line.*',
            r'.*connection.*id.*',
            r'.*device.*id.*',
            r'.*fw.*version.*',
            r'.*firmware.*',
            r'.*region.*code.*',
            r'.*circle.*code.*',
            r'.*order.*id.*',
            r'.*transaction.*id.*',
        ]
    
    def detect_relationships(self, processed_data: List[Dict]) -> List[Dict[str, Any]]:
        """
        Detect potential relationships between sheets across files
        """
        relationships = []
        
        # Get all sheet data
        all_sheets = []
        for file_info in processed_data:
            for sheet_name, sheet_data in file_info['sheets'].items():
                all_sheets.append({
                    'file': file_info['filename'],
                    'sheet': sheet_name,
                    'data': sheet_data
                })
        
        # Compare each pair of sheets
        for i, sheet1 in enumerate(all_sheets):
            for sheet2 in all_sheets[i+1:]:
                # Don't compare sheets from the same file (optional - you can enable this)
                # if sheet1['file'] == sheet2['file']:
                #     continue
                
                # Find potential linking columns
                potential_links = self._find_linking_columns(
                    sheet1['data'], sheet2['data']
                )
                
                for link in potential_links:
                    relationships.append({
                        'source_file': sheet1['file'],
                        'source_sheet': sheet1['sheet'],
                        'source_column': link['col1'],
                        'target_file': sheet2['file'],
                        'target_sheet': sheet2['sheet'],
                        'target_column': link['col2'],
                        'match_score': link['match_score'],
                        'common_values': link['common_values'],
                        'confidence': link['confidence'],
                        'join_type': 'inner'  # default
                    })
        
        # Sort by match score
        relationships.sort(key=lambda x: x['match_score'], reverse=True)
        
        return relationships
    
    def _find_linking_columns(self, sheet1_data: Dict, sheet2_data: Dict) -> List[Dict]:
        """
        Find columns that could be used to link two sheets
        """
        potential_links = []
        
        df1 = sheet1_data['dataframe']
        df2 = sheet2_data['dataframe']
        
        # Compare each column in sheet1 with each column in sheet2
        for col1 in df1.columns:
            for col2 in df2.columns:
                # Check if column names are similar
                name_similarity = self._calculate_name_similarity(col1, col2)
                
                # Check if they match any linking patterns
                is_linking_pattern = self._matches_linking_pattern(col1) or self._matches_linking_pattern(col2)
                
                # Check data overlap
                try:
                    # Convert to same type for comparison
                    values1 = set(df1[col1].dropna().astype(str).str.strip().str.lower())
                    values2 = set(df2[col2].dropna().astype(str).str.strip().str.lower())
                    
                    if len(values1) == 0 or len(values2) == 0:
                        continue
                    
                    # Calculate overlap
                    common = values1.intersection(values2)
                    overlap_ratio = len(common) / min(len(values1), len(values2))
                    
                    # If significant overlap and reasonable conditions
                    if overlap_ratio > 0.1 or (name_similarity > 0.7 and overlap_ratio > 0.05):
                        match_score = (overlap_ratio * 0.6) + (name_similarity * 0.3) + (0.1 if is_linking_pattern else 0)
                        
                        # Determine confidence level
                        if match_score > 0.7:
                            confidence = 'High'
                        elif match_score > 0.4:
                            confidence = 'Medium'
                        else:
                            confidence = 'Low'
                        
                        potential_links.append({
                            'col1': col1,
                            'col2': col2,
                            'match_score': match_score,
                            'overlap_ratio': overlap_ratio,
                            'common_values': len(common),
                            'confidence': confidence
                        })
                
                except Exception as e:
                    continue
        
        return potential_links
    
    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """
        Calculate similarity between two column names
        """
        name1 = name1.lower().strip()
        name2 = name2.lower().strip()
        
        # Exact match
        if name1 == name2:
            return 1.0
        
        # Remove common words and special characters
        common_words = ['id', 'no', 'number', 'code', 'name']
        
        # Tokenize
        tokens1 = set(re.findall(r'\w+', name1))
        tokens2 = set(re.findall(r'\w+', name2))
        
        # Remove common words
        tokens1 = {t for t in tokens1 if t not in common_words}
        tokens2 = {t for t in tokens2 if t not in common_words}
        
        if len(tokens1) == 0 or len(tokens2) == 0:
            return 0.0
        
        # Jaccard similarity
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _matches_linking_pattern(self, column_name: str) -> bool:
        """
        Check if column name matches common linking patterns
        """
        col_lower = column_name.lower().strip()
        
        for pattern in self.linking_patterns:
            if re.match(pattern, col_lower):
                return True
        
        return False
    
    def validate_relationship(self, df1: pd.DataFrame, col1: str, 
                            df2: pd.DataFrame, col2: str) -> Dict[str, Any]:
        """
        Validate a potential relationship between two columns
        """
        try:
            # Get unique values
            values1 = set(df1[col1].dropna().astype(str))
            values2 = set(df2[col2].dropna().astype(str))
            
            # Calculate statistics
            common = values1.intersection(values2)
            only_in_1 = values1 - values2
            only_in_2 = values2 - values1
            
            return {
                'valid': len(common) > 0,
                'common_count': len(common),
                'only_in_source': len(only_in_1),
                'only_in_target': len(only_in_2),
                'overlap_percentage': len(common) / len(values1) * 100 if values1 else 0
            }
        except Exception as e:
            return {
                'valid': False,
                'error': str(e)
            }
