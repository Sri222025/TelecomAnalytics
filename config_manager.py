import json
from typing import Dict, Any

class ConfigManager:
    """Manage configuration and preferences"""
    
    def __init__(self):
        self.config = {}
    
    def save_relationships(self, relationships: list) -> str:
        """Save relationship configuration as JSON"""
        return json.dumps(relationships, indent=2)
    
    def load_relationships(self, json_str: str) -> list:
        """Load relationship configuration from JSON"""
        try:
            return json.loads(json_str)
        except:
            return []
    
    def save_preferences(self, preferences: Dict[str, Any]) -> str:
        """Save user preferences"""
        return json.dumps(preferences, indent=2)
    
    def load_preferences(self, json_str: str) -> Dict[str, Any]:
        """Load user preferences"""
        try:
            return json.loads(json_str)
        except:
            return {}
