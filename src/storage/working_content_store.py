"""
Working Content Store Adapter
Loads content from unified index without ChromaDB dependency
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class ContentItem:
    id: str
    title: str
    content_type: str
    source_path: str
    chunks: List[Dict]
    metadata: Dict
    created_at: float = 0.0
    keywords: List[str] = None
    topic_assignments: List[str] = None

class WorkingContentStore:
    """Content store that actually works with existing content"""
    
    def __init__(self):
        self.content_items = {}
        self._load_from_unified_index()
    
    def _load_from_unified_index(self):
        """Load from the unified content index"""
        index_path = Path("unified_content/index.json")
        
        if not index_path.exists():
            print(f"Unified content index not found at {index_path}")
            return
        
        try:
            with open(index_path, 'r') as f:
                data = json.load(f)
            
            for item_data in data.get('content_items', []):
                content_item = ContentItem(
                    id=item_data.get('id', ''),
                    title=item_data.get('title', ''),
                    content_type=item_data.get('content_type', ''),
                    source_path=item_data.get('source_path', ''),
                    chunks=item_data.get('chunks', []),
                    metadata=item_data.get('metadata', {}),
                    created_at=item_data.get('created_at', 0.0),
                    keywords=item_data.get('keywords', []),
                    topic_assignments=item_data.get('topic_assignments', [])
                )
                
                self.content_items[content_item.id] = content_item
            
            print(f"Loaded {len(self.content_items)} items from unified content")
            
        except Exception as e:
            print(f"Error loading unified content: {e}")
    
    def get_all_content(self) -> List[ContentItem]:
        """Get all content items"""
        return list(self.content_items.values())
    
    def search_content_by_name(self, search_term: str) -> List[ContentItem]:
        """Search content by title"""
        results = []
        search_term_lower = search_term.lower()
        
        for content_item in self.content_items.values():
            if search_term_lower in content_item.title.lower():
                results.append(content_item)
        
        return results
