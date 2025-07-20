"""
Simple Content Store
Uses actual paper titles and video titles instead of complex topic classification
"""

import os
import json
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
import time

@dataclass
class ContentItem:
    """Simple content item with actual names"""
    id: str
    title: str  # Actual paper title or video title
    content_type: str  # 'pdf' or 'youtube'
    source_path: str  # File path or URL
    chunks: List[Dict[str, Any]]  # Text chunks with embeddings
    metadata: Dict[str, Any]
    created_at: float
    
class SimpleContentStore:
    """
    Simple content organization using actual titles instead of topic classification
    """
    
    def __init__(self, storage_dir: str = "simple_content_store"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # Files
        self.content_index_file = self.storage_dir / "content_index.json"
        
        # In-memory cache
        self.content_items: Dict[str, ContentItem] = {}
        
        # Load existing content
        self._load_content_index()
    
    def add_pdf_content(self, pdf_path: str, title: str, chunks: List[Dict], metadata: Dict) -> str:
        """Add PDF content with actual paper title"""
        
        # Clean title for use as ID
        content_id = self._generate_content_id(title, 'pdf')
        
        content_item = ContentItem(
            id=content_id,
            title=title,
            content_type='pdf',
            source_path=pdf_path,
            chunks=chunks,
            metadata=metadata,
            created_at=time.time()
        )
        
        self.content_items[content_id] = content_item
        self._save_content_index()
        
        print(f"📄 Added PDF: '{title}' with {len(chunks)} chunks")
        return content_id
    
    def add_youtube_content(self, url: str, title: str, chunks: List[Dict], metadata: Dict) -> str:
        """Add YouTube content with actual video title"""
        
        # Clean title for use as ID
        content_id = self._generate_content_id(title, 'youtube')
        
        content_item = ContentItem(
            id=content_id,
            title=title,
            content_type='youtube',
            source_path=url,
            chunks=chunks,
            metadata=metadata,
            created_at=time.time()
        )
        
        self.content_items[content_id] = content_item
        self._save_content_index()
        
        print(f"🎥 Added YouTube: '{title}' with {len(chunks)} chunks")
        return content_id
    
    def search_content_by_name(self, search_term: str) -> List[ContentItem]:
        """Search content by title using regex"""
        
        # Create case-insensitive regex pattern
        pattern = re.compile(re.escape(search_term), re.IGNORECASE)
        
        matching_items = []
        for content_item in self.content_items.values():
            if pattern.search(content_item.title):
                matching_items.append(content_item)
        
        print(f"🔍 Found {len(matching_items)} items matching '{search_term}':")
        for item in matching_items:
            print(f"  - {item.content_type.upper()}: {item.title}")
        
        return matching_items
    
    def get_content_chunks(self, content_ids: List[str]) -> List[Dict]:
        """Get all chunks from specified content items"""
        
        all_chunks = []
        for content_id in content_ids:
            if content_id in self.content_items:
                content_item = self.content_items[content_id]
                for chunk in content_item.chunks:
                    # Add source info to chunk
                    chunk_with_source = {
                        **chunk,
                        'source_title': content_item.title,
                        'source_type': content_item.content_type,
                        'content_id': content_id
                    }
                    all_chunks.append(chunk_with_source)
        
        return all_chunks
    
    def analyze_new_information(self, search_term: str, new_content_chunks: List[Dict]) -> Dict[str, Any]:
        """
        Compare new content with existing content matching search term
        Find what's new/different
        """
        
        # Find existing content matching the search term
        existing_items = self.search_content_by_name(search_term)
        existing_chunks = self.get_content_chunks([item.id for item in existing_items])
        
        print(f"🔬 Analyzing new information about '{search_term}'")
        print(f"📚 Existing content: {len(existing_chunks)} chunks from {len(existing_items)} sources")
        print(f"🆕 New content: {len(new_content_chunks)} chunks")
        
        # For now, simple text-based comparison
        # TODO: Use embeddings for semantic similarity
        existing_texts = {chunk.get('text', '') for chunk in existing_chunks}
        
        new_information = []
        for chunk in new_content_chunks:
            chunk_text = chunk.get('text', '')
            
            # Check if this information is truly new
            is_new = True
            for existing_text in existing_texts:
                # Simple overlap check (can be improved with embeddings)
                if len(set(chunk_text.lower().split()) & set(existing_text.lower().split())) > 5:
                    is_new = False
                    break
            
            if is_new:
                new_information.append(chunk)
        
        return {
            'search_term': search_term,
            'existing_sources': [item.title for item in existing_items],
            'new_information_chunks': new_information,
            'total_existing_chunks': len(existing_chunks),
            'total_new_chunks': len(new_information),
            'analysis_summary': f"Found {len(new_information)} new chunks about '{search_term}' not covered in existing {len(existing_items)} sources"
        }
    
    def get_all_content(self) -> List[ContentItem]:
        """Get all content items"""
        return list(self.content_items.values())
    
    def get_content_by_type(self, content_type: str) -> List[ContentItem]:
        """Get content by type (pdf or youtube)"""
        return [item for item in self.content_items.values() if item.content_type == content_type]
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        pdf_count = len(self.get_content_by_type('pdf'))
        youtube_count = len(self.get_content_by_type('youtube'))
        
        total_chunks = sum(len(item.chunks) for item in self.content_items.values())
        
        return {
            'total_items': len(self.content_items),
            'pdf_count': pdf_count,
            'youtube_count': youtube_count,
            'total_chunks': total_chunks,
            'storage_dir': str(self.storage_dir)
        }
    
    def _generate_content_id(self, title: str, content_type: str) -> str:
        """Generate clean content ID from title"""
        
        # Clean title: remove special chars, limit length
        clean_title = re.sub(r'[^\w\s-]', '', title.strip())
        clean_title = re.sub(r'\s+', '_', clean_title)
        clean_title = clean_title.lower()[:50]  # Limit length
        
        # Add type prefix and ensure uniqueness
        base_id = f"{content_type}_{clean_title}"
        
        # Ensure uniqueness
        counter = 1
        content_id = base_id
        while content_id in self.content_items:
            content_id = f"{base_id}_{counter}"
            counter += 1
        
        return content_id
    
    def _load_content_index(self):
        """Load content index from file"""
        if self.content_index_file.exists():
            try:
                with open(self.content_index_file, 'r') as f:
                    data = json.load(f)
                
                # Convert back to ContentItem objects
                for item_data in data.get('content_items', []):
                    content_item = ContentItem(**item_data)
                    self.content_items[content_item.id] = content_item
                
                print(f"📂 Loaded {len(self.content_items)} content items from storage")
                
            except Exception as e:
                print(f"⚠️ Error loading content index: {e}")
                self.content_items = {}
    
    def _save_content_index(self):
        """Save content index to file"""
        try:
            data = {
                'content_items': [asdict(item) for item in self.content_items.values()],
                'last_updated': time.time()
            }
            
            with open(self.content_index_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"⚠️ Error saving content index: {e}")

# Test the simple content store
if __name__ == "__main__":
    store = SimpleContentStore()
    
    # Test adding content
    store.add_pdf_content(
        pdf_path="darkmatter.pdf",
        title="Dark Matter and Galactic Halos - A Quantum Approach",
        chunks=[{"text": "Dark matter constitutes...", "embedding": [0.1, 0.2, 0.3]}],
        metadata={"pages": 53, "author": "A. D. Ernest"}
    )
    
    # Test searching
    results = store.search_content_by_name("dark matter")
    print(f"Search results: {len(results)}")
    
    # Test stats
    stats = store.get_storage_stats()
    print(f"Storage stats: {stats}")