"""
Enhanced Search Service for KnowledgeAgent
Integrates trie-based search with content routing
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time

from search.trie_search import ContentTrieSearch, SearchResult
from storage.unified_content_store import UnifiedContentStore


@dataclass 
class EnhancedSearchResult:
    """Enhanced search result with routing information"""
    content_id: str
    title: str
    content_type: str
    source_path: str
    match_score: float
    matched_terms: List[str]
    preview: str
    routing_strategy: str
    confidence: float


class KnowledgeAgentSearchService:
    """
    Advanced search service that combines:
    1. Trie-based fast search
    2. Content routing logic
    3. Fuzzy matching
    4. Smart ranking
    """
    
    def __init__(self):
        self.trie_search = ContentTrieSearch()
        self.content_store = SimpleContentStore()
        self.last_index_update = 0
        self.index_cache_duration = 300  # 5 minutes
        self._initialize_search_index()
    
    def _initialize_search_index(self):
        """Initialize the search index with current content"""
        try:
            all_content = self.content_store.get_all_content()
            
            print(f"🔍 Indexing {len(all_content)} items for search...")
            
            for item in all_content:
                # Extract preview from first chunk
                preview = ""
                if item.chunks:
                    preview = item.chunks[0].get('text', '')[:200]
                
                # Extract keywords from title and content
                keywords = self._extract_keywords(item.title, preview)
                
                self.trie_search.add_content(
                    content_id=item.id,
                    title=item.title,
                    content_type=item.content_type,
                    source_path=item.source_path,
                    preview=preview,
                    keywords=keywords
                )
            
            self.last_index_update = time.time()
            print(f"✅ Search index initialized with {len(all_content)} items")
            
        except Exception as e:
            print(f"❌ Failed to initialize search index: {e}")
    
    def _extract_keywords(self, title: str, content: str) -> List[str]:
        """Extract important keywords from title and content"""
        import re
        
        # Combine title and content
        text = f"{title} {content}"
        
        # Find capitalized words (likely important terms)
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        
        # Find technical terms (words with numbers, hyphens, underscores)
        technical = re.findall(r'\b\w*[0-9_-]\w*\b', text)
        
        # Find quoted terms
        quoted = re.findall(r'"([^"]*)"', text)
        
        keywords = capitalized + technical + quoted
        
        # Clean and deduplicate
        keywords = [kw.strip() for kw in keywords if len(kw.strip()) > 2]
        return list(set(keywords))
    
    def refresh_index_if_needed(self):
        """Refresh search index if content has changed"""
        current_time = time.time()
        if current_time - self.last_index_update > self.index_cache_duration:
            print("🔄 Refreshing search index...")
            self.trie_search = ContentTrieSearch()  # Reset
            self._initialize_search_index()
    
    def search(self, query: str, max_results: int = 10) -> List[EnhancedSearchResult]:
        """
        Enhanced search that combines trie search with smart routing
        """
        self.refresh_index_if_needed()
        
        if not query.strip():
            return []
        
        print(f"🔍 Enhanced search for: '{query}'")
        
        # Get trie search results
        trie_results = self.trie_search.search(query, max_results * 2)
        
        # Enhance results with routing information
        enhanced_results = []
        
        for result in trie_results:
            # Determine routing strategy
            routing_strategy, confidence = self._determine_routing_strategy(query, result)
            
            enhanced_result = EnhancedSearchResult(
                content_id=result.content_id,
                title=result.title,
                content_type=result.content_type,
                source_path=result.source_path,
                match_score=result.match_score,
                matched_terms=result.matched_terms,
                preview=result.preview,
                routing_strategy=routing_strategy,
                confidence=confidence
            )
            
            enhanced_results.append(enhanced_result)
        
        # Sort by combined score (match_score * confidence)
        enhanced_results.sort(key=lambda x: x.match_score * x.confidence, reverse=True)
        
        return enhanced_results[:max_results]
    
    def _determine_routing_strategy(self, query: str, result: SearchResult) -> Tuple[str, float]:
        """Determine the best routing strategy for this result"""
        query_lower = query.lower()
        title_lower = result.title.lower()
        
        # Direct file mention (highest priority)
        if any(ext in query_lower for ext in ['.pdf', '.doc', '.txt']):
            if any(ext in title_lower for ext in ['.pdf', '.doc', '.txt']):
                return "direct_file", 0.95
        
        # Title keyword match
        query_words = set(query_lower.split())
        title_words = set(title_lower.split())
        overlap = len(query_words.intersection(title_words))
        
        if overlap > 0:
            confidence = min(0.9, 0.6 + 0.1 * overlap)
            return "title_match", confidence
        
        # Content type priority
        if result.content_type == "pdf" and any(word in query_lower for word in ["paper", "research", "study", "document"]):
            return "content_type_match", 0.7
        
        if result.content_type == "youtube" and any(word in query_lower for word in ["video", "watch", "explain", "tutorial"]):
            return "content_type_match", 0.7
        
        # Keyword match in content
        if result.matched_terms:
            return "keyword_match", 0.6
        
        # Default semantic search
        return "semantic_search", 0.4
    
    def suggest_queries(self, partial_query: str, max_suggestions: int = 5) -> List[str]:
        """Get query suggestions based on partial input"""
        self.refresh_index_if_needed()
        return self.trie_search.suggest_completions(partial_query, max_suggestions)
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get comprehensive search statistics"""
        trie_stats = self.trie_search.get_statistics()
        content_stats = self.content_store.get_storage_stats()
        
        return {
            "search_index": trie_stats,
            "content_store": content_stats,
            "last_update": self.last_index_update,
            "index_age_minutes": (time.time() - self.last_index_update) / 60
        }
    
    def find_similar_content(self, content_id: str, max_results: int = 5) -> List[EnhancedSearchResult]:
        """Find content similar to the given content"""
        try:
            # Get the source content
            all_content = self.content_store.get_all_content()
            source_item = None
            
            for item in all_content:
                if item.id == content_id:
                    source_item = item
                    break
            
            if not source_item:
                return []
            
            # Use title words as search query
            title_words = self.trie_search._extract_words(source_item.title)
            query = " ".join(title_words[:3])  # Use first 3 meaningful words
            
            results = self.search(query, max_results + 1)
            
            # Filter out the source content itself
            similar_results = [r for r in results if r.content_id != content_id]
            
            return similar_results[:max_results]
            
        except Exception as e:
            print(f"Error finding similar content: {e}")
            return []