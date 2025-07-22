"""
Simple Unified Search - Like VS Code Search
Finds ALL content matching query keywords and appends everything for RAG
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time


@dataclass
class SearchMatch:
    """A single search match"""
    source_id: str
    source_type: str  # 'pdf' or 'youtube'
    source_path: str
    title: str
    matched_in: str  # 'title' or 'content'
    chunk_text: str
    chunk_id: str
    metadata: Dict[str, Any]


@dataclass
class UnifiedSearchResult:
    """Complete search results for RAG"""
    query: str
    total_matches: int
    title_matches: int
    content_matches: int
    pdf_sources: List[str]
    video_sources: List[str]
    all_content: str  # Combined content for RAG
    match_details: List[SearchMatch]


class SimpleUnifiedSearch:
    """
    VS Code-style search that finds ALL content matching keywords
    """
    
    def __init__(self, content_store):
        self.content_store = content_store
    
    def search_everything(self, query: str) -> UnifiedSearchResult:
        """
        Search like VS Code - find ALL files/chunks containing the keywords
        """
        start_time = time.time()
        
        # Extract search keywords
        keywords = self._extract_keywords(query)
        print(f"🔍 Searching for keywords: {keywords}")
        
        # Get all content
        all_content = self.content_store.get_all_content()
        
        # Group content by source to avoid duplicates
        sources = self._group_by_source(all_content)
        print(f"📁 Searching {len(sources)} unique sources...")
        
        title_matches = []
        content_matches = []
        
        # 1. Search titles first (like VS Code file names)
        for source_id, items in sources.items():
            representative_item = items[0]  # Use first item as representative
            
            if self._matches_title(representative_item.title, keywords):
                print(f"📄 TITLE MATCH: '{representative_item.title}'")
                
                # Add ALL chunks from this source
                for item in items:
                    for i, chunk in enumerate(item.chunks):
                        match = SearchMatch(
                            source_id=source_id,
                            source_type=item.content_type,
                            source_path=item.source_path or "",
                            title=item.title,
                            matched_in="title",
                            chunk_text=chunk.get('text', ''),
                            chunk_id=f"{item.id}_chunk_{i}",
                            metadata=chunk.get('metadata', {})
                        )
                        title_matches.append(match)
        
        # 2. Search content chunks (like VS Code content search)
        for source_id, items in sources.items():
            # Skip if already matched by title to avoid duplicates
            representative_item = items[0]
            if self._matches_title(representative_item.title, keywords):
                continue
                
            for item in items:
                for i, chunk in enumerate(item.chunks):
                    chunk_text = chunk.get('text', '')
                    
                    if self._matches_content(chunk_text, keywords):
                        print(f"📝 CONTENT MATCH: '{item.title}' (chunk {i})")
                        
                        match = SearchMatch(
                            source_id=source_id,
                            source_type=item.content_type,
                            source_path=item.source_path or "",
                            title=item.title,
                            matched_in="content",
                            chunk_text=chunk_text,
                            chunk_id=f"{item.id}_chunk_{i}",
                            metadata=chunk.get('metadata', {})
                        )
                        content_matches.append(match)
        
        # 3. Combine all matches and prepare for RAG
        all_matches = title_matches + content_matches
        combined_content = self._combine_for_rag(all_matches)
        
        # 4. Extract unique sources
        pdf_sources = list(set(m.source_path for m in all_matches if m.source_type == 'pdf' and m.source_path))
        video_sources = list(set(m.source_path for m in all_matches if m.source_type == 'youtube' and m.source_path))
        
        search_time = time.time() - start_time
        
        result = UnifiedSearchResult(
            query=query,
            total_matches=len(all_matches),
            title_matches=len(title_matches),
            content_matches=len(content_matches),
            pdf_sources=pdf_sources,
            video_sources=video_sources,
            all_content=combined_content,
            match_details=all_matches
        )
        
        print(f"✅ Search completed in {search_time:.2f}s")
        print(f"📊 Found {len(all_matches)} total matches ({len(title_matches)} title + {len(content_matches)} content)")
        print(f"📂 Sources: {len(pdf_sources)} PDFs + {len(video_sources)} videos")
        
        return result
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract meaningful keywords from query"""
        # Remove quotes and special chars, split by spaces
        clean_query = re.sub(r'[^\w\s]', ' ', query.lower())
        words = clean_query.split()
        
        # Filter out short words and common stop words
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'what', 'was', 'is', 'are', 'about'}
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]
        
        return keywords
    
    def _group_by_source(self, all_content) -> Dict[str, List]:
        """Group content items by source to avoid duplicates"""
        sources = {}
        for item in all_content:
            source_key = item.source_path or item.id
            if source_key not in sources:
                sources[source_key] = []
            sources[source_key].append(item)
        return sources
    
    def _matches_title(self, title: str, keywords: List[str]) -> bool:
        """Check if title contains any keywords"""
        title_lower = title.lower()
        return any(keyword in title_lower for keyword in keywords)
    
    def _matches_content(self, content: str, keywords: List[str]) -> bool:
        """Check if content contains any keywords"""
        content_lower = content.lower()
        return any(keyword in content_lower for keyword in keywords)
    
    def _combine_for_rag(self, matches: List[SearchMatch]) -> str:
        """Combine all matched content for RAG processing"""
        combined_parts = []
        
        # Group by source for better organization
        by_source = {}
        for match in matches:
            source_key = f"{match.title} ({match.source_type})"
            if source_key not in by_source:
                by_source[source_key] = []
            by_source[source_key].append(match)
        
        # Format for RAG
        for source_title, source_matches in by_source.items():
            combined_parts.append(f"\n=== {source_title} ===")
            
            for i, match in enumerate(source_matches):
                # Add metadata context if available
                context = ""
                if match.source_type == 'pdf' and 'page_number' in match.metadata:
                    context = f"[Page {match.metadata['page_number']}]"
                elif match.source_type == 'youtube' and 'timestamp' in match.metadata:
                    context = f"[{match.metadata['timestamp']}]"
                
                combined_parts.append(f"{context} {match.chunk_text.strip()}")
        
        return "\n\n".join(combined_parts)
    
    def get_search_summary(self, result: UnifiedSearchResult) -> str:
        """Get a human-readable summary of search results"""
        summary_parts = []
        
        summary_parts.append(f"Found {result.total_matches} matches for '{result.query}':")
        
        if result.title_matches > 0:
            summary_parts.append(f"📄 {result.title_matches} matches in file titles")
        
        if result.content_matches > 0:
            summary_parts.append(f"📝 {result.content_matches} matches in content")
        
        if result.pdf_sources:
            summary_parts.append(f"📕 PDF sources: {', '.join([s.split('/')[-1] for s in result.pdf_sources])}")
        
        if result.video_sources:
            summary_parts.append(f"📺 Video sources: {len(result.video_sources)} videos")
        
        return "\n".join(summary_parts)