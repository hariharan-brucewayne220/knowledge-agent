"""
Simple Smart Router - Direct and Efficient
Just does what we need: find the file, get the content, answer the query
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from difflib import SequenceMatcher

class SimpleSmartRouter:
    """
    Super simple router:
    1. Check if query mentions a filename -> search only that file
    2. If no filename, search all content with the query terms
    3. Let OpenAI handle the "why" questions with the retrieved content
    """
    
    def __init__(self, content_store):
        self.content_store = content_store
    
    def route_query(self, query: str) -> Tuple[List[str], List[str], str]:
        """
        Simple routing logic
        
        Returns:
            (pdf_files, youtube_urls, explanation)
        """
        query_lower = query.lower()
        
        # Step 0: Check if query contains a YouTube URL (highest priority)
        youtube_url = self._extract_youtube_url(query)
        if youtube_url:
            return self._handle_direct_youtube_url(youtube_url)
        
        # Step 1: Check if query mentions multiple topics (like "dark matter AND antimatter")
        if self._has_multiple_topics(query):
            return self._search_multiple_topics(query)
        
        # Step 2: Check if query mentions a specific file
        mentioned_file = self._find_mentioned_file(query)
        if mentioned_file:
            return self._get_file_sources(mentioned_file)
        
        # Step 3: No specific file mentioned, search all content
        return self._search_all_content(query)
    
    def _find_mentioned_file(self, query: str) -> Optional[Any]:
        """Find if query mentions a specific file"""
        query_lower = query.lower()
        
        # Get all content
        all_content = self.content_store.get_all_content()
        
        print(f"🔍 Looking for file mention in: '{query}'")
        print(f"📚 Available files: {[item.title for item in all_content]}")
        
        for content_item in all_content:
            title = content_item.title.lower()
            
            # Simple checks - if any part of the title appears in query
            title_words = title.replace('.pdf', '').replace('.', ' ').split()
            
            for word in title_words:
                if len(word) > 3 and word in query_lower:  # Skip short words
                    print(f"✅ Found file mention: '{word}' matches '{content_item.title}'")
                    return content_item
            
            # Also check direct filename mentions
            if any(ext in query_lower for ext in ['.pdf', '.doc', '.txt']):
                # Extract potential filename from query
                filename_match = re.search(r'[\w\-]+\.(pdf|doc|txt|docx)', query_lower)
                if filename_match:
                    potential_filename = filename_match.group()
                    if potential_filename in title:
                        print(f"✅ Found filename match: '{potential_filename}' matches '{content_item.title}'")
                        return content_item
        
        print("❌ No file mention found")
        return None
    
    def _get_file_sources(self, content_item) -> Tuple[List[str], List[str], str]:
        """Get source paths for a specific file"""
        pdf_files = []
        youtube_urls = []
        
        if content_item.content_type == 'pdf':
            pdf_files = [content_item.source_path] if content_item.source_path else []
        elif content_item.content_type == 'youtube':
            youtube_urls = [content_item.source_path] if content_item.source_path else []
        
        explanation = f"🎯 **File-specific search**: Found '{content_item.title}'"
        
        print(f"📄 Routing to specific file: {content_item.title}")
        print(f"📂 PDFs: {pdf_files}")
        print(f"🎥 Videos: {youtube_urls}")
        
        return pdf_files, youtube_urls, explanation
    
    def _search_all_content(self, query: str) -> Tuple[List[str], List[str], str]:
        """Search all content for query terms"""
        query_lower = query.lower()
        
        # Get all content
        all_content = self.content_store.get_all_content()
        
        relevant_pdfs = []
        relevant_videos = []
        
        print(f"🔍 Searching all content for: '{query}'")
        
        # Simple keyword matching in content
        for content_item in all_content:
            # Check title and content chunks for query terms
            title_text = content_item.title.lower()
            content_text = ""
            
            # Get first few chunks to check relevance
            for chunk in content_item.chunks[:5]:  # Check first 5 chunks
                content_text += chunk.get('text', '').lower() + " "
            
            # Simple relevance check - if query words appear in content
            query_words = [word for word in query_lower.split() if len(word) > 3]
            matches = sum(1 for word in query_words if word in title_text or word in content_text)
            
            if matches > 0:
                print(f"📄 Found {matches} matches in '{content_item.title}'")
                if content_item.content_type == 'pdf':
                    relevant_pdfs.append(content_item.source_path)
                elif content_item.content_type == 'youtube':
                    relevant_videos.append(content_item.source_path)
        
        if relevant_pdfs or relevant_videos:
            explanation = f"🔍 **Content search**: Found {len(relevant_pdfs)} PDFs and {len(relevant_videos)} videos with relevant content"
        else:
            explanation = "🌐 **General search**: No specific content found, searching all available content"
            # If no relevant content found, return all content
            for content_item in all_content:
                if content_item.content_type == 'pdf':
                    relevant_pdfs.append(content_item.source_path)
                elif content_item.content_type == 'youtube':
                    relevant_videos.append(content_item.source_path)
        
        print(f"📂 Total PDFs: {len(relevant_pdfs)}")
        print(f"🎥 Total Videos: {len(relevant_videos)}")
        
        return relevant_pdfs, relevant_videos, explanation
    
    def _extract_youtube_url(self, query: str) -> Optional[str]:
        """Extract YouTube URL from query if present"""
        import re
        
        # YouTube URL patterns
        youtube_patterns = [
            r'https?://(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]+)',
            r'https?://(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]+)',
            r'https?://youtu\.be/([a-zA-Z0-9_-]+)',
            r'https?://(?:www\.)?youtube\.com/v/([a-zA-Z0-9_-]+)'
        ]
        
        for pattern in youtube_patterns:
            match = re.search(pattern, query)
            if match:
                video_id = match.group(1)
                full_url = f"https://www.youtube.com/watch?v={video_id}"
                print(f"🎥 Found YouTube URL in query: {full_url}")
                return full_url
        
        return None
    
    def _handle_direct_youtube_url(self, youtube_url: str) -> Tuple[List[str], List[str], str]:
        """Handle direct YouTube URL in query"""
        print(f"🎯 Processing direct YouTube URL: {youtube_url}")
        
        # Return the URL for processing
        explanation = f"🎥 **Direct YouTube URL**: Processing new video from URL"
        
        return [], [youtube_url], explanation
    
    def _has_multiple_topics(self, query: str) -> bool:
        """Check if query asks about multiple topics that should be searched together"""
        query_lower = query.lower()
        
        # Common patterns that indicate multiple topics
        multiple_topic_patterns = [
            ('dark matter', 'antimatter'),
            ('dark matter', 'matter'),
            ('antimatter', 'matter'),
            ('difference', 'between'),
            ('compare', 'contrast'),
            ('how', 'different'),
            ('what', 'difference')
        ]
        
        # Check if query contains multiple topic keywords
        for pattern in multiple_topic_patterns:
            if all(keyword in query_lower for keyword in pattern):
                print(f"🔍 Multiple topics detected: {pattern}")
                return True
        
        return False
    
    def _search_multiple_topics(self, query: str) -> Tuple[List[str], List[str], str]:
        """Search for content covering multiple topics"""
        query_lower = query.lower()
        
        print(f"🔍 Searching for multiple topics in: '{query}'")
        
        # Get all content
        all_content = self.content_store.get_all_content()
        
        relevant_pdfs = []
        relevant_videos = []
        matched_topics = []
        
        # Define topic keywords
        topic_keywords = {
            'dark_matter': ['dark matter', 'dark energy', 'galactic', 'halo'],
            'antimatter': ['antimatter', 'positron', 'antiparticle'],
            'matter': ['matter', 'particle', 'atom', 'physics']
        }
        
        # Find content for each topic
        for topic, keywords in topic_keywords.items():
            for content_item in all_content:
                # Check title and content for topic keywords
                title_text = content_item.title.lower()
                content_text = ""
                
                # Get first few chunks to check relevance
                for chunk in content_item.chunks[:3]:
                    content_text += chunk.get('text', '').lower() + " "
                
                # Check if any topic keywords appear
                if any(keyword in title_text or keyword in content_text for keyword in keywords):
                    if topic not in matched_topics:
                        matched_topics.append(topic)
                        print(f"📄 Found {topic} content in: '{content_item.title}'")
                    
                    if content_item.content_type == 'pdf' and content_item.source_path not in relevant_pdfs:
                        relevant_pdfs.append(content_item.source_path)
                    elif content_item.content_type == 'youtube' and content_item.source_path not in relevant_videos:
                        relevant_videos.append(content_item.source_path)
        
        if relevant_pdfs or relevant_videos:
            explanation = f"🔍 **Multi-topic search**: Found content about {', '.join(matched_topics)} ({len(relevant_pdfs)} PDFs, {len(relevant_videos)} videos)"
        else:
            explanation = "🔍 **Multi-topic search**: No relevant content found for multiple topics"
        
        print(f"📊 Multi-topic results: {len(relevant_pdfs)} PDFs, {len(relevant_videos)} videos")
        print(f"📋 Topics found: {matched_topics}")
        
        return relevant_pdfs, relevant_videos, explanation