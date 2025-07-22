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
        
        # Get all content and group by source
        all_content = self.content_store.get_all_content()
        
        # Group content by source_path to get unique files
        source_files = {}
        for item in all_content:
            source_key = item.source_path or item.id
            if source_key not in source_files:
                source_files[source_key] = item

        print(f"Looking for file mention in: '{query}'")
        print(f"Available files: {list(source_files.keys())}")
        
        # First, try to match exact filenames from query
        if any(ext in query_lower for ext in ['.pdf', '.doc', '.txt']):
            # Extract potential filename from query
            filename_matches = re.findall(r'[\w\-]+\.(pdf|doc|txt|docx)', query_lower)
            
            for potential_filename in filename_matches:
                print(f"Searching for filename: '{potential_filename}'")
                
                # Check if any source path contains this filename
                for source_path, content_item in source_files.items():
                    if source_path and potential_filename in source_path.lower():
                        print(f"Found exact filename match: '{potential_filename}' in '{source_path}'")
                        return content_item
                    
                    # Also check if the filename (without extension) matches the source ID
                    filename_without_ext = potential_filename.rsplit('.', 1)[0]
                    if filename_without_ext in content_item.id.lower():
                        print(f"Found ID match: '{filename_without_ext}' matches '{content_item.id}'")
                        return content_item
        
        # Fallback: NER-based fuzzy matching using extracted keywords and topics
        potential_matches = []
        
        # Use NER-extracted keywords and topics from content metadata
        for source_path, content_item in source_files.items():
            title = content_item.title.lower()
            title_words = title.replace('.pdf', '').replace('.', ' ').split()
            
            # Check regular title word matches
            for word in title_words:
                if len(word) > 3 and word in query_lower:
                    print(f"Found title match: '{word}' in '{content_item.title}'")
                    potential_matches.append((word, content_item))
            
            # NER-based matching using extracted keywords and topics
            content_keywords = getattr(content_item, 'keywords', [])
            content_topics = getattr(content_item, 'topic_assignments', [])
            
            # Extract meaningful terms from query
            query_terms = self._extract_ner_terms(query_lower)
            
            # Check fuzzy matches with NER keywords
            for query_term in query_terms:
                best_keyword_match = 0.0
                matched_keyword = ""
                
                for keyword in content_keywords:
                    # Use fuzzy string matching
                    from difflib import SequenceMatcher
                    similarity = SequenceMatcher(None, query_term, keyword.lower()).ratio()
                    if similarity > best_keyword_match and similarity > 0.7:
                        best_keyword_match = similarity
                        matched_keyword = keyword
                
                if best_keyword_match > 0.7:
                    # Add NER keyword match with similarity scoring
                    ner_match = f"ner_{matched_keyword}_{best_keyword_match:.2f}"
                    print(f"Found NER keyword match: '{query_term}' ~ '{matched_keyword}' ({best_keyword_match:.2f}) in '{content_item.title}'")
                    potential_matches.append((ner_match, content_item))
                
                # Check topic matches
                for topic in content_topics:
                    if query_term in topic.lower() or topic.lower() in query_term:
                        topic_match = f"topic_{topic}"
                        print(f"Found topic match: '{query_term}' matches topic '{topic}' in '{content_item.title}'")
                        potential_matches.append((topic_match, content_item))
        
        # If we found matches, decide whether it's a single-file or multi-file query
        if potential_matches:
            # Check if query likely needs multiple resources
            multi_resource_indicators = [
                'provide', 'enough', 'power', 'supply', 'generate', 'produce',
                'compare', 'contrast', 'versus', 'vs', 'different',
                'both', 'combined', 'together', 'integrate'
            ]
            
            has_multi_indicator = any(indicator in query_lower for indicator in multi_resource_indicators)
            
            # If query asks about providing/powering/comparing, don't limit to single file
            if has_multi_indicator and len(potential_matches) == 1:
                print(f"Multi-resource query detected, not limiting to single file")
                return None  # Let it fall through to search_all_content
            
            # If only one match and no multi-resource indicators, return it
            if len(potential_matches) == 1:
                return potential_matches[0][1]
            
            # Multiple matches - find the best match by counting matching words
            if len(potential_matches) > 1:
                # Score matches by number of matching words and word quality
                match_scores = {}
                for word, content_item in potential_matches:
                    if content_item.id not in match_scores:
                        match_scores[content_item.id] = {'item': content_item, 'score': 0, 'words': []}
                    match_scores[content_item.id]['score'] += len(word)  # Longer words = higher score
                    match_scores[content_item.id]['words'].append(word)
                
                # Find highest scoring match
                best_match = max(match_scores.values(), key=lambda x: x['score'])
                
                # If the best match has significantly more/better words, use it
                # Lower threshold for better specificity with larger document collections
                if best_match['score'] > 3:  # Lowered threshold for good match
                    print(f"Best title match: '{best_match['item'].title}' (score: {best_match['score']})")
                    return best_match['item']
            
            # If no clear best match, use broader search
            print(f"Multiple title matches found ({len(potential_matches)}), using broader search")
            return None
        
        print("No file mention found")
        return None
    
    def _get_file_sources(self, content_item) -> Tuple[List[str], List[str], str]:
        """Get source paths for a specific file"""
        pdf_files = []
        youtube_urls = []
        
        if content_item.content_type == 'pdf':
            pdf_files = [content_item.source_path] if content_item.source_path else []
        elif content_item.content_type == 'youtube':
            youtube_urls = [content_item.source_path] if content_item.source_path else []
        
        explanation = f"File-specific search: Found '{content_item.title}'"
        
        print(f"Routing to specific file: {content_item.title}")
        print(f"PDFs: {pdf_files}")
        print(f"Videos: {youtube_urls}")
        
        return pdf_files, youtube_urls, explanation
    
    def _search_all_content(self, query: str) -> Tuple[List[str], List[str], str]:
        """Search all content for query terms (grouped by unique files)"""
        query_lower = query.lower()
        
        # Get all content and group by source
        all_content = self.content_store.get_all_content()
        
        # Group content by source_path to avoid duplicates
        source_files = {}
        for item in all_content:
            source_key = item.source_path or item.id
            if source_key not in source_files:
                source_files[source_key] = item

        relevant_pdfs = set()  # Use sets to avoid duplicates
        relevant_videos = set()
        matches_found = {}
        
        print(f"Searching {len(source_files)} unique files for: '{query}'")
        
        # Simple keyword matching in content
        for source_path, content_item in source_files.items():
            # Check title and content chunks for query terms
            title_text = content_item.title.lower()
            content_text = ""
            
            # Get first few chunks to check relevance - handle dict format
            chunks = content_item.chunks
            if isinstance(chunks, dict) and 'chunks' in chunks:
                actual_chunks = chunks['chunks'][:3]  # Check first 3 chunks only
            else:
                actual_chunks = chunks[:3] if hasattr(chunks, '__getitem__') else []
            
            for chunk in actual_chunks:
                chunk_text = chunk.get('text', '') if isinstance(chunk, dict) else str(chunk)
                content_text += chunk_text.lower() + " "
            
            # Smart relevance check - both keyword matching AND semantic similarity
            query_words = [word for word in query_lower.split() if len(word) > 3]
            keyword_matches = sum(1 for word in query_words if word in title_text or word in content_text)
            
            # Also check for semantic similarity using embeddings
            semantic_score = self._calculate_semantic_similarity(query_lower, title_text + " " + content_text)
            
            # Dynamic semantic threshold based on document collection size
            if len(source_files) <= 3:
                semantic_threshold = 0.25
            elif len(source_files) <= 5:
                semantic_threshold = 0.4  # Higher threshold for larger collections
            else:
                semantic_threshold = 0.5  # Even higher for 6+ documents
            
            # Content is relevant if it has keyword matches OR high semantic similarity
            if keyword_matches > 0 or semantic_score > semantic_threshold:
                total_score = keyword_matches + (semantic_score * 10)  # Weight semantic score
                matches_found[content_item.title] = total_score
                if content_item.content_type == 'pdf' and content_item.source_path:
                    relevant_pdfs.add(content_item.source_path)
                elif content_item.content_type == 'youtube' and content_item.source_path:
                    relevant_videos.add(content_item.source_path)
        
        # Convert sets back to lists
        relevant_pdfs = list(relevant_pdfs)
        relevant_videos = list(relevant_videos)
        
        # Show only top matches to reduce noise
        if matches_found:
            top_matches = sorted(matches_found.items(), key=lambda x: x[1], reverse=True)[:5]
            for title, match_count in top_matches:
                print(f"Found {match_count} matches in '{title}'")
        
        if relevant_pdfs or relevant_videos:
            explanation = f"Content search: Found {len(relevant_pdfs)} unique PDFs and {len(relevant_videos)} unique videos with relevant content"
        else:
            explanation = "General search: No specific content found, limiting to top content"
            # If no relevant content found, return only a few top files to avoid overload
            pdf_count = 0
            video_count = 0
            for source_path, content_item in source_files.items():
                if content_item.content_type == 'pdf' and pdf_count < 3 and content_item.source_path:
                    relevant_pdfs.append(content_item.source_path)
                    pdf_count += 1
                elif content_item.content_type == 'youtube' and video_count < 2 and content_item.source_path:
                    relevant_videos.append(content_item.source_path)
                    video_count += 1
        
        print(f"Total PDFs: {len(relevant_pdfs)}")
        print(f"Total Videos: {len(relevant_videos)}")
        
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
                print(f"Found YouTube URL in query: {full_url}")
                return full_url
        
        return None
    
    def _handle_direct_youtube_url(self, youtube_url: str) -> Tuple[List[str], List[str], str]:
        """Handle direct YouTube URL in query"""
        print(f"Processing direct YouTube URL: {youtube_url}")
        
        # Return the URL for processing
        explanation = f"Direct YouTube URL: Processing new video from URL"
        
        return [], [youtube_url], explanation
    
    def _has_multiple_topics(self, query: str) -> bool:
        """Check if query asks about multiple topics that should be searched together"""
        query_lower = query.lower()
        
        # Common patterns that indicate multiple topics
        multiple_topic_patterns = [
            ('dark matter', 'antimatter'),
            ('antimatter', 'matter'),
            # Removed ('dark matter', 'matter') as this is too broad
            ('difference', 'between'),
            ('compare', 'contrast'),
            # Climate/Energy multi-resource patterns
            ('solar', 'carbon'),
            ('solar', 'sequestration'),
            ('panel', 'capture'),
            ('efficiency', 'carbon'),
            ('energy', 'carbon'),
            ('renewable', 'sequestration'),
            ('grid', 'carbon'),
            ('provide', 'energy'),
            ('enough', 'energy'),
            ('power', 'carbon'),
            # Technical cross-references
            ('kwh', 'efficiency'),
            ('ton', 'efficiency'),
            ('dac', 'solar'),
            ('capture', 'panel'),
            ('storage', 'generation'),
            ('weathering', 'renewable'),
            ('payback', 'sequestration'),
            ('how', 'different'),
            ('what', 'difference')
        ]
        
        # Check if query contains multiple topic keywords
        for pattern in multiple_topic_patterns:
            if all(keyword in query_lower for keyword in pattern):
                print(f"Multiple topics detected: {pattern}")
                return True
        
        return False
    
    def _extract_ner_terms(self, query: str) -> List[str]:
        """Extract meaningful NER-like terms from query"""
        import re
        
        # Remove common stop words
        stop_words = {
            'what', 'is', 'the', 'how', 'do', 'does', 'can', 'will', 'would', 'could',
            'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'about',
            'and', 'or', 'but', 'so', 'if', 'then', 'than', 'when', 'where',
            'document', 'documents', 'pdf', 'file', 'according', 'mentioned'
        }
        
        # Extract words and multi-word terms
        words = re.findall(r'\b\w+\b', query)
        meaningful_terms = []
        
        # Single words
        for word in words:
            if len(word) > 2 and word.lower() not in stop_words:
                meaningful_terms.append(word.lower())
        
        # Multi-word technical terms (common in scientific text)
        multi_word_patterns = [
            r'\b\w+\s+\w+\b'         # 2-word terms
        ]
        
        for pattern in multi_word_patterns:
            matches = re.findall(pattern, query)
            for match in matches:
                if not any(stop in match.lower() for stop in ['what is', 'how to', 'can you']):
                    meaningful_terms.append(match.lower())
        
        return list(set(meaningful_terms))  # Remove duplicates
    
    def _search_multiple_topics(self, query: str) -> Tuple[List[str], List[str], str]:
        """Search for content covering multiple topics"""
        query_lower = query.lower()
        
        print(f"Searching for multiple topics in: '{query}'")
        
        # Get all content
        all_content = self.content_store.get_all_content()
        
        relevant_pdfs = []
        relevant_videos = []
        matched_topics = []
        
        # Use NER-based fuzzy routing for multi-topic detection
        try:
            from .ner_fuzzy_router import NERFuzzyRouter
            ner_router = NERFuzzyRouter(self.content_store)
            
            # Use NER router for intelligent topic detection
            pdf_files, youtube_urls, ner_explanation = ner_router.route_query(query)
            
            if pdf_files or youtube_urls:
                print(f"NER fuzzy routing found: {len(pdf_files)} PDFs, {len(youtube_urls)} videos")
                print(f"NER explanation: {ner_explanation}")
                return pdf_files, youtube_urls, f"Multi-topic NER routing: {ner_explanation}"
            
        except ImportError:
            print("NER fuzzy router not available, using fallback topic detection")
        
        # Fallback: Build dynamic topic keywords from content metadata
        topic_keywords = {}
        for content_item in all_content:
            # Use NER-detected primary topics if available
            primary_topics = getattr(content_item, 'topic_assignments', [])
            keywords = getattr(content_item, 'keywords', [])
            
            for topic in primary_topics:
                if topic not in topic_keywords:
                    topic_keywords[topic] = set()
                # Add content keywords to this topic
                topic_keywords[topic].update(keywords[:10])  # Top 10 keywords per topic
        
        # Convert sets to lists and add fallback topics if none found
        for topic, keyword_set in topic_keywords.items():
            topic_keywords[topic] = list(keyword_set)
        
        # Fallback topics if NER hasn't detected any yet
        if not topic_keywords:
            topic_keywords = {
                'solar_energy': ['solar', 'panel', 'photovoltaic', 'silicon', 'efficiency'],
                'carbon_sequestration': ['carbon', 'sequestration', 'capture', 'dac', 'co2'],
                'energy_systems': ['energy', 'power', 'kwh', 'generation', 'electricity']
            }
        
        # Find content for each topic
        for topic, keywords in topic_keywords.items():
            for content_item in all_content:
                # Check title and content for topic keywords
                title_text = content_item.title.lower()
                content_text = ""
                
                # Get first few chunks to check relevance - handle dict format
                chunks = content_item.chunks
                if isinstance(chunks, dict) and 'chunks' in chunks:
                    actual_chunks = chunks['chunks'][:3]  # Check first 3 chunks only
                else:
                    actual_chunks = chunks[:3] if hasattr(chunks, '__getitem__') else []
                
                for chunk in actual_chunks:
                    chunk_text = chunk.get('text', '') if isinstance(chunk, dict) else str(chunk)
                    content_text += chunk_text.lower() + " "
                
                # Check if any topic keywords appear OR semantic similarity to the topic
                keyword_match = any(keyword in title_text or keyword in content_text for keyword in keywords)
                
                # Also check semantic similarity to the topic concept
                topic_query = " ".join(keywords)
                semantic_match = self._calculate_semantic_similarity(topic_query, title_text + " " + content_text) > 0.25
                
                if keyword_match or semantic_match:
                    if topic not in matched_topics:
                        matched_topics.append(topic)
                        print(f"Found {topic} content in: '{content_item.title}'")
                    
                    if content_item.content_type == 'pdf' and content_item.source_path not in relevant_pdfs:
                        relevant_pdfs.append(content_item.source_path)
                    elif content_item.content_type == 'youtube' and content_item.source_path not in relevant_videos:
                        relevant_videos.append(content_item.source_path)
        
        if relevant_pdfs or relevant_videos:
            explanation = f"Multi-topic search: Found content about {', '.join(matched_topics)} ({len(relevant_pdfs)} PDFs, {len(relevant_videos)} videos)"
        else:
            explanation = "Multi-topic search: No relevant content found for multiple topics"
        
        print(f"Multi-topic results: {len(relevant_pdfs)} PDFs, {len(relevant_videos)} videos")
        print(f"Topics found: {matched_topics}")
        
        return relevant_pdfs, relevant_videos, explanation
    
    def _calculate_semantic_similarity(self, query: str, content: str) -> float:
        """Calculate semantic similarity between query and content using embeddings"""
        try:
            # Import here to avoid circular imports
            from sentence_transformers import SentenceTransformer
            import numpy as np
            
            # Use a lightweight model for quick similarity calculation
            if not hasattr(self, '_similarity_model'):
                self._similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Handle empty content
            if not content.strip() or not query.strip():
                return 0.0
            
            # Limit content length for efficiency
            content = content[:1000] if len(content) > 1000 else content
            
            # Get embeddings
            query_embedding = self._similarity_model.encode([query])
            content_embedding = self._similarity_model.encode([content])
            
            # Calculate cosine similarity
            similarity = np.dot(query_embedding[0], content_embedding[0]) / (
                np.linalg.norm(query_embedding[0]) * np.linalg.norm(content_embedding[0])
            )
            
            return float(similarity)
            
        except Exception as e:
            print(f"Error calculating semantic similarity: {e}")
            return 0.0