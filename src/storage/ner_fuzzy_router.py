"""
NER-Based Fuzzy Search Router
Uses extracted NER keywords and topic classifications for intelligent routing
"""

from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from difflib import SequenceMatcher
import re
from collections import Counter
import numpy as np

class NERFuzzyRouter:
    """
    Smart router that uses NER-extracted keywords and fuzzy matching
    No hard-coded domain knowledge - learns from document content
    """
    
    def __init__(self, content_store):
        self.content_store = content_store
        self._build_ner_index()
    
    def _build_ner_index(self):
        """Build searchable index from NER-extracted keywords and topics"""
        self.document_profiles = {}
        all_content = self.content_store.get_all_content()
        
        for item in all_content:
            profile = {
                'title': item.title.lower(),
                'keywords': [kw.lower().strip() for kw in item.keywords],
                'topics': [topic.lower() for topic in item.topic_assignments],
                'full_text_sample': item.full_text[:500].lower(),  # First 500 chars for context
                'source_path': item.source_path,
                'content_item': item
            }
            self.document_profiles[item.id] = profile
        
        print(f"Built NER index for {len(self.document_profiles)} documents")
    
    def route_query(self, query: str) -> Tuple[List[str], List[str], str]:
        """
        Route query using NER-based fuzzy matching
        
        Returns:
            (pdf_files, youtube_urls, explanation)
        """
        query_lower = query.lower()
        
        # Step 1: Extract query terms and clean them
        query_terms = self._extract_query_terms(query_lower)
        
        # Step 2: Score documents using multiple NER-based methods
        document_scores = {}
        
        for doc_id, profile in self.document_profiles.items():
            score = self._calculate_ner_score(query_terms, profile)
            if score > 0:
                document_scores[doc_id] = score
        
        # Step 3: Determine how many documents to return
        num_docs = self._determine_result_count(query_lower, document_scores)
        
        # Step 4: Select top documents
        top_docs = sorted(document_scores.items(), key=lambda x: x[1], reverse=True)[:num_docs]
        
        # Step 5: Convert to file paths
        pdf_files = []
        youtube_urls = []
        
        for doc_id, score in top_docs:
            profile = self.document_profiles[doc_id]
            if profile['content_item'].content_type == 'pdf':
                pdf_files.append(profile['source_path'])
            elif profile['content_item'].content_type == 'youtube':
                youtube_urls.append(profile['source_path'])
        
        # Step 6: Generate explanation
        explanation = self._generate_explanation(query, top_docs, document_scores)
        
        return pdf_files, youtube_urls, explanation
    
    def _extract_query_terms(self, query: str) -> List[str]:
        """Extract meaningful terms from query using NER principles"""
        
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
            r'\b\w+\s+\w+\s+\w+\b',  # 3-word terms
            r'\b\w+\s+\w+\b'         # 2-word terms
        ]
        
        for pattern in multi_word_patterns:
            matches = re.findall(pattern, query)
            for match in matches:
                if not any(stop in match.lower() for stop in ['what is', 'how to', 'can you']):
                    meaningful_terms.append(match.lower())
        
        return list(set(meaningful_terms))  # Remove duplicates
    
    def _calculate_ner_score(self, query_terms: List[str], profile: Dict) -> float:
        """Calculate relevance score using NER-extracted features"""
        
        total_score = 0.0
        
        # 1. Exact keyword matches (highest weight)
        exact_matches = 0
        for term in query_terms:
            if term in profile['keywords']:
                exact_matches += 1
        
        exact_score = exact_matches * 10.0
        total_score += exact_score
        
        # 2. Fuzzy keyword matches (medium weight)
        fuzzy_score = 0.0
        for term in query_terms:
            best_fuzzy = 0.0
            for keyword in profile['keywords']:
                # Use fuzzy string matching
                similarity = SequenceMatcher(None, term, keyword).ratio()
                if similarity > 0.8:  # High similarity threshold
                    best_fuzzy = max(best_fuzzy, similarity)
            fuzzy_score += best_fuzzy * 5.0
        
        total_score += fuzzy_score
        
        # 3. Topic relevance (medium weight)
        topic_score = 0.0
        for term in query_terms:
            for topic in profile['topics']:
                if term in topic or topic in term:
                    topic_score += 3.0
        
        total_score += topic_score
        
        # 4. Title relevance (high weight)
        title_score = 0.0
        for term in query_terms:
            if term in profile['title']:
                title_score += 8.0
            else:
                # Fuzzy title matching
                title_similarity = SequenceMatcher(None, term, profile['title']).ratio()
                if title_similarity > 0.6:
                    title_score += title_similarity * 4.0
        
        total_score += title_score
        
        # 5. Context relevance (low weight but important for disambiguation)
        context_score = 0.0
        for term in query_terms:
            if term in profile['full_text_sample']:
                context_score += 1.0
        
        total_score += context_score
        
        return total_score
    
    def _determine_result_count(self, query: str, scores: Dict) -> int:
        """Intelligently determine how many documents to return"""
        
        if not scores:
            return 0
        
        # Check for comparative/multi-document indicators
        comparative_indicators = [
            'compare', 'comparison', 'versus', 'vs', 'difference', 'different',
            'both', 'all', 'multiple', 'various', 'several'
        ]
        
        is_comparative = any(indicator in query for indicator in comparative_indicators)
        
        # Get score distribution
        score_values = list(scores.values())
        max_score = max(score_values)
        
        if is_comparative:
            # For comparative queries, return documents with scores > 50% of max
            threshold = max_score * 0.5
            return len([s for s in score_values if s >= threshold])
        else:
            # For specific queries, be more selective
            if max_score > 20:  # Strong match exists
                threshold = max_score * 0.7
                count = len([s for s in score_values if s >= threshold])
                return max(1, min(count, 3))  # 1-3 documents
            else:
                # Weaker matches, return top 2
                return min(2, len(score_values))
    
    def _generate_explanation(self, query: str, top_docs: List[Tuple[str, float]], all_scores: Dict) -> str:
        """Generate human-readable explanation of routing decision"""
        
        if not top_docs:
            return "No relevant documents found using NER-based matching"
        
        doc_count = len(top_docs)
        total_docs = len(all_scores)
        
        # Get document titles for explanation
        doc_titles = []
        for doc_id, score in top_docs:
            profile = self.document_profiles[doc_id]
            title = profile['content_item'].title
            doc_titles.append(title.split(' -')[0])  # Truncate long titles
        
        if doc_count == 1:
            return f"NER-based routing: Specific match found for '{doc_titles[0]}' (score: {top_docs[0][1]:.1f})"
        elif doc_count <= 3:
            return f"NER-based routing: Found {doc_count} relevant documents from {total_docs} total using fuzzy keyword matching"
        else:
            return f"NER-based routing: Multi-document query matched {doc_count} sources using topic classification"
    
    def get_routing_debug_info(self, query: str) -> Dict:
        """Get detailed debugging information about routing decisions"""
        
        query_terms = self._extract_query_terms(query.lower())
        debug_info = {
            'query': query,
            'extracted_terms': query_terms,
            'document_scores': {}
        }
        
        for doc_id, profile in self.document_profiles.items():
            score = self._calculate_ner_score(query_terms, profile)
            debug_info['document_scores'][profile['content_item'].title] = {
                'total_score': score,
                'matched_keywords': [kw for kw in profile['keywords'] if any(term in kw for term in query_terms)],
                'matched_topics': [topic for topic in profile['topics'] if any(term in topic for term in query_terms)]
            }
        
        return debug_info

# Test the NER-based router
if __name__ == "__main__":
    # This would be tested with actual content store
    print("NER-based Fuzzy Router implementation complete")
    print("Features:")
    print("- Uses NER-extracted keywords for matching")
    print("- Fuzzy string matching for term variations") 
    print("- Topic-based relevance scoring")
    print("- Adaptive result count based on query type")
    print("- No hard-coded domain knowledge")