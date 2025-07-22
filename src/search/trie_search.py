"""
Trie-based Search Implementation for KnowledgeAgent
Provides fast prefix matching and fuzzy search capabilities
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class SearchResult:
    """Result from trie search"""
    content_id: str
    title: str
    content_type: str
    source_path: str
    match_score: float
    matched_terms: List[str]
    preview: str


class TrieNode:
    """Node in the trie structure"""
    
    def __init__(self):
        self.children: Dict[str, 'TrieNode'] = {}
        self.is_end_of_word = False
        self.content_refs: List[Dict[str, Any]] = []  # References to content containing this word
        self.frequency = 0


class ContentTrieSearch:
    """
    Trie-based search engine for content titles and keywords
    Provides fast prefix matching and ranking
    """
    
    def __init__(self):
        self.root = TrieNode()
        self.content_index: Dict[str, Dict[str, Any]] = {}
        self.total_content = 0
        
    def _normalize_text(self, text: str) -> str:
        """Normalize text for indexing and searching"""
        # Convert to lowercase and remove special characters
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _extract_words(self, text: str) -> List[str]:
        """Extract meaningful words from text"""
        normalized = self._normalize_text(text)
        words = normalized.split()
        
        # Filter out very short words and common stop words
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        meaningful_words = [word for word in words if len(word) > 2 and word not in stop_words]
        
        return meaningful_words
    
    def _insert_word(self, word: str, content_ref: Dict[str, Any]):
        """Insert a word into the trie with content reference"""
        node = self.root
        
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        
        node.is_end_of_word = True
        node.content_refs.append(content_ref)
        node.frequency += 1
    
    def add_content(self, content_id: str, title: str, content_type: str, source_path: str, 
                   preview: str = "", keywords: List[str] = None):
        """Add content to the search index"""
        content_ref = {
            'content_id': content_id,
            'title': title,
            'content_type': content_type,
            'source_path': source_path,
            'preview': preview
        }
        
        # Store content details
        self.content_index[content_id] = content_ref
        
        # Extract words from title
        title_words = self._extract_words(title)
        
        # Add title words with higher weight
        for word in title_words:
            self._insert_word(word, {**content_ref, 'match_type': 'title', 'weight': 3})
        
        # Add keywords if provided
        if keywords:
            for keyword in keywords:
                keyword_words = self._extract_words(keyword)
                for word in keyword_words:
                    self._insert_word(word, {**content_ref, 'match_type': 'keyword', 'weight': 2})
        
        # Add words from preview with lower weight
        if preview:
            preview_words = self._extract_words(preview)[:20]  # Limit preview words
            for word in preview_words:
                self._insert_word(word, {**content_ref, 'match_type': 'content', 'weight': 1})
        
        self.total_content += 1
    
    def _search_prefix(self, prefix: str) -> List[TrieNode]:
        """Find all nodes that match the given prefix"""
        node = self.root
        
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        
        # Collect all end nodes from this prefix
        end_nodes = []
        self._collect_end_nodes(node, end_nodes)
        return end_nodes
    
    def _collect_end_nodes(self, node: TrieNode, end_nodes: List[TrieNode]):
        """Recursively collect all end-of-word nodes"""
        if node.is_end_of_word:
            end_nodes.append(node)
        
        for child in node.children.values():
            self._collect_end_nodes(child, end_nodes)
    
    def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Search for content matching the query"""
        query_words = self._extract_words(query)
        
        if not query_words:
            return []
        
        # Score content based on matches
        content_scores: Dict[str, float] = defaultdict(float)
        content_matches: Dict[str, List[str]] = defaultdict(list)
        
        for word in query_words:
            # Search for exact matches
            exact_nodes = self._search_prefix(word)
            
            for node in exact_nodes:
                for content_ref in node.content_refs:
                    content_id = content_ref['content_id']
                    weight = content_ref.get('weight', 1)
                    
                    # Score based on match type and frequency
                    score = weight * (1.0 + 0.1 * node.frequency)
                    content_scores[content_id] += score
                    content_matches[content_id].append(word)
            
            # Search for partial matches (prefix search)
            if len(word) > 3:
                for i in range(3, len(word)):
                    prefix = word[:i]
                    prefix_nodes = self._search_prefix(prefix)
                    
                    for node in prefix_nodes:
                        for content_ref in node.content_refs:
                            content_id = content_ref['content_id']
                            weight = content_ref.get('weight', 1)
                            
                            # Lower score for partial matches
                            score = weight * 0.5 * (i / len(word))
                            content_scores[content_id] += score
                            if word not in content_matches[content_id]:
                                content_matches[content_id].append(f"{prefix}*")
        
        # Create search results
        results = []
        for content_id, score in content_scores.items():
            if content_id in self.content_index:
                content = self.content_index[content_id]
                
                result = SearchResult(
                    content_id=content_id,
                    title=content['title'],
                    content_type=content['content_type'],
                    source_path=content['source_path'],
                    match_score=score,
                    matched_terms=content_matches[content_id],
                    preview=content.get('preview', '')
                )
                results.append(result)
        
        # Sort by score (highest first) and limit results
        results.sort(key=lambda x: x.match_score, reverse=True)
        return results[:max_results]
    
    def suggest_completions(self, partial_query: str, max_suggestions: int = 5) -> List[str]:
        """Suggest query completions based on partial input"""
        if not partial_query:
            return []
        
        words = self._extract_words(partial_query)
        if not words:
            return []
        
        last_word = words[-1]
        suggestions = []
        
        # Find all words that start with the last partial word
        prefix_nodes = self._search_prefix(last_word)
        
        # Collect complete words
        word_suggestions = []
        for node in prefix_nodes:
            self._collect_words_from_node(node, last_word, word_suggestions)
        
        # Sort by frequency and limit
        word_suggestions.sort(key=lambda x: x[1], reverse=True)
        
        # Build complete suggestions
        base_query = ' '.join(words[:-1])
        for word, freq in word_suggestions[:max_suggestions]:
            if base_query:
                suggestions.append(f"{base_query} {word}")
            else:
                suggestions.append(word)
        
        return suggestions
    
    def _collect_words_from_node(self, node: TrieNode, current_word: str, 
                                word_list: List[Tuple[str, int]], max_depth: int = 20):
        """Collect complete words from a trie node"""
        if len(current_word) > max_depth:
            return
        
        if node.is_end_of_word:
            word_list.append((current_word, node.frequency))
        
        for char, child_node in node.children.items():
            self._collect_words_from_node(child_node, current_word + char, word_list, max_depth)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get search index statistics"""
        return {
            'total_content': self.total_content,
            'total_words_indexed': self._count_total_words(),
            'average_words_per_content': self._count_total_words() / max(1, self.total_content)
        }
    
    def _count_total_words(self) -> int:
        """Count total words in the trie"""
        return self._count_words_from_node(self.root)
    
    def _count_words_from_node(self, node: TrieNode) -> int:
        """Recursively count words from a node"""
        count = 1 if node.is_end_of_word else 0
        for child in node.children.values():
            count += self._count_words_from_node(child)
        return count