"""
Dynamic Topic Classification System for KnowAgent
Uses semantic embeddings for intelligent content clustering
"""

import numpy as np
from typing import Dict, List, Set, Optional, Tuple, DefaultDict, Any
from collections import defaultdict
from dataclasses import dataclass, field
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import json
import time
from pathlib import Path

@dataclass
class ContentItem:
    """Represents a piece of content (PDF chunk or video segment)"""
    content_id: str
    content_type: str  # 'pdf' or 'video'
    source_id: str  # document_id or video_id
    text: str
    embedding: np.ndarray
    metadata: Dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

@dataclass
class Topic:
    """Represents a discovered topic cluster"""
    topic_id: str
    name: str
    description: str
    centroid_embedding: np.ndarray
    content_items: Set[str] = field(default_factory=set)  # content_ids
    confidence_score: float = 0.0
    created_timestamp: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)

class DynamicTopicClassifier:
    """
    Intelligent topic classification system that:
    1. Discovers topics from content automatically
    2. Assigns content to multiple topics if relevant
    3. Recalibrates when new content is added
    4. Maintains topic relationships and hierarchies
    """
    
    def __init__(self, 
                 similarity_threshold: float = 0.7,
                 min_cluster_size: int = 3,
                 eps: float = 0.4,
                 max_topics: int = 10,
                 min_topic_size: int = 2,
                 storage_path: str = "topic_classification"):
        """
        Initialize the ML-based topic classifier
        
        Args:
            similarity_threshold: Minimum similarity to merge topics (0.7 = 70%)
            min_cluster_size: Minimum items needed to form a topic cluster  
            eps: DBSCAN clustering parameter (higher = fewer clusters)
            max_topics: Maximum number of topics to maintain
            min_topic_size: Minimum items per topic (smaller topics get merged)
            storage_path: Directory to store topic classifications
        """
        self.similarity_threshold = similarity_threshold
        self.min_cluster_size = min_cluster_size
        self.eps = eps
        self.max_topics = max_topics
        self.min_topic_size = min_topic_size
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Core data structures
        self.content_items: Dict[str, ContentItem] = {}
        self.topics: Dict[str, Topic] = {}
        self.content_to_topics: DefaultDict[str, Set[str]] = defaultdict(set)  # content_id -> topic_ids
        self.topic_to_contents: DefaultDict[str, Set[str]] = defaultdict(set)  # topic_id -> content_ids
        
        # For efficient similarity computation
        self.topic_embeddings_matrix: Optional[np.ndarray] = None
        self.topic_ids_order: List[str] = []
        
        print(f"Topic classifier initialized (threshold: {similarity_threshold}, storage: {storage_path})")
        self._load_existing_classifications()
    
    def add_content(self, 
                   content_id: str,
                   content_type: str,
                   source_id: str, 
                   text: str,
                   embedding: np.ndarray,
                   metadata: Dict = None) -> List[str]:
        """
        Add new content and assign to source-level topic (one topic per source)
        
        Returns:
            List of topic IDs the content was assigned to
        """
        # Create content item
        content_item = ContentItem(
            content_id=content_id,
            content_type=content_type,
            source_id=source_id,
            text=text,
            embedding=embedding,
            metadata=metadata or {}
        )
        
        self.content_items[content_id] = content_item
        
        # Check if we already have a topic for this source
        source_topic_id = None
        for topic_id, topic in self.topics.items():
            if hasattr(topic, 'source_id') and topic.source_id == source_id:
                source_topic_id = topic_id
                break
        
        # If no topic exists for this source, create one
        if not source_topic_id:
            # Use full transcript context if available in metadata for better topic naming
            topic_text = metadata.get('full_transcript', text) if metadata else text
            source_topic_id = self._create_source_topic(source_id, content_type, topic_text, embedding)
        
        # Assign content to the source topic
        assigned_topics = [source_topic_id]
        
        # Update mappings
        self.content_to_topics[content_id].add(source_topic_id)
        self.topic_to_contents[source_topic_id].add(content_id)
        self.topics[source_topic_id].content_items.add(content_id)
        
        # Update topic centroid with new content
        self._update_topic_centroid(source_topic_id, embedding)
        
        print(f"Content '{content_id}' assigned to source topic: {self.topics[source_topic_id].name}")
        
        # Save periodically
        if len(self.content_items) % 20 == 0:
            self._save_classifications()
        
        return assigned_topics
    
    def _classify_content(self, content_item: ContentItem) -> List[str]:
        """
        Classify a content item into existing topics based on similarity
        """
        if not self.topics:
            return []
        
        assigned_topics = []
        content_embedding = content_item.embedding.reshape(1, -1)
        
        # Calculate similarities to all existing topics
        if self.topic_embeddings_matrix is not None:
            similarities = cosine_similarity(content_embedding, self.topic_embeddings_matrix)[0]
            
            # Assign to topics above threshold
            for i, similarity in enumerate(similarities):
                if similarity >= self.similarity_threshold:
                    topic_id = self.topic_ids_order[i]
                    assigned_topics.append(topic_id)
                    
                    # Update topic centroid (incremental learning)
                    self._update_topic_centroid(topic_id, content_item.embedding)
        
        return assigned_topics
    
    def _create_source_topic(self, source_id: str, content_type: str, sample_text: str, sample_embedding: np.ndarray) -> str:
        """
        Create a single topic for an entire source (video or PDF)
        """
        # Generate topic name from source content
        topic_name, description = self._generate_contextual_topic(sample_text[:1000], 1)  # Use more text for better analysis
        
        # Create unique topic ID for this source
        topic_id = f"{content_type}_{source_id}_{int(time.time() * 1000)}"
        
        # Create topic
        topic = Topic(
            topic_id=topic_id,
            name=topic_name,
            description=description,
            centroid_embedding=sample_embedding.copy(),
            content_items=set(),
            confidence_score=0.9
        )
        
        # Add source_id for easy lookup
        topic.source_id = source_id
        topic.content_type = content_type
        
        self.topics[topic_id] = topic
        
        print(f"Created source topic: '{topic_name}' for {content_type} source: {source_id}")
        return topic_id
    
    def find_relevant_sources(self, query_embedding: np.ndarray, max_sources: int = 5) -> List[Tuple[str, float]]:
        """
        Find relevant sources for a query using semantic similarity
        This is where we use ML clustering/similarity for query-time matching
        """
        if not self.topics:
            return []
        
        query_embedding = query_embedding.reshape(1, -1)
        relevant_sources = []
        
        for topic_id, topic in self.topics.items():
            if hasattr(topic, 'source_id'):
                topic_embedding = topic.centroid_embedding.reshape(1, -1)
                similarity = cosine_similarity(query_embedding, topic_embedding)[0][0]
                
                relevant_sources.append((
                    topic.source_id,
                    similarity,
                    topic.name,
                    topic.content_type
                ))
        
        # Sort by similarity and return top sources
        relevant_sources.sort(key=lambda x: x[1], reverse=True)
        return relevant_sources[:max_sources]
    
    def _group_and_assign_content(self) -> List[str]:
        """
        Group unassigned content using semantic clustering and create meaningful topics
        """
        # Get unassigned content
        unassigned_content = []
        for content_id, content_item in self.content_items.items():
            if not self.content_to_topics.get(content_id):
                unassigned_content.append((content_id, content_item))
        
        if len(unassigned_content) < 2:
            # If only one item, create individual topic
            if unassigned_content:
                return self._create_individual_topic(unassigned_content[0])
            return []
        
        print(f"Grouping {len(unassigned_content)} unassigned items using semantic clustering...")
        
        # Prepare embeddings for clustering
        embeddings = np.array([item.embedding for _, item in unassigned_content])
        content_ids = [content_id for content_id, _ in unassigned_content]
        
        # Use semantic clustering to find meaningful groups
        # First try DBSCAN for natural clusters
        clustering = DBSCAN(eps=0.3, min_samples=max(3, len(unassigned_content) // 8), metric='cosine')
        cluster_labels = clustering.fit_predict(embeddings)
        
        # If DBSCAN finds too few clusters, use KMeans for better topic distribution
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        
        if n_clusters < 2 and len(unassigned_content) >= 10:
            # Use KMeans to create meaningful topic groups
            target_clusters = min(5, max(2, len(unassigned_content) // 8))
            kmeans = KMeans(n_clusters=target_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)
            print(f"Used KMeans clustering with {target_clusters} target clusters")
        
        # Group content by clusters
        clusters = defaultdict(list)
        noise_items = []  # Items that don't fit any cluster (label = -1)
        
        for i, label in enumerate(cluster_labels):
            if label == -1:
                noise_items.append((content_ids[i], unassigned_content[i][1]))
            else:
                clusters[label].append((content_ids[i], unassigned_content[i][1]))
        
        assigned_topics = []
        
        # Create topics for clusters
        for cluster_id, cluster_items in clusters.items():
            if len(cluster_items) >= self.min_topic_size:
                topic_id = self._create_cluster_topic(cluster_items)
                assigned_topics.extend([topic_id] * len(cluster_items))
                print(f"Created cluster topic with {len(cluster_items)} items")
        
        # Handle noise items (create individual topics or merge with existing)
        for content_id, content_item in noise_items:
            # Try to assign to existing topics
            similar_topics = self._find_similar_topics(content_item.embedding)
            if similar_topics:
                topic_id = similar_topics[0][0]  # Best match
                assigned_topics.append(topic_id)
                print(f"Assigned noise item to existing topic: {topic_id}")
            else:
                # Create individual topic for unmatched content
                topic_id = self._create_individual_topic((content_id, content_item))[0]
                assigned_topics.append(topic_id)
        
        # Limit total topics to prevent explosion
        if len(self.topics) > self.max_topics:
            self._merge_smallest_topics()
        
        self._rebuild_topic_embeddings_matrix()
        return assigned_topics
    
    def _create_cluster_topic(self, cluster_items: List[Tuple[str, ContentItem]]) -> str:
        """
        Create a topic from a cluster of semantically similar content
        """
        content_ids = [content_id for content_id, _ in cluster_items]
        embeddings = np.array([item.embedding for _, item in cluster_items])
        texts = [item.text for _, item in cluster_items]
        
        # Calculate centroid
        centroid = np.mean(embeddings, axis=0)
        
        # Generate topic name from cluster content
        topic_name, description = self._extract_topic_info(texts)
        
        # Calculate confidence from cluster cohesion
        confidences = cosine_similarity(embeddings, centroid.reshape(1, -1)).flatten()
        confidence_score = np.mean(confidences)
        
        # Create topic
        topic_id = f"cluster_topic_{int(time.time() * 1000)}_{len(self.topics)}"
        topic = Topic(
            topic_id=topic_id,
            name=topic_name,
            description=description,
            centroid_embedding=centroid,
            content_items=set(content_ids),
            confidence_score=confidence_score
        )
        
        self.topics[topic_id] = topic
        
        print(f"Created cluster topic: '{topic_name}' with {len(content_ids)} items (confidence: {confidence_score:.3f})")
        return topic_id
    
    def _create_individual_topic(self, content_item_tuple: Tuple[str, ContentItem]) -> List[str]:
        """
        Create an individual topic for a single content item
        """
        content_id, content_item = content_item_tuple
        
        topic_name, description = self._generate_contextual_topic(content_item.text, 1)
        topic_id = f"individual_topic_{int(time.time() * 1000)}_{len(self.topics)}"
        
        topic = Topic(
            topic_id=topic_id,
            name=topic_name,
            description=description,
            centroid_embedding=content_item.embedding.copy(),
            content_items={content_id},
            confidence_score=0.8  # Lower confidence for individual topics
        )
        
        self.topics[topic_id] = topic
        print(f"Created individual topic: '{topic_name}' for {content_id}")
        return [topic_id]
    
    def _find_similar_topics(self, embedding: np.ndarray, threshold: float = None) -> List[Tuple[str, float]]:
        """
        Find existing topics similar to the given embedding
        """
        if not self.topics or self.topic_embeddings_matrix is None:
            return []
        
        threshold = threshold or self.similarity_threshold
        embedding = embedding.reshape(1, -1)
        
        similarities = cosine_similarity(embedding, self.topic_embeddings_matrix)[0]
        
        similar_topics = []
        for i, similarity in enumerate(similarities):
            if similarity >= threshold:
                topic_id = self.topic_ids_order[i]
                similar_topics.append((topic_id, similarity))
        
        # Sort by similarity (highest first)
        similar_topics.sort(key=lambda x: x[1], reverse=True)
        return similar_topics
    
    def _merge_smallest_topics(self):
        """
        Merge the smallest topics to stay under max_topics limit
        """
        if len(self.topics) <= self.max_topics:
            return
        
        # Sort topics by size (smallest first)
        topic_sizes = [(topic_id, len(topic.content_items)) for topic_id, topic in self.topics.items()]
        topic_sizes.sort(key=lambda x: x[1])
        
        # Merge smallest topics until we're under the limit
        topics_to_remove = len(self.topics) - self.max_topics
        
        for i in range(0, min(topics_to_remove * 2, len(topic_sizes) - 1), 2):
            if i + 1 < len(topic_sizes):
                small_topic_id = topic_sizes[i][0]
                merge_target_id = topic_sizes[i + 1][0]
                
                self._merge_topics(small_topic_id, merge_target_id)
                print(f"Merged small topic {small_topic_id} into {merge_target_id}")
    
    def _merge_topics(self, source_topic_id: str, target_topic_id: str):
        """
        Merge source topic into target topic
        """
        if source_topic_id not in self.topics or target_topic_id not in self.topics:
            return
        
        source_topic = self.topics[source_topic_id]
        target_topic = self.topics[target_topic_id]
        
        # Update content mappings
        for content_id in source_topic.content_items:
            self.content_to_topics[content_id].discard(source_topic_id)
            self.content_to_topics[content_id].add(target_topic_id)
            self.topic_to_contents[target_topic_id].add(content_id)
        
        # Update target topic
        target_topic.content_items.update(source_topic.content_items)
        target_topic.last_updated = time.time()
        
        # Update centroid (weighted average)
        source_weight = len(source_topic.content_items)
        target_weight = len(target_topic.content_items) - source_weight
        total_weight = source_weight + target_weight
        
        if total_weight > 0:
            target_topic.centroid_embedding = (
                (target_weight * target_topic.centroid_embedding + 
                 source_weight * source_topic.centroid_embedding) / total_weight
            )
        
        # Remove source topic
        del self.topics[source_topic_id]
        if source_topic_id in self.topic_to_contents:
            del self.topic_to_contents[source_topic_id]
    
    def _generate_topic_from_cluster(self, content_ids: List[str], embeddings: np.ndarray) -> Topic:
        """
        Generate a topic from a cluster of content items
        """
        # Calculate centroid
        centroid = np.mean(embeddings, axis=0)
        
        # Analyze content to generate topic name and description
        texts = [self.content_items[cid].text for cid in content_ids]
        topic_name, topic_description = self._extract_topic_info(texts)
        
        # Calculate confidence based on cluster cohesion
        confidences = cosine_similarity(embeddings, centroid.reshape(1, -1)).flatten()
        confidence_score = np.mean(confidences)
        
        return Topic(
            topic_id="",  # Will be set by caller
            name=topic_name,
            description=topic_description,
            centroid_embedding=centroid,
            content_items=set(content_ids),
            confidence_score=confidence_score
        )
    
    def _extract_topic_info(self, texts: List[str]) -> Tuple[str, str]:
        """
        Extract topic name and description using contextual analysis
        Similar to how ChatGPT generates chat titles from conversation content
        """
        # Combine all texts for analysis
        all_text = " ".join(texts)
        
        # Use contextual topic generation
        topic_name, description = self._generate_contextual_topic(all_text, len(texts))
        
        return topic_name, description
    
    def _generate_contextual_topic(self, text: str, source_count: int) -> Tuple[str, str]:
        """
        Generate contextual topic name like ChatGPT chat titles
        Uses domain detection and key phrase extraction
        """
        text_lower = text.lower()
        
        # Domain detection with key phrases
        domain_patterns = {
            'space_astronomy': {
                'keywords': ['space', 'astronomy', 'telescope', 'observatory', 'celestial', 'stellar', 'galaxy', 'cosmic'],
                'phrases': ['space observation', 'astronomical phenomena', 'celestial mechanics', 'deep space'],
                'name': 'Space Science and Astronomy'
            },
            'mars_planetary': {
                'keywords': ['mars', 'planetary', 'rover', 'red planet', 'surface', 'geology', 'exploration'],
                'phrases': ['mars exploration', 'planetary surface', 'robotic rovers', 'martian geology'],
                'name': 'Mars Exploration and Planetary Science'
            },
            'machine_learning': {
                'keywords': ['machine learning', 'neural network', 'deep learning', 'training', 'model', 'algorithm', 'ai'],
                'phrases': ['neural networks', 'deep learning', 'machine learning models', 'training algorithms'],
                'name': 'Machine Learning and Neural Networks'
            },
            'computer_vision': {
                'keywords': ['computer vision', 'image', 'visual', 'detection', 'recognition', 'processing', 'opencv'],
                'phrases': ['computer vision', 'image processing', 'object detection', 'visual recognition'],
                'name': 'Computer Vision and Image Processing'
            },
            'natural_language': {
                'keywords': ['natural language', 'nlp', 'text', 'linguistic', 'language model', 'processing'],
                'phrases': ['natural language processing', 'text analysis', 'language models', 'linguistic processing'],
                'name': 'Natural Language Processing'
            },
            'robotics': {
                'keywords': ['robot', 'robotics', 'autonomous', 'control', 'sensor', 'manipulation', 'navigation'],
                'phrases': ['robotic systems', 'autonomous control', 'robot navigation', 'sensor fusion'],
                'name': 'Robotics and Autonomous Systems'
            },
            'data_science': {
                'keywords': ['data', 'analysis', 'statistics', 'dataset', 'analytics', 'visualization', 'mining'],
                'phrases': ['data analysis', 'statistical analysis', 'data visualization', 'data mining'],
                'name': 'Data Science and Analytics'
            },
            'programming_software': {
                'keywords': ['programming', 'coding', 'software', 'development', 'python', 'javascript', 'framework'],
                'phrases': ['software development', 'programming tutorial', 'code review', 'web development'],
                'name': 'Programming and Software Development'
            },
            'business_tech': {
                'keywords': ['business', 'startup', 'entrepreneurship', 'technology', 'innovation', 'strategy'],
                'phrases': ['business strategy', 'tech innovation', 'startup advice', 'digital transformation'],
                'name': 'Business and Technology'
            },
            'education_tutorial': {
                'keywords': ['tutorial', 'learning', 'education', 'course', 'lesson', 'teaching', 'guide'],
                'phrases': ['how to', 'step by step', 'beginner guide', 'tutorial series'],
                'name': 'Educational Content and Tutorials'
            },
            'research_academic': {
                'keywords': ['research', 'academic', 'study', 'paper', 'findings', 'methodology', 'experiment'],
                'phrases': ['research paper', 'academic study', 'scientific method', 'experimental results'],
                'name': 'Academic Research and Studies'
            }
        }
        
        # Calculate domain scores
        domain_scores = {}
        for domain, patterns in domain_patterns.items():
            score = 0
            
            # Keyword matching
            for keyword in patterns['keywords']:
                if keyword in text_lower:
                    score += 2
            
            # Phrase matching (higher weight)
            for phrase in patterns['phrases']:
                if phrase in text_lower:
                    score += 5
            
            domain_scores[domain] = score
        
        # Find best domain
        best_domain = max(domain_scores, key=domain_scores.get) if domain_scores else None
        best_score = domain_scores.get(best_domain, 0) if best_domain else 0
        
        # Generate topic name and description
        if best_score >= 3:  # Confident domain match
            topic_name = domain_patterns[best_domain]['name']
            description = f"Content focused on {topic_name.lower()} from {source_count} sources (confidence: {best_score})"
        else:
            # Fallback: extract key terms from text
            topic_name = self._extract_key_phrase_topic(text)
            description = f"Research content covering {topic_name.lower()} from {source_count} sources"
        
        return topic_name, description
    
    def _extract_key_phrase_topic(self, text: str) -> str:
        """
        Extract topic from key phrases when domain detection fails
        """
        text_lower = text.lower()
        words = text_lower.split()
        
        # Look for meaningful noun phrases
        important_terms = []
        
        # First, try to extract from the beginning of the text (titles, headings)
        first_50_words = ' '.join(words[:50])
        
        # Look for title-like patterns
        title_indicators = ['lecture', 'introduction to', 'tutorial on', 'guide to', 'overview of', 
                          'understanding', 'exploring', 'discussion on', 'presentation on']
        
        for indicator in title_indicators:
            if indicator in first_50_words:
                # Extract what comes after the indicator
                parts = first_50_words.split(indicator, 1)
                if len(parts) > 1 and parts[1].strip():
                    topic_part = parts[1].strip()[:30]  # Take first 30 chars
                    # Clean up and capitalize
                    topic_part = ' '.join(topic_part.split()[:4])  # Max 4 words
                    if len(topic_part) > 3:
                        return topic_part.title()
        
        # Multi-word technical terms
        technical_patterns = [
            'machine learning', 'deep learning', 'computer vision', 'natural language',
            'neural network', 'data analysis', 'image processing', 'text processing',
            'knowledge graph', 'expert system', 'decision making', 'pattern recognition',
            'artificial intelligence', 'data science', 'software engineering', 'web development'
        ]
        
        for pattern in technical_patterns:
            if pattern in text_lower:
                important_terms.append(pattern.title())
        
        # Single important words
        key_words = [
            'algorithm', 'system', 'framework', 'model', 'analysis', 'research',
            'optimization', 'evaluation', 'implementation', 'methodology'
        ]
        
        found_words = [word.title() for word in key_words if word in words]
        important_terms.extend(found_words[:2])  # Top 2 single words
        
        if important_terms:
            # Combine terms intelligently
            if len(important_terms) == 1:
                return f"{important_terms[0]} Research"
            elif len(important_terms) == 2:
                return f"{important_terms[0]} and {important_terms[1]}"
            else:
                return f"{important_terms[0]}, {important_terms[1]} and Related Topics"
        else:
            return "Research Content"
    
    def _is_duplicate_topic(self, new_topic: Topic) -> bool:
        """
        Check if a new topic is too similar to existing topics
        Uses both semantic similarity and name matching
        """
        if not self.topics:
            return False
        
        new_embedding = new_topic.centroid_embedding.reshape(1, -1)
        
        for existing_topic in self.topics.values():
            # Check semantic similarity
            existing_embedding = existing_topic.centroid_embedding.reshape(1, -1)
            similarity = cosine_similarity(new_embedding, existing_embedding)[0][0]
            
            # Check name similarity (prevent multiple similar topics)
            name_similarity = self._calculate_name_similarity(new_topic.name, existing_topic.name)
            
            # Merge if very similar semantically OR if names are too similar
            if similarity > 0.75 or name_similarity:
                print(f"Merging similar topic: {new_topic.name} -> {existing_topic.name} (similarity: {similarity:.3f})")
                return True
        
        return False
    
    def _calculate_name_similarity(self, name1: str, name2: str) -> bool:
        """
        Calculate similarity between topic names to prevent duplicates
        """
        name1_lower = name1.lower()
        name2_lower = name2.lower()
        
        # Exact match
        if name1_lower == name2_lower:
            return True
        
        # Both are generic research topics
        if ("research" in name1_lower and "research" in name2_lower):
            return True
        
        # Check for domain overlap
        domain_keywords = {
            'space': ['space', 'astronomy', 'mars', 'planetary', 'celestial', 'observatory'],
            'ai': ['machine', 'learning', 'neural', 'artificial', 'intelligence', 'deep'],
            'vision': ['computer', 'vision', 'image', 'visual', 'processing'],
            'language': ['natural', 'language', 'nlp', 'text', 'linguistic'],
            'robotics': ['robot', 'robotics', 'autonomous', 'control']
        }
        
        for domain, keywords in domain_keywords.items():
            name1_has_domain = any(keyword in name1_lower for keyword in keywords)
            name2_has_domain = any(keyword in name2_lower for keyword in keywords)
            
            if name1_has_domain and name2_has_domain:
                # Both topics are in the same domain
                return True
        
        return False
    
    def _update_topic_centroid(self, topic_id: str, new_embedding: np.ndarray):
        """
        Update topic centroid with new content (incremental learning)
        """
        if topic_id not in self.topics:
            return
        
        topic = self.topics[topic_id]
        current_centroid = topic.centroid_embedding
        content_count = len(topic.content_items)
        
        # Weighted average update
        alpha = 1.0 / (content_count + 1)  # Learning rate decreases with more content
        updated_centroid = (1 - alpha) * current_centroid + alpha * new_embedding
        
        topic.centroid_embedding = updated_centroid
        topic.last_updated = time.time()
    
    def _rebuild_topic_embeddings_matrix(self):
        """
        Rebuild the topic embeddings matrix for efficient similarity computation
        """
        if not self.topics:
            self.topic_embeddings_matrix = None
            self.topic_ids_order = []
            return
        
        self.topic_ids_order = list(self.topics.keys())
        embeddings = [self.topics[tid].centroid_embedding for tid in self.topic_ids_order]
        self.topic_embeddings_matrix = np.array(embeddings)
    
    def get_content_topics(self, content_id: str) -> List[str]:
        """Get topics assigned to a specific content item"""
        return list(self.content_to_topics.get(content_id, set()))
    
    def get_topic_contents(self, topic_id: str) -> List[ContentItem]:
        """Get all content items in a specific topic"""
        content_ids = self.topic_to_contents.get(topic_id, set())
        return [self.content_items[cid] for cid in content_ids if cid in self.content_items]
    
    def get_cross_topic_content(self) -> Dict[str, List[str]]:
        """
        Get content that appears in multiple topics
        Returns: {content_id: [topic_ids]}
        """
        cross_topic = {}
        for content_id, topics in self.content_to_topics.items():
            if len(topics) > 1:
                cross_topic[content_id] = list(topics)
        return cross_topic
    
    def get_topic_hierarchy(self) -> Dict[str, Dict]:
        """
        Get source-level topics for UI display
        Returns organized topic structure by source
        """
        topic_hierarchy = {}
        
        for topic_id, topic in self.topics.items():
            if hasattr(topic, 'source_id'):
                content_items = self.get_topic_contents(topic_id)
                content_type = getattr(topic, 'content_type', 'unknown')
                
                # Count PDFs and videos
                pdfs = 1 if content_type == 'pdf' else 0
                videos = 1 if content_type == 'video' else 0
                
                topic_hierarchy[topic_id] = {
                    "name": topic.name,
                    "description": topic.description,
                    "confidence": topic.confidence_score,
                    "total_content": len(content_items),
                    "content_type": content_type,
                    "source_id": getattr(topic, 'source_id', 'unknown'),
                    "pdfs": pdfs,
                    "videos": videos,
                    "created": topic.created_timestamp,
                    "updated": topic.last_updated
                }
        
        return topic_hierarchy
    
    def suggest_related_content(self, content_id: str, max_suggestions: int = 5) -> List[Tuple[str, float]]:
        """
        Suggest related content based on topic similarity
        Returns: [(content_id, similarity_score)]
        """
        if content_id not in self.content_items:
            return []
        
        content_item = self.content_items[content_id]
        content_embedding = content_item.embedding.reshape(1, -1)
        
        suggestions = []
        
        for other_id, other_item in self.content_items.items():
            if other_id == content_id:
                continue
            
            other_embedding = other_item.embedding.reshape(1, -1)
            similarity = cosine_similarity(content_embedding, other_embedding)[0][0]
            
            if similarity > 0.5:  # Minimum relevance threshold
                suggestions.append((other_id, similarity))
        
        # Sort by similarity and return top suggestions
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return suggestions[:max_suggestions]
    
    def _save_classifications(self):
        """Save topic classifications to disk"""
        try:
            # Save to JSON files
            classifications_file = self.storage_path / "classifications.json"
            
            # Prepare data for serialization
            save_data = {
                "content_items": {},
                "topics": {},
                "content_to_topics": {},
                "topic_to_contents": {},
                "timestamp": time.time()
            }
            
            # Convert content items to serializable format
            for content_id, item in self.content_items.items():
                save_data["content_items"][content_id] = {
                    "content_id": item.content_id,
                    "content_type": item.content_type,
                    "source_id": item.source_id,
                    "text": item.text,
                    "embedding": item.embedding.tolist() if item.embedding is not None else None,
                    "metadata": item.metadata,
                    "timestamp": item.timestamp
                }
            
            # Convert topics to serializable format
            for topic_id, topic in self.topics.items():
                save_data["topics"][topic_id] = {
                    "topic_id": topic.topic_id,
                    "name": topic.name,
                    "description": topic.description,
                    "centroid_embedding": topic.centroid_embedding.tolist() if topic.centroid_embedding is not None else None,
                    "content_items": list(topic.content_items),
                    "confidence_score": topic.confidence_score,
                    "created_timestamp": topic.created_timestamp,
                    "last_updated": topic.last_updated,
                    "source_id": getattr(topic, 'source_id', None),
                    "content_type": getattr(topic, 'content_type', None)
                }
            
            # Convert mappings to serializable format
            save_data["content_to_topics"] = {k: list(v) for k, v in self.content_to_topics.items()}
            save_data["topic_to_contents"] = {k: list(v) for k, v in self.topic_to_contents.items()}
            
            # Save to file
            with open(classifications_file, 'w') as f:
                json.dump(save_data, f, indent=2)
            
            print(f"Saved topic classifications ({len(self.topics)} topics, {len(self.content_items)} items)")
            
        except Exception as e:
            print(f"Error saving classifications: {e}")
    
    def _load_existing_classifications(self):
        """Load existing topic classifications from disk"""
        try:
            classifications_file = self.storage_path / "classifications.json"
            
            if not classifications_file.exists():
                print("No existing classifications found, starting fresh")
                return
            
            with open(classifications_file, 'r') as f:
                save_data = json.load(f)
            
            # Load content items
            for content_id, item_data in save_data.get("content_items", {}).items():
                embedding = np.array(item_data["embedding"]) if item_data["embedding"] else None
                self.content_items[content_id] = ContentItem(
                    content_id=item_data["content_id"],
                    content_type=item_data["content_type"],
                    source_id=item_data["source_id"],
                    text=item_data["text"],
                    embedding=embedding,
                    metadata=item_data["metadata"],
                    timestamp=item_data["timestamp"]
                )
            
            # Load topics
            for topic_id, topic_data in save_data.get("topics", {}).items():
                centroid_embedding = np.array(topic_data["centroid_embedding"]) if topic_data["centroid_embedding"] else None
                topic = Topic(
                    topic_id=topic_data["topic_id"],
                    name=topic_data["name"],
                    description=topic_data["description"],
                    centroid_embedding=centroid_embedding,
                    content_items=set(topic_data.get("content_items", [])),
                    confidence_score=topic_data.get("confidence_score", 0.0),
                    created_timestamp=topic_data.get("created_timestamp", time.time()),
                    last_updated=topic_data.get("last_updated", time.time())
                )
                
                # Restore source_id and content_type
                if topic_data.get("source_id"):
                    topic.source_id = topic_data["source_id"]
                if topic_data.get("content_type"):
                    topic.content_type = topic_data["content_type"]
                
                self.topics[topic_id] = topic
            
            # Load mappings
            for content_id, topic_ids in save_data.get("content_to_topics", {}).items():
                self.content_to_topics[content_id] = set(topic_ids)
            
            for topic_id, content_ids in save_data.get("topic_to_contents", {}).items():
                self.topic_to_contents[topic_id] = set(content_ids)
            
            # Rebuild embeddings matrix
            self._rebuild_embeddings_matrix()
            
            print(f"Loaded existing classifications ({len(self.topics)} topics, {len(self.content_items)} items)")
            
        except Exception as e:
            print(f"Error loading classifications: {e}")
            print("Starting with fresh classifications")
    
    def _rebuild_embeddings_matrix(self):
        """Rebuild the embeddings matrix from loaded topics"""
        if not self.topics:
            self.topic_embeddings_matrix = None
            self.topic_ids_order = []
            return
        
        # Build matrix from topic embeddings
        valid_topics = [(topic_id, topic) for topic_id, topic in self.topics.items() 
                       if topic.centroid_embedding is not None]
        
        if not valid_topics:
            self.topic_embeddings_matrix = None
            self.topic_ids_order = []
            return
        
        self.topic_ids_order = [topic_id for topic_id, _ in valid_topics]
        embeddings = [topic.centroid_embedding for _, topic in valid_topics]
        self.topic_embeddings_matrix = np.array(embeddings)
    
    def get_classification_stats(self) -> Dict[str, Any]:
        """Get statistics about current classification state"""
        cross_topic_items = len(self.get_cross_topic_content())
        
        return {
            "total_topics": len(self.topics),
            "total_content_items": len(self.content_items),
            "cross_topic_items": cross_topic_items,
            "avg_content_per_topic": np.mean([len(items) for items in self.topic_to_contents.values()]) if self.topics else 0,
            "similarity_threshold": self.similarity_threshold,
            "recalibration_count": getattr(self, '_recalibration_count', 0)
        }