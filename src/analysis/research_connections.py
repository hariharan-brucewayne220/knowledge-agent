"""
Research Connections Analysis Engine
Finds contradictions, confirmations, extensions, and gaps across research content
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Any, Optional
from dataclasses import dataclass, field
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import re
import time

@dataclass
class ContentChunk:
    """Represents a piece of content for analysis"""
    id: str
    text: str
    embedding: np.ndarray
    source_id: str
    content_type: str  # 'pdf' or 'video'
    metadata: Dict = field(default_factory=dict)

@dataclass
class ResearchConnection:
    """Represents a connection between two pieces of content"""
    chunk1_id: str
    chunk2_id: str
    connection_type: str  # 'contradiction', 'confirmation', 'extension'
    confidence: float
    explanation: str
    chunk1_text: str
    chunk2_text: str
    source1: str
    source2: str

@dataclass
class ResearchGap:
    """Represents a potential gap in research"""
    topic_area: str
    description: str
    related_chunks: List[str]
    confidence: float

@dataclass
class ResearchConnectionsResult:
    """Complete analysis result"""
    contradictions: List[ResearchConnection]
    confirmations: List[ResearchConnection]
    extensions: List[ResearchConnection]
    gaps: List[ResearchGap]
    analysis_metadata: Dict[str, Any]

class ResearchConnectionsAnalyzer:
    """
    Analyzes research content to find meaningful connections and gaps
    """
    
    def __init__(self, 
                 contradiction_threshold: float = 0.15,
                 confirmation_threshold: float = 0.85,
                 extension_threshold: float = 0.7,
                 max_connections_per_type: int = 10):
        """
        Initialize the research connections analyzer
        
        Args:
            contradiction_threshold: Similarity threshold below which we look for contradictions
            confirmation_threshold: Similarity threshold above which we consider confirmations
            extension_threshold: Similarity threshold for detecting extensions/building-upon
            max_connections_per_type: Maximum connections to return per type
        """
        self.contradiction_threshold = contradiction_threshold
        self.confirmation_threshold = confirmation_threshold
        self.extension_threshold = extension_threshold
        self.max_connections_per_type = max_connections_per_type
        
        # Semantic patterns for detecting different connection types
        self.contradiction_patterns = [
            r'\b(but|however|although|despite|nevertheless|contradicts?|disagrees?|opposes?)\b',
            r'\b(not|no|never|none|neither|false|incorrect|wrong|inaccurate)\b',
            r'\b(unlike|different|contrary|opposite|versus|vs\.?|against)\b'
        ]
        
        self.confirmation_patterns = [
            r'\b(confirms?|supports?|agrees?|validates?|proves?|demonstrates?)\b',
            r'\b(similarly|likewise|also|too|as well|additionally|furthermore)\b',
            r'\b(consistent|compatible|aligns?|matches|corresponds?)\b'
        ]
        
        self.extension_patterns = [
            r'\b(builds? on|extends?|expands?|develops?|elaborates?|adds? to)\b',
            r'\b(furthermore|moreover|additionally|in addition|beyond)\b',
            r'\b(therefore|thus|consequently|as a result|leads? to)\b'
        ]
        
        self.uncertainty_patterns = [
            r'\b(uncertain|unclear|unknown|needs? research|further study|investigate)\b',
            r'\b(question|debate|controversy|dispute|disagreement)\b',
            r'\b(might|may|could|possibly|perhaps|potentially|probably)\b'
        ]

    def analyze_topic_connections(self, content_chunks: List[ContentChunk]) -> ResearchConnectionsResult:
        """
        Analyze connections within a topic's content
        
        Args:
            content_chunks: List of content chunks to analyze
            
        Returns:
            ResearchConnectionsResult with all discovered connections
        """
        start_time = time.time()
        
        if len(content_chunks) < 2:
            return ResearchConnectionsResult([], [], [], [], {
                "total_chunks": len(content_chunks),
                "analysis_time": time.time() - start_time,
                "message": "Need at least 2 content chunks for connection analysis"
            })
        
        # Find all pairwise connections
        contradictions = self._find_contradictions(content_chunks)
        confirmations = self._find_confirmations(content_chunks)
        extensions = self._find_extensions(content_chunks)
        gaps = self._find_research_gaps(content_chunks)
        
        # Sort by confidence and limit results
        contradictions = sorted(contradictions, key=lambda x: x.confidence, reverse=True)[:self.max_connections_per_type]
        confirmations = sorted(confirmations, key=lambda x: x.confidence, reverse=True)[:self.max_connections_per_type]
        extensions = sorted(extensions, key=lambda x: x.confidence, reverse=True)[:self.max_connections_per_type]
        gaps = sorted(gaps, key=lambda x: x.confidence, reverse=True)[:self.max_connections_per_type]
        
        analysis_metadata = {
            "total_chunks": len(content_chunks),
            "pairs_analyzed": len(content_chunks) * (len(content_chunks) - 1) // 2,
            "contradictions_found": len(contradictions),
            "confirmations_found": len(confirmations),
            "extensions_found": len(extensions),
            "gaps_found": len(gaps),
            "analysis_time": time.time() - start_time
        }
        
        return ResearchConnectionsResult(
            contradictions=contradictions,
            confirmations=confirmations,
            extensions=extensions,
            gaps=gaps,
            analysis_metadata=analysis_metadata
        )

    def _find_contradictions(self, chunks: List[ContentChunk]) -> List[ResearchConnection]:
        """Find contradictory content between chunks"""
        contradictions = []
        
        for i in range(len(chunks)):
            for j in range(i + 1, len(chunks)):
                chunk1, chunk2 = chunks[i], chunks[j]
                
                # Skip if from same source (less likely to contradict)
                if chunk1.source_id == chunk2.source_id:
                    continue
                
                # Calculate semantic similarity
                similarity = cosine_similarity(
                    chunk1.embedding.reshape(1, -1), 
                    chunk2.embedding.reshape(1, -1)
                )[0][0]
                
                # Look for contradictions: semantically related but textually opposing
                if 0.3 <= similarity <= 0.7:  # Related topic but not identical
                    contradiction_score = self._detect_textual_contradiction(chunk1.text, chunk2.text)
                    
                    if contradiction_score > 0.3:  # Found contradiction indicators
                        confidence = contradiction_score * (1 - abs(similarity - 0.5) * 2)  # Peak at 0.5 similarity
                        
                        explanation = self._generate_contradiction_explanation(chunk1.text, chunk2.text)
                        
                        contradictions.append(ResearchConnection(
                            chunk1_id=chunk1.id,
                            chunk2_id=chunk2.id,
                            connection_type='contradiction',
                            confidence=confidence,
                            explanation=explanation,
                            chunk1_text=chunk1.text[:200] + "..." if len(chunk1.text) > 200 else chunk1.text,
                            chunk2_text=chunk2.text[:200] + "..." if len(chunk2.text) > 200 else chunk2.text,
                            source1=chunk1.source_id,
                            source2=chunk2.source_id
                        ))
        
        return contradictions

    def _find_confirmations(self, chunks: List[ContentChunk]) -> List[ResearchConnection]:
        """Find content that confirms or supports each other"""
        confirmations = []
        
        for i in range(len(chunks)):
            for j in range(i + 1, len(chunks)):
                chunk1, chunk2 = chunks[i], chunks[j]
                
                # Calculate semantic similarity
                similarity = cosine_similarity(
                    chunk1.embedding.reshape(1, -1), 
                    chunk2.embedding.reshape(1, -1)
                )[0][0]
                
                # Look for confirmations: high semantic similarity + supporting language
                if similarity >= self.confirmation_threshold:
                    confirmation_score = self._detect_textual_confirmation(chunk1.text, chunk2.text)
                    
                    # Boost score if from different sources (more valuable confirmation)
                    source_diversity_bonus = 0.2 if chunk1.source_id != chunk2.source_id else 0
                    
                    confidence = min(1.0, similarity * 0.7 + confirmation_score * 0.3 + source_diversity_bonus)
                    
                    if confidence > 0.7:
                        explanation = self._generate_confirmation_explanation(chunk1.text, chunk2.text, similarity)
                        
                        confirmations.append(ResearchConnection(
                            chunk1_id=chunk1.id,
                            chunk2_id=chunk2.id,
                            connection_type='confirmation',
                            confidence=confidence,
                            explanation=explanation,
                            chunk1_text=chunk1.text[:200] + "..." if len(chunk1.text) > 200 else chunk1.text,
                            chunk2_text=chunk2.text[:200] + "..." if len(chunk2.text) > 200 else chunk2.text,
                            source1=chunk1.source_id,
                            source2=chunk2.source_id
                        ))
        
        return confirmations

    def _find_extensions(self, chunks: List[ContentChunk]) -> List[ResearchConnection]:
        """Find content that extends or builds upon other content"""
        extensions = []
        
        for i in range(len(chunks)):
            for j in range(i + 1, len(chunks)):
                chunk1, chunk2 = chunks[i], chunks[j]
                
                # Calculate semantic similarity
                similarity = cosine_similarity(
                    chunk1.embedding.reshape(1, -1), 
                    chunk2.embedding.reshape(1, -1)
                )[0][0]
                
                # Look for extensions: moderate similarity + building/extending language
                if 0.5 <= similarity <= 0.8:
                    extension_score = self._detect_textual_extension(chunk1.text, chunk2.text)
                    
                    if extension_score > 0.3:
                        confidence = similarity * 0.6 + extension_score * 0.4
                        
                        explanation = self._generate_extension_explanation(chunk1.text, chunk2.text)
                        
                        extensions.append(ResearchConnection(
                            chunk1_id=chunk1.id,
                            chunk2_id=chunk2.id,
                            connection_type='extension',
                            confidence=confidence,
                            explanation=explanation,
                            chunk1_text=chunk1.text[:200] + "..." if len(chunk1.text) > 200 else chunk1.text,
                            chunk2_text=chunk2.text[:200] + "..." if len(chunk2.text) > 200 else chunk2.text,
                            source1=chunk1.source_id,
                            source2=chunk2.source_id
                        ))
        
        return extensions

    def _find_research_gaps(self, chunks: List[ContentChunk]) -> List[ResearchGap]:
        """Identify potential gaps in the research"""
        gaps = []
        
        # Analyze content for uncertainty indicators
        uncertain_areas = defaultdict(list)
        
        for chunk in chunks:
            uncertainty_score = self._detect_uncertainty(chunk.text)
            if uncertainty_score > 0.3:
                # Extract the uncertain topic area
                topic_area = self._extract_uncertain_topic(chunk.text)
                uncertain_areas[topic_area].append((chunk.id, uncertainty_score))
        
        # Create gaps from uncertain areas
        for topic_area, chunk_data in uncertain_areas.items():
            if len(chunk_data) >= 1:  # At least one mention of uncertainty
                confidence = min(1.0, sum(score for _, score in chunk_data) / len(chunk_data))
                related_chunks = [chunk_id for chunk_id, _ in chunk_data]
                
                description = f"Multiple sources indicate uncertainty or need for further research in {topic_area.lower()}"
                
                gaps.append(ResearchGap(
                    topic_area=topic_area,
                    description=description,
                    related_chunks=related_chunks,
                    confidence=confidence
                ))
        
        return gaps

    def _detect_textual_contradiction(self, text1: str, text2: str) -> float:
        """Detect contradiction indicators in text pairs"""
        text1_lower = text1.lower()
        text2_lower = text2.lower()
        
        score = 0.0
        matches = 0
        
        for pattern in self.contradiction_patterns:
            matches1 = len(re.findall(pattern, text1_lower))
            matches2 = len(re.findall(pattern, text2_lower))
            if matches1 > 0 or matches2 > 0:
                score += 0.3
                matches += matches1 + matches2
        
        # Normalize score
        return min(1.0, score + min(0.4, matches * 0.1))

    def _detect_textual_confirmation(self, text1: str, text2: str) -> float:
        """Detect confirmation indicators in text pairs"""
        text1_lower = text1.lower()
        text2_lower = text2.lower()
        
        score = 0.0
        matches = 0
        
        for pattern in self.confirmation_patterns:
            matches1 = len(re.findall(pattern, text1_lower))
            matches2 = len(re.findall(pattern, text2_lower))
            if matches1 > 0 or matches2 > 0:
                score += 0.2
                matches += matches1 + matches2
        
        return min(1.0, score + min(0.3, matches * 0.05))

    def _detect_textual_extension(self, text1: str, text2: str) -> float:
        """Detect extension indicators in text pairs"""
        text1_lower = text1.lower()
        text2_lower = text2.lower()
        
        score = 0.0
        matches = 0
        
        for pattern in self.extension_patterns:
            matches1 = len(re.findall(pattern, text1_lower))
            matches2 = len(re.findall(pattern, text2_lower))
            if matches1 > 0 or matches2 > 0:
                score += 0.25
                matches += matches1 + matches2
        
        return min(1.0, score + min(0.3, matches * 0.08))

    def _detect_uncertainty(self, text: str) -> float:
        """Detect uncertainty indicators in text"""
        text_lower = text.lower()
        
        score = 0.0
        matches = 0
        
        for pattern in self.uncertainty_patterns:
            pattern_matches = len(re.findall(pattern, text_lower))
            if pattern_matches > 0:
                score += 0.3
                matches += pattern_matches
        
        return min(1.0, score + min(0.4, matches * 0.1))

    def _extract_uncertain_topic(self, text: str) -> str:
        """Extract the topic area that shows uncertainty"""
        # Simple heuristic: look for key terms near uncertainty indicators
        words = text.lower().split()
        
        # Find uncertainty words
        uncertainty_positions = []
        for i, word in enumerate(words):
            for pattern in self.uncertainty_patterns:
                if re.search(pattern, word):
                    uncertainty_positions.append(i)
                    break
        
        # Extract nearby topic words
        topic_words = []
        for pos in uncertainty_positions:
            # Look at words around the uncertainty indicator
            start = max(0, pos - 5)
            end = min(len(words), pos + 5)
            
            for j in range(start, end):
                word = words[j].strip('.,!?;:')
                if len(word) > 3 and word.isalpha():  # Skip short and non-alphabetic words
                    topic_words.append(word)
        
        if topic_words:
            # Return most common topic word, capitalized
            from collections import Counter
            most_common = Counter(topic_words).most_common(1)[0][0]
            return most_common.title()
        
        return "Research Area"

    def _generate_contradiction_explanation(self, text1: str, text2: str) -> str:
        """Generate human-readable explanation for contradiction"""
        return f"These sources present conflicting information on related topics. Source 1 suggests one perspective while Source 2 indicates a different viewpoint."

    def _generate_confirmation_explanation(self, text1: str, text2: str, similarity: float) -> str:
        """Generate human-readable explanation for confirmation"""
        return f"These sources strongly support each other (similarity: {similarity:.1%}). They present consistent information from different perspectives."

    def _generate_extension_explanation(self, text1: str, text2: str) -> str:
        """Generate human-readable explanation for extension"""
        return f"These sources build upon each other. One provides foundational information while the other extends or develops the ideas further."