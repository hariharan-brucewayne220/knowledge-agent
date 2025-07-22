"""
Unified Content Storage System
Single schema for all content types with ChromaDB integration
"""

import os
import json
import time
import hashlib
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field, asdict
import numpy as np
import re

@dataclass
class UnifiedChunk:
    """Individual content chunk with embedding"""
    chunk_id: int                     # 0, 1, 2...
    text: str                         # Chunk text content
    embedding: List[float]            # Vector embedding
    char_count: int                   # Character count
    metadata: Dict[str, Any] = field(default_factory=dict)  # Page, timestamps, etc.

@dataclass
class UnifiedContentItem:
    """Unified content item for all storage systems"""
    # === IDENTITY ===
    id: str                           # Unique: "pdf_solar_efficiency"
    title: str                        # Actual document/video title
    content_type: str                 # "pdf" | "youtube"
    source_path: str                  # File path or YouTube URL
    
    # === CONTENT ===
    chunks: List[UnifiedChunk]        # Processed text chunks with embeddings
    full_text: str                    # Complete text content
    
    # === METADATA ===
    metadata: Dict[str, Any]          # File size, pages, duration, etc.
    created_at: float                 # Unix timestamp
    processing_status: str            # "pending" | "completed" | "failed"
    
    # === SEARCH & RETRIEVAL ===
    keywords: List[str] = field(default_factory=list)  # Extracted keywords
    
    # === TOPICS (Optional) ===
    topic_assignments: List[str] = field(default_factory=list)  # Topic IDs
    confidence_scores: List[float] = field(default_factory=list)  # Per topic

class UnifiedContentStore:
    """
    Unified storage system handling both file storage and vector search
    Single source of truth for all content with ChromaDB integration
    """
    
    def __init__(self, storage_dir: str = "unified_content"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # Storage files
        self.content_index_file = self.storage_dir / "index.json"
        self.vector_db_dir = self.storage_dir / "vectors"
        
        # In-memory cache
        self.content_items: Dict[str, UnifiedContentItem] = {}
        
        # Initialize ChromaDB
        self._init_vector_store()
        
        # Load existing content
        self._load_content_index()
    
    def _init_vector_store(self):
        """Initialize ChromaDB for vector storage"""
        try:
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.vector_db_dir),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Single unified collection
            self.collection = self.chroma_client.get_or_create_collection(
                name="unified_content",
                metadata={"description": "Unified content storage for all document types"}
            )
            
            print(f"Vector store initialized at: {self.vector_db_dir}")
            
        except Exception as e:
            print(f"Vector store initialization failed: {e}")
            self.chroma_client = None
            self.collection = None
    
    def add_pdf_content(self, pdf_path: str, title: str, chunks: List[Dict], metadata: Dict) -> str:
        """Add PDF content with unified schema"""
        content_id = self._generate_content_id(title, 'pdf')
        
        # Convert chunks to unified format
        unified_chunks = []
        full_text_parts = []
        
        for i, chunk_data in enumerate(chunks):
            chunk_text = chunk_data.get('text', '')
            chunk_embedding = chunk_data.get('embedding', [])
            
            unified_chunk = UnifiedChunk(
                chunk_id=i,
                text=chunk_text,
                embedding=chunk_embedding,
                char_count=len(chunk_text),
                metadata={
                    'page_number': chunk_data.get('page_number'),
                    'section': chunk_data.get('section'),
                    'keywords': chunk_data.get('keywords', [])
                }
            )
            unified_chunks.append(unified_chunk)
            full_text_parts.append(chunk_text)
        
        # Advanced NER-based processing
        full_text = ' '.join(full_text_parts)
        ner_keywords = self._extract_keywords(title + ' ' + full_text)
        topic_scores = self._categorize_content(full_text, title)
        
        # Determine primary topics (confidence > 0.3)
        primary_topics = [topic for topic, score in topic_scores.items() if score > 0.3]
        
        # Enhanced metadata with NER insights
        enhanced_metadata = metadata.copy()
        enhanced_metadata.update({
            'ner_topics': topic_scores,
            'primary_topics': primary_topics,
            'confidence_scores': [topic_scores[topic] for topic in primary_topics],
            'keyword_extraction_method': 'advanced_ner'
        })
        
        # Create unified content item
        content_item = UnifiedContentItem(
            id=content_id,
            title=title,
            content_type='pdf',
            source_path=pdf_path,
            chunks=unified_chunks,
            full_text=full_text,
            metadata=enhanced_metadata,
            created_at=time.time(),
            processing_status='completed',
            keywords=ner_keywords,
            topic_assignments=primary_topics,
            confidence_scores=[topic_scores[topic] for topic in primary_topics]
        )
        
        # Store in both systems
        self.content_items[content_id] = content_item
        self._save_content_index()
        self._store_in_vector_db(content_item)
        
        print(f"Added PDF content: {title} ({len(unified_chunks)} chunks)")
        return content_id
    
    def add_youtube_content(self, url: str, title: str, transcript_segments: List[Dict], metadata: Dict) -> str:
        """Add YouTube content with unified schema"""
        content_id = self._generate_content_id(title, 'youtube')
        
        # Convert segments to unified format
        unified_chunks = []
        full_text_parts = []
        
        for i, segment_data in enumerate(transcript_segments):
            segment_text = segment_data.get('text', '')
            segment_embedding = segment_data.get('embedding', [])
            
            unified_chunk = UnifiedChunk(
                chunk_id=i,
                text=segment_text,
                embedding=segment_embedding,
                char_count=len(segment_text),
                metadata={
                    'start_time': segment_data.get('start_time'),
                    'end_time': segment_data.get('end_time'),
                    'keywords': segment_data.get('keywords', [])
                }
            )
            unified_chunks.append(unified_chunk)
            full_text_parts.append(segment_text)
        
        # Advanced NER-based processing
        full_text = ' '.join(full_text_parts)
        ner_keywords = self._extract_keywords(title + ' ' + full_text)
        topic_scores = self._categorize_content(full_text, title)
        
        # Determine primary topics (confidence > 0.3)
        primary_topics = [topic for topic, score in topic_scores.items() if score > 0.3]
        
        # Enhanced metadata with NER insights
        enhanced_metadata = metadata.copy()
        enhanced_metadata.update({
            'ner_topics': topic_scores,
            'primary_topics': primary_topics,
            'confidence_scores': [topic_scores[topic] for topic in primary_topics],
            'keyword_extraction_method': 'advanced_ner'
        })
        
        # Create unified content item
        content_item = UnifiedContentItem(
            id=content_id,
            title=title,
            content_type='youtube',
            source_path=url,
            chunks=unified_chunks,
            full_text=full_text,
            metadata=enhanced_metadata,
            created_at=time.time(),
            processing_status='completed',
            keywords=ner_keywords,
            topic_assignments=primary_topics,
            confidence_scores=[topic_scores[topic] for topic in primary_topics]
        )
        
        # Store in both systems
        self.content_items[content_id] = content_item
        self._save_content_index()
        self._store_in_vector_db(content_item)
        
        print(f"Added YouTube content: {title} ({len(unified_chunks)} chunks)")
        return content_id
    
    def _store_in_vector_db(self, content_item: UnifiedContentItem):
        """Store content chunks in ChromaDB"""
        if not self.collection:
            return
        
        try:
            ids = []
            documents = []
            embeddings = []
            metadatas = []
            
            for chunk in content_item.chunks:
                chunk_id = f"{content_item.id}_chunk_{chunk.chunk_id}"
                
                ids.append(chunk_id)
                documents.append(chunk.text)
                embeddings.append(chunk.embedding)
                metadatas.append({
                    'content_id': content_item.id,
                    'title': content_item.title,
                    'content_type': content_item.content_type,
                    'source_path': content_item.source_path,
                    'chunk_id': chunk.chunk_id,
                    'char_count': chunk.char_count,
                    **chunk.metadata
                })
            
            # Add to ChromaDB
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
        except Exception as e:
            print(f"Failed to store in vector DB: {e}")
    
    def semantic_search(self, query: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """Perform semantic search across all content"""
        if not self.collection:
            return []
        
        try:
            # Query ChromaDB
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            search_results = []
            for i in range(len(results['ids'][0])):
                search_results.append({
                    'chunk_id': results['ids'][0][i],
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'similarity_score': 1 - results['distances'][0][i],  # Convert distance to similarity
                })
            
            return search_results
            
        except Exception as e:
            print(f"Semantic search failed: {e}")
            return []
    
    def get_content_by_type(self, content_type: str) -> List[UnifiedContentItem]:
        """Get all content of specific type"""
        return [item for item in self.content_items.values() if item.content_type == content_type]
    
    def get_all_content(self) -> List[UnifiedContentItem]:
        """Get all content items"""
        return list(self.content_items.values())
    
    def search_content_by_name(self, search_term: str) -> List[UnifiedContentItem]:
        """Search content by title - compatibility method"""
        results = []
        search_term_lower = search_term.lower()
        
        for content_item in self.content_items.values():
            if search_term_lower in content_item.title.lower():
                results.append(content_item)
        
        return results
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        pdf_count = len([item for item in self.content_items.values() if item.content_type == 'pdf'])
        youtube_count = len([item for item in self.content_items.values() if item.content_type == 'youtube'])
        total_chunks = sum(len(item.chunks) for item in self.content_items.values())
        
        return {
            'total_items': len(self.content_items),
            'pdf_count': pdf_count,
            'youtube_count': youtube_count,
            'total_chunks': total_chunks,
            'vector_db_active': self.collection is not None
        }
    
    def _generate_content_id(self, title: str, content_type: str) -> str:
        """Generate unique content ID"""
        # Clean title for ID
        clean_title = "".join(c for c in title if c.isalnum() or c in (' ', '_')).rstrip()
        clean_title = "_".join(clean_title.split()).lower()
        return f"{content_type}_{clean_title}"
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Advanced NER-based keyword extraction"""
        return self._ner_extract_keywords(text)
    
    def _ner_extract_keywords(self, text: str) -> List[str]:
        """Extract technical entities and keywords using NER patterns"""
        
        # Technical term patterns for different domains
        patterns = {
            # Solar/Energy terms
            'solar_tech': r'\b(solar|photovoltaic|panel|silicon|perovskite|efficiency|grid|renewable|inverter|monocrystalline|polycrystalline)\b',
            'energy_measurements': r'\b(\d+(?:\.\d+)?)\s*(?:%|kWh|MW|GW|watts?)\b',
            'energy_systems': r'\b(energy|power|electricity|generation|supply|storage|battery|lithium)\b',
            
            # Carbon/Sequestration terms  
            'carbon_tech': r'\b(carbon|sequestration|capture|DAC|CO₂|CO2|storage|geological|biological|weathering)\b',
            'carbon_processes': r'\b(direct\s+air\s+capture|enhanced\s+weathering|ocean\s+alkalinization|biochar|forest\s+sequestration)\b',
            'carbon_measurements': r'\b(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:tons?|kg|Gt)\s*(?:CO₂|CO2|carbon)?\b',
            
            # Materials and compounds
            'materials': r'\b(silicon|lithium|olivine|tandem|perovskite|oxide|mineral|compound)\b',
            'chemicals': r'\b([A-Z][a-z]*₂?|CO₂|CO2|H₂O|NaCl)\b',
            
            # Technologies and methods
            'technologies': r'\b([A-Z][a-z]*(?:\s+[A-Z][a-z]*)*)\s+(?:technology|method|system|process|technique)\b',
            'equipment': r'\b(cells?|panels?|arrays?|inverters?|transformers?|pumps?|reactors?)\b',
            
            # Performance metrics
            'percentages': r'\b(\d+(?:\.\d+)?)\s*%\b',
            'costs': r'\$(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:per|/)?\s*(kWh|ton|kg|MW)?\b',
            'temperatures': r'\b(\d+(?:\.\d+)?)\s*°?C\b',
            'times': r'\b(\d+(?:\.\d+)?)\s*(?:years?|months?|days?|hours?|minutes?)\b'
        }
        
        extracted = set()
        text_lower = text.lower()
        
        # Extract using patterns
        for category, pattern in patterns.items():
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if matches:
                # Handle different match types
                for match in matches:
                    if isinstance(match, tuple):
                        # Multiple groups - join them
                        extracted.add(' '.join(str(m) for m in match if m).strip())
                    else:
                        # Single match
                        extracted.add(str(match).strip())
        
        # Add title words (filtered)
        title_words = [word.lower().strip() for word in re.split(r'[^\w]+', text) 
                      if len(word) > 3 and word.lower() not in 
                      {'this', 'that', 'with', 'from', 'they', 'have', 'been', 'were', 'will', 'would', 'could', 'should'}]
        extracted.update(title_words[:10])  # Top 10 title words
        
        # Clean and filter results
        keywords = []
        for keyword in extracted:
            keyword = keyword.strip().lower()
            if len(keyword) >= 2 and keyword.replace(' ', '').isalnum():
                keywords.append(keyword)
        
        # Return top 25 most relevant keywords
        return list(set(keywords))[:25]
    
    def _categorize_content(self, text: str, title: str) -> Dict[str, float]:
        """Categorize content into topic areas with confidence scores using NER"""
        
        combined_text = f"{title} {text}".lower()
        
        # Define topic signatures with weights
        topic_signatures = {
            'solar_energy': {
                'primary': ['solar', 'photovoltaic', 'panel', 'silicon', 'efficiency'],
                'secondary': ['grid', 'renewable', 'inverter', 'monocrystalline', 'polycrystalline'],
                'technical': [r'\d+(?:\.\d+)?%', r'\d+\s*kWh', r'theoretical\s+limit']
            },
            'carbon_sequestration': {
                'primary': ['carbon', 'sequestration', 'capture', 'dac', 'co2'],
                'secondary': ['storage', 'geological', 'biological', 'weathering'],
                'technical': [r'\d+(?:,\d{3})*\s*tons?', r'direct\s+air\s+capture', r'enhanced\s+weathering']
            },
            'energy_systems': {
                'primary': ['energy', 'power', 'electricity', 'generation'],
                'secondary': ['supply', 'storage', 'battery', 'lithium', 'grid'],
                'technical': [r'\d+\s*(?:MW|GW|kWh)', r'energy\s+requirements', r'power\s+generation']
            },
            'materials_science': {
                'primary': ['silicon', 'lithium', 'perovskite', 'materials'],
                'secondary': ['compound', 'substrate', 'crystal', 'tandem'],
                'technical': [r'theoretical\s+limit', r'temperature\s+coefficient', r'degradation\s+rate']
            },
            'environmental': {
                'primary': ['climate', 'emissions', 'environmental'],
                'secondary': ['atmosphere', 'greenhouse', 'mitigation'],
                'technical': [r'climate\s+change', r'greenhouse\s+gas', r'environmental\s+impact']
            }
        }
        
        scores = {}
        
        for topic, signature in topic_signatures.items():
            score = 0.0
            
            # Primary terms (high weight)
            primary_matches = sum(1 for term in signature['primary'] if term in combined_text)
            score += primary_matches * 3.0
            
            # Secondary terms (medium weight)  
            secondary_matches = sum(1 for term in signature['secondary'] if term in combined_text)
            score += secondary_matches * 1.5
            
            # Technical patterns (very high weight)
            technical_matches = sum(1 for pattern in signature['technical'] 
                                  if re.search(pattern, combined_text, re.IGNORECASE))
            score += technical_matches * 5.0
            
            # Normalize by total possible score
            max_possible = len(signature['primary']) * 3.0 + len(signature['secondary']) * 1.5 + len(signature['technical']) * 5.0
            confidence = min(score / max_possible, 1.0) if max_possible > 0 else 0.0
            
            scores[topic] = confidence
        
        return scores
    
    def _load_content_index(self):
        """Load content index from file"""
        try:
            if self.content_index_file.exists():
                with open(self.content_index_file, 'r') as f:
                    data = json.load(f)
                    
                for item_data in data.get('content_items', []):
                    # Convert chunk dictionaries back to UnifiedChunk objects
                    chunks = []
                    for chunk_data in item_data.get('chunks', []):
                        chunk = UnifiedChunk(**chunk_data)
                        chunks.append(chunk)
                    
                    # Create UnifiedContentItem
                    item_data['chunks'] = chunks
                    content_item = UnifiedContentItem(**item_data)
                    self.content_items[content_item.id] = content_item
                    
                print(f"Loaded {len(self.content_items)} content items from index")
            else:
                print("No existing content index found, starting fresh")
                
        except Exception as e:
            print(f"Error loading content index: {e}")
    
    def _save_content_index(self):
        """Save content index to file"""
        try:
            data = {
                'content_items': [asdict(item) for item in self.content_items.values()],
                'metadata': {
                    'total_items': len(self.content_items),
                    'last_updated': time.time(),
                    'schema_version': '2.0'
                }
            }
            
            with open(self.content_index_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"Error saving content index: {e}")

# Test functionality
if __name__ == "__main__":
    store = UnifiedContentStore()
    
    print("Unified Content Store initialized!")
    print(f"Storage stats: {store.get_storage_stats()}")