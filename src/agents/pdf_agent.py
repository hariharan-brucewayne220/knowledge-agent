"""
PDF Processing Agent for KnowAgent

This agent handles all PDF-related actions from our knowledge base:
- extract_text: Extract text from PDF files
- chunk_document: Split text into semantic chunks
- create_embeddings: Generate vector embeddings
- extract_citations: Find academic citations

This is where the rubber meets the road - turning knowledge base actions
into actual executable code.
"""

import asyncio
import time
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import re

# PDF processing
import PyPDF2
from io import BytesIO

# Embeddings
from sentence_transformers import SentenceTransformer
import numpy as np

# Our base agent
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.base_agent import BaseAgent, ExecutionResult

# Topic classification
try:
    from classification.topic_classifier import DynamicTopicClassifier
    TOPIC_CLASSIFICATION_AVAILABLE = True
except ImportError:
    TOPIC_CLASSIFICATION_AVAILABLE = False
    print("Topic classification not available - install sklearn")

class PDFAgent(BaseAgent):
    """
    Specialized agent for PDF document processing.
    
    This agent implements all the PDF-related actions from our
    ActionKnowledgeBase, turning them into real functionality.
    """
    
    def __init__(self):
        # Initialize embedding model
        print("Loading sentence transformer model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Embedding model loaded!")
        
        # Initialize topic classifier
        if TOPIC_CLASSIFICATION_AVAILABLE:
            self.topic_classifier = DynamicTopicClassifier(
                similarity_threshold=0.7,  # 70% similarity threshold as requested
                storage_path="topic_classifications"
            )
            print("Topic classifier initialized!")
        else:
            self.topic_classifier = None
        
        # Storage for processed documents
        self.document_chunks = {}
        self.document_embeddings = {}
        
        super().__init__("PDFAgent")
    
    def _get_supported_actions(self) -> List[str]:
        """PDF Agent supports these actions from the knowledge base"""
        return [
            "extract_text",
            "chunk_document", 
            "create_embeddings",
            "extract_citations",
            "semantic_search",
            "classify_topics"
        ]
    
    async def execute_action(self, action: str, target: str, **kwargs) -> ExecutionResult:
        """
        Execute PDF processing actions.
        
        This is where we translate knowledge base actions into actual code.
        """
        start_time = time.time()
        
        try:
            if action == "extract_text":
                pdf_path = target.get("pdf_path") if isinstance(target, dict) else target
                result = await self._extract_text(pdf_path, **kwargs)
            elif action == "chunk_document":
                result = await self._chunk_document(target, **kwargs)
            elif action == "create_embeddings":
                result = await self._create_embeddings(target, **kwargs)
            elif action == "extract_citations":
                result = await self._extract_citations(target, **kwargs)
            elif action == "semantic_search":
                result = await self._semantic_search(target, **kwargs)
            elif action == "classify_topics":
                result = await self._classify_topics(target, **kwargs)
            else:
                return ExecutionResult(
                    success=False,
                    output=None,
                    error_message=f"Unknown action: {action}"
                )
            
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            return result
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                output=None,
                error_message=f"Error in {action}: {str(e)}",
                execution_time=time.time() - start_time
            )
    
    async def _extract_text(self, pdf_path: str, **kwargs) -> ExecutionResult:
        """
        Extract text from PDF file.
        
        This implements the 'extract_text' action from our knowledge base.
        """
        print(f"Extracting text from: {pdf_path}")
        
        # Check if we have simulated content (for testing)
        simulated_content = kwargs.get('simulated_content')
        if simulated_content:
            # Use simulated content for testing
            extracted_text = simulated_content
            page_texts = [{
                "page_number": 1,
                "text": simulated_content,
                "char_count": len(simulated_content)
            }]
        else:
            # Check if file exists
            if not os.path.exists(pdf_path):
                return ExecutionResult(
                    success=False,
                    output=None,
                    error_message=f"PDF file not found: {pdf_path}"
                )
        
        try:
            if not simulated_content:
                extracted_text = ""
                page_texts = []
                
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    
                    print(f"PDF has {len(pdf_reader.pages)} pages")
                    
                    for page_num, page in enumerate(pdf_reader.pages):
                        try:
                            page_text = page.extract_text()
                            page_texts.append({
                                "page_number": page_num + 1,
                                "text": page_text,
                                "char_count": len(page_text)
                            })
                            extracted_text += f"\n[Page {page_num + 1}]\n{page_text}\n"
                        except Exception as e:
                            print(f"Warning: Error extracting page {page_num + 1}: {e}")
                            continue
            
            # Store results
            doc_id = Path(pdf_path).stem
            result_data = {
                "document_id": doc_id,
                "file_path": pdf_path,
                "total_pages": len(page_texts),
                "total_characters": len(extracted_text),
                "extracted_text": extracted_text,
                "page_texts": page_texts,
                "extraction_timestamp": time.time()
            }
            
            # Cache the extracted text
            self.document_chunks[f"{doc_id}_raw"] = extracted_text
            
            print(f"Extracted {len(extracted_text)} characters from {len(page_texts)} pages")
            
            return ExecutionResult(
                success=True,
                output=result_data,
                metadata={
                    "action": "extract_text",
                    "pages_processed": len(page_texts),
                    "total_chars": len(extracted_text)
                }
            )
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                output=None,
                error_message=f"PDF extraction failed: {str(e)}"
            )
    
    async def _chunk_document(self, target: str, **kwargs) -> ExecutionResult:
        """
        Split document into semantic chunks.
        
        This implements the 'chunk_document' action from our knowledge base.
        """
        print(f"Chunking document: {target}")
        
        # Get previous results or raw text
        previous_results = kwargs.get('previous_results', {})
        
        # Find the extracted text
        extracted_text = None
        doc_id = None
        
        # Check if we have extraction results
        for key, result in previous_results.items():
            if 'extract_text' in key and isinstance(result, dict):
                extracted_text = result.get('extracted_text')
                doc_id = result.get('document_id')
                break
        
        # Fallback: check our cache
        if not extracted_text:
            doc_id = Path(target).stem if '.' in target else target
            cache_key = f"{doc_id}_raw"
            extracted_text = self.document_chunks.get(cache_key)
        
        if not extracted_text:
            return ExecutionResult(
                success=False,
                output=None,
                error_message="No extracted text found for chunking"
            )
        
        try:
            # Chunking strategy: Split by paragraphs and combine into ~500 char chunks
            paragraphs = [p.strip() for p in extracted_text.split('\n\n') if p.strip()]
            
            chunks = []
            current_chunk = ""
            chunk_size = 500  # Target chunk size
            overlap = 50      # Overlap between chunks
            
            for paragraph in paragraphs:
                # If adding this paragraph would exceed chunk size
                if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
                    # Save current chunk
                    chunks.append({
                        "chunk_id": len(chunks),
                        "text": current_chunk.strip(),
                        "char_count": len(current_chunk),
                        "paragraph_count": current_chunk.count('\n\n') + 1
                    })
                    
                    # Start new chunk with overlap
                    if len(current_chunk) > overlap:
                        current_chunk = current_chunk[-overlap:] + " " + paragraph
                    else:
                        current_chunk = paragraph
                else:
                    # Add paragraph to current chunk
                    if current_chunk:
                        current_chunk += "\n\n" + paragraph
                    else:
                        current_chunk = paragraph
            
            # Don't forget the last chunk
            if current_chunk:
                chunks.append({
                    "chunk_id": len(chunks),
                    "text": current_chunk.strip(),
                    "char_count": len(current_chunk),
                    "paragraph_count": current_chunk.count('\n\n') + 1
                })
            
            # Store chunks
            chunk_key = f"{doc_id}_chunks"
            self.document_chunks[chunk_key] = chunks
            
            result_data = {
                "document_id": doc_id,
                "total_chunks": len(chunks),
                "chunks": chunks,
                "average_chunk_size": sum(c['char_count'] for c in chunks) / len(chunks),
                "chunking_timestamp": time.time()
            }
            
            print(f"Created {len(chunks)} chunks (avg size: {result_data['average_chunk_size']:.0f} chars)")
            
            return ExecutionResult(
                success=True,
                output=result_data,
                metadata={
                    "action": "chunk_document",
                    "chunk_count": len(chunks),
                    "strategy": "paragraph_based_with_overlap"
                }
            )
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                output=None,
                error_message=f"Chunking failed: {str(e)}"
            )
    
    async def _create_embeddings(self, target: str, **kwargs) -> ExecutionResult:
        """
        Generate vector embeddings for text chunks.
        
        This implements the 'create_embeddings' action from our knowledge base.
        """
        print(f"Creating embeddings for: {target}")
        
        # Get previous results
        previous_results = kwargs.get('previous_results', {})
        
        # Find the chunks
        chunks_data = None
        doc_id = None
        
        for key, result in previous_results.items():
            if 'chunk_document' in key and isinstance(result, dict):
                chunks_data = result
                doc_id = result.get('document_id')
                break
        
        if not chunks_data:
            # Fallback: check cache
            doc_id = Path(target).stem if '.' in target else target
            cache_key = f"{doc_id}_chunks"
            if cache_key in self.document_chunks:
                chunks_data = {"chunks": self.document_chunks[cache_key]}
        
        if not chunks_data or 'chunks' not in chunks_data:
            return ExecutionResult(
                success=False,
                output=None,
                error_message="No chunks found for embedding"
            )
        
        try:
            chunks = chunks_data['chunks']
            chunk_texts = [chunk['text'] for chunk in chunks]
            
            print(f"Generating embeddings for {len(chunk_texts)} chunks...")
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(
                chunk_texts,
                batch_size=8,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            # Combine chunks with embeddings
            enriched_chunks = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                enriched_chunks.append({
                    **chunk,
                    "embedding": embedding.tolist(),  # Convert to list for JSON serialization
                    "embedding_model": "all-MiniLM-L6-v2",
                    "embedding_dimension": len(embedding)
                })
            
            # Store embeddings
            embedding_key = f"{doc_id}_embeddings"
            self.document_embeddings[embedding_key] = {
                "embeddings": embeddings,
                "chunks": enriched_chunks,
                "model": "all-MiniLM-L6-v2",
                "dimension": embeddings.shape[1]
            }
            
            # Automatically classify topics if topic classifier is available
            topic_assignments = {}
            if self.topic_classifier:
                print("Classifying chunks into topics...")
                for i, chunk in enumerate(enriched_chunks):
                    content_id = f"{doc_id}_chunk_{chunk['chunk_id']}"
                    assigned_topics = self.topic_classifier.add_content(
                        content_id=content_id,
                        content_type="pdf",
                        source_id=doc_id,
                        text=chunk['text'],
                        embedding=embeddings[i],
                        metadata={
                            "chunk_id": chunk['chunk_id'],
                            "char_count": chunk['char_count'],
                            "paragraph_count": chunk['paragraph_count']
                        }
                    )
                    topic_assignments[content_id] = assigned_topics
                    # Add topic info to chunk
                    chunk['topics'] = assigned_topics
                
                print(f"Classified {len(enriched_chunks)} chunks into topics")
            
            result_data = {
                "document_id": doc_id,
                "embedding_count": len(embeddings),
                "embedding_dimension": embeddings.shape[1],
                "model_name": "all-MiniLM-L6-v2",
                "enriched_chunks": enriched_chunks,
                "topic_assignments": topic_assignments,
                "embedding_timestamp": time.time()
            }
            
            print(f"Generated {len(embeddings)} embeddings (dim: {embeddings.shape[1]})")
            
            return ExecutionResult(
                success=True,
                output=result_data,
                metadata={
                    "action": "create_embeddings",
                    "embedding_count": len(embeddings),
                    "model": "all-MiniLM-L6-v2",
                    "topic_classification": len(topic_assignments) > 0
                }
            )
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                output=None,
                error_message=f"Embedding generation failed: {str(e)}"
            )
    
    async def _extract_citations(self, target: str, **kwargs) -> ExecutionResult:
        """
        Extract academic citations from document text.
        
        This implements the 'extract_citations' action from our knowledge base.
        """
        print(f"Extracting citations from: {target}")
        
        # Get previous results
        previous_results = kwargs.get('previous_results', {})
        
        # Find extracted text
        extracted_text = None
        doc_id = None
        
        for key, result in previous_results.items():
            if 'extract_text' in key and isinstance(result, dict):
                extracted_text = result.get('extracted_text')
                doc_id = result.get('document_id')
                break
        
        if not extracted_text:
            return ExecutionResult(
                success=False,
                output=None,
                error_message="No extracted text found for citation analysis"
            )
        
        try:
            # Citation patterns (basic academic citation detection)
            citation_patterns = [
                # Author, Year format: "Smith et al. (2020)"
                r'([A-Z][a-z]+(?:\s+et\s+al\.)?)\s*\((\d{4})\)',
                # Author Year format: "Smith 2020"
                r'([A-Z][a-z]+)\s+(\d{4})',
                # DOI pattern
                r'doi:\s*(10\.\d+/[^\s]+)',
                # URL pattern for papers
                r'https?://(?:www\.)?(?:arxiv\.org|scholar\.google|pubmed|ieee|acm)[^\s]+',
                # Reference list entries
                r'^[^\n]*\(\d{4}\)[^\n]*$'
            ]
            
            citations = []
            
            for i, pattern in enumerate(citation_patterns):
                matches = re.finditer(pattern, extracted_text, re.MULTILINE | re.IGNORECASE)
                
                for match in matches:
                    citation_text = match.group(0)
                    citations.append({
                        "citation_id": len(citations),
                        "text": citation_text,
                        "pattern_type": f"pattern_{i}",
                        "start_position": match.start(),
                        "end_position": match.end(),
                        "context": extracted_text[max(0, match.start()-50):match.end()+50]
                    })
            
            # Remove duplicates and sort by position
            unique_citations = []
            seen_texts = set()
            
            for citation in sorted(citations, key=lambda x: x['start_position']):
                if citation['text'] not in seen_texts:
                    unique_citations.append(citation)
                    seen_texts.add(citation['text'])
            
            result_data = {
                "document_id": doc_id,
                "citation_count": len(unique_citations),
                "citations": unique_citations,
                "extraction_patterns_used": len(citation_patterns),
                "citation_timestamp": time.time()
            }
            
            print(f"Found {len(unique_citations)} potential citations")
            
            return ExecutionResult(
                success=True,
                output=result_data,
                metadata={
                    "action": "extract_citations",
                    "citation_count": len(unique_citations),
                    "patterns_used": len(citation_patterns)
                }
            )
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                output=None,
                error_message=f"Citation extraction failed: {str(e)}"
            )
    
    async def _semantic_search(self, target: str, **kwargs) -> ExecutionResult:
        """
        Perform semantic search across processed documents.
        
        This implements the 'semantic_search' action from our knowledge base.
        """
        print(f"Performing semantic search for: {target}")
        
        # Get previous results
        previous_results = kwargs.get('previous_results', {})
        
        # Collect embeddings from all documents
        all_chunks = []
        all_embeddings = []
        
        for key, result in previous_results.items():
            if 'create_embeddings' in key and isinstance(result, dict):
                chunks = result.get('enriched_chunks', [])
                all_chunks.extend(chunks)
                
                # Extract embeddings from chunks
                for chunk in chunks:
                    if 'embedding' in chunk:
                        all_embeddings.append(chunk['embedding'])
        
        if not all_chunks or not all_embeddings:
            return ExecutionResult(
                success=False,
                output=None,
                error_message="No embeddings found for semantic search"
            )
        
        try:
            # Encode the search query
            query_embedding = self.embedding_model.encode([target])
            
            # Calculate similarities
            all_embeddings_array = np.array(all_embeddings)
            similarities = np.dot(all_embeddings_array, query_embedding.T).flatten()
            
            # Get top 5 most similar chunks
            top_indices = np.argsort(similarities)[::-1][:5]
            
            search_results = []
            for i, idx in enumerate(top_indices):
                chunk_data = all_chunks[idx]
                similarity_score = similarities[idx]
                
                search_results.append({
                    "rank": i + 1,
                    "chunk_id": chunk_data.get('chunk_id'),
                    "text": chunk_data.get('text', ''),
                    "document_id": chunk_data.get('document_id'),
                    "similarity_score": float(similarity_score),
                    "metadata": chunk_data.get('metadata', {})
                })
            
            result_data = {
                "query": target,
                "total_chunks_searched": len(all_chunks),
                "results_returned": len(search_results),
                "search_results": search_results,
                "search_timestamp": time.time()
            }
            
            print(f"Found {len(search_results)} relevant passages")
            
            return ExecutionResult(
                success=True,
                output=result_data,
                metadata={
                    "action": "semantic_search",
                    "query": target,
                    "results_count": len(search_results)
                }
            )
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                output=None,
                error_message=f"Semantic search failed: {str(e)}"
            )
    
    async def _classify_topics(self, target: str, **kwargs) -> ExecutionResult:
        """
        Classify document content into topics using the topic classifier.
        
        This implements the 'classify_topics' action from our knowledge base.
        """
        print(f"Classifying topics for: {target}")
        
        if not self.topic_classifier:
            return ExecutionResult(
                success=False,
                output=None,
                error_message="Topic classifier not available - install sklearn"
            )
        
        # Get previous results
        previous_results = kwargs.get('previous_results', {})
        
        # Find embeddings data
        embeddings_data = None
        doc_id = None
        
        for key, result in previous_results.items():
            if 'create_embeddings' in key and isinstance(result, dict):
                embeddings_data = result
                doc_id = result.get('document_id')
                break
        
        if not embeddings_data:
            return ExecutionResult(
                success=False,
                output=None,
                error_message="No embeddings found for topic classification"
            )
        
        try:
            enriched_chunks = embeddings_data.get('enriched_chunks', [])
            topic_assignments = {}
            
            # If chunks don't already have topics, classify them
            if not any('topics' in chunk for chunk in enriched_chunks):
                print("Classifying chunks into topics...")
                for chunk in enriched_chunks:
                    content_id = f"{doc_id}_chunk_{chunk['chunk_id']}"
                    embedding = np.array(chunk['embedding'])
                    
                    assigned_topics = self.topic_classifier.add_content(
                        content_id=content_id,
                        content_type="pdf",
                        source_id=doc_id,
                        text=chunk['text'],
                        embedding=embedding,
                        metadata={
                            "chunk_id": chunk['chunk_id'],
                            "char_count": chunk['char_count'],
                            "paragraph_count": chunk['paragraph_count']
                        }
                    )
                    topic_assignments[content_id] = assigned_topics
                    chunk['topics'] = assigned_topics
            else:
                # Extract existing topic assignments
                for chunk in enriched_chunks:
                    content_id = f"{doc_id}_chunk_{chunk['chunk_id']}"
                    topic_assignments[content_id] = chunk.get('topics', [])
            
            # Get topic hierarchy and statistics
            topic_hierarchy = self.topic_classifier.get_topic_hierarchy()
            cross_topic_content = self.topic_classifier.get_cross_topic_content()
            classification_stats = self.topic_classifier.get_classification_stats()
            
            result_data = {
                "document_id": doc_id,
                "topic_assignments": topic_assignments,
                "topic_hierarchy": topic_hierarchy,
                "cross_topic_content": cross_topic_content,
                "classification_stats": classification_stats,
                "total_chunks_classified": len(topic_assignments),
                "classification_timestamp": time.time()
            }
            
            print(f"Classified {len(topic_assignments)} chunks into {len(topic_hierarchy)} topics")
            
            return ExecutionResult(
                success=True,
                output=result_data,
                metadata={
                    "action": "classify_topics",
                    "chunks_classified": len(topic_assignments),
                    "topics_discovered": len(topic_hierarchy),
                    "cross_topic_items": len(cross_topic_content)
                }
            )
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                output=None,
                error_message=f"Topic classification failed: {str(e)}"
            )

# Test the PDF agent
if __name__ == "__main__":
    async def test_pdf_agent():
        agent = PDFAgent()
        print(f"PDF Agent initialized with actions: {agent.supported_actions}")
        
        # Test with a dummy file (would need actual PDF for real test)
        print("\nPDF Agent ready for testing!")
        print("To test with real PDF:")
        print("1. Place a PDF file in the project directory")
        print("2. Call: agent.execute_action('extract_text', 'your_file.pdf')")
    
    asyncio.run(test_pdf_agent())