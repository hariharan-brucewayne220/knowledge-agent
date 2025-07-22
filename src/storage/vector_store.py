"""
Persistent Vector Storage for KnowledgeAgent
Uses ChromaDB for efficient vector similarity search
"""

import chromadb
from chromadb.config import Settings
import os
import json
import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np

class KnowledgeAgentVectorStore:
    """
    Persistent vector storage for KnowledgeAgent research system
    """
    
    def __init__(self, persist_directory: str = "knowledgeagent_vectordb"):
        """
        Initialize the vector store
        
        Args:
            persist_directory: Directory to store the vector database
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(exist_ok=True)
        
        # Initialize ChromaDB with persistence and explicit settings to avoid encoding issues
        try:
            # Set environment variable to avoid .env file reading issues
            os.environ['CHROMA_ENV_FILE'] = ''
            
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(
                    anonymized_telemetry=False,
                    is_persistent=True,
                    allow_reset=True
                )
            )
        except UnicodeDecodeError as e:
            print(f"Unicode error in ChromaDB initialization: {e}")
            # Try to create a fresh client without existing data
            import shutil
            if self.persist_directory.exists():
                shutil.rmtree(self.persist_directory)
                self.persist_directory.mkdir(exist_ok=True)
            
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(
                    anonymized_telemetry=False,
                    is_persistent=True,
                    allow_reset=True
                )
            )
        
        # Create collections for different content types
        self.pdf_collection = self._get_or_create_collection("pdf_documents")
        self.video_collection = self._get_or_create_collection("video_transcripts")
        
        print(f"Vector store initialized at: {self.persist_directory}")
    
    def _get_or_create_collection(self, name: str):
        """Get existing collection or create new one"""
        try:
            return self.client.get_collection(name)
        except Exception:
            # Collection doesn't exist, create it
            return self.client.create_collection(
                name=name,
                metadata={"description": f"KnowAgent {name} collection"}
            )
    
    def _generate_chunk_id(self, document_id: str, chunk_index: int) -> str:
        """Generate unique ID for document chunk"""
        return f"{document_id}_chunk_{chunk_index}"
    
    def store_pdf_document(self, 
                          document_id: str, 
                          chunks: List[Dict[str, Any]], 
                          embeddings: np.ndarray,
                          metadata: Dict[str, Any] = None,
                          topic_assignments: Dict[str, List[str]] = None) -> bool:
        """
        Store PDF document chunks and embeddings
        
        Args:
            document_id: Unique identifier for the document
            chunks: List of text chunks with metadata
            embeddings: Numpy array of embeddings for each chunk
            metadata: Additional document metadata
            topic_assignments: Dictionary mapping content_id to list of topic IDs
        
        Returns:
            Success status
        """
        try:
            # Prepare data for storage
            chunk_ids = []
            chunk_texts = []
            chunk_embeddings = []
            chunk_metadatas = []
            
            for i, chunk in enumerate(chunks):
                chunk_id = self._generate_chunk_id(document_id, i)
                chunk_ids.append(chunk_id)
                chunk_texts.append(chunk.get('text', ''))
                chunk_embeddings.append(embeddings[i].tolist())
                
                # Combine chunk metadata with document metadata
                chunk_metadata = {
                    "document_id": document_id,
                    "chunk_index": i,
                    "chunk_size": len(chunk.get('text', '')),
                    "page_number": chunk.get('page_number', 0),
                    "content_type": "pdf",
                    **(metadata or {}),
                    **(chunk.get('metadata', {}))
                }
                
                # Add topic assignments if available
                if topic_assignments:
                    content_id = f"{document_id}_chunk_{i}"
                    if content_id in topic_assignments:
                        chunk_metadata["topics"] = ",".join(topic_assignments[content_id])
                        chunk_metadata["topic_count"] = len(topic_assignments[content_id])
                    else:
                        chunk_metadata["topics"] = ""
                        chunk_metadata["topic_count"] = 0
                
                # Include topics from chunk if available
                if 'topics' in chunk:
                    chunk_metadata["topics"] = ",".join(chunk['topics'])
                    chunk_metadata["topic_count"] = len(chunk['topics'])
                
                chunk_metadatas.append(chunk_metadata)
            
            # Store in ChromaDB
            self.pdf_collection.add(
                ids=chunk_ids,
                documents=chunk_texts,
                embeddings=chunk_embeddings,
                metadatas=chunk_metadatas
            )
            
            print(f"Stored PDF document '{document_id}' with {len(chunks)} chunks")
            return True
            
        except Exception as e:
            print(f"Error storing PDF document: {str(e)}")
            return False
    
    def store_video_transcript(self,
                              video_id: str,
                              segments: List[Dict[str, Any]],
                              embeddings: np.ndarray,
                              metadata: Dict[str, Any] = None,
                              topic_assignments: Dict[str, List[str]] = None) -> bool:
        """
        Store video transcript segments and embeddings
        
        Args:
            video_id: Unique identifier for the video
            segments: List of transcript segments with timestamps
            embeddings: Numpy array of embeddings for each segment
            metadata: Additional video metadata
            topic_assignments: Dictionary mapping content_id to list of topic IDs
        
        Returns:
            Success status
        """
        try:
            segment_ids = []
            segment_texts = []
            segment_embeddings = []
            segment_metadatas = []
            
            for i, segment in enumerate(segments):
                segment_id = f"{video_id}_segment_{i}"
                segment_ids.append(segment_id)
                segment_texts.append(segment.get('text', ''))
                segment_embeddings.append(embeddings[i].tolist())
                
                segment_metadata = {
                    "video_id": video_id,
                    "segment_index": i,
                    "start_time": segment.get('start_time', 0),
                    "end_time": segment.get('end_time', 0),
                    "duration": segment.get('duration', 0),
                    "content_type": "video",
                    **(metadata or {}),
                    **(segment.get('metadata', {}))
                }
                
                # Add topic assignments if available
                if topic_assignments:
                    content_id = f"{video_id}_segment_{i}"
                    if content_id in topic_assignments:
                        segment_metadata["topics"] = ",".join(topic_assignments[content_id])
                        segment_metadata["topic_count"] = len(topic_assignments[content_id])
                    else:
                        segment_metadata["topics"] = ""
                        segment_metadata["topic_count"] = 0
                
                # Include topics from segment if available
                if 'topics' in segment:
                    segment_metadata["topics"] = ",".join(segment['topics'])
                    segment_metadata["topic_count"] = len(segment['topics'])
                
                segment_metadatas.append(segment_metadata)
            
            # Store in ChromaDB
            self.video_collection.add(
                ids=segment_ids,
                documents=segment_texts,
                embeddings=segment_embeddings,
                metadatas=segment_metadatas
            )
            
            print(f"Stored video '{video_id}' with {len(segments)} segments")
            return True
            
        except Exception as e:
            print(f"Error storing video transcript: {str(e)}")
            return False
    
    def search_documents(self, 
                        query_embedding: np.ndarray, 
                        n_results: int = 10,
                        document_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar document chunks
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            document_filter: Optional filter for document metadata
        
        Returns:
            List of search results with metadata
        """
        try:
            results = self.pdf_collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
                where=document_filter
            )
            
            return self._format_search_results(results)
            
        except Exception as e:
            print(f"Error searching documents: {str(e)}")
            return []
    
    def search_videos(self,
                     query_embedding: np.ndarray,
                     n_results: int = 10,
                     video_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar video segments
        """
        try:
            results = self.video_collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
                where=video_filter
            )
            
            return self._format_search_results(results)
            
        except Exception as e:
            print(f"Error searching videos: {str(e)}")
            return []
    
    def search_all_content(self,
                          query_embedding: np.ndarray,
                          n_results: int = 10) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search across all content types
        
        Returns:
            Dictionary with 'documents' and 'videos' results
        """
        doc_results = self.search_documents(query_embedding, n_results // 2)
        video_results = self.search_videos(query_embedding, n_results // 2)
        
        return {
            "documents": doc_results,
            "videos": video_results
        }
    
    def search_by_topic(self,
                       topic_id: str,
                       query_embedding: np.ndarray = None,
                       n_results: int = 10) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search for content within a specific topic
        
        Args:
            topic_id: Topic ID to filter by
            query_embedding: Optional query embedding for semantic search within topic
            n_results: Number of results to return
        
        Returns:
            Dictionary with 'documents' and 'videos' results from the topic
        """
        # Filter by topic (topics are stored as comma-separated strings)
        topic_filter = {"topics": {"$contains": topic_id}}
        
        if query_embedding is not None:
            # Semantic search within topic
            doc_results = self.search_documents(query_embedding, n_results // 2, topic_filter)
            video_results = self.search_videos(query_embedding, n_results // 2, topic_filter)
        else:
            # Just get all content from topic
            doc_results = self._get_content_by_filter(self.pdf_collection, topic_filter, n_results // 2)
            video_results = self._get_content_by_filter(self.video_collection, topic_filter, n_results // 2)
        
        return {
            "documents": doc_results,
            "videos": video_results
        }
    
    def get_topics_for_content(self, content_id: str) -> List[str]:
        """
        Get topics assigned to a specific content item
        
        Args:
            content_id: Content ID to look up
            
        Returns:
            List of topic IDs
        """
        try:
            # Try PDF collection first
            results = self.pdf_collection.get(
                ids=[content_id],
                include=["metadatas"]
            )
            
            if results['metadatas']:
                topics_str = results['metadatas'][0].get('topics', '')
                return topics_str.split(',') if topics_str else []
            
            # Try video collection
            results = self.video_collection.get(
                ids=[content_id],
                include=["metadatas"]
            )
            
            if results['metadatas']:
                topics_str = results['metadatas'][0].get('topics', '')
                return topics_str.split(',') if topics_str else []
            
            return []
            
        except Exception as e:
            print(f"Error getting topics for content: {str(e)}")
            return []
    
    def get_content_by_topics(self, topic_ids: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all content that belongs to any of the specified topics
        
        Args:
            topic_ids: List of topic IDs to search for
            
        Returns:
            Dictionary with 'documents' and 'videos' results
        """
        # Create a filter that matches any of the topic IDs (comma-separated format)
        if len(topic_ids) == 1:
            topic_filter = {"topics": {"$contains": topic_ids[0]}}
        else:
            # For multiple topics, use OR logic
            topic_filter = {"$or": [{"topics": {"$contains": topic_id}} for topic_id in topic_ids]}
        
        doc_results = self._get_content_by_filter(self.pdf_collection, topic_filter, 100)
        video_results = self._get_content_by_filter(self.video_collection, topic_filter, 100)
        
        return {
            "documents": doc_results,
            "videos": video_results
        }
    
    def _get_content_by_filter(self, collection, filter_dict: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        """
        Get content from a collection using metadata filter
        """
        try:
            results = collection.get(
                where=filter_dict,
                limit=limit,
                include=["documents", "metadatas"]
            )
            
            formatted_results = []
            if results['documents']:
                documents = results['documents']
                metadatas = results['metadatas'] if results['metadatas'] else [{}] * len(documents)
                ids = results['ids'] if results['ids'] else list(range(len(documents)))
                
                for i, doc in enumerate(documents):
                    formatted_results.append({
                        'id': ids[i],
                        'text': doc,
                        'metadata': metadatas[i],
                        'similarity_score': 1.0,  # No similarity score for filter-based results
                        'distance': 0.0
                    })
            
            return formatted_results
            
        except Exception as e:
            print(f"Error getting content by filter: {str(e)}")
            return []
    
    def _format_search_results(self, results) -> List[Dict[str, Any]]:
        """Format ChromaDB results into consistent structure"""
        formatted_results = []
        
        if not results['documents'] or not results['documents'][0]:
            return formatted_results
        
        documents = results['documents'][0]
        metadatas = results['metadatas'][0] if results['metadatas'] else [{}] * len(documents)
        distances = results['distances'][0] if results['distances'] else [0.0] * len(documents)
        ids = results['ids'][0] if results['ids'] else list(range(len(documents)))
        
        for i, doc in enumerate(documents):
            formatted_results.append({
                'id': ids[i],
                'text': doc,
                'metadata': metadatas[i],
                'similarity_score': 1.0 - distances[i],  # Convert distance to similarity
                'distance': distances[i]
            })
        
        return formatted_results
    
    def document_exists(self, document_id: str) -> bool:
        """Check if document is already stored"""
        try:
            results = self.pdf_collection.get(
                where={"document_id": document_id},
                limit=1
            )
            return len(results['ids']) > 0
        except:
            return False
    
    def video_exists(self, video_id: str) -> bool:
        """Check if video is already stored"""
        try:
            results = self.video_collection.get(
                where={"video_id": video_id},
                limit=1
            )
            return len(results['ids']) > 0
        except:
            return False
    
    def get_document_info(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a stored document"""
        try:
            results = self.pdf_collection.get(
                where={"document_id": document_id},
                limit=1,
                include=["metadatas"]
            )
            
            if results['metadatas']:
                return results['metadatas'][0]
            return None
            
        except Exception as e:
            print(f"Error getting document info: {str(e)}")
            return None
    
    def list_documents(self) -> List[str]:
        """List all stored document IDs"""
        try:
            results = self.pdf_collection.get(include=["metadatas"])
            document_ids = set()
            
            for metadata in results['metadatas']:
                if 'document_id' in metadata:
                    document_ids.add(metadata['document_id'])
            
            return list(document_ids)
            
        except Exception as e:
            print(f"Error listing documents: {str(e)}")
            return []
    
    def list_videos(self) -> List[str]:
        """List all stored video IDs"""
        try:
            results = self.video_collection.get(include=["metadatas"])
            video_ids = set()
            
            for metadata in results['metadatas']:
                if 'video_id' in metadata:
                    video_ids.add(metadata['video_id'])
            
            return list(video_ids)
            
        except Exception as e:
            print(f"Error listing videos: {str(e)}")
            return []
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        try:
            pdf_count = self.pdf_collection.count()
            video_count = self.video_collection.count()
            
            # Get topic statistics
            topic_stats = self._get_topic_stats()
            
            return {
                "pdf_chunks": pdf_count,
                "video_segments": video_count,
                "total_chunks": pdf_count + video_count,
                "documents": len(self.list_documents()),
                "videos": len(self.list_videos()),
                "storage_path": str(self.persist_directory),
                "topic_statistics": topic_stats
            }
            
        except Exception as e:
            print(f"Error getting storage stats: {str(e)}")
            return {}
    
    def _get_topic_stats(self) -> Dict[str, Any]:
        """Get statistics about topic distribution"""
        try:
            # Get all PDF metadatas
            pdf_results = self.pdf_collection.get(include=["metadatas"])
            video_results = self.video_collection.get(include=["metadatas"])
            
            all_topics = set()
            topic_counts = {}
            cross_topic_content = 0
            
            # Analyze PDF topics
            for metadata in pdf_results.get('metadatas', []):
                topics_str = metadata.get('topics', '')
                topics = topics_str.split(',') if topics_str else []
                if len(topics) > 1:
                    cross_topic_content += 1
                for topic in topics:
                    if topic:  # Skip empty strings
                        all_topics.add(topic)
                        topic_counts[topic] = topic_counts.get(topic, 0) + 1
            
            # Analyze video topics  
            for metadata in video_results.get('metadatas', []):
                topics_str = metadata.get('topics', '')
                topics = topics_str.split(',') if topics_str else []
                if len(topics) > 1:
                    cross_topic_content += 1
                for topic in topics:
                    if topic:  # Skip empty strings
                        all_topics.add(topic)
                        topic_counts[topic] = topic_counts.get(topic, 0) + 1
            
            return {
                "unique_topics": len(all_topics),
                "cross_topic_content": cross_topic_content,
                "topic_distribution": topic_counts,
                "most_common_topics": sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            }
            
        except Exception as e:
            print(f"Error getting topic stats: {str(e)}")
            return {}