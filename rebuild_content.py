"""
Quick script to rebuild content store from ChromaDB
"""

import sys
sys.path.append('src')

from storage.vector_store import KnowledgeAgentVectorStore
from storage.simple_content_store import SimpleContentStore
import json

def rebuild_content_store():
    """Rebuild content store from existing ChromaDB"""
    print("Rebuilding content store from ChromaDB...")
    
    try:
        # Initialize stores
        vector_store = KnowledgeAgentVectorStore("knowledgeagent_vectordb")
        content_store = SimpleContentStore()
        
        # Get all documents and videos from ChromaDB
        try:
            pdf_collection = vector_store.pdf_collection
            video_collection = vector_store.video_collection
            
            # Get PDFs
            pdf_data = pdf_collection.get()
            print(f"PDF: Found {len(pdf_data['ids'])} PDF chunks in ChromaDB")
            
            # Group PDF chunks by document
            pdf_docs = {}
            for i, doc_id in enumerate(pdf_data['ids']):
                # Extract document name from chunk ID
                if '_chunk_' in doc_id:
                    doc_name = doc_id.split('_chunk_')[0]
                else:
                    doc_name = doc_id
                
                if doc_name not in pdf_docs:
                    pdf_docs[doc_name] = {
                        'chunks': [],
                        'metadata': pdf_data['metadatas'][i] if pdf_data['metadatas'] else {},
                        'documents': []
                    }
                
                chunk_data = {
                    'chunk_id': i,
                    'text': pdf_data['documents'][i] if pdf_data['documents'] else '',
                    'metadata': pdf_data['metadatas'][i] if pdf_data['metadatas'] else {}
                }
                pdf_docs[doc_name]['chunks'].append(chunk_data)
            
            # Add PDFs to content store
            for doc_name, doc_data in pdf_docs.items():
                print(f"Adding PDF: {doc_name}")
                
                # Try to get title from metadata or use doc_name
                title = doc_data['metadata'].get('title', doc_name)
                source_path = doc_data['metadata'].get('file_path', f"uploads/{doc_name}.pdf")
                
                content_store.add_pdf_content(
                    pdf_path=source_path,
                    title=title,
                    chunks=doc_data['chunks'],
                    metadata=doc_data['metadata']
                )
            
            # Get Videos
            video_data = video_collection.get()
            print(f"VIDEO: Found {len(video_data['ids'])} video chunks in ChromaDB")
            
            # Group video chunks by video
            video_docs = {}
            for i, doc_id in enumerate(video_data['ids']):
                # Extract video ID from chunk ID
                if '_segment_' in doc_id:
                    video_id = doc_id.split('_segment_')[0]
                else:
                    video_id = doc_id
                
                if video_id not in video_docs:
                    video_docs[video_id] = {
                        'chunks': [],
                        'metadata': video_data['metadatas'][i] if video_data['metadatas'] else {},
                        'documents': []
                    }
                
                chunk_data = {
                    'chunk_id': i,
                    'text': video_data['documents'][i] if video_data['documents'] else '',
                    'metadata': video_data['metadatas'][i] if video_data['metadatas'] else {}
                }
                video_docs[video_id]['chunks'].append(chunk_data)
            
            # Add videos to content store
            for video_id, video_data in video_docs.items():
                print(f"Adding Video: {video_id}")
                
                # Try to get title and URL from metadata
                title = video_data['metadata'].get('title', f"Video {video_id}")
                url = video_data['metadata'].get('url', f"https://www.youtube.com/watch?v={video_id}")
                
                content_store.add_youtube_content(
                    url=url,
                    title=title,
                    chunks=video_data['chunks'],
                    metadata=video_data['metadata']
                )
            
            print(f"SUCCESS: Rebuilt content store with {len(pdf_docs)} PDFs and {len(video_docs)} videos")
            
            # Test the content
            all_content = content_store.get_all_content()
            print(f"Content store now has {len(all_content)} items")
            
        except Exception as e:
            print(f"ERROR accessing ChromaDB: {e}")
            print("Creating empty content store...")
            
    except Exception as e:
        print(f"ERROR rebuilding content store: {e}")

if __name__ == "__main__":
    rebuild_content_store()