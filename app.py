"""
KnowledgeAgent Web Interface - FastAPI Backend
Claude-style UI with ChatGPT colors
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import asyncio
import json
import re
import os
import sys
import time
import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path
import aiofiles
import redis
from urllib.parse import urlparse
import httpx
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append('src')

# Import our Enhanced KnowledgeAgent system
from agents.research_executor import ResearchExecutor
from agents.enhanced_research_executor import EnhancedResearchExecutor
from agents.youtube_agent import YouTubeAgent

app = FastAPI(title="KnowledgeAgent Research Assistant", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize systems
research_executor = None
enhanced_research_executor = None
youtube_agent = None
cache_client = None

# OpenAI API configuration
OPENAI_API_KEY = ""

USE_ENHANCED_EXECUTOR = True  # Re-enabled - hallucination fixed

# Cache setup
try:
    cache_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    cache_client.ping()
    print("Redis cache connected successfully")
except:
    print("Redis not available, using in-memory cache")
    cache_client = {}

# In-memory cache fallback
memory_cache = {}

class CacheManager:
    """Robust caching system with Redis fallback to memory"""
    
    def __init__(self):
        self.redis_available = isinstance(cache_client, redis.Redis)
    
    async def get(self, key: str) -> Optional[str]:
        try:
            if self.redis_available:
                return cache_client.get(key)
            else:
                return memory_cache.get(key)
        except:
            return memory_cache.get(key)
    
    async def set(self, key: str, value: str, expire: int = 3600):
        try:
            if self.redis_available:
                cache_client.setex(key, expire, value)
            memory_cache[key] = value
        except:
            memory_cache[key] = value
    
    def cache_key(self, prefix: str, data: str) -> str:
        """Generate cache key from data hash"""
        hash_obj = hashlib.md5(data.encode())
        return f"{prefix}:{hash_obj.hexdigest()}"

cache_manager = CacheManager()

class YouTubeAPIClient:
    """YouTube API client with transcript checking"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('YOUTUBE_API_KEY')
        self.base_url = "https://www.googleapis.com/youtube/v3"
    
    async def get_video_info(self, video_id: str) -> Dict[str, Any]:
        """Get video metadata from YouTube API"""
        cache_key = cache_manager.cache_key("video_info", video_id)
        cached = await cache_manager.get(cache_key)
        
        if cached:
            return json.loads(cached)
        
        if not self.api_key:
            return {"error": "YouTube API key not configured"}
        
        url = f"{self.base_url}/videos"
        params = {
            'part': 'snippet,contentDetails,statistics',
            'id': video_id,
            'key': self.api_key
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params)
                data = response.json()
                
                await cache_manager.set(cache_key, json.dumps(data), 3600)
                return data
        except Exception as e:
            return {"error": str(e)}
    
    async def check_captions_available(self, video_id: str) -> Dict[str, Any]:
        """Check if captions/transcripts are available"""
        cache_key = cache_manager.cache_key("captions", video_id)
        cached = await cache_manager.get(cache_key)
        
        if cached:
            return json.loads(cached)
        
        if not self.api_key:
            return {"available": False, "reason": "API key not configured"}
        
        url = f"{self.base_url}/captions"
        params = {
            'part': 'snippet',
            'videoId': video_id,
            'key': self.api_key
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params)
                data = response.json()
                
                result = {
                    "available": len(data.get('items', [])) > 0,
                    "captions": data.get('items', []),
                    "reason": "API check completed"
                }
                
                await cache_manager.set(cache_key, json.dumps(result), 3600)
                return result
        except Exception as e:
            return {"available": False, "reason": str(e)}

youtube_api = YouTubeAPIClient()

def clean_for_json(obj):
    """Clean object for JSON serialization"""
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif hasattr(obj, '__dict__'):
        return clean_for_json(obj.__dict__)
    elif isinstance(obj, (datetime,)):
        return obj.isoformat()
    else:
        try:
            json.dumps(obj)
            return obj
        except:
            return str(obj)

class LinkTextExtractor:
    """Extract links and text from user input"""
    
    @staticmethod
    def extract_youtube_urls(text: str) -> List[str]:
        """Extract YouTube URLs from text"""
        youtube_patterns = [
            r'https?://(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]+)',
            r'https?://(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]+)',
            r'https?://youtu\.be/([a-zA-Z0-9_-]+)',
            r'https?://(?:www\.)?youtube\.com/v/([a-zA-Z0-9_-]+)'
        ]
        
        urls = []
        for pattern in youtube_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                urls.append(f"https://www.youtube.com/watch?v={match}")
        
        # Also find full URLs
        url_pattern = r'https?://(?:www\.)?(?:youtube\.com|youtu\.be)[^\s]+'
        full_urls = re.findall(url_pattern, text)
        urls.extend(full_urls)
        
        return list(set(urls))  # Remove duplicates
    
    @staticmethod
    def extract_other_urls(text: str) -> List[str]:
        """Extract non-YouTube URLs"""
        url_pattern = r'https?://[^\s]+'
        all_urls = re.findall(url_pattern, text)
        youtube_urls = LinkTextExtractor.extract_youtube_urls(text)
        
        return [url for url in all_urls if url not in youtube_urls]
    
    @staticmethod
    def extract_text_without_urls(text: str) -> str:
        """Extract text with URLs removed"""
        url_pattern = r'https?://[^\s]+'
        clean_text = re.sub(url_pattern, '', text)
        # Clean up extra whitespace
        clean_text = ' '.join(clean_text.split())
        return clean_text.strip()
    
    @staticmethod
    def parse_input(user_input: str) -> Dict[str, Any]:
        """Parse user input into components"""
        return {
            "original_text": user_input,
            "youtube_urls": LinkTextExtractor.extract_youtube_urls(user_input),
            "other_urls": LinkTextExtractor.extract_other_urls(user_input),
            "clean_text": LinkTextExtractor.extract_text_without_urls(user_input)
        }

async def initialize_system():
    """Initialize the research executor"""
    global research_executor, enhanced_research_executor, youtube_agent
    
    if research_executor is None:
        print("Initializing KnowledgeAgent Research System...")
        
        # Initialize enhanced executor (normal for now - anti-hallucination at API level)
        if USE_ENHANCED_EXECUTOR:
            print("[ENHANCED] Initializing Enhanced KnowledgeAgent with OpenAI integration...")
            enhanced_research_executor = EnhancedResearchExecutor(
                openai_api_key=OPENAI_API_KEY,
                openai_model="gpt-3.5-turbo"
            )
            print(f"[SUCCESS] Enhanced KnowledgeAgent ready! OpenAI available: {enhanced_research_executor.openai_synthesizer.is_available()}")
        else:
            print("[STANDARD] Using standard KnowledgeAgent executor...")
        
        # Keep original executor as fallback
        research_executor = ResearchExecutor()
        youtube_agent = YouTubeAgent()
        print("KnowledgeAgent system ready!")

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main UI"""
    return await get_ui_html()

@app.post("/api/parse-input")
async def parse_input(data: dict, background_tasks: BackgroundTasks):
    """Parse user input to extract links and text"""
    try:
        user_input = data.get("input", "")
        parsed = LinkTextExtractor.parse_input(user_input)
        
        # Check YouTube videos for transcript availability
        youtube_analysis = []
        for url in parsed["youtube_urls"]:
            video_id = extract_video_id(url)
            if video_id:
                captions_info = await youtube_api.check_captions_available(video_id)
                video_info = await youtube_api.get_video_info(video_id)
                
                youtube_analysis.append({
                    "url": url,
                    "video_id": video_id,
                    "captions_available": captions_info.get("available", False),
                    "captions_reason": captions_info.get("reason", "Unknown"),
                    "title": video_info.get("items", [{}])[0].get("snippet", {}).get("title", "Unknown"),
                    "download_needed": not captions_info.get("available", False)
                })
        
        # Process YouTube URLs in background (similar to PDF processing)
        if parsed["youtube_urls"]:
            for url in parsed["youtube_urls"]:
                # Check if this video is already processed
                video_id = extract_video_id(url)
                if video_id and not is_video_already_processed(video_id):
                    background_tasks.add_task(process_youtube_to_database, url)
                    print(f"[PROCESSING] Started background processing for YouTube video: {video_id}")
                else:
                    print(f"[SKIP] Video {video_id} already processed")
        
        return {
            "success": True,
            "parsed": parsed,
            "youtube_analysis": youtube_analysis,
            "processing_started": len(parsed["youtube_urls"]) > 0
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload-pdf")
async def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Handle PDF file upload and processing"""
    try:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Save uploaded file
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        
        file_path = upload_dir / file.filename
        
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        # Process PDF immediately and store in database
        background_tasks.add_task(process_pdf_to_database, str(file_path), file.filename)
        
        return {
            "success": True,
            "filename": file.filename,
            "path": str(file_path),
            "size": len(content),
            "message": "PDF uploaded and processing started"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def process_pdf_to_database(file_path: str, filename: str):
    """Process uploaded PDF and store in database"""
    try:
        print(f"[PROCESSING] Processing PDF: {filename}")
        
        # Initialize system if needed
        await initialize_system()
        
        if not research_executor or not research_executor.pdf_agent:
            print(f"[ERROR] PDF agent not available")
            return
        
        # Extract text from PDF
        extract_result = await research_executor.pdf_agent.execute_action("extract_text", file_path)
        if not extract_result.success:
            print(f"[ERROR] Failed to extract text from {filename}: {extract_result.error_message}")
            return
        
        # Chunk the document
        chunk_result = await research_executor.pdf_agent.execute_action("chunk_document", file_path, previous_results={"extract_text": extract_result.output})
        if not chunk_result.success:
            print(f"[ERROR] Failed to chunk {filename}: {chunk_result.error_message}")
            return
        
        # Create embeddings and classify topics
        embedding_result = await research_executor.pdf_agent.execute_action("create_embeddings", file_path, previous_results={"chunk_document": chunk_result.output})
        if not embedding_result.success:
            print(f"[ERROR] Failed to create embeddings for {filename}: {embedding_result.error_message}")
            return
        
        # Store in vector database
        from storage.vector_store import KnowledgeAgentVectorStore
        vector_store = KnowledgeAgentVectorStore("knowledgeagent_vectordb")
        
        chunks = chunk_result.output.get("chunks", [])
        embeddings_data = embedding_result.output.get("enriched_chunks", [])
        
        # Convert to numpy array
        import numpy as np
        embeddings_array = np.array([chunk.get("embedding", []) for chunk in embeddings_data])
        
        # Store in vector database
        success = vector_store.store_pdf_document(
            document_id=Path(filename).stem,
            chunks=chunks,
            embeddings=embeddings_array,
            metadata={"title": filename, "file_path": file_path},
            topic_assignments=embedding_result.output.get("topic_assignments", {})
        )
        
        if success:
            print(f"[SUCCESS] PDF {filename} processed and stored in database")
        else:
            print(f"[ERROR] Failed to store {filename} in vector database")
            
    except Exception as e:
        print(f"[ERROR] Failed to process PDF {filename}: {str(e)}")
        import traceback
        traceback.print_exc()

async def process_youtube_to_database(youtube_url: str):
    """Process YouTube URL and store in database"""
    try:
        print(f"[PROCESSING] Processing YouTube video: {youtube_url}")
        
        # Extract video ID for identification
        video_id = extract_video_id(youtube_url)
        if not video_id:
            print(f"[ERROR] Could not extract video ID from {youtube_url}")
            return
        
        # Check if already processed
        if is_video_already_processed(video_id):
            print(f"[SKIP] Video {video_id} already processed, skipping...")
            return
        
        # Initialize system if needed
        await initialize_system()
        
        if not research_executor or not research_executor.youtube_agent:
            print(f"[ERROR] YouTube agent not available")
            return
        
        # Try to get captions first (faster than audio download)
        captions_result = await research_executor.youtube_agent.execute_action("get_captions", {"url": youtube_url})
        
        transcript_result = None
        video_info = {}
        
        if captions_result.success:
            print(f"[SUCCESS] Got captions for {youtube_url}")
            transcript_result = captions_result
            video_info = captions_result.output.get("video_info", {})
        else:
            print(f"[INFO] Captions not available for {youtube_url}, downloading audio...")
            
            # Download audio as fallback
            download_result = await research_executor.youtube_agent.execute_action("download_audio", {"url": youtube_url})
            if not download_result.success:
                print(f"[ERROR] Failed to download audio from {youtube_url}: {download_result.error_message}")
                return
            
            # Transcribe audio
            transcript_result = await research_executor.youtube_agent.execute_action(
                "transcribe_audio", 
                {"url": youtube_url},
                previous_results={"download_audio": download_result.output}
            )
            video_info = download_result.output.get("video_info", {})
        
        if not transcript_result or not transcript_result.success:
            print(f"[ERROR] Failed to get transcript for {youtube_url}: {transcript_result.error_message if transcript_result else 'No result'}")
            return
        
        # Extract video title
        video_title = video_info.get("title", f"YouTube Video {video_id}")
        print(f"[INFO] Processing video: '{video_title}'")
        
        # Get transcript segments and text
        segments = transcript_result.output.get("segments", [])
        transcript_text = transcript_result.output.get("transcript", "")
        
        if not segments or not transcript_text:
            print(f"[ERROR] No transcript content found for {youtube_url}")
            return
        
        # Store in vector database
        from storage.vector_store import KnowledgeAgentVectorStore
        vector_store = KnowledgeAgentVectorStore("knowledgeagent_vectordb")
        
        # Convert segments to chunks format
        chunks = []
        for i, segment in enumerate(segments):
            chunk = {
                "chunk_id": i,
                "text": segment.get("text", ""),
                "start_time": segment.get("start", 0),
                "duration": segment.get("duration", 0),
                "metadata": {
                    "segment_index": i,
                    "timestamp": f"{segment.get('start', 0):.1f}s"
                }
            }
            chunks.append(chunk)
        
        # Store video in vector database with topic classification
        # First, get embeddings for the segments
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        segment_texts = [segment.get('text', '') for segment in segments]
        embeddings = embedding_model.encode(segment_texts)
        
        success = vector_store.store_video_transcript(
            video_id=video_id,
            segments=segments,
            embeddings=embeddings,
            metadata={
                "title": video_title,
                "url": youtube_url,
                "processing_method": "captions" if captions_result.success else "transcription",
                "total_segments": len(segments),
                "transcript_length": len(transcript_text)
            }
        )
        
        if success:
            print(f"[SUCCESS] YouTube video '{video_title}' processed and stored in database")
        else:
            print(f"[ERROR] Failed to store {youtube_url} in vector database")
            
    except Exception as e:
        print(f"[ERROR] Failed to process YouTube video {youtube_url}: {str(e)}")
        import traceback
        traceback.print_exc()

@app.post("/api/research")
async def execute_research(data: dict, background_tasks: BackgroundTasks):
    """Execute research query with KnowledgeAgent and smart content routing"""
    try:
        # Initialize system if needed
        await initialize_system()
        query = data.get("query", "")
        pdf_files = data.get("pdf_files", [])
        youtube_urls = data.get("youtube_urls", [])
        
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")
        
        # Check cache first
        cache_key = cache_manager.cache_key("research", f"{query}:{pdf_files}:{youtube_urls}")
        cached_result = await cache_manager.get(cache_key)
        
        if cached_result:
            return {
                "success": True,
                "cached": True,
                "result": json.loads(cached_result)
            }
        
        print(f"🔍 Executing smart research query: {query}")
        
        # Use NER-based fuzzy routing for intelligent content discovery
        from storage.ner_fuzzy_router import NERFuzzyRouter
        from storage.simple_smart_router import SimpleSmartRouter
        from storage.unified_content_store import UnifiedContentStore
        
        # Initialize content store
        content_store = UnifiedContentStore()
        
        # Try Enhanced NER-based fuzzy router first (most intelligent)
        try:
            from storage.enhanced_ner_fuzzy_router import EnhancedNERFuzzyRouter
            enhanced_router = EnhancedNERFuzzyRouter(content_store)
            relevant_pdfs, relevant_videos, explanation = enhanced_router.route_query(query)
            print(f"📊 Enhanced NER routing: {explanation}")
        except Exception as e:
            print(f"⚠️ Enhanced router failed, falling back to standard NER router: {e}")
            # Fallback to standard NER router
            try:
                ner_router = NERFuzzyRouter(content_store)
                relevant_pdfs, relevant_videos, explanation = ner_router.route_query(query)
                print(f"📊 NER fuzzy routing: {explanation}")
            except Exception as e2:
                print(f"⚠️ NER router failed, falling back to simple router: {e2}")
                # Final fallback to simple router
                simple_router = SimpleSmartRouter(content_store)
                relevant_pdfs, relevant_videos, explanation = simple_router.route_query(query)
        
        # Override provided files with smart-detected content
        if relevant_pdfs or relevant_videos:
            pdf_files = relevant_pdfs
            youtube_urls = relevant_videos
            print(f"📊 Smart routing found: {len(relevant_pdfs)} PDFs, {len(relevant_videos)} videos")
            print(f"📋 {explanation}")
            
            # If this is a new YouTube URL, process it immediately
            if "Direct YouTube URL" in explanation and youtube_urls:
                new_url = youtube_urls[0]
                video_id = extract_video_id(new_url)
                if video_id and not is_video_already_processed(video_id):
                    print(f"🎥 New YouTube URL detected - processing immediately: {video_id}")
                    # Process the video immediately rather than in background
                    # This ensures it's available for the research query
                    try:
                        await process_youtube_to_database(new_url)
                    except Exception as e:
                        print(f"❌ Failed to process new YouTube URL: {e}")
        else:
            print("⚠️  No specific content found, using general search")
        
        # Execute research with enhanced executor if available
        if USE_ENHANCED_EXECUTOR and enhanced_research_executor:
            print("[RESEARCH] Using Enhanced KnowledgeAgent...")
            use_openai = data.get("use_openai", False)  # Force fallback to prevent hallucination
            
            enhanced_result = await enhanced_research_executor.execute_research_query(
                query=query,
                pdf_files=pdf_files,
                youtube_urls=youtube_urls,
                use_openai=use_openai
            )
            
            # Convert enhanced result to compatible format
            # Extract source citations from synthesis result
            source_citations = []
            reasoning_steps = []
            
            if hasattr(enhanced_result.synthesis_result, 'sources_used'):
                source_citations = enhanced_result.synthesis_result.sources_used
                
            if hasattr(enhanced_result.synthesis_result, 'reasoning_steps'):
                reasoning_steps = enhanced_result.synthesis_result.reasoning_steps
                
            result = {
                "final_answer": {
                    "answer_summary": enhanced_result.final_answer,
                    "research_confidence": enhanced_result.research_confidence,
                    "sources_processed": enhanced_result.sources_processed,
                    "next_steps": enhanced_result.next_steps,
                    "source_provenance": source_citations,
                    "reasoning_steps": reasoning_steps
                },
                "execution": {
                    "total_steps": len(enhanced_result.action_path.steps),
                    "successful_steps": sum(1 for step in enhanced_result.action_path.steps if step.success),
                    "total_execution_time": enhanced_result.total_execution_time,
                    "action_path": [step.action_type.value for step in enhanced_result.action_path.steps],
                    "cost_breakdown": enhanced_result.cost_breakdown
                },
                "enhanced": True,
                "synthesis_model": getattr(enhanced_result.synthesis_result, 'model_used', 'local_fallback'),
                "smart_routing": {
                    "strategy": "simple_routing",
                    "confidence": 1.0 if relevant_pdfs or relevant_videos else 0.5,
                    "matched_terms": [],
                    "explanation": explanation
                }
            }
        else:
            print("📚 Using standard KnowledgeAgent executor...")
            result = await research_executor.execute_research_query(
                query=query,
                pdf_files=pdf_files,
                youtube_urls=youtube_urls
            )
            result["enhanced"] = False
            result["smart_routing"] = {
                "strategy": "simple_routing",
                "confidence": 1.0 if relevant_pdfs or relevant_videos else 0.5,
                "matched_terms": [],
                "explanation": explanation
            }
        
        # Clean result for JSON serialization
        clean_result = clean_for_json(result)
        
        # Cache result
        await cache_manager.set(cache_key, json.dumps(clean_result), 7200)  # 2 hours
        
        return {
            "success": True,
            "cached": False,
            "result": clean_result
        }
    
    except Exception as e:
        print(f"❌ Research query failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status")
async def get_status():
    """Get system status"""
    enhanced_stats = {}
    if enhanced_research_executor:
        enhanced_stats = enhanced_research_executor.get_statistics()
    
    return {
        "status": "healthy",
        "research_executor": research_executor is not None,
        "enhanced_executor": enhanced_research_executor is not None,
        "youtube_agent": youtube_agent is not None,
        "cache": "redis" if isinstance(cache_client, redis.Redis) else "memory",
        "youtube_api": youtube_api.api_key is not None,
        "openai_available": enhanced_research_executor.openai_synthesizer.is_available() if enhanced_research_executor else False,
        "enhanced_knowagent": USE_ENHANCED_EXECUTOR,
        "enhanced_stats": enhanced_stats
    }

@app.get("/api/simple-content")
async def get_simple_content():
    """Get content using simple title-based organization"""
    try:
        from storage.unified_content_store import UnifiedContentStore
        
        content_store = UnifiedContentStore()
        all_content = content_store.get_all_content()
        storage_stats = content_store.get_storage_stats()
        
        # Format for frontend
        content_list = []
        for item in all_content:
            content_list.append({
                "id": item.id,
                "title": item.title,
                "type": item.content_type,
                "source_path": item.source_path,
                "chunks_count": len(item.chunks),
                "created_at": item.created_at,
                "metadata": item.metadata
            })
        
        # Sort by creation time, newest first
        content_list.sort(key=lambda x: x["created_at"], reverse=True)
        
        return {
            "content": content_list,
            "stats": storage_stats
        }
        
    except Exception as e:
        print(f"Error loading simple content: {e}")
        return {
            "content": [],
            "stats": {"total_items": 0, "pdf_count": 0, "youtube_count": 0, "total_chunks": 0}
        }

@app.post("/api/search-content")
async def search_content(data: dict):
    """Enhanced search content using trie-based search"""
    try:
        search_term = data.get("search_term", "")
        max_results = data.get("max_results", 10)
        
        if not search_term:
            raise HTTPException(status_code=400, detail="Search term is required")
        
        # Use enhanced search service
        from search.search_service import KnowledgeAgentSearchService
        
        search_service = KnowledgeAgentSearchService()
        enhanced_results = search_service.search(search_term, max_results)
        
        # Format results for frontend
        results = []
        for result in enhanced_results:
            results.append({
                "id": result.content_id,
                "title": result.title,
                "type": result.content_type,
                "source_path": result.source_path,
                "match_score": round(result.match_score, 3),
                "matched_terms": result.matched_terms,
                "preview": result.preview,
                "routing_strategy": result.routing_strategy,
                "confidence": round(result.confidence, 3),
                "combined_score": round(result.match_score * result.confidence, 3)
            })
        
        return {
            "search_term": search_term,
            "results": results,
            "total_found": len(results),
            "search_type": "enhanced_trie",
            "statistics": search_service.get_search_statistics()
        }
        
    except Exception as e:
        print(f"Enhanced search error: {e}")
        # Fallback to simple search
        try:
            from storage.unified_content_store import UnifiedContentStore
            
            content_store = UnifiedContentStore()
            matching_items = content_store.search_content_by_name(search_term)
            
            results = []
            for item in matching_items:
                results.append({
                    "id": item.id,
                    "title": item.title,
                    "type": item.content_type,
                    "source_path": item.source_path,
                    "chunks_count": len(item.chunks),
                    "created_at": item.created_at,
                    "match_score": 1.0,
                    "routing_strategy": "fallback_regex",
                    "confidence": 0.8
                })
            
            return {
                "search_term": search_term,
                "results": results,
                "total_found": len(results),
                "search_type": "fallback_regex"
            }
        except Exception as fallback_error:
            raise HTTPException(status_code=500, detail=str(fallback_error))

@app.post("/api/search-suggestions")
async def get_search_suggestions(data: dict):
    """Get search query suggestions based on partial input"""
    try:
        partial_query = data.get("partial_query", "")
        max_suggestions = data.get("max_suggestions", 5)
        
        if not partial_query:
            return {"suggestions": []}
        
        from search.search_service import KnowledgeAgentSearchService
        
        search_service = KnowledgeAgentSearchService()
        suggestions = search_service.suggest_queries(partial_query, max_suggestions)
        
        return {
            "partial_query": partial_query,
            "suggestions": suggestions,
            "total_suggestions": len(suggestions)
        }
        
    except Exception as e:
        print(f"Search suggestions error: {e}")
        return {"suggestions": []}

def generate_clean_content_title(content_item, source_metadata=None):
    """Generate clean, user-friendly title for content items"""
    content_type = content_item.content_type
    source_id = content_item.source_id
    
    # For PDF content
    if content_type == "pdf":
        if source_metadata and source_metadata.get("title"):
            return source_metadata["title"]
        # Clean up the source_id to make it readable
        clean_name = source_id.replace("_", " ").replace("-", " ")
        if clean_name.endswith(".pdf"):
            clean_name = clean_name[:-4]
        return clean_name.title() + ".pdf"
    
    # For video content
    elif content_type == "video":
        if source_metadata and source_metadata.get("title"):
            return source_metadata["title"]
        # Use video ID or create a readable title
        return f"Video: {source_id}"
    
    # Fallback: use a preview of the content
    preview = content_item.text[:60].strip()
    if len(content_item.text) > 60:
        preview += "..."
    return preview

def get_content_metadata(content_item):
    """Get metadata for content items"""
    content_type = content_item.content_type
    metadata = content_item.metadata if hasattr(content_item, 'metadata') else {}
    
    if content_type == "pdf":
        return {
            "type": "pdf",
            "icon": "fa-file-pdf",
            "preview": content_item.text[:100] + "..." if len(content_item.text) > 100 else content_item.text,
            "page_info": metadata.get("page_number", "Unknown page"),
            "char_count": len(content_item.text)
        }
    elif content_type == "video":
        return {
            "type": "video", 
            "icon": "fa-video",
            "preview": content_item.text[:100] + "..." if len(content_item.text) > 100 else content_item.text,
            "timestamp": metadata.get("start_time", "Unknown time"),
            "duration": metadata.get("duration", "Unknown duration")
        }
    
    return {
        "type": "unknown",
        "icon": "fa-file",
        "preview": content_item.text[:100] + "..." if len(content_item.text) > 100 else content_item.text
    }

@app.get("/api/topics")
async def get_topics():
    """Get topic hierarchy and content organization"""
    try:
        from storage.vector_store import KnowledgeAgentVectorStore
        from classification.topic_classifier import DynamicTopicClassifier
        
        # Initialize vector store and topic classifier with error handling
        try:
            vector_store = KnowledgeAgentVectorStore("knowledgeagent_vectordb")
        except Exception as ve:
            print(f"Vector store initialization error: {ve}")
            # Fall back to simple content store
            from storage.unified_content_store import UnifiedContentStore
            return await get_simple_content()
        
        try:
            topic_classifier = DynamicTopicClassifier(storage_path="topic_classifications")
        except Exception as te:
            print(f"Topic classifier initialization error: {te}")
            # Return basic structure if topic classifier fails
            return {
                "topics": [],
                "stats": {
                    "total_topics": 0,
                    "total_content": 0,
                    "cross_topic_items": 0,
                    "storage_stats": {}
                }
            }
        
        # Get topic hierarchy
        topic_hierarchy = topic_classifier.get_topic_hierarchy()
        cross_topic_content = topic_classifier.get_cross_topic_content()
        classification_stats = topic_classifier.get_classification_stats()
        storage_stats = vector_store.get_storage_stats()
        
        # Group content by source to avoid showing individual chunks
        source_content_map = {}
        
        # Format topics for frontend
        topics = []
        for topic_id, topic_info in topic_hierarchy.items():
            # Get content for this topic
            topic_content = topic_classifier.get_topic_contents(topic_id)
            
            # Group content by source_id to aggregate chunks
            source_groups = {}
            for content_item in topic_content:
                source_id = content_item.source_id
                if source_id not in source_groups:
                    source_groups[source_id] = {
                        "items": [],
                        "type": content_item.content_type,
                        "is_cross_topic": content_item.content_id in cross_topic_content
                    }
                source_groups[source_id]["items"].append(content_item)
            
            # Create clean content items grouped by source
            content_items = []
            for source_id, group in source_groups.items():
                first_item = group["items"][0]
                metadata = get_content_metadata(first_item)
                
                # Generate clean title
                clean_title = generate_clean_content_title(first_item)
                
                content_items.append({
                    "id": source_id,  # Use source_id instead of chunk ID
                    "title": clean_title,
                    "type": group["type"],
                    "source_id": source_id,
                    "cross_topic": group["is_cross_topic"],
                    "chunk_count": len(group["items"]),
                    "metadata": metadata,
                    "timestamp": first_item.timestamp if hasattr(first_item, 'timestamp') else None
                })
            
            # Sort content items by title
            content_items.sort(key=lambda x: x["title"])
            
            topics.append({
                "id": topic_id,
                "name": topic_info["name"],
                "description": topic_info["description"],
                "confidence": topic_info["confidence"],
                "total_content": len(content_items),  # Use aggregated count
                "pdfs": topic_info["pdfs"],
                "videos": topic_info["videos"],
                "content": content_items,
                "created": topic_info["created"],
                "updated": topic_info["updated"]
            })
        
        # Sort topics by confidence and content count
        topics.sort(key=lambda x: (x["confidence"], x["total_content"]), reverse=True)
        
        return {
            "topics": topics,
            "stats": {
                "total_topics": classification_stats.get("total_topics", 0),
                "total_content": classification_stats.get("total_content_items", 0),
                "cross_topic_items": classification_stats.get("cross_topic_items", 0),
                "storage_stats": storage_stats
            }
        }
        
    except Exception as e:
        print(f"Error loading topics: {e}")
        import traceback
        traceback.print_exc()
        
        # Return empty state if error
        return {
            "topics": [],
            "stats": {
                "total_topics": 0,
                "total_content": 0,
                "cross_topic_items": 0,
                "storage_stats": {}
            }
        }

@app.get("/api/research-connections/{topic_id}")
async def get_research_connections(topic_id: str):
    """Get research connections analysis for a specific topic"""
    try:
        from storage.vector_store import KnowledgeAgentVectorStore
        from classification.topic_classifier import DynamicTopicClassifier
        from analysis.research_connections import ResearchConnectionsAnalyzer, ContentChunk
        
        # Initialize components
        vector_store = KnowledgeAgentVectorStore("knowledgeagent_vectordb")
        topic_classifier = DynamicTopicClassifier(storage_path="topic_classifications")
        connections_analyzer = ResearchConnectionsAnalyzer()
        
        # Get topic content
        topic_content = topic_classifier.get_topic_contents(topic_id)
        
        if not topic_content:
            return {
                "success": False,
                "error": f"Topic {topic_id} not found or has no content"
            }
        
        # Convert to ContentChunk format for analysis
        content_chunks = []
        for content_item in topic_content:
            if content_item.embedding is not None:
                chunk = ContentChunk(
                    id=content_item.content_id,
                    text=content_item.text,
                    embedding=content_item.embedding,
                    source_id=content_item.source_id,
                    content_type=content_item.content_type,
                    metadata=content_item.metadata
                )
                content_chunks.append(chunk)
        
        if len(content_chunks) < 2:
            return {
                "success": True,
                "connections": {
                    "contradictions": [],
                    "confirmations": [],
                    "extensions": [],
                    "gaps": []
                },
                "metadata": {
                    "total_chunks": len(content_chunks),
                    "message": "Need at least 2 content chunks for meaningful analysis"
                }
            }
        
        # Analyze connections
        print(f"Analyzing research connections for topic {topic_id} with {len(content_chunks)} chunks")
        result = connections_analyzer.analyze_topic_connections(content_chunks)
        
        # Format for JSON response
        response_data = {
            "success": True,
            "topic_id": topic_id,
            "connections": {
                "contradictions": [
                    {
                        "chunk1_id": conn.chunk1_id,
                        "chunk2_id": conn.chunk2_id,
                        "confidence": round(conn.confidence, 3),
                        "explanation": conn.explanation,
                        "chunk1_text": conn.chunk1_text,
                        "chunk2_text": conn.chunk2_text,
                        "source1": conn.source1,
                        "source2": conn.source2
                    }
                    for conn in result.contradictions
                ],
                "confirmations": [
                    {
                        "chunk1_id": conn.chunk1_id,
                        "chunk2_id": conn.chunk2_id,
                        "confidence": round(conn.confidence, 3),
                        "explanation": conn.explanation,
                        "chunk1_text": conn.chunk1_text,
                        "chunk2_text": conn.chunk2_text,
                        "source1": conn.source1,
                        "source2": conn.source2
                    }
                    for conn in result.confirmations
                ],
                "extensions": [
                    {
                        "chunk1_id": conn.chunk1_id,
                        "chunk2_id": conn.chunk2_id,
                        "confidence": round(conn.confidence, 3),
                        "explanation": conn.explanation,
                        "chunk1_text": conn.chunk1_text,
                        "chunk2_text": conn.chunk2_text,
                        "source1": conn.source1,
                        "source2": conn.source2
                    }
                    for conn in result.extensions
                ],
                "gaps": [
                    {
                        "topic_area": gap.topic_area,
                        "description": gap.description,
                        "related_chunks": gap.related_chunks,
                        "confidence": round(gap.confidence, 3)
                    }
                    for gap in result.gaps
                ]
            },
            "metadata": clean_for_json(result.analysis_metadata)
        }
        
        return response_data
        
    except Exception as e:
        print(f"Error analyzing research connections: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/api/delete-all-data")
async def delete_all_data():
    """Delete all stored data and embeddings"""
    try:
        import shutil
        from pathlib import Path
        
        # Directories and files to clear
        clear_targets = [
            # Vector databases
            "knowledgeagent_vectordb",
            "knowagent_vectordb",  # Legacy name for cleanup
            "chroma_db", 
            "vector_db",
            
            # Topic classifications
            "topic_classifications",
            "topic_classifications.json",
            
            # Simple content store
            "simple_content_store",
            
            # Any .db files in current directory
            Path(".").glob("*.db"),
            Path(".").glob("*.sqlite"),
            Path(".").glob("*.sqlite3"),
        ]
        
        cleared_items = []
        
        for target in clear_targets:
            if isinstance(target, Path):
                # Handle glob results
                continue
            
            target_path = Path(target)
            
            try:
                if target_path.exists():
                    if target_path.is_dir():
                        shutil.rmtree(target_path)
                        cleared_items.append(f"Directory: {target_path}")
                    else:
                        target_path.unlink()
                        cleared_items.append(f"File: {target_path}")
            except Exception as e:
                print(f"Could not delete {target_path}: {e}")
        
        # Handle glob patterns
        for pattern in [Path(".").glob("*.db"), Path(".").glob("*.sqlite"), Path(".").glob("*.sqlite3")]:
            for file_path in pattern:
                try:
                    file_path.unlink()
                    cleared_items.append(f"File: {file_path}")
                except Exception as e:
                    print(f"Could not delete {file_path}: {e}")
        
        # Clear Redis cache if available
        try:
            if isinstance(cache_client, redis.Redis):
                cache_client.flushdb()
                cleared_items.append("Redis cache")
        except:
            pass
        
        # Clear memory cache
        global memory_cache
        memory_cache = {}
        cleared_items.append("Memory cache")
        
        return {
            "success": True,
            "message": f"Successfully deleted all data",
            "cleared_items": cleared_items,
            "total_cleared": len(cleared_items)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to delete all data"
        }

@app.get("/api/export")
async def export_library():
    """Export the research library as JSON"""
    try:
        from storage.vector_store import KnowledgeAgentVectorStore
        from classification.topic_classifier import DynamicTopicClassifier
        
        vector_store = KnowledgeAgentVectorStore("knowledgeagent_vectordb")
        topic_classifier = DynamicTopicClassifier(storage_path="topic_classifications")
        
        # Get all data
        topics = topic_classifier.get_topic_hierarchy()
        storage_stats = vector_store.get_storage_stats()
        documents = vector_store.list_documents()
        videos = vector_store.list_videos()
        
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "version": "1.0",
            "topics": topics,
            "documents": documents,
            "videos": videos,
            "storage_stats": storage_stats
        }
        
        return JSONResponse(
            content=export_data,
            headers={"Content-Disposition": "attachment; filename=knowledgeagent_library.json"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

def extract_video_id(url: str) -> Optional[str]:
    """Extract YouTube video ID from URL"""
    patterns = [
        r'(?:v=|\/)([a-zA-Z0-9_-]{11})',
        r'(?:embed\/)([a-zA-Z0-9_-]{11})',
        r'(?:v\/)([a-zA-Z0-9_-]{11})'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def is_video_already_processed(video_id: str) -> bool:
    """Check if YouTube video is already processed"""
    try:
        from storage.unified_content_store import UnifiedContentStore
        content_store = UnifiedContentStore()
        all_content = content_store.get_all_content()
        
        for content_item in all_content:
            if content_item.content_type == 'youtube':
                # Check if this video ID is in the content
                if video_id in content_item.id or video_id in content_item.source_path:
                    return True
        
        return False
    except Exception as e:
        print(f"[ERROR] Could not check if video is processed: {e}")
        return False  # If we can't check, allow processing

async def get_ui_html() -> str:
    """Generate the UI HTML"""
    html_file = Path("static/index.html")
    if html_file.exists():
        async with aiofiles.open(html_file, 'r') as f:
            return await f.read()
    else:
        # Return basic HTML if static file doesn't exist
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>KnowledgeAgent Research Assistant</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
        </head>
        <body>
            <div id="app">
                <h1>KnowledgeAgent Research Assistant</h1>
                <p>Loading...</p>
            </div>
        </body>
        </html>
        """

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)