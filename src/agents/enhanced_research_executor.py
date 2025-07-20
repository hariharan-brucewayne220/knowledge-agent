"""
Enhanced Research Executor
Combines KnowledgeAgent planning methodology with our multi-modal system and OpenAI synthesis
"""

import asyncio
import time
import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from planning.action_graph import (
    KnowledgeAgentActionGraph, ActionType, ActionStep, ActionPath, ActionPathOptimizer
)
from synthesis.openai_synthesizer import OpenAISynthesizer, FallbackSynthesizer
from agents.pdf_agent import PDFAgent
from agents.youtube_agent import YouTubeAgent
from storage.simple_content_store import SimpleContentStore

logger = logging.getLogger(__name__)

@dataclass
class EnhancedResearchResult:
    """Enhanced research result with KnowAgent methodology"""
    query: str
    final_answer: str
    action_path: ActionPath
    synthesis_result: Any  # SynthesisResult from OpenAI or fallback
    sources_processed: Dict[str, int]
    total_execution_time: float
    research_confidence: float
    next_steps: List[str]
    cost_breakdown: Dict[str, float]

class EnhancedResearchExecutor:
    """
    Enhanced research executor combining:
    - KnowAgent's structured planning
    - Our multi-modal processing (PDF + YouTube)
    - OpenAI intelligent synthesis
    - Dynamic topic classification
    """
    
    def __init__(self, openai_api_key: Optional[str] = None, openai_model: str = "gpt-3.5-turbo"):
        """
        Initialize enhanced research executor
        
        Args:
            openai_api_key: Optional OpenAI API key for enhanced synthesis
            openai_model: OpenAI model to use (gpt-3.5-turbo, gpt-4)
        """
        # Initialize core components
        self.action_graph = KnowledgeAgentActionGraph()
        self.path_optimizer = ActionPathOptimizer()
        
        # Initialize agents (our existing multi-modal system)
        self.pdf_agent = PDFAgent()
        self.youtube_agent = YouTubeAgent()
        
        # Initialize simple content store
        self.content_store = SimpleContentStore()
        
        # Initialize synthesizers
        self.openai_synthesizer = OpenAISynthesizer(openai_api_key, openai_model)
        self.fallback_synthesizer = FallbackSynthesizer()
        
        # Track processing statistics
        self.stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "openai_synthesis_used": 0,
            "total_cost": 0.0
        }
        
        logger.info(f"Enhanced Research Executor initialized")
        logger.info(f"OpenAI available: {self.openai_synthesizer.is_available()}")
    
    async def execute_research_query(
        self,
        query: str,
        pdf_files: Optional[List[str]] = None,
        youtube_urls: Optional[List[str]] = None,
        use_openai: bool = True,
        max_steps: int = 10
    ) -> EnhancedResearchResult:
        """
        Execute enhanced research query with KnowledgeAgent planning
        
        Args:
            query: Research question
            pdf_files: List of PDF file paths
            youtube_urls: List of YouTube URLs
            use_openai: Whether to use OpenAI for synthesis
            max_steps: Maximum action steps
            
        Returns:
            EnhancedResearchResult with comprehensive analysis
        """
        start_time = time.time()
        self.stats["total_queries"] += 1
        
        logger.info(f"Starting enhanced research query: {query}")
        
        try:
            # Step 1: Plan research path using KnowAgent methodology
            available_resources = {
                "pdf_files": pdf_files or [],
                "youtube_urls": youtube_urls or []
            }
            
            planned_actions = self.action_graph.plan_research_path(query, available_resources)
            action_path = ActionPath(query=query, steps=[])
            
            logger.info(f"Planned action path: {[action.value for action in planned_actions]}")
            
            # Step 2: Execute planned actions
            execution_results = await self._execute_action_path(
                query, planned_actions, available_resources, max_steps
            )
            
            action_path.steps = execution_results["steps"]
            
            # Step 3: Synthesize results
            synthesis_result = await self._synthesize_results(
                query, execution_results, use_openai
            )
            
            # Step 4: Create final result
            research_result = self._create_research_result(
                query, action_path, synthesis_result, execution_results, start_time
            )
            
            # Step 5: Learn from this trajectory
            action_path.final_answer = research_result.final_answer
            action_path.success = True
            action_path.confidence_score = research_result.research_confidence
            action_path.total_execution_time = research_result.total_execution_time
            
            self.path_optimizer.record_trajectory(action_path)
            self.stats["successful_queries"] += 1
            
            logger.info(f"Research query completed successfully in {research_result.total_execution_time:.2f}s")
            return research_result
            
        except Exception as e:
            logger.error(f"Research query failed: {e}")
            
            # Create failure result
            error_result = EnhancedResearchResult(
                query=query,
                final_answer=f"Research failed: {str(e)}",
                action_path=ActionPath(query=query, steps=[], success=False),
                synthesis_result=None,
                sources_processed={"errors": 1},
                total_execution_time=time.time() - start_time,
                research_confidence=0.0,
                next_steps=["Please try rephrasing your query or check your sources"],
                cost_breakdown={"error": 0.0}
            )
            
            return error_result
    
    async def _execute_action_path(
        self,
        query: str,
        planned_actions: List[ActionType],
        available_resources: Dict[str, Any],
        max_steps: int
    ) -> Dict[str, Any]:
        """Execute the planned action path"""
        
        steps = []
        results = {
            "pdf_results": [],
            "youtube_results": [],
            "vector_results": [],
            "steps": steps
        }
        
        step_number = 1
        
        for action_type in planned_actions:
            if step_number > max_steps:
                break
                
            step = self.action_graph.create_action_step(step_number, action_type, query)
            
            try:
                # Execute the action
                observation = await self._execute_single_action(
                    action_type, query, available_resources, results
                )
                
                step.observation = observation
                step.success = True
                
                logger.info(f"Action {step_number} ({action_type.value}): Success")
                
            except Exception as e:
                step.observation = f"Action failed: {str(e)}"
                step.success = False
                step.error_message = str(e)
                
                logger.warning(f"Action {step_number} ({action_type.value}): Failed - {e}")
            
            steps.append(step)
            step_number += 1
        
        return results
    
    async def _execute_single_action(
        self,
        action_type: ActionType,
        query: str,
        available_resources: Dict[str, Any],
        results: Dict[str, Any]
    ) -> str:
        """Execute a single action and return observation"""
        
        if action_type == ActionType.START:
            return "Research session initiated"
            
        elif action_type == ActionType.SEARCH_PDF:
            if available_resources.get("pdf_files"):
                return f"Found {len(available_resources['pdf_files'])} PDF files for processing"
            else:
                return "No PDF files available for processing"
                
        elif action_type == ActionType.PROCESS_PDF:
            if available_resources.get("pdf_files"):
                pdf_results = []
                for pdf_file in available_resources["pdf_files"]:
                    try:
                        logger.info(f"Starting full PDF processing pipeline for: {pdf_file}")
                        
                        # Step 1: Extract text
                        extract_result = await self.pdf_agent.execute_action("extract_text", {"pdf_path": pdf_file})
                        
                        if not extract_result.success:
                            logger.warning(f"PDF text extraction failed for {pdf_file}: {extract_result.error_message}")
                            continue
                        
                        logger.info(f"✅ Text extraction successful: {len(extract_result.output.get('extracted_text', ''))} characters")
                        
                        # Step 2: Chunk document
                        chunk_result = await self.pdf_agent.execute_action(
                            "chunk_document", 
                            pdf_file,
                            previous_results={"extract_text": extract_result.output}
                        )
                        
                        if not chunk_result.success:
                            logger.warning(f"PDF chunking failed for {pdf_file}: {chunk_result.error_message}")
                            continue
                        
                        logger.info(f"✅ Chunking successful: {chunk_result.output.get('total_chunks', 0)} chunks created")
                        
                        # Step 3: Create embeddings
                        embed_result = await self.pdf_agent.execute_action(
                            "create_embeddings",
                            pdf_file,
                            previous_results={"chunk_document": chunk_result.output}
                        )
                        
                        if not embed_result.success:
                            logger.warning(f"PDF embeddings creation failed for {pdf_file}: {embed_result.error_message}")
                            continue
                        
                        logger.info(f"✅ Embeddings creation successful: {embed_result.output.get('embedding_count', 0)} embeddings created")
                        
                        # Extract actual paper title from the content
                        extracted_text = extract_result.output.get("extracted_text", "")
                        paper_title = self._extract_paper_title(extracted_text, pdf_file)
                        
                        # Store content in simple content store using actual title
                        chunks_with_embeddings = embed_result.output.get("enriched_chunks", [])
                        content_metadata = {
                            "source_path": pdf_file,
                            "total_pages": extract_result.output.get("total_pages", 0),
                            "total_chunks": chunk_result.output.get("total_chunks", 0),
                            "embedding_count": embed_result.output.get("embedding_count", 0),
                            "extraction": extract_result.metadata,
                            "chunking": chunk_result.metadata,
                            "embeddings": embed_result.metadata
                        }
                        
                        content_id = self.content_store.add_pdf_content(
                            pdf_path=pdf_file,
                            title=paper_title,
                            chunks=chunks_with_embeddings,
                            metadata=content_metadata
                        )
                        
                        # Convert to expected format with actual paper title
                        pdf_result = {
                            "title": paper_title,
                            "content": extracted_text,
                            "content_id": content_id,
                            "chunks": chunk_result.output.get("chunks", []),
                            "embeddings": chunks_with_embeddings,
                            "metadata": content_metadata,
                            "source_path": pdf_file,
                            "processing_complete": True
                        }
                        pdf_results.append(pdf_result)
                        logger.info(f"✅ Full PDF processing completed: '{paper_title}'")
                        
                    except Exception as e:
                        logger.warning(f"PDF processing failed for {pdf_file}: {e}")
                        import traceback
                        traceback.print_exc()
                
                results["pdf_results"] = pdf_results
                return f"Fully processed {len(pdf_results)} PDF documents (extraction → chunking → embeddings)"
            else:
                return "No PDF files to process"
                
        elif action_type == ActionType.SEARCH_YOUTUBE:
            if available_resources.get("youtube_urls"):
                return f"Found {len(available_resources['youtube_urls'])} YouTube URLs for processing"
            else:
                return "No YouTube URLs available for processing"
                
        elif action_type == ActionType.PROCESS_YOUTUBE:
            if available_resources.get("youtube_urls"):
                youtube_results = []
                for youtube_url in available_resources["youtube_urls"]:
                    try:
                        logger.info(f"Processing YouTube URL: {youtube_url}")
                        
                        # Step 1: Try to get captions first (much faster!)
                        captions_result = await self.youtube_agent.execute_action("get_captions", {"url": youtube_url})
                        
                        transcript_result = None
                        video_info = {}
                        
                        if captions_result.success:
                            logger.info(f"Got captions for {youtube_url} - no audio download needed!")
                            transcript_result = captions_result
                            video_info = captions_result.output.get("video_info", {})
                            
                            # Also classify topics for the captions
                            logger.info(f"Classifying topics for {youtube_url}...")
                            topic_result = await self.youtube_agent.execute_action(
                                "classify_topics", 
                                {"url": youtube_url},
                                previous_results={"get_captions": captions_result.output}
                            )
                        else:
                            logger.info(f"Captions not available for {youtube_url}, falling back to audio transcription...")
                            
                            # Step 1 Fallback: Download audio
                            download_result = await self.youtube_agent.execute_action("download_audio", {"url": youtube_url})
                            
                            if not download_result.success:
                                logger.warning(f"Audio download failed for {youtube_url}: {download_result.error_message}")
                                continue
                            
                            # Step 2 Fallback: Transcribe audio with previous results
                            transcript_result = await self.youtube_agent.execute_action(
                                "transcribe_audio", 
                                {"url": youtube_url},
                                previous_results={"download_audio": download_result.output}
                            )
                            
                            video_info = download_result.output.get("video_info", {})
                        
                        if transcript_result and transcript_result.success:
                            # Extract actual video title
                            video_title = self._extract_video_title(video_info)
                            
                            # Create chunks from transcript segments
                            segments = transcript_result.output.get("segments", [])
                            transcript_text = transcript_result.output.get("transcript", "")
                            
                            # Convert segments to chunks format
                            video_chunks = []
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
                                video_chunks.append(chunk)
                            
                            # Store video content in simple content store
                            content_metadata = {
                                "url": youtube_url,
                                "video_info": video_info,
                                "total_segments": len(segments),
                                "transcript_length": len(transcript_text),
                                "processing_method": "captions" if captions_result.success else "transcription",
                                "metadata": transcript_result.metadata
                            }
                            
                            content_id = self.content_store.add_youtube_content(
                                url=youtube_url,
                                title=video_title,
                                chunks=video_chunks,
                                metadata=content_metadata
                            )
                            
                            # Convert to expected format
                            youtube_result = {
                                "title": video_title,
                                "content_id": content_id,
                                "transcript": transcript_text,
                                "url": youtube_url,
                                "video_info": video_info,
                                "segments": segments,
                                "chunks": video_chunks,
                                "metadata": content_metadata
                            }
                            youtube_results.append(youtube_result)
                            logger.info(f"✅ Successfully processed YouTube: '{video_title}'")
                        else:
                            logger.warning(f"YouTube transcription failed for {youtube_url}: {transcript_result.error_message}")
                            
                    except Exception as e:
                        logger.warning(f"YouTube processing failed for {youtube_url}: {e}")
                        import traceback
                        traceback.print_exc()
                
                results["youtube_results"] = youtube_results
                return f"Processed {len(youtube_results)} YouTube videos successfully"
            else:
                return "No YouTube URLs to process"
                
        elif action_type == ActionType.SEARCH_VECTOR:
            # TODO: Integrate with vector database search
            results["vector_results"] = []
            return "Vector database search completed (not yet implemented)"
            
        elif action_type == ActionType.RETRIEVE_CONTENT:
            # TODO: Integrate with vector database retrieval
            return "Content retrieval from knowledge base completed"
            
        elif action_type == ActionType.LOOKUP_CONTENT:
            # TODO: Implement content lookup functionality
            return "Content lookup completed"
            
        elif action_type == ActionType.SYNTHESIZE:
            return "Ready for synthesis of collected information"
            
        elif action_type == ActionType.FINISH:
            return "Research task completed"
            
        else:
            return f"Unknown action type: {action_type.value}"
    
    async def _synthesize_results(
        self,
        query: str,
        execution_results: Dict[str, Any],
        use_openai: bool
    ) -> Any:
        """Synthesize results using OpenAI or fallback"""
        
        pdf_results = execution_results.get("pdf_results", [])
        youtube_results = execution_results.get("youtube_results", [])
        vector_results = execution_results.get("vector_results", [])
        
        # Choose synthesizer
        if use_openai and self.openai_synthesizer.is_available():
            synthesizer = self.openai_synthesizer
            self.stats["openai_synthesis_used"] += 1
        else:
            synthesizer = self.fallback_synthesizer
        
        # Perform synthesis
        synthesis_result = await synthesizer.synthesize_research_results(
            query, pdf_results, youtube_results, vector_results
        )
        
        # Track costs
        if hasattr(synthesis_result, 'cost_estimate'):
            self.stats["total_cost"] += synthesis_result.cost_estimate
        
        return synthesis_result
    
    def _create_research_result(
        self,
        query: str,
        action_path: ActionPath,
        synthesis_result: Any,
        execution_results: Dict[str, Any],
        start_time: float
    ) -> EnhancedResearchResult:
        """Create the final research result"""
        
        return EnhancedResearchResult(
            query=query,
            final_answer=synthesis_result.answer,
            action_path=action_path,
            synthesis_result=synthesis_result,
            sources_processed={
                "pdf_documents": len(execution_results.get("pdf_results", [])),
                "youtube_videos": len(execution_results.get("youtube_results", [])),
                "vector_results": len(execution_results.get("vector_results", []))
            },
            total_execution_time=time.time() - start_time,
            research_confidence=getattr(synthesis_result, 'confidence_score', 0.8),
            next_steps=getattr(synthesis_result, 'follow_up_suggestions', [
                "Consider exploring related topics",
                "Search for more recent sources"
            ]),
            cost_breakdown={
                "openai_synthesis": getattr(synthesis_result, 'cost_estimate', 0.0),
                "total": getattr(synthesis_result, 'cost_estimate', 0.0)
            }
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get executor statistics"""
        success_rate = 0.0
        if self.stats["total_queries"] > 0:
            success_rate = self.stats["successful_queries"] / self.stats["total_queries"]
        
        return {
            **self.stats,
            "success_rate": success_rate,
            "openai_usage_rate": (
                self.stats["openai_synthesis_used"] / max(self.stats["total_queries"], 1)
            )
        }
    
    async def optimize_future_queries(self):
        """Use learned trajectories to optimize future queries"""
        # Save trajectory data
        self.path_optimizer.save_trajectories()
        
        # Log optimization insights
        stats = self.get_statistics()
        logger.info(f"Research Executor Statistics: {stats}")
    
    def _extract_paper_title(self, extracted_text: str, pdf_file: str) -> str:
        """Extract actual paper title from PDF text"""
        
        lines = extracted_text.split('\n')
        
        # Look for title patterns in first few pages
        for line in lines[:50]:  # Check first 50 lines
            line = line.strip()
            
            # Skip empty lines, page numbers, headers
            if not line or line.isdigit() or len(line) < 10:
                continue
            
            # Skip common non-title patterns
            if any(skip in line.lower() for skip in ['abstract', 'introduction', 'page', 'author', 'university', 'school', 'department']):
                continue
            
            # Look for title-like lines (not too long, has meaningful content)
            if 20 <= len(line) <= 100 and not line.startswith('['):
                # Clean up the title
                title = re.sub(r'\s+', ' ', line)
                title = re.sub(r'[^\w\s\-\:\.]', '', title)
                
                if len(title.strip()) > 15:
                    logger.info(f"📄 Extracted paper title: '{title.strip()}'")
                    return title.strip()
        
        # Fallback: use filename
        fallback_title = Path(pdf_file).stem.replace('_', ' ').replace('-', ' ').title()
        logger.info(f"📄 Using filename as title: '{fallback_title}'")
        return fallback_title
    
    def _extract_video_title(self, video_info: Dict[str, Any]) -> str:
        """Extract actual video title"""
        
        title = video_info.get('title', 'Unknown Video')
        
        # Clean up title
        title = re.sub(r'\s+', ' ', title)
        title = title.strip()
        
        if len(title) > 100:
            title = title[:97] + "..."
        
        logger.info(f"🎥 Extracted video title: '{title}'")
        return title
    
    async def search_existing_content(self, search_term: str) -> Dict[str, Any]:
        """
        Search existing content using regex on titles
        Used for queries like "what about darkmatter that is not in previous papers"
        """
        
        logger.info(f"🔍 Searching existing content for: '{search_term}'")
        
        matching_items = self.content_store.search_content_by_name(search_term)
        
        if not matching_items:
            return {
                'search_term': search_term,
                'matching_items': [],
                'total_chunks': 0,
                'summary': f"No existing content found matching '{search_term}'"
            }
        
        # Get all chunks from matching content
        matching_chunks = self.content_store.get_content_chunks([item.id for item in matching_items])
        
        return {
            'search_term': search_term,
            'matching_items': [{'title': item.title, 'type': item.content_type, 'id': item.id} for item in matching_items],
            'total_chunks': len(matching_chunks),
            'chunks': matching_chunks,
            'summary': f"Found {len(matching_items)} sources with {len(matching_chunks)} chunks about '{search_term}'"
        }
    
    async def analyze_new_vs_existing(self, search_term: str, new_content: List[Dict]) -> Dict[str, Any]:
        """
        Compare new content with existing content
        Find what's new that wasn't in previous sources
        """
        
        logger.info(f"🔬 Analyzing new information about '{search_term}' vs existing content")
        
        analysis = self.content_store.analyze_new_information(search_term, new_content)
        
        return analysis

# Test the enhanced executor
if __name__ == "__main__":
    async def test_enhanced_executor():
        # Test with fallback (no API key)
        executor = EnhancedResearchExecutor()
        
        test_query = "What is the main contribution of this research paper?"
        test_pdf_files = ["test_paper.pdf"]  # Would need actual file
        
        result = await executor.execute_research_query(
            query=test_query,
            pdf_files=test_pdf_files,
            use_openai=False  # Use fallback for testing
        )
        
        print(f"Query: {result.query}")
        print(f"Answer: {result.final_answer[:200]}...")
        print(f"Execution time: {result.total_execution_time:.2f}s")
        print(f"Confidence: {result.research_confidence}")
        print(f"Sources processed: {result.sources_processed}")
        print(f"Action path: {[step.action_type.value for step in result.action_path.steps]}")
        
        # Print statistics
        stats = executor.get_statistics()
        print(f"Executor stats: {stats}")
    
    # Run test
    asyncio.run(test_enhanced_executor())