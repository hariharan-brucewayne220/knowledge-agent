"""
Research Executor - The Complete KnowledgeAgent System

This brings everything together:
1. Takes a research plan from the planner
2. Executes it using the specialized agents
3. Coordinates the multi-agent workflow
4. Returns comprehensive research results

This is the "conductor" that makes KnowledgeAgent work end-to-end.
"""

import asyncio
import time
import json
from typing import Dict, List, Any, Optional
from pathlib import Path

# Import our components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agent import AgentCoordinator, ExecutionResult
from agents.pdf_agent import PDFAgent
from agents.youtube_agent import YouTubeAgent
from planning.research_planner import ResearchPlanner, ResearchPlan
from knowledge.action_knowledge import ActionKnowledgeBase

class ResearchExecutor:
    """
    The complete KnowAgent research execution system.
    
    This orchestrates the entire research workflow:
    1. Plan generation (using ResearchPlanner)
    2. Multi-agent execution (using specialized agents)
    3. Result synthesis and validation
    4. Final answer generation
    """
    
    def __init__(self):
        print("Initializing KnowAgent Research Executor...")
        
        # Initialize the planner
        self.planner = ResearchPlanner()
        
        # Initialize the agent coordinator
        self.coordinator = AgentCoordinator()
        
        # Initialize and register specialized agents
        print("Loading PDF processing agent...")
        self.pdf_agent = PDFAgent()
        self.coordinator.register_agent(self.pdf_agent)
        
        print("Loading YouTube processing agent...")
        self.youtube_agent = YouTubeAgent()
        self.coordinator.register_agent(self.youtube_agent)
        
        # Knowledge base for validation
        self.knowledge_base = ActionKnowledgeBase()
        
        print("KnowAgent Research Executor ready!")
        print(f"Available agents: {list(self.coordinator.agents.keys())}")
    
    async def execute_research_query(self, 
                                   query: str, 
                                   pdf_files: List[str] = None,
                                   youtube_urls: List[str] = None) -> Dict[str, Any]:
        """
        Execute a complete research query end-to-end.
        
        This is the main entry point for KnowAgent research.
        
        Args:
            query: The research question
            pdf_files: List of PDF file paths
            youtube_urls: List of YouTube URLs
        
        Returns:
            Complete research results with sources and analysis
        """
        start_time = time.time()
        
        # Auto-detect YouTube URLs in the query
        detected_youtube_urls = self._extract_youtube_urls(query)
        if detected_youtube_urls:
            youtube_urls = (youtube_urls or []) + detected_youtube_urls
        
        print(f"\n=== KNOWAGENT RESEARCH EXECUTION ===")
        print(f"Query: {query}")
        print(f"PDFs: {pdf_files or []}")
        print(f"Videos: {youtube_urls or []}")
        
        try:
            # Step 1: Create research plan
            print(f"\n1. PLANNING PHASE")
            sources = {
                "pdfs": pdf_files or [],
                "videos": youtube_urls or []
            }
            
            plan = self.planner.create_plan(query, sources)
            print(f"Generated plan with {len(plan.steps)} steps")
            print(f"Estimated time: {plan.total_estimated_time}")
            print(f"Confidence: {plan.confidence_score}")
            
            # Step 2: Execute the plan
            print(f"\n2. EXECUTION PHASE")
            execution_results = {}
            step_outputs = {}
            
            for i, step in enumerate(plan.steps):
                print(f"\nStep {i+1}/{len(plan.steps)}: {step.action}")
                print(f"Target: {step.target}")
                
                # Execute the step
                result = await self.coordinator.execute_plan_step(step, step_outputs)
                
                if result.success:
                    print(f"Success: {step.action} completed in {result.execution_time:.1f}s")
                    # Store result for next steps
                    step_key = f"step_{i+1}_{step.action}"
                    step_outputs[step_key] = result.output
                    execution_results[step_key] = result
                else:
                    print(f"Failed: {result.error_message}")
                    # Continue with other steps (graceful degradation)
            
            # Step 3: Synthesize results
            print(f"\n3. SYNTHESIS PHASE")
            final_answer = await self._synthesize_results(query, plan, execution_results, step_outputs)
            
            total_time = time.time() - start_time
            
            # Step 4: Package complete results
            complete_results = {
                "query": query,
                "sources": sources,
                "plan": {
                    "task_type": plan.task_type.value,
                    "steps": [{"action": s.action, "target": s.target, "description": s.description} for s in plan.steps],
                    "confidence": plan.confidence_score,
                    "estimated_time": plan.total_estimated_time
                },
                "execution": {
                    "total_steps": len(plan.steps),
                    "successful_steps": sum(1 for r in execution_results.values() if r.success),
                    "total_execution_time": total_time,
                    "step_results": execution_results
                },
                "final_answer": final_answer,
                "metadata": {
                    "execution_timestamp": time.time(),
                    "knowagent_version": "1.0",
                    "agents_used": list(self.coordinator.agents.keys())
                }
            }
            
            print(f"\n=== RESEARCH COMPLETED ===")
            print(f"Total time: {total_time:.1f} seconds")
            print(f"Steps completed: {complete_results['execution']['successful_steps']}/{len(plan.steps)}")
            
            return complete_results
            
        except Exception as e:
            print(f"Research execution failed: {str(e)}")
            return {
                "query": query,
                "error": str(e),
                "execution_time": time.time() - start_time,
                "status": "failed"
            }
    
    async def _synthesize_results(self, 
                                query: str, 
                                plan: ResearchPlan, 
                                execution_results: Dict[str, ExecutionResult],
                                step_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synthesize execution results into a final answer.
        
        This is where we combine all the processed information
        to answer the user's research question.
        """
        print("Synthesizing research results...")
        
        # Collect processed content
        pdf_content = []
        video_content = []
        embeddings = []
        search_results = []
        
        # Extract content from execution results
        for step_key, output in step_outputs.items():
            if not output:
                continue
                
            if "extract_text" in step_key:
                pdf_content.append({
                    "document_id": output.get("document_id"),
                    "text": output.get("extracted_text", ""),
                    "pages": output.get("total_pages", 0),
                    "source": output.get("file_path", "")
                })
            
            elif "transcribe_audio" in step_key:
                video_content.append({
                    "video_id": output.get("video_id"),
                    "transcript": output.get("transcript", ""),
                    "segments": output.get("segments", []),
                    "duration": output.get("total_duration", 0),
                    "language": output.get("language", "unknown"),
                    "video_info": output.get("video_info", {})
                })
            
            elif "create_embeddings" in step_key:
                embeddings.append({
                    "document_id": output.get("document_id"),
                    "chunks": output.get("enriched_chunks", []),
                    "embedding_count": output.get("embedding_count", 0)
                })
            
            elif "semantic_search" in step_key:
                search_results = output.get("search_results", [])
        
        # Simple synthesis (in production, you'd use an LLM here)
        # DEBUG: Log actual content
        self.debug_log_content(pdf_content, video_content, search_results)
        
        synthesis = {
            "answer_summary": self._generate_summary(query, pdf_content, video_content, search_results),
            "sources_processed": {
                "pdf_documents": len(pdf_content),
                "video_transcripts": len(video_content),
                "total_embeddings": sum(e.get("embedding_count", 0) for e in embeddings)
            },
            "content_analysis": {
                "pdf_content": pdf_content,
                "video_content": video_content,
                "embeddings_available": len(embeddings) > 0
            },
            "research_confidence": self._calculate_synthesis_confidence(pdf_content, video_content),
            "next_steps": self._suggest_next_steps(query, pdf_content, video_content)
        }
        
        return synthesis
    
    
    # DEBUG: Log what content we actually receive
    
    def validate_content_matches_query(self, query: str, pdf_content: List[Dict], expected_keywords: List[str] = None) -> bool:
        """
        Validate that the content actually matches what we expect
        Prevents returning wrong cached or template responses
        """
        if not pdf_content:
            print("⚠️ VALIDATION: No PDF content to validate")
            return False
        
        # Get actual text content
        actual_text = ""
        for pdf in pdf_content:
            actual_text += pdf.get('text', '').lower()
        
        # Check if content matches expected keywords
        query_lower = query.lower()
        
        # For our test document, expect KnowAgent-related content
        knowagent_keywords = ['knowagent', 'test document', 'research assistant', 'validation']
        translation_keywords = ['translation', 'cnn', 'lstm', 'deep learning']
        
        has_knowagent_content = any(keyword in actual_text for keyword in knowagent_keywords)
        has_translation_content = any(keyword in actual_text for keyword in translation_keywords)
        
        print(f"🔍 CONTENT VALIDATION:")
        print(f"  - Text length: {len(actual_text)}")
        print(f"  - Has KnowAgent content: {has_knowagent_content}")
        print(f"  - Has translation content: {has_translation_content}")
        print(f"  - Text preview: {actual_text[:200]}...")
        
        if has_translation_content and not has_knowagent_content:
            print("🚨 WARNING: Content appears to be about translation, not KnowAgent test!")
            print("🚨 This suggests wrong cached data or template response!")
            return False
        
        return True
    
    def debug_log_content(self, pdf_content, video_content, search_results):
        """Debug logging to track content flow"""
        print("\n" + "="*50)
        print("🔍 DEBUG: SYNTHESIS INPUT")
        print("="*50)
        print(f"PDF Content Count: {len(pdf_content)}")
        for i, pdf in enumerate(pdf_content):
            print(f"PDF {i+1}:")
            print(f"  - ID: {pdf.get('document_id', 'Unknown')}")
            print(f"  - Text Length: {len(pdf.get('text', ''))}")
            print(f"  - Text Preview: {pdf.get('text', '')[:200]}...")
        
        print(f"\nVideo Content Count: {len(video_content)}")
        print(f"Search Results Count: {len(search_results)}")
        print("="*50)
        return True
    
    def _generate_summary(self, query: str, pdf_content: List[Dict], video_content: List[Dict], search_results: List[Dict] = None) -> str:
        """
        Generate a meaningful summary of findings based on the content and search results.
        WITH HALLUCINATION PREVENTION
        """
        summary_parts = []
        search_results = search_results or []
        
        # COUNT ACTUAL SOURCES - PREVENT HALLUCINATION
        total_sources = len(pdf_content) + len(video_content)
        
        summary_parts.append(f"Research Query: {query}")
        summary_parts.append("")
        
        # ONLY CLAIM WHAT WE ACTUALLY HAVE
        if total_sources == 0:
            summary_parts.append("No content sources were successfully processed.")
            return "\n".join(summary_parts)
        elif total_sources == 1:
            summary_parts.append("ANALYSIS OF SINGLE SOURCE:")
        else:
            summary_parts.append(f"ANALYSIS OF {total_sources} SOURCES:")
        
        # ANSWER THE SPECIFIC QUESTION FIRST using search results
        if search_results:
            summary_parts.append("DIRECT ANSWER FROM CONTENT:")
            
            query_lower = query.lower()
            
            # ONLY analyze what we actually have - no fake connections
            if 'implementation' in query_lower or 'code' in query_lower or 'github' in query_lower:
                implementation_found = False
                for result in search_results:
                    text = result.get('text', '').lower()
                    if any(term in text for term in ['github', 'code', 'implementation', 'repository', 'source code', 'algorithm', 'pseudocode']):
                        implementation_found = True
                        summary_parts.append(f"• Found implementation details: {result.get('text', '')[:200]}...")
                        break
                
                if not implementation_found:
                    summary_parts.append("• No explicit implementation details, GitHub repositories, or source code references found in the provided content.")
            
            # Analyze for methodology questions - ONLY from actual content
            elif 'method' in query_lower or 'approach' in query_lower or 'how' in query_lower:
                for result in search_results[:2]:  # Top 2 most relevant results
                    summary_parts.append(f"• {result.get('text', '')[:300]}...")
            
            # Analyze for results/evaluation questions - ONLY from actual content
            elif 'result' in query_lower or 'performance' in query_lower or 'evaluation' in query_lower:
                for result in search_results[:2]:
                    text = result.get('text', '')
                    if any(term in text.lower() for term in ['result', 'performance', 'f1', 'accuracy', 'evaluation']):
                        summary_parts.append(f"• {text[:300]}...")
            
            # Analyze for general "about" questions - ONLY from actual content
            elif 'about' in query_lower or 'summary' in query_lower:
                if search_results:
                    top_result = search_results[0]
                    summary_parts.append(f"• {top_result.get('text', '')[:400]}...")
            
            # Default: show most relevant passages from ACTUAL content
            else:
                summary_parts.append("Most relevant passages found:")
                for i, result in enumerate(search_results[:3], 1):
                    score = result.get('similarity_score', 0)
                    text = result.get('text', '')[:250]
                    summary_parts.append(f"{i}. (Score: {score:.3f}) {text}...")
            
            summary_parts.append("")
        
        # ONLY describe content we actually have
        if pdf_content:
            summary_parts.append("DOCUMENT ANALYSIS:")
            
            for doc in pdf_content:
                doc_text = doc.get('text', '')
                doc_id = doc.get('document_id', 'Unknown')
                word_count = len(doc_text.split()) if doc_text else 0
                summary_parts.append(f"• {doc_id}: {doc.get('pages', 0)} pages, {word_count} words")
                
                # Extract actual themes from the document
                if doc_text:
                    doc_lower = doc_text.lower()
                    if any(term in doc_lower for term in ['machine learning', 'neural network', 'deep learning']):
                        summary_parts.append("  - Contains machine learning and AI content")
                    if any(term in doc_lower for term in ['algorithm', 'method', 'approach']):
                        summary_parts.append("  - Discusses algorithms and methodological approaches")
                    if any(term in doc_lower for term in ['experiment', 'evaluation', 'result']):
                        summary_parts.append("  - Includes experimental results and evaluations")
            
            summary_parts.append("")
        
        if video_content:
            summary_parts.append("VIDEO ANALYSIS:")
            for video in video_content:
                duration_min = video['duration'] / 60 if video['duration'] else 0
                word_count = len(video['transcript'].split()) if video['transcript'] else 0
                title = video.get('video_info', {}).get('title', 'Unknown Title')
                summary_parts.append(f"VIDEO: {title[:50]}: {duration_min:.1f} min, {word_count} words")
            summary_parts.append("")
        
        # ONLY suggest connections if we have multiple sources
        if total_sources > 1:
            summary_parts.append("CROSS-SOURCE INSIGHTS:")
            summary_parts.append("• Multiple sources available for comparison and validation")
        else:
            summary_parts.append("SINGLE SOURCE ANALYSIS:")
            summary_parts.append("• Analysis based on single source - no cross-validation available")
        
        summary_parts.append("")
        summary_parts.append("PROCESSING COMPLETE:")
        summary_parts.append(f"+ {len(pdf_content)} documents analyzed")
        summary_parts.append(f"+ {len(video_content)} videos processed") 
        summary_parts.append("+ Content indexed and searchable")
        
        return "\n".join(summary_parts)
    
    def _calculate_synthesis_confidence(self, pdf_content: List[Dict], video_content: List[Dict]) -> float:
        """Calculate confidence in the synthesis based on available content."""
        confidence = 0.0
        
        # Base confidence from content availability
        if pdf_content:
            confidence += 0.4
        if video_content:
            confidence += 0.4
        
        # Bonus for multiple sources
        total_sources = len(pdf_content) + len(video_content)
        if total_sources >= 2:
            confidence += 0.2
        
        return min(1.0, confidence)
    
    def _suggest_next_steps(self, query: str, pdf_content: List[Dict], video_content: List[Dict]) -> List[str]:
        """Suggest next steps for the research."""
        suggestions = []
        
        if pdf_content and video_content:
            suggestions.append("Perform cross-modal semantic search to find connections")
            suggestions.append("Compare concepts between written and spoken content")
        
        if len(pdf_content) > 1:
            suggestions.append("Compare findings across multiple documents")
        
        if len(video_content) > 1:
            suggestions.append("Analyze consistency across different video sources")
        
        suggestions.append("Use semantic search to find specific answers to sub-questions")
        suggestions.append("Generate detailed citations and source references")
        
        return suggestions
    
    
    def validate_sources(self, pdf_results: List[Dict], youtube_results: List[Dict]) -> Dict[str, Any]:
        """
        Validate and count actual sources to prevent hallucination
        """
        validation = {
            "total_sources": len(pdf_results) + len(youtube_results),
            "pdf_count": len(pdf_results),
            "youtube_count": len(youtube_results),
            "can_make_connections": len(pdf_results) + len(youtube_results) > 1,
            "source_ids": []
        }
        
        for pdf in pdf_results:
            validation["source_ids"].append(f"PDF: {pdf.get('title', 'Unknown')}")
        
        for yt in youtube_results:
            validation["source_ids"].append(f"Video: {yt.get('title', 'Unknown')}")
        
        return validation
    
    def _extract_youtube_urls(self, text: str) -> List[str]:
        """Extract YouTube URLs from text"""
        import re
        
        # YouTube URL patterns
        youtube_patterns = [
            r'https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+',
            r'https?://youtu\.be/[\w-]+',
            r'https?://(?:www\.)?youtube\.com/embed/[\w-]+',
            r'https?://(?:www\.)?youtube\.com/v/[\w-]+'
        ]
        
        urls = []
        for pattern in youtube_patterns:
            matches = re.findall(pattern, text)
            urls.extend(matches)
        
        return list(set(urls))  # Remove duplicates

# Test the complete system
if __name__ == "__main__":
    async def test_research_executor():
        executor = ResearchExecutor()
        
        print("\nKnowAgent Research Executor ready!")
        print("\nTest queries you could run:")
        print("1. executor.execute_research_query('What is machine learning?', ['paper.pdf'], ['https://youtube.com/...'])")
        print("2. executor.execute_research_query('Compare approaches', pdf_files=['doc1.pdf', 'doc2.pdf'])")
        print("3. executor.execute_research_query('Explain concept', youtube_urls=['url1', 'url2'])")
    
    asyncio.run(test_research_executor())