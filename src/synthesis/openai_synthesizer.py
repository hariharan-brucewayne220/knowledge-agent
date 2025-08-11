"""
OpenAI API Integration for Enhanced Synthesis
Combines KnowAgent planning with GPT-3.5/GPT-4 intelligent reasoning
"""

import os
import json
import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Try to import OpenAI, gracefully handle if not available
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None

logger = logging.getLogger(__name__)

@dataclass 
class SynthesisResult:
    """Result from OpenAI synthesis"""
    answer: str
    confidence_score: float
    reasoning_steps: List[str]
    sources_used: List[str]
    model_used: str
    tokens_used: int
    cost_estimate: float
    timestamp: datetime

class OpenAISynthesizer:
    """
    Enhanced synthesis using OpenAI GPT models
    Integrates with our existing KnowAgent system
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        """
        Initialize OpenAI synthesizer
        
        Args:
            api_key: OpenAI API key (if None, tries environment variable)
            model: Model to use (gpt-3.5-turbo, gpt-4, gpt-4-turbo)
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.model = model
        self.available = False
        
        if not OPENAI_AVAILABLE:
            logger.info("OpenAI library not installed, synthesis will use local models")
            self.available = False
        elif self.api_key:
            try:
                openai.api_key = self.api_key
                self.client = openai.OpenAI(api_key=self.api_key)
                self.available = True
                logger.info(f"OpenAI API initialized with model: {model}")
            except Exception as e:
                logger.warning(f"OpenAI API initialization failed: {e}")
                self.available = False
        else:
            logger.info("No OpenAI API key provided, synthesis will use local models")
            
        # Token cost estimates (approximate, update as needed)
        self.cost_per_1k_tokens = {
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03}
        }
    
    def is_available(self) -> bool:
        """Check if OpenAI API is available"""
        return self.available
    
    async def synthesize_research_results(
        self, 
        query: str, 
        pdf_results: List[Dict[str, Any]], 
        youtube_results: List[Dict[str, Any]],
        vector_results: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> SynthesisResult:
        """
        Synthesize research results using OpenAI
        
        Args:
            query: Original research query
            pdf_results: Results from PDF processing
            youtube_results: Results from YouTube processing  
            vector_results: Results from vector database
            context: Additional context information
            
        Returns:
            SynthesisResult with comprehensive answer
        """
        if not self.available:
            raise ValueError("OpenAI API not available. Please provide valid API key.")
        
        # Prepare context for OpenAI
        context_text = self._prepare_context(query, pdf_results, youtube_results, vector_results, context)
        
        # Create the synthesis prompt
        system_prompt = self._create_system_prompt()
        user_prompt = self._create_user_prompt(query, context_text)
        
        try:
            # Call OpenAI API
            response = await self._call_openai_api(system_prompt, user_prompt)
            
            # Parse and structure the response
            synthesis_result = self._parse_synthesis_response(response, query)
            
            logger.info(f"OpenAI synthesis completed for query: {query[:50]}...")
            return synthesis_result
            
        except Exception as e:
            logger.error(f"OpenAI synthesis failed: {e}")
            raise
    
    async def _call_openai_api(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Make async call to OpenAI API"""
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=2000,
                    top_p=0.9
                )
            )
            
            return {
                "content": response.choices[0].message.content,
                "tokens_used": response.usage.total_tokens,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "model": response.model
            }
            
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise
    
    def _prepare_context(
        self, 
        query: str,
        pdf_results: List[Dict[str, Any]], 
        youtube_results: List[Dict[str, Any]],
        vector_results: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Prepare context text for OpenAI"""
        
        context_parts = []
        
        # Add PDF results
        if pdf_results:
            context_parts.append("=== PDF Document Analysis ===")
            for i, pdf_result in enumerate(pdf_results[:3]):  # Limit to top 3
                title = pdf_result.get('title', 'Unknown Document')
                source_path = pdf_result.get('source_path', 'Unknown Path')
                context_parts.append(f"PDF {i+1}: {title}")
                context_parts.append(f"Source ID: {title}")
                context_parts.append(f"File Path: {source_path}")
                context_parts.append(f"Source Type: pdf")
                
                # Provide much more content for analysis (up to 8000 chars for meaningful analysis)
                full_content = pdf_result.get('content', '')
                if len(full_content) > 8000:
                    # Take beginning, middle, and end sections for comprehensive view
                    beginning = full_content[:3000]
                    middle_start = len(full_content) // 2 - 1500
                    middle = full_content[middle_start:middle_start + 3000]
                    end = full_content[-2000:]
                    content_summary = f"{beginning}\n\n[... MIDDLE SECTION ...]\n\n{middle}\n\n[... END SECTION ...]\n\n{end}"
                    context_parts.append(f"Content Structure: [BEGINNING SECTION] + [MIDDLE SECTION] + [END SECTION]")
                else:
                    content_summary = full_content
                    context_parts.append(f"Content Structure: [FULL DOCUMENT]")
                
                context_parts.append(f"Content: {content_summary}")
                if pdf_result.get('metadata'):
                    metadata = pdf_result['metadata']
                    if 'page_count' in metadata:
                        context_parts.append(f"Total Pages: {metadata['page_count']}")
                    if 'chunks_count' in metadata:
                        context_parts.append(f"Content Chunks: {metadata['chunks_count']}")
                    context_parts.append(f"Additional Metadata: {json.dumps(metadata, indent=2)}")
                context_parts.append("")
        
        # Add YouTube results  
        if youtube_results:
            context_parts.append("=== YouTube Video Analysis ===")
            for i, yt_result in enumerate(youtube_results[:3]):  # Limit to top 3
                title = yt_result.get('title', 'Unknown Video')
                url = yt_result.get('url', 'Unknown URL')
                context_parts.append(f"Video {i+1}: {title}")
                context_parts.append(f"Source ID: {title}")
                context_parts.append(f"URL: {url}")
                context_parts.append(f"Source Type: youtube")
                
                # Smart transcript chunking to fit within token limits
                full_transcript = yt_result.get('transcript', '')
                transcript_summary = self._create_transcript_summary(full_transcript)
                context_parts.append(f"Transcript Summary: {transcript_summary}")
                context_parts.append(f"Note: Reference specific timestamps when citing this video (format: 'at 3:45' or '2:15-3:30')")
                
                if yt_result.get('duration'):
                    context_parts.append(f"Video Duration: {yt_result['duration']}")
                context_parts.append("")
        
        # Add vector database results
        if vector_results:
            context_parts.append("=== Knowledge Base Results ===")
            for i, vector_result in enumerate(vector_results[:5]):  # Limit to top 5
                context_parts.append(f"Result {i+1} (Score: {vector_result.get('score', 'N/A')}):")
                context_parts.append(f"Content: {vector_result.get('content', '')[:300]}...")
                context_parts.append("")
        
        # Add additional context
        if context:
            context_parts.append("=== Additional Context ===")
            context_parts.append(json.dumps(context, indent=2))
        
        return "\n".join(context_parts)
    
    def _create_transcript_summary(self, full_transcript: str) -> str:
        """
        Create an intelligent summary of a long transcript to fit within token limits
        Uses key sections: beginning, middle sections, and end
        """
        if len(full_transcript) <= 3000:  # Short enough to include fully
            return full_transcript
        
        # For long transcripts, create strategic excerpts
        words = full_transcript.split()
        total_words = len(words)
        
        if total_words <= 2000:  # Use full transcript
            return full_transcript
        
        # Extract key sections for comprehensive understanding
        beginning_words = 400   # Opening announcements + lecture start
        middle_words = 800      # Core content
        end_words = 400         # Conclusions
        
        beginning = " ".join(words[:beginning_words])
        
        # Multiple middle sections to capture different topics
        middle_sections = []
        section_size = middle_words // 3
        
        for i in range(3):
            start_idx = beginning_words + (total_words - beginning_words - end_words) * i // 3
            end_idx = start_idx + section_size
            if end_idx < total_words - end_words:
                section = " ".join(words[start_idx:end_idx])
                middle_sections.append(f"[Section {i+1}]: {section}")
        
        end = " ".join(words[-end_words:])
        
        # Combine sections with clear markers
        summary_parts = [
            f"[BEGINNING]: {beginning}",
            "\n".join(middle_sections),
            f"[CONCLUSION]: {end}"
        ]
        
        summary = "\n\n".join(summary_parts)
        
        return f"[Lecture Transcript - {len(words)} words total]:\n{summary}"
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for OpenAI"""
        return """You are an advanced research assistant that synthesizes information from multiple sources to provide comprehensive, detailed, and thorough answers.

Your capabilities:
- Analyze PDF documents for academic and technical content with deep comprehension
- Process YouTube video transcripts for multimedia insights  
- Query knowledge bases for relevant background information
- Synthesize cross-modal information into coherent, detailed explanations

Guidelines:
1. ONLY answer based on the provided source materials - do not use external knowledge
2. Provide COMPREHENSIVE and DETAILED analysis - don't give brief summaries
3. Extract and explain key concepts, theories, methodologies, and findings from the sources
4. Include specific details, examples, and explanations found in the source materials
5. Always cite the specific source where you found each piece of information
6. Never make up facts, equations, quotes, or details not explicitly mentioned in the sources
7. If asked for specific details (like equations, quotes, numbers) that aren't in the sources, say so clearly
8. Provide detailed reasoning steps that show your analytical process
9. Acknowledge limitations and uncertainties, but provide rich analysis of what IS available
10. Distinguish between different types of sources (academic papers, videos, general knowledge)
11. Your answer should be substantive and informative - aim for depth and comprehensiveness

Format your response as JSON with these fields:
{
  "answer": "DETAILED and COMPREHENSIVE answer to the query with thorough analysis and explanation",
  "confidence_score": 0.85,
  "reasoning_steps": ["Detailed step 1", "Detailed step 2", "Detailed step 3"],
  "sources_cited": [
    {
      "source_id": "document_title_or_video_title",
      "source_type": "pdf|youtube", 
      "specific_reference": "page 5, section 2.1" or "timestamp 3:45-4:20",
      "content_excerpt": "exact quote or key finding from source",
      "relevance": "how this source supports the answer"
    }
  ],
  "limitations": "Any limitations or uncertainties",
  "follow_up_suggestions": ["Suggestion 1", "Suggestion 2"]
}"""
    
    def _create_user_prompt(self, query: str, context: str) -> str:
        """Create user prompt with query and context"""
        return f"""Research Query: {query}

Available Information:
{context}

Please synthesize this information to provide a COMPREHENSIVE, DETAILED, and THOROUGH answer to the research query. 

CRITICAL REQUIREMENTS:
- Provide a detailed, substantive analysis that fully explores the available content
- Extract and explain key concepts, theories, methods, findings, and insights from the sources
- Include specific examples, details, and explanations found in the source materials
- Your answer should be rich in content and informative - avoid brief or superficial responses
- Only use information explicitly provided in the sources above
- If information is not in the sources, state "This information is not available in the provided sources"
- For EVERY fact, number, or claim you mention, provide specific source citation with:
  * PDF sources: Include page numbers, section titles, or chunk identifiers
  * YouTube sources: Include timestamps (e.g., "at 3:45" or "between 2:15-3:30")
  * Always include exact quotes or specific data points from the source
- Do not make up equations, quotes, numbers, or any details not in the sources
- If asked for specific details not in the sources, clearly state they are not available
- In the sources_cited array, provide detailed provenance for each source used
- Aim for depth and comprehensiveness rather than brevity"""
    
    def _parse_synthesis_response(self, response: Dict[str, Any], query: str) -> SynthesisResult:
        """Parse OpenAI response into structured result"""
        try:
            # Try to parse as JSON first
            content = response["content"]
            if content.strip().startswith("{"):
                parsed = json.loads(content)
                answer = parsed.get("answer", content)
                confidence_score = parsed.get("confidence_score", 0.8)
                reasoning_steps = parsed.get("reasoning_steps", [])
                sources_used = parsed.get("sources_cited", [])
            else:
                # Fallback to plain text
                answer = content
                confidence_score = 0.8
                reasoning_steps = ["OpenAI synthesis"]
                sources_used = ["Multiple sources"]
                
        except json.JSONDecodeError:
            # Plain text response
            answer = response["content"]
            confidence_score = 0.8
            reasoning_steps = ["OpenAI synthesis"]
            sources_used = ["Multiple sources"]
        
        # Calculate cost estimate
        cost_estimate = self._calculate_cost(response["tokens_used"])
        
        return SynthesisResult(
            answer=answer,
            confidence_score=confidence_score,
            reasoning_steps=reasoning_steps,
            sources_used=sources_used,
            model_used=response["model"],
            tokens_used=response["tokens_used"],
            cost_estimate=cost_estimate,
            timestamp=datetime.now()
        )
    
    def _calculate_cost(self, tokens_used: int) -> float:
        """Calculate approximate cost for API usage"""
        if self.model not in self.cost_per_1k_tokens:
            return 0.0
            
        # Rough estimate assuming 70% input, 30% output tokens
        input_tokens = int(tokens_used * 0.7)
        output_tokens = int(tokens_used * 0.3)
        
        costs = self.cost_per_1k_tokens[self.model]
        input_cost = (input_tokens / 1000) * costs["input"]
        output_cost = (output_tokens / 1000) * costs["output"]
        
        return input_cost + output_cost
    
    async def quick_synthesis(self, query: str, content: str) -> str:
        """Quick synthesis for simple queries"""
        if not self.available:
            return content  # Fallback to original content
            
        try:
            response = await self._call_openai_api(
                "You are a helpful research assistant. Provide clear, concise answers.",
                f"Query: {query}\n\nContent: {content}\n\nPlease provide a direct answer to the query based on the content."
            )
            return response["content"]
        except Exception as e:
            logger.warning(f"Quick synthesis failed: {e}")
            return content

class FallbackSynthesizer:
    """
    Fallback synthesizer when OpenAI is not available
    Uses our existing local synthesis methods
    """
    
    def __init__(self):
        self.available = True
    
    def is_available(self) -> bool:
        return True
    
    async def synthesize_research_results(
        self, 
        query: str, 
        pdf_results: List[Dict[str, Any]], 
        youtube_results: List[Dict[str, Any]],
        vector_results: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> SynthesisResult:
        """Local synthesis fallback"""
        
        # Combine all results
        all_content = []
        sources = []
        
        for pdf in pdf_results:
            all_content.append(pdf.get('content', ''))
            sources.append(f"PDF: {pdf.get('title', 'Unknown')}")
            
        for yt in youtube_results:
            all_content.append(yt.get('transcript', ''))
            sources.append(f"Video: {yt.get('title', 'Unknown')}")
            
        for vector in vector_results:
            all_content.append(vector.get('content', ''))
            sources.append(f"Knowledge Base (Score: {vector.get('score', 'N/A')})")
        
        # Simple synthesis with query-aware content selection
        combined_content = "\n".join(all_content)
        
        # If content is long, try to show most relevant parts first
        if len(combined_content) > 2000:
            # Look for query-relevant content first
            query_words = query.lower().split()
            content_parts = []
            
            for content in all_content:
                if any(word in content.lower() for word in query_words if len(word) > 3):
                    content_parts.insert(0, content)  # Put relevant content first
                else:
                    content_parts.append(content)
            
            combined_content = "\n\n".join(content_parts)
            answer = f"Based on the available sources:\n\n{combined_content[:2000]}..."
        else:
            answer = f"Based on the available sources:\n\n{combined_content}"
        
        return SynthesisResult(
            answer=answer,
            confidence_score=0.6,
            reasoning_steps=["Local content aggregation"],
            sources_used=sources,
            model_used="local_fallback",
            tokens_used=len(combined_content.split()),
            cost_estimate=0.0,
            timestamp=datetime.now()
        )

# Test the synthesizer
if __name__ == "__main__":
    async def test_synthesizer():
        # Test with and without API key
        synthesizer = OpenAISynthesizer()
        
        if synthesizer.is_available():
            print("✅ OpenAI API available")
        else:
            print("❌ OpenAI API not available, using fallback")
            synthesizer = FallbackSynthesizer()
        
        # Test synthesis
        test_query = "What is machine learning?"
        test_pdf_results = [{"title": "ML Paper", "content": "Machine learning is..."}]
        test_youtube_results = []
        test_vector_results = []
        
        result = await synthesizer.synthesize_research_results(
            test_query, test_pdf_results, test_youtube_results, test_vector_results
        )
        
        print(f"\nSynthesis Result:")
        print(f"Answer: {result.answer[:200]}...")
        print(f"Confidence: {result.confidence_score}")
        print(f"Model: {result.model_used}")
        print(f"Cost: ${result.cost_estimate:.4f}")
    
    # Run test
    asyncio.run(test_synthesizer())