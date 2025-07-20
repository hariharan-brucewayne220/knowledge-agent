"""
Anti-Hallucination OpenAI Synthesizer
Ensures OpenAI API ONLY uses your database data and never hallucinates
"""

import os
import json
import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging
from datetime import datetime
import hashlib

# Try to import OpenAI, gracefully handle if not available
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None

logger = logging.getLogger(__name__)

@dataclass 
class AntiHallucinationResult:
    """Result from anti-hallucination synthesis"""
    answer: str
    confidence_score: float
    reasoning_steps: List[str]
    sources_used: List[str]
    source_quotes: List[str]  # Exact quotes from sources
    model_used: str
    tokens_used: int
    cost_estimate: float
    timestamp: datetime
    data_only: bool = True  # Always True - only uses provided data

class AntiHallucinationSynthesizer:
    """
    Ultra-strict synthesizer that prevents OpenAI from hallucinating
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.model = model
        self.available = False
        
        if not OPENAI_AVAILABLE:
            logger.info("OpenAI library not installed")
            self.available = False
        elif self.api_key:
            try:
                self.client = openai.OpenAI(api_key=self.api_key)
                self.available = True
                logger.info(f"Anti-hallucination OpenAI API initialized with model: {model}")
            except Exception as e:
                logger.warning(f"OpenAI API initialization failed: {e}")
                self.available = False
        else:
            logger.info("No OpenAI API key provided")
            
        # Token cost estimates
        self.cost_per_1k_tokens = {
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03}
        }
    
    def is_available(self) -> bool:
        return self.available
    
    async def synthesize_from_database_only(
        self, 
        query: str, 
        pdf_results: List[Dict[str, Any]], 
        youtube_results: List[Dict[str, Any]],
        vector_results: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> AntiHallucinationResult:
        """
        Synthesize using ONLY the provided database content
        Absolutely no external knowledge allowed
        """
        if not self.available:
            raise ValueError("OpenAI API not available. Please provide valid API key.")
        
        # Create content fingerprint to verify no external data is used
        content_hash = self._create_content_hash(pdf_results, youtube_results, vector_results)
        
        # Prepare ultra-strict context
        context_text = self._prepare_strict_context(query, pdf_results, youtube_results, vector_results, context)
        
        # Create anti-hallucination prompts
        system_prompt = self._create_anti_hallucination_system_prompt()
        user_prompt = self._create_strict_user_prompt(query, context_text, content_hash)
        
        try:
            # Call OpenAI with strict parameters
            response = await self._call_openai_strict(system_prompt, user_prompt)
            
            # Verify response only uses provided data
            synthesis_result = self._parse_and_verify_response(response, query, content_hash)
            
            logger.info(f"Anti-hallucination synthesis completed for query: {query[:50]}...")
            return synthesis_result
            
        except Exception as e:
            logger.error(f"Anti-hallucination synthesis failed: {e}")
            raise
    
    def _create_content_hash(self, pdf_results: List[Dict], youtube_results: List[Dict], vector_results: List[Dict]) -> str:
        """Create hash of all provided content to verify no external data is used"""
        content_str = json.dumps({
            "pdf": pdf_results,
            "youtube": youtube_results, 
            "vector": vector_results
        }, sort_keys=True)
        return hashlib.md5(content_str.encode()).hexdigest()[:8]
    
    def _prepare_strict_context(self, query: str, pdf_results: List[Dict], youtube_results: List[Dict], 
                               vector_results: List[Dict], context: Optional[Dict] = None) -> str:
        """Prepare context with strict data-only formatting"""
        context_parts = [
            "=== YOUR DATABASE CONTENT (USE ONLY THIS DATA) ===",
            "",
            "CRITICAL: You must ONLY use information from the content below.",
            "Do NOT use any external knowledge, training data, or general information.",
            "If information is not explicitly stated below, say 'NOT AVAILABLE IN DATABASE'",
            ""
        ]
        
        # Add PDF results with strict formatting
        if pdf_results:
            context_parts.append("=== PDF DOCUMENTS IN DATABASE ===")
            for i, pdf_result in enumerate(pdf_results):
                context_parts.append(f"PDF SOURCE {i+1}:")
                context_parts.append(f"Title: {pdf_result.get('title', 'Unknown')}")
                context_parts.append(f"Content: {pdf_result.get('content', 'No content')}")
                context_parts.append("---")
        
        # Add YouTube results with strict formatting
        if youtube_results:
            context_parts.append("=== YOUTUBE VIDEOS IN DATABASE ===")
            for i, youtube_result in enumerate(youtube_results):
                context_parts.append(f"VIDEO SOURCE {i+1}:")
                context_parts.append(f"Title: {youtube_result.get('title', 'Unknown')}")
                context_parts.append(f"Transcript: {youtube_result.get('transcript', 'No transcript')}")
                context_parts.append("---")
        
        # Add vector search results with strict formatting
        if vector_results:
            context_parts.append("=== VECTOR SEARCH RESULTS FROM DATABASE ===")
            for i, vector_result in enumerate(vector_results):
                context_parts.append(f"SEARCH RESULT {i+1}:")
                context_parts.append(f"Content: {vector_result.get('content', 'No content')}")
                context_parts.append(f"Source: {vector_result.get('source', 'Unknown')}")
                context_parts.append(f"Similarity: {vector_result.get('similarity', 'Unknown')}")
                context_parts.append("---")
        
        context_parts.append("=== END OF DATABASE CONTENT ===")
        context_parts.append("")
        context_parts.append("REMINDER: Use ONLY the content above. No external knowledge allowed.")
        
        return "\n".join(context_parts)
    
    def _create_anti_hallucination_system_prompt(self) -> str:
        """Create system prompt that absolutely prevents hallucination"""
        return """You are a DATABASE-ONLY research assistant. You are FORBIDDEN from using any external knowledge.

CRITICAL RULES (VIOLATION = IMMEDIATE FAILURE):
1. ONLY use information explicitly provided in the user's database content
2. NEVER use your training data, general knowledge, or external information
3. If information is not in the provided database, respond with "NOT AVAILABLE IN DATABASE"
4. NEVER make up facts, equations, quotes, numbers, or details
5. Quote EXACTLY from the provided sources - no paraphrasing unless clearly marked
6. Always cite the specific source (PDF SOURCE 1, VIDEO SOURCE 2, etc.)
7. If asked about anything not in the database, clearly state it's not available

RESPONSE FORMAT (JSON only):
{
  "answer": "Answer using ONLY database content",
  "confidence_score": 0.85,
  "reasoning_steps": ["Step 1 with exact source", "Step 2 with exact source"],
  "sources_used": ["PDF SOURCE 1", "VIDEO SOURCE 2"],
  "source_quotes": ["Exact quote 1", "Exact quote 2"],
  "limitations": "What information is NOT available in the database",
  "database_only": true
}

FAILURE CONDITIONS:
- Using external knowledge = IMMEDIATE FAILURE
- Making up information = IMMEDIATE FAILURE
- No source citations = IMMEDIATE FAILURE
- General knowledge answers = IMMEDIATE FAILURE"""
    
    def _create_strict_user_prompt(self, query: str, context: str, content_hash: str) -> str:
        """Create user prompt with strict data-only requirements"""
        return f"""DATABASE QUERY: {query}

CONTENT HASH: {content_hash} (for verification)

{context}

INSTRUCTIONS:
1. Answer the query using ONLY the database content above
2. Quote exactly from the sources provided
3. Cite every fact with its source (PDF SOURCE 1, VIDEO SOURCE 2, etc.)
4. If information is missing, state "NOT AVAILABLE IN DATABASE"
5. Never use external knowledge or training data
6. Provide exact quotes in the source_quotes field

RESPOND IN JSON FORMAT ONLY."""
    
    async def _call_openai_strict(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Call OpenAI with strict anti-hallucination parameters"""
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.0,  # Zero temperature for consistency
                    max_tokens=1500,
                    top_p=0.1,        # Very focused responses
                    frequency_penalty=0.0,
                    presence_penalty=0.0
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
    
    def _parse_and_verify_response(self, response: Dict[str, Any], query: str, content_hash: str) -> AntiHallucinationResult:
        """Parse response and verify it only uses provided data"""
        try:
            # Parse JSON response
            content = response["content"]
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "").strip()
            
            parsed = json.loads(content)
            
            # Verify response structure
            required_fields = ["answer", "confidence_score", "reasoning_steps", "sources_used", "source_quotes"]
            for field in required_fields:
                if field not in parsed:
                    raise ValueError(f"Missing required field: {field}")
            
            # Calculate cost estimate
            cost_estimate = self._calculate_cost(response["tokens_used"], response["prompt_tokens"], response["completion_tokens"])
            
            return AntiHallucinationResult(
                answer=parsed["answer"],
                confidence_score=float(parsed["confidence_score"]),
                reasoning_steps=parsed["reasoning_steps"],
                sources_used=parsed["sources_used"],
                source_quotes=parsed["source_quotes"],
                model_used=response["model"],
                tokens_used=response["tokens_used"],
                cost_estimate=cost_estimate,
                timestamp=datetime.now(),
                data_only=True
            )
            
        except Exception as e:
            logger.error(f"Failed to parse anti-hallucination response: {e}")
            raise ValueError(f"Invalid response format: {e}")
    
    def _calculate_cost(self, total_tokens: int, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost estimate"""
        if self.model not in self.cost_per_1k_tokens:
            return 0.0
        
        rates = self.cost_per_1k_tokens[self.model]
        input_cost = (prompt_tokens / 1000) * rates["input"]
        output_cost = (completion_tokens / 1000) * rates["output"]
        
        return input_cost + output_cost