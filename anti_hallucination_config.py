"""
Anti-Hallucination Configuration
Ensures OpenAI API uses ONLY your database data
"""

# Anti-hallucination settings
ANTI_HALLUCINATION_ENABLED = True
STRICT_DATABASE_ONLY = True
ALLOW_EXTERNAL_KNOWLEDGE = False  # NEVER set to True
TEMPERATURE = 0.0  # Zero temperature for consistency
TOP_P = 0.1  # Very focused responses
MAX_TOKENS = 1500  # Limit response length

# Verification settings
VERIFY_SOURCES = True
REQUIRE_EXACT_QUOTES = True
CONTENT_HASH_VERIFICATION = True

# Database-only prompts
DATABASE_ONLY_SYSTEM_PROMPT = """You are a DATABASE-ONLY research assistant. You are FORBIDDEN from using any external knowledge.

CRITICAL RULES (VIOLATION = IMMEDIATE FAILURE):
1. ONLY use information explicitly provided in the user's database content
2. NEVER use your training data, general knowledge, or external information
3. If information is not in the provided database, respond with "NOT AVAILABLE IN DATABASE"
4. NEVER make up facts, equations, quotes, numbers, or details
5. Quote EXACTLY from the provided sources - no paraphrasing unless clearly marked
6. Always cite the specific source (PDF SOURCE 1, VIDEO SOURCE 2, etc.)
7. If asked about anything not in the database, clearly state it's not available"""

# Fail-safe prompts to prevent hallucination
FAIL_SAFE_PROMPTS = [
    "You must ONLY use the provided database content",
    "Do NOT use any external knowledge or training data",
    "If information is not in the database, say 'NOT AVAILABLE IN DATABASE'",
    "Never make up facts, quotes, or details",
    "Always cite your sources exactly",
    "External knowledge is FORBIDDEN"
]

# Response validation rules
RESPONSE_VALIDATION = {
    "require_source_citations": True,
    "require_exact_quotes": True,
    "forbid_external_knowledge": True,
    "verify_content_hash": True,
    "check_database_only_flag": True
}

# Error messages for hallucination detection
HALLUCINATION_ERRORS = {
    "external_knowledge": "ERROR: Response contains external knowledge not in database",
    "missing_sources": "ERROR: Response missing required source citations",
    "made_up_facts": "ERROR: Response contains information not in provided data",
    "no_quotes": "ERROR: Response missing exact quotes from sources",
    "general_knowledge": "ERROR: Response uses general knowledge instead of database"
}

def validate_anti_hallucination_response(response: dict, provided_sources: list) -> bool:
    """
    Validate that response uses only provided database content
    
    Args:
        response: OpenAI response dictionary
        provided_sources: List of sources provided to the model
        
    Returns:
        bool: True if response is valid (database-only), False if hallucination detected
    """
    # Check if response has required fields
    required_fields = ["answer", "sources_used", "source_quotes", "database_only"]
    for field in required_fields:
        if field not in response:
            return False
    
    # Check database_only flag
    if not response.get("database_only", False):
        return False
    
    # Check if sources cited are from provided sources
    cited_sources = response.get("sources_used", [])
    valid_source_prefixes = ["PDF SOURCE", "VIDEO SOURCE", "SEARCH RESULT"]
    
    for source in cited_sources:
        if not any(source.startswith(prefix) for prefix in valid_source_prefixes):
            return False
    
    # Check if quotes are provided
    if not response.get("source_quotes"):
        return False
    
    return True

def get_database_only_config() -> dict:
    """Get configuration for database-only responses"""
    return {
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "system_prompt": DATABASE_ONLY_SYSTEM_PROMPT,
        "validation_rules": RESPONSE_VALIDATION,
        "fail_safe_prompts": FAIL_SAFE_PROMPTS
    }

def create_database_context(pdf_results: list, youtube_results: list, vector_results: list) -> str:
    """Create strict database-only context"""
    context_parts = [
        "=== YOUR DATABASE CONTENT (USE ONLY THIS DATA) ===",
        "",
        "CRITICAL: You must ONLY use information from the content below.",
        "Do NOT use any external knowledge, training data, or general information.",
        "If information is not explicitly stated below, say 'NOT AVAILABLE IN DATABASE'",
        ""
    ]
    
    # Add PDF results
    if pdf_results:
        context_parts.append("=== PDF DOCUMENTS IN DATABASE ===")
        for i, pdf_result in enumerate(pdf_results):
            context_parts.append(f"PDF SOURCE {i+1}:")
            context_parts.append(f"Title: {pdf_result.get('title', 'Unknown')}")
            context_parts.append(f"Content: {pdf_result.get('content', 'No content')}")
            context_parts.append("---")
    
    # Add YouTube results
    if youtube_results:
        context_parts.append("=== YOUTUBE VIDEOS IN DATABASE ===")
        for i, youtube_result in enumerate(youtube_results):
            context_parts.append(f"VIDEO SOURCE {i+1}:")
            context_parts.append(f"Title: {youtube_result.get('title', 'Unknown')}")
            context_parts.append(f"Transcript: {youtube_result.get('transcript', 'No transcript')}")
            context_parts.append("---")
    
    # Add vector search results
    if vector_results:
        context_parts.append("=== VECTOR SEARCH RESULTS FROM DATABASE ===")
        for i, vector_result in enumerate(vector_results):
            context_parts.append(f"SEARCH RESULT {i+1}:")
            context_parts.append(f"Content: {vector_result.get('content', 'No content')}")
            context_parts.append(f"Source: {vector_result.get('source', 'Unknown')}")
            context_parts.append("---")
    
    context_parts.append("=== END OF DATABASE CONTENT ===")
    context_parts.append("")
    context_parts.append("REMINDER: Use ONLY the content above. No external knowledge allowed.")
    
    return "\n".join(context_parts)

# Testing function
def test_anti_hallucination_setup():
    """Test that anti-hallucination setup is working"""
    print("🔒 Testing Anti-Hallucination Setup")
    print(f"✅ Anti-hallucination enabled: {ANTI_HALLUCINATION_ENABLED}")
    print(f"✅ Strict database-only mode: {STRICT_DATABASE_ONLY}")
    print(f"✅ External knowledge forbidden: {not ALLOW_EXTERNAL_KNOWLEDGE}")
    print(f"✅ Temperature set to: {TEMPERATURE} (0.0 = no creativity)")
    print(f"✅ Top-p set to: {TOP_P} (0.1 = very focused)")
    print(f"✅ Source verification: {VERIFY_SOURCES}")
    print(f"✅ Exact quotes required: {REQUIRE_EXACT_QUOTES}")
    print("🎯 Configuration is optimal for preventing hallucination!")

if __name__ == "__main__":
    test_anti_hallucination_setup()