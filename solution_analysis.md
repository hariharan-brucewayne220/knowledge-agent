# Smart Routing Scalability Solutions

## Problem Analysis
- With 5+ PDFs, routing becomes too broad (all queries return all documents)
- Synthesizer overwhelmed with 2000+ chars from multiple documents
- Specific queries get diluted in multi-document responses
- Battery query now returns all 5 PDFs instead of just battery PDF

## Recommended Solutions (Priority Order)

### 1. ENHANCED TITLE MATCHING (Quick Fix - 30 mins)
**Problem**: Current title matching falls back to broad search too easily
**Solution**: Improve scoring algorithm to be more confident in best matches

```python
# In simple_smart_router.py _find_mentioned_file()
# Lower the threshold for "good match" and add document-specific keywords
if best_match['score'] > 3:  # Lower from 6 to 3
    return best_match['item']

# Add domain-specific keyword boosting
domain_keywords = {
    'battery': ['energy', 'density', 'lithium', 'storage', 'wh/kg'],
    'solar': ['efficiency', 'silicon', 'panel', 'photovoltaic'],
    'carbon': ['sequestration', 'capture', 'dac', 'co2'],
    'dark': ['matter', 'galactic', 'quantum', 'physics']
}
```

### 2. SEMANTIC SIMILARITY THRESHOLD (Medium Fix - 1 hour)
**Problem**: Semantic similarity is too permissive (0.3 threshold)
**Solution**: Implement adaptive thresholds based on document count

```python
# Dynamic threshold based on collection size
if len(source_files) <= 3:
    semantic_threshold = 0.25
elif len(source_files) <= 5:
    semantic_threshold = 0.4  # Higher threshold for larger collections
else:
    semantic_threshold = 0.5  # Even higher for 6+ documents
```

### 3. QUERY INTENT CLASSIFICATION (Advanced Fix - 2 hours)
**Problem**: System doesn't distinguish between specific vs comparative queries
**Solution**: Add query intent detection

```python
def classify_query_intent(query):
    specific_indicators = ['what is', 'according to', 'in the document', 'from the']
    comparative_indicators = ['compare', 'difference', 'versus', 'both', 'all']
    
    if any(indicator in query.lower() for indicator in specific_indicators):
        return 'specific'  # Should return 1-2 documents
    elif any(indicator in query.lower() for indicator in comparative_indicators):
        return 'comparative'  # Can return multiple documents
    else:
        return 'general'  # Default behavior
```

### 4. CONTENT PRIORITIZATION IN SYNTHESIS (Medium Fix - 1 hour)
**Problem**: Fallback synthesizer truncates randomly instead of prioritizing relevant content
**Solution**: Implement relevance-based content ordering

```python
def prioritize_content_by_relevance(query, all_content):
    scored_content = []
    query_words = set(query.lower().split())
    
    for content in all_content:
        # Score based on query term frequency
        content_words = set(content.lower().split())
        overlap = len(query_words.intersection(content_words))
        score = overlap / len(query_words) if query_words else 0
        
        scored_content.append((score, content))
    
    # Sort by relevance and return top content first
    scored_content.sort(reverse=True)
    return [content for score, content in scored_content]
```

## Recommended Implementation Plan

### PHASE 1: Quick Wins (30 minutes)
1. **Lower title matching threshold** from 6 to 3
2. **Add domain-specific keyword boosting** for battery/solar/carbon/dark matter
3. **Test with existing queries** to ensure they route to specific documents

### PHASE 2: Smart Thresholds (1 hour) 
1. **Implement adaptive semantic thresholds** based on document count
2. **Add query intent detection** for specific vs comparative queries
3. **Route specific queries to max 2 documents**, comparative to max 4

### PHASE 3: Enhanced Synthesis (1 hour)
1. **Implement content prioritization** in fallback synthesizer
2. **Increase content limit** from 2000 to 4000 chars but with smart ordering
3. **Add query-specific content extraction** instead of just showing all content

## Expected Results After Implementation
- ✅ "Battery energy density" → Battery PDF only (not all 5)
- ✅ "What is dark matter" → Dark matter PDF primarily  
- ✅ "Compare solar and battery" → Solar + Battery PDFs only
- ✅ Faster responses with more focused content
- ✅ System scales to 10+ documents without quality loss

## Alternative: Document Tags/Categories
If routing improvements aren't sufficient, we could add explicit document categorization:
```python
document_categories = {
    'energy_storage': ['Advanced Battery Technologies'],
    'renewable_energy': ['Solar Panel Efficiency'],
    'carbon_tech': ['Carbon Sequestration', 'Atmospheric Carbon Removal'],
    'physics': ['Dark Matter and Galactic Halos']
}
```

## Recommendation
**Start with Phase 1** (quick wins) to immediately improve routing specificity, then assess if Phase 2 is needed based on results.