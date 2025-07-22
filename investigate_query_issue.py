#!/usr/bin/env python3
"""
Investigate the quantum physics/dark matter query issue
"""

import sys
sys.path.append('src')

from storage.unified_content_store import UnifiedContentStore
from storage.enhanced_ner_fuzzy_router import EnhancedNERFuzzyRouter

def investigate_quantum_dark_matter_query():
    """Investigate what went wrong with the quantum physics/dark matter query"""
    
    print("=== INVESTIGATING QUANTUM PHYSICS/DARK MATTER QUERY ===\n")
    
    content_store = UnifiedContentStore()
    enhanced_router = EnhancedNERFuzzyRouter(content_store)
    
    # The query that was problematic
    query = "what does the sources say about quantum physics how is it related to dark matter"
    
    print(f"Query: '{query}'")
    
    # Test routing
    try:
        pdfs, videos, explanation = enhanced_router.route_query(query)
        print(f"\nRouting Results:")
        print(f"PDFs found: {len(pdfs)}")
        print(f"Videos found: {len(videos)}")
        print(f"Explanation: {explanation}")
        
        # Show what was actually routed
        print(f"\nActual content routed:")
        if pdfs:
            print(f"PDFs:")
            for pdf in pdfs:
                print(f"  - {pdf}")
        if videos:
            print(f"Videos:")
            for video in videos:
                print(f"  - {video}")
        
    except Exception as e:
        print(f"ERROR in routing: {e}")
    
    # Now let's check what content we actually have about quantum physics and dark matter
    print(f"\n=== CONTENT ANALYSIS ===")
    
    all_content = content_store.get_all_content()
    
    print(f"Checking all {len(all_content)} items for quantum/dark matter content:")
    
    for item in all_content:
        content_text = item.full_text.lower()
        
        # Check for quantum physics content
        quantum_terms = ['quantum', 'physics', 'theory', 'energy', 'principle']
        quantum_count = sum(content_text.count(term) for term in quantum_terms)
        
        # Check for dark matter content  
        dark_matter_terms = ['dark', 'matter', 'galactic', 'halos', 'gravitational']
        dark_matter_count = sum(content_text.count(term) for term in dark_matter_terms)
        
        if quantum_count > 5 or dark_matter_count > 5:
            print(f"\n{item.title} ({item.content_type}):")
            print(f"  Quantum-related terms: {quantum_count}")
            print(f"  Dark matter terms: {dark_matter_count}")
            
            # Show specific content excerpts
            if 'quantum' in content_text:
                quantum_pos = content_text.find('quantum')
                if quantum_pos >= 0:
                    context = item.full_text[max(0, quantum_pos-100):quantum_pos+200]
                    print(f"  Quantum context: '...{context}...'")
            
            if 'dark matter' in content_text:
                dm_pos = content_text.find('dark matter')
                if dm_pos >= 0:
                    context = item.full_text[max(0, dm_pos-100):dm_pos+200]
                    print(f"  Dark matter context: '...{context}...'")

def test_specific_queries():
    """Test specific queries that should work"""
    
    print(f"\n=== TESTING SPECIFIC QUERIES ===")
    
    content_store = UnifiedContentStore()
    enhanced_router = EnhancedNERFuzzyRouter(content_store)
    
    test_queries = [
        "dark matter",
        "quantum physics", 
        "quantum theory",
        "dark matter and quantum",
        "physics theory",
        "gravitational effects",
        "quantum mechanics"
    ]
    
    for query in test_queries:
        try:
            pdfs, videos, explanation = enhanced_router.route_query(query)
            print(f"'{query}' -> {len(pdfs)} PDFs, {len(videos)} videos")
            print(f"  {explanation}")
            
            # Check if we get the right content
            if pdfs or videos:
                for item in content_store.get_all_content():
                    if ((item.content_type == 'pdf' and item.source_path in pdfs) or 
                        (item.content_type == 'youtube' and item.source_path in videos)):
                        
                        content = item.full_text.lower()
                        query_words = query.lower().split()
                        matches = sum(1 for word in query_words if len(word) > 3 and word in content)
                        print(f"    {item.title}: {matches}/{len([w for w in query_words if len(w) > 3])} query terms found")
            print()
            
        except Exception as e:
            print(f"ERROR with '{query}': {e}")

def check_actual_quantum_dark_matter_content():
    """Check the actual content about quantum physics and dark matter"""
    
    print(f"\n=== ACTUAL QUANTUM/DARK MATTER CONTENT ===")
    
    content_store = UnifiedContentStore()
    all_content = content_store.get_all_content()
    
    # Find the dark matter PDF
    dark_matter_pdf = None
    quantum_video = None
    
    for item in all_content:
        if 'dark matter' in item.title.lower():
            dark_matter_pdf = item
        elif item.content_type == 'youtube' and 'quantum' in item.full_text.lower():
            if quantum_video is None or item.full_text.lower().count('quantum') > quantum_video.full_text.lower().count('quantum'):
                quantum_video = item
    
    if dark_matter_pdf:
        print(f"DARK MATTER PDF: {dark_matter_pdf.title}")
        content = dark_matter_pdf.full_text
        
        # Look for quantum mentions in dark matter PDF
        quantum_mentions = content.lower().count('quantum')
        print(f"Quantum mentions in dark matter PDF: {quantum_mentions}")
        
        if quantum_mentions > 0:
            quantum_pos = content.lower().find('quantum')
            context = content[max(0, quantum_pos-200):quantum_pos+300]
            print(f"Quantum context in dark matter PDF:")
            print(f"'{context}'")
        
        # Show key dark matter concepts
        print(f"\nKey dark matter concepts in PDF:")
        dm_context = content[:1000]
        print(f"'{dm_context}...'")
    
    if quantum_video:
        print(f"\nQUANTUM VIDEO: {quantum_video.title}")
        content = quantum_video.full_text
        
        quantum_mentions = content.lower().count('quantum')
        physics_mentions = content.lower().count('physics')
        print(f"Quantum mentions: {quantum_mentions}")
        print(f"Physics mentions: {physics_mentions}")
        
        if quantum_mentions > 0:
            quantum_pos = content.lower().find('quantum')
            context = content[max(0, quantum_pos-200):quantum_pos+300]
            print(f"Quantum context in video:")
            print(f"'{context}'")

def main():
    print("INVESTIGATING QUANTUM PHYSICS/DARK MATTER QUERY ISSUE")
    print("="*60)
    
    # 1. Investigate the specific query
    investigate_quantum_dark_matter_query()
    
    # 2. Test related queries
    test_specific_queries()
    
    # 3. Check actual content
    check_actual_quantum_dark_matter_content()
    
    print("\n" + "="*60)
    print("INVESTIGATION COMPLETE")

if __name__ == "__main__":
    main()