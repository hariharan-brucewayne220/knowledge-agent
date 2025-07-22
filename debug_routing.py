"""
Debug Smart Router Logic
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'src'))

def debug_smart_router():
    """Debug the smart router step by step"""
    
    try:
        from storage.unified_content_store import UnifiedContentStore
        from storage.simple_smart_router import SimpleSmartRouter
        
        content_store = UnifiedContentStore()
        router = SimpleSmartRouter(content_store)
        
        print("SMART ROUTER DEBUG")
        print("="*50)
        
        # Test query
        test_query = "Could solar panels provide enough energy for carbon sequestration projects?"
        print(f"Test Query: {test_query}")
        
        # Check content store first
        all_content = content_store.get_all_content()
        print(f"\nContent Store has {len(all_content)} items:")
        for item in all_content:
            print(f"  - {item.title} ({item.content_type})")
            keywords = item.keywords[:5]  # First 5 keywords
            print(f"    Keywords: {keywords}")
            chunks_preview = item.full_text[:100] + "..." if len(item.full_text) > 100 else item.full_text
            print(f"    Content: {chunks_preview}")
            print()
        
        # Test multiple topics detection
        print(f"Testing multiple topics detection:")
        has_multiple = router._has_multiple_topics(test_query)
        print(f"  Has multiple topics: {has_multiple}")
        
        if has_multiple:
            print(f"  -> Should route to _search_multiple_topics")
        else:
            print(f"  -> Will check for mentioned file")
            
        # Test mentioned file detection
        mentioned_file = router._find_mentioned_file(test_query)
        if mentioned_file:
            print(f"  Found mentioned file: {mentioned_file.title}")
        else:
            print(f"  No specific file mentioned -> will search all content")
        
        # Test full routing
        print(f"\nFull routing test:")
        pdfs, videos, explanation = router.route_query(test_query)
        print(f"  PDFs found: {len(pdfs)} - {pdfs}")
        print(f"  Videos found: {len(videos)} - {videos}")
        print(f"  Explanation: {explanation}")
        
        # Test with carbon-specific query
        carbon_query = "What are DAC energy requirements per ton CO2?"
        print(f"\nTesting Carbon-specific query: {carbon_query}")
        pdfs2, videos2, explanation2 = router.route_query(carbon_query)
        print(f"  PDFs found: {len(pdfs2)} - {pdfs2}")
        print(f"  Explanation: {explanation2}")
        
    except Exception as e:
        print(f"Debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_smart_router()