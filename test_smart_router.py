"""
Test Smart Router with Unified Content Store
"""

import sys
from pathlib import Path

# Add src to path  
sys.path.append(str(Path(__file__).parent / 'src'))

def test_smart_router():
    """Test smart router with unified content"""
    
    try:
        from storage.unified_content_store import UnifiedContentStore
        from storage.simple_smart_router import SimpleSmartRouter
        
        # Initialize stores
        content_store = UnifiedContentStore()
        smart_router = SimpleSmartRouter(content_store)
        
        print("Initialized UnifiedContentStore and SmartRouter")
        print(f"Content items loaded: {len(content_store.get_all_content())}")
        
        # Test smart routing
        test_query = "What are the efficiency rates of silicon solar cells?"
        print(f"\\nTesting query: '{test_query}'")
        
        relevant_pdfs, relevant_videos, explanation = smart_router.route_query(test_query)
        
        print(f"Smart routing results:")
        print(f"  - Found PDFs: {len(relevant_pdfs)}")
        print(f"  - Found Videos: {len(relevant_videos)}")  
        print(f"  - Explanation: {explanation}")
        
        if relevant_pdfs:
            print(f"  - PDF files: {relevant_pdfs}")
            
        return True
        
    except Exception as e:
        print(f"Error testing smart router: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_smart_router()