#!/usr/bin/env python3
"""
Test NER-based fuzzy routing integration in the KnowledgeAgent app
"""

import sys
sys.path.append('src')

from storage.unified_content_store import UnifiedContentStore
from storage.ner_fuzzy_router import NERFuzzyRouter

def test_ner_routing():
    print("=== Testing NER-based Fuzzy Routing ===\n")
    
    # Initialize content store
    content_store = UnifiedContentStore()
    print(f"Content store loaded with {len(content_store.get_all_content())} documents\n")
    
    # Initialize NER fuzzy router
    ner_router = NERFuzzyRouter(content_store)
    
    # Test queries that user mentioned
    test_queries = [
        "battery energy density",
        "solar panel efficiency", 
        "carbon sequestration technologies",
        "dark matter physics",
        "compare solar and battery storage",
        "how much energy can batteries store",
        "atmospheric carbon removal methods"
    ]
    
    for query in test_queries:
        print(f"Query: '{query}'")
        print("-" * 50)
        
        try:
            # Route the query using NER fuzzy router
            relevant_pdfs, relevant_videos, explanation = ner_router.route_query(query)
            
            print(f"PDFs found: {len(relevant_pdfs)}")
            for pdf in relevant_pdfs:
                filename = pdf.split('/')[-1] if '/' in pdf else pdf.split('\\')[-1] 
                print(f"  - {filename}")
            
            print(f"Videos found: {len(relevant_videos)}")
            for video in relevant_videos:
                print(f"  - {video}")
                
            print(f"Routing explanation: {explanation}")
            
            # Show debug info for interesting cases
            if len(relevant_pdfs) > 1:
                debug_info = ner_router.get_routing_debug_info(query)
                print(f"Query terms extracted: {debug_info['extracted_terms']}")
                print("Top document scores:")
                sorted_scores = sorted(debug_info['document_scores'].items(), 
                                     key=lambda x: x[1]['total_score'], reverse=True)
                for doc_title, score_info in sorted_scores[:3]:
                    if score_info['total_score'] > 0:
                        print(f"  {doc_title}: {score_info['total_score']:.1f}")
                        
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            
        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    test_ner_routing()