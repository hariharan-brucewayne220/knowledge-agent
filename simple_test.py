"""
Simple Source Provenance Test - No Emojis
"""

import json
import requests
import time

def test_single_query():
    """Test a single query and show detailed results"""
    
    server_url = "http://localhost:8080"
    
    # Test server
    try:
        response = requests.get(f"{server_url}/api/status", timeout=5)
        if response.status_code == 200:
            print("Server is running")
        else:
            print(f"Server error: {response.status_code}")
            return
    except Exception as e:
        print(f"Cannot connect to server: {e}")
        return
    
    # Test query
    query = "What is the efficiency of silicon solar cells mentioned in the documents?"
    print(f"\nQuery: {query}")
    
    try:
        payload = {"query": query}
        response = requests.post(
            f"{server_url}/api/research", 
            json=payload, 
            timeout=60
        )
        
        if response.status_code != 200:
            print(f"API Error: {response.status_code}")
            print(f"Response: {response.text[:500]}")
            return
        
        result = response.json()
        
        if not result.get('success'):
            print(f"Query failed: {result}")
            return
        
        # Extract information
        final_answer = result['result']['final_answer']
        answer_summary = final_answer.get('answer_summary', '')
        source_provenance = final_answer.get('source_provenance', [])
        sources_processed = final_answer.get('sources_processed', {})
        
        print(f"\nRESULTS:")
        print(f"PDFs Analyzed: {sources_processed.get('pdf_documents', 0)}")
        print(f"Videos Analyzed: {sources_processed.get('youtube_videos', 0)}")
        print(f"Source Citations: {len(source_provenance)}")
        print(f"Answer Length: {len(answer_summary)} chars")
        
        print(f"\nANSWER SUMMARY:")
        print(answer_summary[:500] + "..." if len(answer_summary) > 500 else answer_summary)
        
        print(f"\nSOURCE CITATIONS:")
        for i, citation in enumerate(source_provenance):
            print(f"  [{i+1}] Source: {citation.get('source_id', 'Unknown')}")
            print(f"      Type: {citation.get('source_type', 'unknown')}")
            print(f"      Reference: {citation.get('specific_reference', 'None')}")
            excerpt = citation.get('content_excerpt', '')
            if excerpt:
                print(f"      Excerpt: {excerpt[:150]}...")
            print()
        
        # Analysis
        print("ANALYSIS:")
        if sources_processed.get('pdf_documents', 0) > 0:
            print("GOOD: PDFs were actually analyzed")
        else:
            print("BAD: No PDFs analyzed - content discovery failed")
        
        if len(source_provenance) > 0:
            print("GOOD: Source citations exist")
        else:
            print("BAD: No source citations")
        
        real_sources = ['Solar Panel Efficiency', 'Carbon Sequestration']
        real_citations = 0
        for citation in source_provenance:
            source_id = citation.get('source_id', '')
            if any(real_source in source_id for real_source in real_sources):
                real_citations += 1
        
        if real_citations > 0:
            print(f"GOOD: Found {real_citations} real citations")
        else:
            print("BAD: No real citations - likely hallucinated")
            
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_single_query()