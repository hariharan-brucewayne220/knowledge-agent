"""
Test Multiple Queries to Verify Source Provenance
"""

import json
import requests
import time

def test_query(query, test_name):
    """Test a single query"""
    
    server_url = "http://localhost:8080"
    
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print(f"Query: {query}")
    print(f"{'='*60}")
    
    try:
        payload = {"query": query}
        response = requests.post(
            f"{server_url}/api/research", 
            json=payload, 
            timeout=60
        )
        
        if response.status_code != 200:
            print(f"API Error: {response.status_code}")
            return
        
        result = response.json()
        
        if not result.get('success'):
            print(f"Query failed")
            return
        
        # Extract information
        final_answer = result['result']['final_answer']
        answer_summary = final_answer.get('answer_summary', '')
        source_provenance = final_answer.get('source_provenance', [])
        sources_processed = final_answer.get('sources_processed', {})
        
        print(f"\nSTATS:")
        print(f"  PDFs Analyzed: {sources_processed.get('pdf_documents', 0)}")
        print(f"  Source Citations: {len(source_provenance)}")
        
        print(f"\nSOURCE CITATIONS:")
        for i, citation in enumerate(source_provenance):
            source_id = citation.get('source_id', 'Unknown')
            print(f"  [{i+1}] {source_id}")
            excerpt = citation.get('content_excerpt', '')[:100]
            if excerpt:
                print(f"      Quote: \"{excerpt}...\"")
        
        print(f"\nANSWER:")
        print(answer_summary[:300] + "..." if len(answer_summary) > 300 else answer_summary)
        
        # Quick analysis
        real_source_count = sum(1 for c in source_provenance 
                              if any(real in c.get('source_id', '') for real in 
                                   ['Solar Panel Efficiency', 'Carbon Sequestration']))
        
        if sources_processed.get('pdf_documents', 0) > 0 and real_source_count > 0:
            print(f"\nRESULT: SUCCESS - {real_source_count} real citations")
        else:
            print(f"\nRESULT: ISSUE - Only {real_source_count} real citations")
        
    except Exception as e:
        print(f"Test failed: {e}")

def run_multiple_tests():
    """Run several different test queries"""
    
    print("COMPREHENSIVE SOURCE PROVENANCE TEST")
    
    test_cases = [
        ("Solar Efficiency", "What are the current efficiency rates of silicon solar cells?"),
        ("Carbon Capture Energy", "How much energy does Direct Air Capture require per ton of CO2?"),
        ("Cross-Document", "Could solar panels provide enough energy for carbon sequestration based on the documents?"),
        ("Technical Details", "What is the theoretical limit efficiency for silicon cells and what are DAC energy requirements?"),
        ("Grid Integration", "What challenges exist for integrating solar panels into the electrical grid?")
    ]
    
    for test_name, query in test_cases:
        test_query(query, test_name)
        time.sleep(2)  # Brief pause
    
    print(f"\n{'='*60}")
    print("TEST COMPLETE")

if __name__ == "__main__":
    run_multiple_tests()