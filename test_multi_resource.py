"""
Test Multi-Resource Detection and Routing
"""

import json
import requests
import time

def test_multi_resource_queries():
    """Test queries that should use multiple documents"""
    
    server_url = "http://localhost:8080"
    
    print("MULTI-RESOURCE ROUTING TEST")
    print("="*60)
    
    # Test cases that should find BOTH PDFs
    test_cases = [
        {
            'name': "Cross-Domain Energy Question",
            'query': "Could solar panels provide enough energy for carbon sequestration projects?",
            'expected_pdfs': 2,  # Should find both Solar + Carbon PDFs
            'reason': "Contains 'provide energy' + 'solar' + 'carbon sequestration'"
        },
        {
            'name': "Technical Cross-Reference", 
            'query': "What is the theoretical efficiency limit for silicon cells and what are DAC energy requirements per ton CO2?",
            'expected_pdfs': 2,
            'reason': "Contains 'efficiency' + 'dac' + technical terms from both domains"
        },
        {
            'name': "Carbon Payback Analysis",
            'query': "What's the carbon payback time when manufacturing solar panels for powering enhanced weathering operations?",
            'expected_pdfs': 2,
            'reason': "Contains 'payback' + 'solar' + 'weathering' (sequestration method)"
        },
        {
            'name': "Energy Balance Question",
            'query': "How much kWh do solar panels generate versus how much kWh does DAC require per ton?",
            'expected_pdfs': 2,
            'reason': "Contains 'kwh' + 'solar' + 'dac' cross-reference"
        },
        {
            'name': "Single Domain (Control)",
            'query': "What is the efficiency of silicon solar cells?",
            'expected_pdfs': 1,
            'reason': "Only solar-specific, should find just Solar PDF"
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        print(f"\nTEST: {test_case['name']}")
        print(f"Query: {test_case['query']}")
        print(f"Expected PDFs: {test_case['expected_pdfs']} ({test_case['reason']})")
        
        try:
            payload = {"query": test_case['query']}
            response = requests.post(f"{server_url}/api/research", json=payload, timeout=60)
            
            if response.status_code != 200:
                print(f"ERROR: {response.status_code}")
                continue
            
            result = response.json()
            if not result.get('success'):
                print("ERROR: Query failed")
                continue
            
            # Extract key metrics
            final_answer = result['result']['final_answer']
            sources_processed = final_answer.get('sources_processed', {})
            source_provenance = final_answer.get('source_provenance', [])
            
            pdfs_analyzed = sources_processed.get('pdf_documents', 0)
            citations_count = len(source_provenance)
            
            # Check if it found the expected number of PDFs
            success = pdfs_analyzed == test_case['expected_pdfs']
            
            print(f"RESULT:")
            print(f"  PDFs Analyzed: {pdfs_analyzed} (expected: {test_case['expected_pdfs']})")
            print(f"  Citations: {citations_count}")
            print(f"  Status: {'SUCCESS' if success else 'FAILED'}")
            
            # Show which sources were cited
            cited_sources = []
            for citation in source_provenance:
                source_name = citation.get('source_id', 'Unknown')
                if 'Solar' in source_name:
                    cited_sources.append('Solar PDF')
                elif 'Carbon' in source_name:
                    cited_sources.append('Carbon PDF')
                else:
                    cited_sources.append('Other')
            
            print(f"  Sources Cited: {', '.join(cited_sources) if cited_sources else 'None'}")
            
            results.append({
                'name': test_case['name'],
                'expected': test_case['expected_pdfs'],
                'actual': pdfs_analyzed,
                'success': success,
                'citations': citations_count
            })
            
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                'name': test_case['name'],
                'expected': test_case['expected_pdfs'],
                'actual': 0,
                'success': False,
                'citations': 0
            })
        
        time.sleep(2)
    
    # Summary
    print(f"\n{'='*60}")
    print("MULTI-RESOURCE TEST SUMMARY")
    print(f"{'='*60}")
    
    successful_tests = sum(1 for r in results if r['success'])
    total_tests = len(results)
    
    print(f"Overall: {successful_tests}/{total_tests} tests passed")
    
    for result in results:
        status = "PASS" if result['success'] else "FAIL"
        print(f"{status} - {result['name']}: {result['actual']}/{result['expected']} PDFs, {result['citations']} citations")
    
    if successful_tests >= 4:  # Allow 1 test to fail
        print(f"\nSUCCESS: Multi-resource routing is working well!")
        return True
    else:
        print(f"\nNEEDS WORK: Multi-resource routing needs improvement")
        return False

if __name__ == "__main__":
    success = test_multi_resource_queries()
    exit(0 if success else 1)