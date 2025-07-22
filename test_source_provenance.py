"""
Source Provenance Verification Test
Tests the complete flow: Query → Content Discovery → Source Citations
"""

import json
import requests
import time
import sys
from pathlib import Path

class SourceProvenanceTest:
    """Test class for verifying source provenance functionality"""
    
    def __init__(self, server_url="http://localhost:8080"):
        self.server_url = server_url
        self.test_results = []
        
    def test_server_status(self):
        """Test if server is running and responsive"""
        try:
            response = requests.get(f"{self.server_url}/api/status", timeout=5)
            if response.status_code == 200:
                print("✅ Server is running and responsive")
                return True
            else:
                print(f"❌ Server responded with status: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Cannot connect to server: {e}")
            return False
    
    def run_query_test(self, test_name, query, expected_sources):
        """Run a query test and verify source provenance"""
        
        print(f"\nTEST: {test_name}")
        print(f"Query: {query}")
        
        try:
            # Send research query
            payload = {"query": query}
            response = requests.post(
                f"{self.server_url}/api/research", 
                json=payload, 
                timeout=60
            )
            
            if response.status_code != 200:
                print(f"❌ API Error: {response.status_code}")
                return False
            
            result = response.json()
            
            # Verify basic response structure
            if not result.get('success'):
                print(f"❌ Query failed: {result.get('error', 'Unknown error')}")
                return False
            
            # Extract key information
            final_answer = result['result']['final_answer']
            answer_summary = final_answer.get('answer_summary', '')
            source_provenance = final_answer.get('source_provenance', [])
            sources_processed = final_answer.get('sources_processed', {})
            
            print(f"\n📊 RESPONSE ANALYSIS:")
            print(f"  - PDFs Analyzed: {sources_processed.get('pdf_documents', 0)}")
            print(f"  - Videos Analyzed: {sources_processed.get('youtube_videos', 0)}")
            print(f"  - Source Citations: {len(source_provenance)}")
            print(f"  - Answer Length: {len(answer_summary)} chars")
            
            # Test 1: Check if PDFs were actually analyzed
            pdfs_analyzed = sources_processed.get('pdf_documents', 0)
            test_passed_pdf_count = pdfs_analyzed > 0
            
            if test_passed_pdf_count:
                print(f"✅ PDFs were analyzed ({pdfs_analyzed} found)")
            else:
                print(f"❌ No PDFs analyzed (expected > 0)")
            
            # Test 2: Check source provenance exists
            test_passed_citations = len(source_provenance) > 0
            
            if test_passed_citations:
                print(f"✅ Source citations found ({len(source_provenance)} citations)")
            else:
                print(f"❌ No source citations (expected > 0)")
            
            # Test 3: Verify source citations are real (not hallucinated)
            print(f"\n📝 SOURCE CITATION ANALYSIS:")
            real_citations = 0
            fake_citations = 0
            
            for i, citation in enumerate(source_provenance):
                source_id = citation.get('source_id', 'Unknown')
                source_type = citation.get('source_type', 'unknown')
                specific_ref = citation.get('specific_reference', 'None')
                content_excerpt = citation.get('content_excerpt', 'None')
                
                print(f"  [{i+1}] Source: {source_id}")
                print(f"      Type: {source_type}")
                print(f"      Reference: {specific_ref}")
                print(f"      Excerpt: {content_excerpt[:100]}..." if content_excerpt else "      Excerpt: None")
                
                # Check if this is a real source from our expected sources
                is_real_source = any(expected in source_id for expected in expected_sources)
                
                if is_real_source:
                    real_citations += 1
                    print(f"      Status: ✅ REAL SOURCE")
                else:
                    fake_citations += 1
                    print(f"      Status: ❌ UNKNOWN/FAKE SOURCE")
                    
                print()
            
            # Test 4: Verify majority of citations are real
            test_passed_real_sources = real_citations > fake_citations
            
            if test_passed_real_sources:
                print(f"✅ Real sources dominate ({real_citations} real vs {fake_citations} fake)")
            else:
                print(f"❌ Too many fake sources ({real_citations} real vs {fake_citations} fake)")
            
            # Test 5: Check answer quality (contains technical details)
            technical_indicators = ['%', 'kWh', 'CO₂', 'CO2', 'efficiency', 'temperature', 'cost']
            technical_count = sum(1 for indicator in technical_indicators if indicator.lower() in answer_summary.lower())
            test_passed_technical = technical_count >= 2
            
            if test_passed_technical:
                print(f"✅ Answer contains technical details ({technical_count} indicators found)")
            else:
                print(f"❌ Answer lacks technical details ({technical_count} indicators found)")
            
            # Overall test result
            all_tests = [
                test_passed_pdf_count,
                test_passed_citations, 
                test_passed_real_sources,
                test_passed_technical
            ]
            
            overall_passed = all(all_tests)
            
            print(f"\n🎯 TEST SUMMARY: {test_name}")
            print(f"  - PDFs Analyzed: {'✅' if test_passed_pdf_count else '❌'}")
            print(f"  - Has Citations: {'✅' if test_passed_citations else '❌'}")
            print(f"  - Real Sources: {'✅' if test_passed_real_sources else '❌'}")
            print(f"  - Technical Details: {'✅' if test_passed_technical else '❌'}")
            print(f"  - Overall: {'✅ PASSED' if overall_passed else '❌ FAILED'}")
            
            # Store result
            self.test_results.append({
                'test_name': test_name,
                'query': query,
                'passed': overall_passed,
                'pdfs_analyzed': pdfs_analyzed,
                'citations_count': len(source_provenance),
                'real_citations': real_citations,
                'fake_citations': fake_citations,
                'technical_indicators': technical_count
            })
            
            return overall_passed
            
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            return False
    
    def run_all_tests(self):
        """Run all source provenance tests"""
        
        print("SOURCE PROVENANCE TEST SUITE")
        print("=" * 50)
        
        # Check server status
        if not self.test_server_status():
            print("❌ Cannot proceed - server not available")
            return False
        
        # Define test cases with expected sources
        expected_climate_sources = [
            "Solar Panel Efficiency",
            "Carbon Sequestration", 
            "solar_panel_efficiency",
            "carbon_sequestration"
        ]
        
        test_cases = [
            {
                'name': "Solar Panel Efficiency Query",
                'query': "What are the current efficiency rates of silicon solar cells mentioned in the documents?",
                'expected_sources': expected_climate_sources
            },
            {
                'name': "Carbon Sequestration Query", 
                'query': "What are the energy requirements for Direct Air Capture according to the research?",
                'expected_sources': expected_climate_sources
            },
            {
                'name': "Cross-Document Query",
                'query': "Could solar panels provide enough energy for carbon sequestration technologies based on the efficiency and energy requirement data?",
                'expected_sources': expected_climate_sources
            },
            {
                'name': "Specific Technical Query",
                'query': "What is the theoretical efficiency limit of silicon solar cells and how much energy does DAC require per ton of CO2?",
                'expected_sources': expected_climate_sources
            }
        ]
        
        # Run each test
        passed_tests = 0
        total_tests = len(test_cases)
        
        for test_case in test_cases:
            if self.run_query_test(
                test_case['name'], 
                test_case['query'], 
                test_case['expected_sources']
            ):
                passed_tests += 1
            time.sleep(2)  # Brief pause between tests
        
        # Final summary
        print("\n🏁 FINAL TEST RESULTS")
        print("=" * 50)
        print(f"Passed: {passed_tests}/{total_tests} tests")
        
        for result in self.test_results:
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            print(f"{status} {result['test_name']}")
            print(f"    PDFs: {result['pdfs_analyzed']}, Citations: {result['citations_count']}, Real: {result['real_citations']}")
        
        success_rate = (passed_tests / total_tests) * 100
        print(f"\nSuccess Rate: {success_rate:.1f}%")
        
        if success_rate >= 75:
            print("🎉 SOURCE PROVENANCE WORKING WELL!")
        elif success_rate >= 50:
            print("⚠️  Source provenance partially working - needs improvement")
        else:
            print("❌ Source provenance system needs major fixes")
        
        return success_rate >= 75

if __name__ == "__main__":
    print("Starting Source Provenance Test...")
    
    tester = SourceProvenanceTest()
    success = tester.run_all_tests()
    
    if success:
        print("\n✅ All systems working correctly!")
        sys.exit(0)
    else:
        print("\n❌ Issues detected - check output above")
        sys.exit(1)