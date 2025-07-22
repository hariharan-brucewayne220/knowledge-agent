"""
Comprehensive testing suite for the NER-enhanced knowledge agent system
"""

import requests
import time
import sys
import os
sys.path.append('src')

def test_system_status():
    """Test if the system is running and responsive"""
    print("=== SYSTEM STATUS TEST ===")
    try:
        response = requests.get("http://localhost:8080", timeout=5)
        if response.status_code == 200:
            print("✓ Server is running")
            return True
        else:
            print(f"✗ Server responded with status {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Server not responding: {e}")
        return False

def test_simple_query():
    """Test basic query functionality"""
    print("\n=== BASIC QUERY TEST ===")
    query = "What is solar panel efficiency?"
    
    try:
        response = requests.post(
            "http://localhost:8080/api/research",
            json={"query": query},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                answer = result['result']['final_answer']['answer_summary']
                print(f"✓ Query successful")
                print(f"  Answer length: {len(answer)} characters")
                print(f"  Preview: {answer[:100]}...")
                return True
            else:
                print(f"✗ Query failed: {result.get('error', 'Unknown error')}")
                return False
        else:
            print(f"✗ HTTP error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"✗ Request failed: {e}")
        return False

def test_multi_resource_query():
    """Test complex multi-resource query"""
    print("\n=== MULTI-RESOURCE QUERY TEST ===")
    query = "How do battery storage systems compare with solar panel efficiency for powering carbon capture operations?"
    
    try:
        response = requests.post(
            "http://localhost:8080/api/research", 
            json={"query": query},
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                answer = result['result']['final_answer']['answer_summary']
                
                # Check for content from multiple PDFs
                battery_terms = ['battery', 'lithium', 'storage', 'energy density', 'Wh/kg']
                solar_terms = ['solar', 'efficiency', 'panel', 'photovoltaic']
                carbon_terms = ['carbon', 'capture', 'sequestration', 'CO2']
                
                battery_found = any(term.lower() in answer.lower() for term in battery_terms)
                solar_found = any(term.lower() in answer.lower() for term in solar_terms)
                carbon_found = any(term.lower() in answer.lower() for term in carbon_terms)
                
                content_types = sum([battery_found, solar_found, carbon_found])
                
                print(f"✓ Multi-resource query successful")
                print(f"  Answer length: {len(answer)} characters")
                print(f"  Battery content: {'YES' if battery_found else 'NO'}")
                print(f"  Solar content: {'YES' if solar_found else 'NO'}")
                print(f"  Carbon content: {'YES' if carbon_found else 'NO'}")
                print(f"  Content types found: {content_types}/3")
                print(f"  Status: {'EXCELLENT' if content_types >= 2 else 'PARTIAL' if content_types >= 1 else 'FAILED'}")
                
                return content_types >= 2
                
            else:
                print(f"✗ Query failed: {result.get('error', 'Unknown error')}")
                return False
        else:
            print(f"✗ HTTP error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"✗ Request failed: {e}")
        return False

def test_technical_query():
    """Test technical query with specific measurements"""
    print("\n=== TECHNICAL QUERY TEST ===")
    query = "What are the energy requirements in kWh per ton CO2 for direct air capture compared to lithium-ion battery energy density?"
    
    try:
        response = requests.post(
            "http://localhost:8080/api/research",
            json={"query": query},
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                answer = result['result']['final_answer']['answer_summary']
                
                # Check for specific technical details
                technical_terms = ['kWh', 'ton', 'CO2', 'Wh/kg', '250-300', '1,500', '2,000']
                found_terms = [term for term in technical_terms if term in answer]
                
                print(f"✓ Technical query successful")
                print(f"  Answer length: {len(answer)} characters")
                print(f"  Technical terms found: {found_terms}")
                print(f"  Specificity: {'HIGH' if len(found_terms) >= 4 else 'MEDIUM' if len(found_terms) >= 2 else 'LOW'}")
                
                return len(found_terms) >= 2
                
            else:
                print(f"✗ Query failed: {result.get('error', 'Unknown error')}")
                return False
        else:
            print(f"✗ HTTP error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"✗ Request failed: {e}")
        return False

def test_ner_routing():
    """Test NER-based smart routing"""
    print("\n=== NER ROUTING TEST ===")
    
    from storage.unified_content_store import UnifiedContentStore
    from storage.simple_smart_router import SimpleSmartRouter
    
    try:
        store = UnifiedContentStore()
        router = SimpleSmartRouter(store)
        
        test_cases = [
            {
                "query": "vanadium redox flow batteries for atmospheric carbon removal",
                "expected_pdfs": 2,
                "description": "Should find battery + atmospheric PDFs"
            },
            {
                "query": "silicon solar cell efficiency",
                "expected_pdfs": 1, 
                "description": "Should find solar PDF only"
            },
            {
                "query": "pyrolysis temperature for biochar production",
                "expected_pdfs": 1,
                "description": "Should find atmospheric/carbon PDF"
            }
        ]
        
        routing_success = 0
        
        for case in test_cases:
            pdfs, videos, explanation = router.route_query(case["query"])
            expected = case["expected_pdfs"]
            actual = len(pdfs)
            
            success = (actual == expected) or (expected > 1 and actual >= expected)
            if success:
                routing_success += 1
                
            print(f"  Query: {case['query'][:50]}...")
            print(f"    Expected: {expected} PDFs, Got: {actual} PDFs")
            print(f"    Result: {'✓ PASS' if success else '✗ FAIL'}")
            print(f"    Files: {[os.path.basename(p) for p in pdfs]}")
        
        overall_success = routing_success >= len(test_cases) * 0.7  # 70% pass rate
        print(f"\n  Routing success: {routing_success}/{len(test_cases)}")
        print(f"  Overall: {'✓ PASS' if overall_success else '✗ FAIL'}")
        
        return overall_success
        
    except Exception as e:
        print(f"✗ NER routing test failed: {e}")
        return False

def test_content_analysis():
    """Test content analysis and NER extraction"""
    print("\n=== CONTENT ANALYSIS TEST ===")
    
    try:
        from storage.unified_content_store import UnifiedContentStore
        store = UnifiedContentStore()
        
        all_content = store.get_all_content()
        
        print(f"  Total documents: {len(all_content)}")
        
        if len(all_content) >= 4:
            print("✓ All expected PDFs loaded")
            
            # Check NER analysis quality
            total_keywords = sum(len(item.keywords) for item in all_content)
            total_topics = sum(len(item.topic_assignments) for item in all_content)
            
            print(f"  Total keywords extracted: {total_keywords}")
            print(f"  Total topic assignments: {total_topics}")
            
            # Check for specific expected topics
            found_topics = set()
            for item in all_content:
                found_topics.update(item.topic_assignments)
            
            expected_topics = {'solar_energy', 'carbon_sequestration', 'energy_systems', 'materials_science'}
            topics_found = len(found_topics.intersection(expected_topics))
            
            print(f"  Topics found: {sorted(found_topics)}")
            print(f"  Expected topics coverage: {topics_found}/{len(expected_topics)}")
            
            success = (len(all_content) >= 4 and 
                      total_keywords >= 20 and 
                      total_topics >= 4 and 
                      topics_found >= 3)
            
            print(f"  Status: {'✓ EXCELLENT' if success else '✗ NEEDS IMPROVEMENT'}")
            return success
            
        else:
            print(f"✗ Expected 4+ PDFs, found {len(all_content)}")
            return False
            
    except Exception as e:
        print(f"✗ Content analysis failed: {e}")
        return False

def run_comprehensive_tests():
    """Run all tests and provide summary"""
    print("=" * 60)
    print("COMPREHENSIVE SYSTEM TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("System Status", test_system_status),
        ("Basic Query", test_simple_query),
        ("Multi-Resource Query", test_multi_resource_query),
        ("Technical Query", test_technical_query),
        ("NER Routing", test_ner_routing),
        ("Content Analysis", test_content_analysis)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        start_time = time.time()
        success = test_func()
        duration = time.time() - start_time
        
        results.append({
            'name': test_name,
            'success': success,
            'duration': duration
        })
    
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for r in results if r['success'])
    total = len(results)
    
    for result in results:
        status = "PASS" if result['success'] else "FAIL"
        duration = result['duration']
        print(f"{result['name']:<20} | {status:<4} | {duration:.2f}s")
    
    print("-" * 60)
    print(f"Overall Success Rate: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - SYSTEM WORKING PERFECTLY!")
    elif passed >= total * 0.8:
        print("✅ EXCELLENT - System mostly working with minor issues")
    elif passed >= total * 0.6:
        print("⚠️  GOOD - System working but needs improvements")
    else:
        print("❌ SYSTEM ISSUES - Multiple tests failed")
    
    return passed, total

if __name__ == "__main__":
    run_comprehensive_tests()