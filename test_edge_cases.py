"""
Test NER system with edge cases
"""

import sys
import os
sys.path.append('src')
from storage.unified_content_store import UnifiedContentStore
from storage.simple_smart_router import SimpleSmartRouter

def test_edge_cases():
    """Test NER system with challenging edge cases"""
    
    store = UnifiedContentStore()
    router = SimpleSmartRouter(store)
    
    edge_cases = [
        {
            "name": "Highly Technical Query",
            "query": "What are temperature coefficients and degradation rates for perovskite tandem cells versus olivine dissolution in enhanced weathering?",
            "expected": "Should find both PDFs due to technical terms"
        },
        {
            "name": "Measurement-Specific Query", 
            "query": "How much energy in kWh per ton CO2 do different carbon capture methods require compared to solar panel power generation in MW?",
            "expected": "Should detect multi-resource need for energy units comparison"
        },
        {
            "name": "Single Topic Query",
            "query": "What is silicon solar cell efficiency?",
            "expected": "Should route to solar PDF only"
        },
        {
            "name": "Unrelated Query",
            "query": "What is the capital of France and how do you cook pasta?",
            "expected": "Should fallback to general search"
        },
        {
            "name": "Chemical Formula Query",
            "query": "How does CO2 storage in geological formations work?", 
            "expected": "Should route to carbon sequestration PDF"
        },
        {
            "name": "Cost Comparison Query",
            "query": "What are the cost differences between solar installation and direct air capture per dollar invested?",
            "expected": "Should need both PDFs for cost comparison"
        },
        {
            "name": "Technical Process Query",
            "query": "Explain the pyrolysis process for biochar production and photovoltaic cell manufacturing",
            "expected": "Should detect both carbon and solar processes"
        }
    ]
    
    print("=== EDGE CASE TESTING FOR NER SYSTEM ===\n")
    
    for i, case in enumerate(edge_cases, 1):
        try:
            print(f"{i}. {case['name']}")
            print(f"Query: {case['query']}")
            print(f"Expected: {case['expected']}")
            
            pdfs, videos, explanation = router.route_query(case['query'])
            
            print(f"Result: {len(pdfs)} PDFs, {len(videos)} videos")
            print(f"PDFs: {[os.path.basename(p) for p in pdfs]}")
            print(f"Routing: {explanation}")
            print(f"Status: {'MULTI-RESOURCE' if len(pdfs) > 1 else 'SINGLE-RESOURCE' if len(pdfs) == 1 else 'NO-RESOURCE'}")
            print("-" * 80)
            
        except Exception as e:
            print(f"ERROR in {case['name']}: {e}")
            print("-" * 80)

def test_ner_keyword_extraction():
    """Test NER keyword extraction capabilities"""
    
    print("\n=== NER KEYWORD EXTRACTION TEST ===\n")
    
    store = UnifiedContentStore()
    
    # Test various text types
    test_texts = [
        {
            "name": "Solar Technical Text",
            "text": "Perovskite tandem solar cells achieve 31.3% efficiency with silicon substrates at 25°C operating temperature, degrading at 0.5% annually over 25-year warranties.",
            "expected_keywords": ["perovskite", "tandem", "efficiency", "silicon", "degradation"]
        },
        {
            "name": "Carbon Sequestration Text", 
            "text": "Direct air capture requires 1,500-2,000 kWh per ton CO2 captured, while enhanced weathering of olivine minerals costs $150-600 per ton CO2.",
            "expected_keywords": ["direct air capture", "kwh", "ton", "co2", "weathering", "olivine"]
        },
        {
            "name": "Mixed Technical Text",
            "text": "Solar-powered DAC facilities could integrate 500 MW photovoltaic arrays with carbon capture systems requiring 2.5 GWh storage capacity.",
            "expected_keywords": ["solar", "dac", "photovoltaic", "carbon", "capture", "storage"]
        }
    ]
    
    for test_case in test_texts:
        print(f"Testing: {test_case['name']}")
        print(f"Text: {test_case['text'][:100]}...")
        
        keywords = store._ner_extract_keywords(test_case['text'])
        topics = store._categorize_content(test_case['text'], test_case['name'])
        
        print(f"NER Keywords: {keywords[:10]}")
        print(f"Topics: {[(topic, f'{score:.3f}') for topic, score in topics.items() if score > 0.1]}")
        
        # Check if expected keywords were found
        found_expected = [exp for exp in test_case['expected_keywords'] 
                         if any(exp.lower() in kw.lower() for kw in keywords)]
        print(f"Expected found: {found_expected}")
        print(f"Success rate: {len(found_expected)}/{len(test_case['expected_keywords'])}")
        print("-" * 60)

if __name__ == "__main__":
    test_edge_cases()
    test_ner_keyword_extraction()