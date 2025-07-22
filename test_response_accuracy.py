"""
Test if responses accurately use content from PDFs without hallucinating
"""

import requests
import sys
import os
sys.path.append('src')

def get_pdf_content():
    """Get actual content from PDFs to verify against responses"""
    
    from storage.unified_content_store import UnifiedContentStore
    store = UnifiedContentStore()
    
    pdf_contents = {}
    
    for item in store.get_all_content():
        pdf_contents[item.title] = {
            'full_text': item.full_text,
            'keywords': item.keywords,
            'topics': item.topic_assignments
        }
    
    return pdf_contents

def test_specific_fact_queries():
    """Test queries that should return specific facts from PDFs"""
    
    print("TESTING RESPONSE ACCURACY - NO HALLUCINATION CHECK")
    print("=" * 60)
    
    pdf_contents = get_pdf_content()
    
    # Print what's actually in each PDF first
    print("\nACTUAL PDF CONTENTS:")
    print("-" * 40)
    
    for title, content in pdf_contents.items():
        print(f"\n{title}:")
        print(f"  Text preview: {content['full_text'][:200]}...")
        print(f"  Length: {len(content['full_text'])} chars")
    
    test_cases = [
        {
            "query": "What is the energy density of lithium-ion batteries according to the battery document?",
            "expected_source": "Advanced Battery Technologies",
            "expected_facts": ["250-300 Wh/kg", "lithium-ion"],
            "should_not_contain": ["500 Wh/kg", "400 Wh/kg", "theoretical limit"]  # Things not in battery PDF
        },
        {
            "query": "What are the energy requirements for direct air capture in kWh per ton CO2?",
            "expected_source": "Atmospheric Carbon Removal",
            "expected_facts": ["1,200-2,500 kWh/tCO2", "artificial tree", "DAC"],
            "should_not_contain": ["1,500-2,000 kWh", "3,000 kWh", "nuclear power"]  # Things not in atmospheric PDF
        },
        {
            "query": "What is the theoretical efficiency limit for silicon solar cells?",
            "expected_source": "Solar Panel Efficiency",
            "expected_facts": ["29%", "theoretical limit", "silicon"],
            "should_not_contain": ["35%", "40%", "perovskite tandem"]  # Check for hallucination
        },
        {
            "query": "What is the temperature range for biochar pyrolysis?",
            "expected_source": "Atmospheric Carbon Removal",
            "expected_facts": ["450-650°C", "pyrolysis", "biochar"],
            "should_not_contain": ["800°C", "1000°C", "combustion"]  # Check for hallucination
        }
    ]
    
    print("\n" + "=" * 60)
    print("ACCURACY TESTING")
    print("=" * 60)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{i}. Testing Specific Fact Query")
        print(f"Query: {test['query']}")
        print(f"Expected source: {test['expected_source']}")
        print(f"Expected facts: {test['expected_facts']}")
        
        try:
            response = requests.post(
                "http://localhost:8080/api/research",
                json={"query": test['query']},
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    answer = result['result']['final_answer']['answer_summary']
                    
                    print(f"\nResponse ({len(answer)} chars):")
                    print(f"{answer}")
                    
                    # Check for expected facts
                    found_facts = []
                    for fact in test['expected_facts']:
                        if fact.lower() in answer.lower():
                            found_facts.append(fact)
                    
                    # Check for hallucinated content
                    hallucinated_content = []
                    for should_not in test['should_not_contain']:
                        if should_not.lower() in answer.lower():
                            hallucinated_content.append(should_not)
                    
                    # Verify against actual PDF content
                    pdf_verification = "UNKNOWN"
                    for pdf_title, pdf_content in pdf_contents.items():
                        if test['expected_source'].lower() in pdf_title.lower():
                            # Check if the facts in the response are actually in the PDF
                            facts_in_pdf = []
                            for fact in found_facts:
                                if fact.lower() in pdf_content['full_text'].lower():
                                    facts_in_pdf.append(fact)
                            
                            if len(facts_in_pdf) == len(found_facts) and len(found_facts) > 0:
                                pdf_verification = "ACCURATE"
                            elif len(facts_in_pdf) > 0:
                                pdf_verification = "PARTIAL"
                            else:
                                pdf_verification = "INACCURATE"
                            break
                    
                    print(f"\nACCURACY ANALYSIS:")
                    print(f"  Expected facts found: {found_facts} ({len(found_facts)}/{len(test['expected_facts'])})")
                    print(f"  Hallucinated content: {hallucinated_content}")
                    print(f"  PDF verification: {pdf_verification}")
                    
                    if pdf_verification == "ACCURATE" and len(hallucinated_content) == 0:
                        print(f"  RESULT: EXCELLENT - Accurate response from correct PDF")
                    elif pdf_verification == "PARTIAL" and len(hallucinated_content) == 0:
                        print(f"  RESULT: GOOD - Mostly accurate, no hallucination")
                    elif len(hallucinated_content) > 0:
                        print(f"  RESULT: POOR - Contains hallucinated information")
                    else:
                        print(f"  RESULT: FAILED - Inaccurate or no relevant facts found")
                        
                else:
                    print(f"Query failed: {result.get('error', 'Unknown error')}")
            else:
                print(f"HTTP Error: {response.status_code}")
                
        except Exception as e:
            print(f"Request failed: {e}")
        
        print("-" * 60)

def test_cross_pdf_hallucination():
    """Test if system incorrectly mixes content between PDFs"""
    
    print("\nCROSS-PDF CONTAMINATION TEST")
    print("=" * 40)
    
    contamination_tests = [
        {
            "query": "Does the battery document mention solar panel efficiency?",
            "should_answer": "No - battery document should not contain solar info"
        },
        {
            "query": "Does the solar document mention vanadium redox batteries?",
            "should_answer": "No - solar document should not contain battery info"
        },
        {
            "query": "According to the atmospheric carbon removal document, what is silicon solar cell efficiency?",
            "should_answer": "Atmospheric document should not have solar cell data"
        }
    ]
    
    for i, test in enumerate(contamination_tests, 1):
        print(f"\n{i}. {test['query']}")
        print(f"Expected: {test['should_answer']}")
        
        try:
            response = requests.post(
                "http://localhost:8080/api/research",
                json={"query": test['query']},
                timeout=45
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    answer = result['result']['final_answer']['answer_summary']
                    print(f"Response: {answer[:200]}...")
                    
                    # Analyze if response correctly identifies document boundaries
                    correct_boundaries = any(phrase in answer.lower() for phrase in [
                        "does not mention", "not found in", "not covered", 
                        "not discussed", "outside the scope", "not included"
                    ])
                    
                    if correct_boundaries:
                        print("Result: CORRECT - Properly identified document boundaries")
                    else:
                        print("Result: POTENTIAL CONTAMINATION - Check for cross-PDF mixing")
                        
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_specific_fact_queries()
    test_cross_pdf_hallucination()