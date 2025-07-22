"""
Test if both PDFs' content is actually being used in answers
"""

import requests

def test_content_presence():
    """Test if answers contain information from both PDFs"""
    
    # This query should require both PDFs
    query = "What is the theoretical efficiency limit for silicon cells and what are DAC energy requirements per ton CO2?"
    
    print(f"Testing query: {query}")
    print("Expected: Should mention both 29% efficiency AND 1,500-2,000 kWh per ton")
    
    try:
        response = requests.post(
            "http://localhost:8080/api/research",
            json={"query": query}, 
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                final_answer = result['result']['final_answer']
                answer_text = final_answer.get('answer_summary', '')
                
                print(f"\nAnswer received ({len(answer_text)} chars):")
                print(answer_text)
                
                # Check for Solar PDF content
                solar_indicators = ['29%', '29-33%', 'theoretical limit', 'silicon']
                solar_found = [indicator for indicator in solar_indicators if indicator in answer_text]
                
                # Check for Carbon PDF content  
                carbon_indicators = ['1,500', '2,000', 'kWh per ton', 'DAC', 'Direct Air Capture']
                carbon_found = [indicator for indicator in carbon_indicators if indicator in answer_text]
                
                print(f"\nContent Analysis:")
                print(f"Solar content found: {solar_found} ({'YES' if solar_found else 'NO'})")
                print(f"Carbon content found: {carbon_found} ({'YES' if carbon_found else 'NO'})")
                
                sources_processed = final_answer.get('sources_processed', {})
                print(f"\nStats claim: {sources_processed.get('pdf_documents', 0)} PDFs processed")
                
                if solar_found and carbon_found:
                    print(f"\n✅ SUCCESS: Answer contains content from BOTH PDFs!")
                    print("   Issue is just with the stats reporting, not actual processing")
                elif solar_found:
                    print(f"\n⚠️  PARTIAL: Only solar content found")
                elif carbon_found:
                    print(f"\n⚠️  PARTIAL: Only carbon content found")  
                else:
                    print(f"\n❌ FAILURE: No specific content from either PDF")
                    
            else:
                print("Query failed")
        else:
            print(f"HTTP Error: {response.status_code}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_content_presence()