"""
Quick test to see server logs during multi-resource query
"""

import requests

def test_with_logs():
    """Test one query and observe server behavior"""
    
    query = "Could solar panels provide enough energy for carbon sequestration projects?"
    
    print(f"Sending query: {query}")
    
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
                sources = final_answer.get('sources_processed', {})
                print(f"PDFs processed: {sources.get('pdf_documents', 0)}")
                print(f"Citations: {len(final_answer.get('source_provenance', []))}")
            else:
                print("Query failed")
        else:
            print(f"HTTP Error: {response.status_code}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_with_logs()