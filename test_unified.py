"""
Test Unified Content Store
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

# Test without ChromaDB first - create a simple test
import json

def test_unified_content():
    """Test if unified content loads correctly"""
    
    # Read the unified content index directly
    try:
        with open('unified_content/index.json', 'r', encoding='utf-8') as f:
            content_data = json.load(f)
        
        print(f"Unified content loaded successfully!")
        print(f"Total items: {len(content_data['content_items'])}")
        
        for item in content_data['content_items']:
            print(f"  - {item['title']} ({item['content_type']})")
            print(f"    Chunks: {len(item['chunks'])}")
            print(f"    Keywords: {item['keywords']}")
            print()
        
        # Test search-like functionality
        query = "solar panel efficiency"
        print(f"Testing search for: '{query}'")
        
        for item in content_data['content_items']:
            title_match = query.lower() in item['title'].lower()
            content_match = any(query.lower() in chunk['text'].lower() for chunk in item['chunks'])
            
            if title_match or content_match:
                print(f"  MATCH: {item['title']}")
                if title_match:
                    print(f"    - Title match")
                if content_match:
                    print(f"    - Content match")
                    # Show matching text excerpt
                    for chunk in item['chunks']:
                        if query.lower() in chunk['text'].lower():
                            text = chunk['text'][:200] + "..."
                            print(f"    - Text: {text}")
                            break
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    test_unified_content()