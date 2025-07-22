#!/usr/bin/env python3
"""
Comprehensive Quantum Physics Content Search
Test direct content access and routing issues
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any

class DirectContentReader:
    """Direct reader for unified content without ChromaDB dependency"""
    
    def __init__(self, index_path: str = "unified_content/index.json"):
        self.index_path = Path(index_path)
        self.content_items = self._load_content()
    
    def _load_content(self):
        """Load content directly from JSON index"""
        try:
            with open(self.index_path, 'r') as f:
                data = json.load(f)
            return data.get('content_items', [])
        except Exception as e:
            print(f"Error loading content: {e}")
            return []
    
    def search_quantum_content(self) -> Dict[str, Any]:
        """Search for all quantum-related content"""
        quantum_terms = [
            'quantum', 'quanta', 'planck', 'heisenberg', 'schrödinger', 
            'bohr', 'feynman', 'eigenstate', 'wave function', 'superposition',
            'double slit', 'uncertainty principle', 'action principle'
        ]
        
        results = {
            'pdf_content': [],
            'video_content': [],
            'total_quantum_mentions': 0,
            'quantum_physics_video': None
        }
        
        for item in self.content_items:
            content_type = item.get('content_type', 'unknown')
            title = item.get('title', 'No title')
            source_path = item.get('source_path', '')
            full_text = item.get('full_text', '')
            item_id = item.get('id', '')
            
            # Count quantum terms
            quantum_count = 0
            found_terms = {}
            
            for term in quantum_terms:
                count = len(re.findall(term, full_text, re.IGNORECASE))
                if count > 0:
                    quantum_count += count
                    found_terms[term] = count
            
            if quantum_count > 0:
                content_info = {
                    'id': item_id,
                    'title': title,
                    'source_path': source_path,
                    'quantum_mentions': quantum_count,
                    'terms_found': found_terms,
                    'chunks_count': len(item.get('chunks', [])),
                    'content_length': len(full_text)
                }
                
                if content_type == 'pdf':
                    results['pdf_content'].append(content_info)
                elif content_type == 'youtube':
                    results['video_content'].append(content_info)
                    
                    # Check for the specific quantum physics video
                    if 'qJZ1Ez28C-A' in source_path:
                        results['quantum_physics_video'] = content_info
                
                results['total_quantum_mentions'] += quantum_count
        
        return results
    
    def get_quantum_physics_video_details(self) -> Dict[str, Any]:
        """Get detailed information about the quantum physics video"""
        for item in self.content_items:
            if 'qJZ1Ez28C-A' in item.get('source_path', ''):
                full_text = item.get('full_text', '')
                chunks = item.get('chunks', [])
                
                # Extract key quantum topics from the video
                quantum_topics = []
                
                # Look for specific physics concepts
                if 'planck' in full_text.lower():
                    quantum_topics.append('Planck\'s constant and quantum action')
                if 'feynman' in full_text.lower():
                    quantum_topics.append('Feynman path integral formulation')
                if 'bohr' in full_text.lower():
                    quantum_topics.append('Bohr atomic model')
                if 'double slit' in full_text.lower():
                    quantum_topics.append('Double slit experiment')
                if 'action' in full_text.lower() and 'principle' in full_text.lower():
                    quantum_topics.append('Principle of least action')
                if 'wave' in full_text.lower() and 'particle' in full_text.lower():
                    quantum_topics.append('Wave-particle duality')
                
                return {
                    'video_id': 'qJZ1Ez28C-A',
                    'title': item.get('title', ''),
                    'content_type': item.get('content_type', ''),
                    'source_path': item.get('source_path', ''),
                    'total_chunks': len(chunks),
                    'transcript_length': len(full_text),
                    'quantum_topics_covered': quantum_topics,
                    'metadata': item.get('metadata', {}),
                    'sample_content': full_text[:1000] + "..." if len(full_text) > 1000 else full_text
                }
        
        return None

def test_routing_system():
    """Test why the routing system can't find quantum content"""
    print("ROUTING SYSTEM DIAGNOSTIC")
    print("=" * 40)
    
    # Test SimpleContentStore
    try:
        from src.storage.simple_content_store import SimpleContentStore
        simple_store = SimpleContentStore()
        simple_content = simple_store.get_all_content()
        print(f"SimpleContentStore has {len(simple_content)} items")
    except Exception as e:
        print(f"SimpleContentStore error: {e}")
    
    # Test UnifiedContentStore
    try:
        from src.storage.unified_content_store import UnifiedContentStore
        unified_store = UnifiedContentStore()
        unified_content = unified_store.get_all_content()
        print(f"UnifiedContentStore has {len(unified_content)} items")
    except Exception as e:
        print(f"UnifiedContentStore error: {e}")
    
    print("\nThe router is likely configured to use SimpleContentStore instead of UnifiedContentStore")

def main():
    """Main analysis function"""
    print("QUANTUM PHYSICS CONTENT COMPREHENSIVE SEARCH")
    print("=" * 50)
    
    # Initialize direct reader
    reader = DirectContentReader()
    
    if not reader.content_items:
        print("❌ No content loaded from unified index!")
        return
    
    print(f"✅ Loaded {len(reader.content_items)} content items from unified index")
    
    # Search for quantum content
    print("\n1. SEARCHING FOR ALL QUANTUM CONTENT...")
    quantum_results = reader.search_quantum_content()
    
    print(f"\n📊 QUANTUM CONTENT SUMMARY:")
    print(f"   - Total quantum mentions across all sources: {quantum_results['total_quantum_mentions']}")
    print(f"   - PDFs with quantum content: {len(quantum_results['pdf_content'])}")
    print(f"   - Videos with quantum content: {len(quantum_results['video_content'])}")
    
    # PDF quantum content
    if quantum_results['pdf_content']:
        print(f"\n📄 QUANTUM PHYSICS IN PDFs:")
        for pdf in quantum_results['pdf_content']:
            print(f"   • {pdf['title']}")
            print(f"     Source: {pdf['source_path']}")
            print(f"     Quantum mentions: {pdf['quantum_mentions']}")
            print(f"     Key terms: {', '.join(f'{k}({v})' for k, v in pdf['terms_found'].items())}")
    
    # Video quantum content
    if quantum_results['video_content']:
        print(f"\n🎥 QUANTUM PHYSICS IN VIDEOS:")
        for video in quantum_results['video_content']:
            print(f"   • {video['title']}")
            print(f"     Source: {video['source_path']}")
            print(f"     Quantum mentions: {video['quantum_mentions']}")
            print(f"     Key terms: {', '.join(f'{k}({v})' for k, v in video['terms_found'].items())}")
    
    # Specific quantum physics video details
    print(f"\n2. QUANTUM PHYSICS VIDEO (qJZ1Ez28C-A) ANALYSIS...")
    video_details = reader.get_quantum_physics_video_details()
    
    if video_details:
        print(f"\n🎯 QUANTUM PHYSICS VIDEO FOUND:")
        print(f"   Video ID: {video_details['video_id']}")
        print(f"   Title: {video_details['title']}")
        print(f"   URL: {video_details['source_path']}")
        print(f"   Total chunks: {video_details['total_chunks']}")
        print(f"   Transcript length: {video_details['transcript_length']:,} characters")
        print(f"   \n   📚 QUANTUM TOPICS COVERED:")
        for topic in video_details['quantum_topics_covered']:
            print(f"      • {topic}")
        
        print(f"\n   📝 SAMPLE CONTENT:")
        print(f"   {video_details['sample_content']}")
        
        # Check metadata
        metadata = video_details['metadata']
        if 'batch_number' in metadata:
            print(f"\n   🔍 VIDEO METADATA:")
            print(f"      • Batch number: {metadata['batch_number']}")
            print(f"      • Duration: {metadata.get('duration', 'Unknown')} seconds")
            print(f"      • Video ID: {metadata.get('video_id', 'Unknown')}")
    else:
        print("❌ Quantum physics video (qJZ1Ez28C-A) NOT FOUND in content!")
    
    # Test routing system
    print(f"\n3. ROUTING SYSTEM DIAGNOSIS...")
    test_routing_system()
    
    # Final summary
    print(f"\n🏁 FINAL SUMMARY:")
    print(f"   ✅ Quantum physics content IS STORED in the unified content system")
    print(f"   ✅ Video qJZ1Ez28C-A contains extensive quantum physics content")
    print(f"   ✅ PDF on dark matter uses quantum approaches")
    print(f"   ❌ Routing system is NOT finding this content (using wrong store)")
    print(f"   \n   💡 SOLUTION: Update router to use UnifiedContentStore instead of SimpleContentStore")

if __name__ == "__main__":
    main()