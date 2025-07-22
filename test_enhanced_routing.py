#!/usr/bin/env python3
"""
Test enhanced routing improvements while ensuring backward compatibility
"""

import sys
sys.path.append('src')

from storage.unified_content_store import UnifiedContentStore
from storage.ner_fuzzy_router import NERFuzzyRouter
from storage.enhanced_ner_fuzzy_router import EnhancedNERFuzzyRouter

def compare_routing_results():
    """Compare original vs enhanced routing to ensure improvements"""
    
    print("=== ROUTING COMPARISON: Original vs Enhanced ===\n")
    
    content_store = UnifiedContentStore()
    original_router = NERFuzzyRouter(content_store)
    enhanced_router = EnhancedNERFuzzyRouter(content_store)
    
    # Test cases that had issues
    test_cases = [
        {
            'query': 'quantum physics energy',
            'expected_improvement': 'Should route to quantum video instead of solar PDF'
        },
        {
            'query': 'solar system planets',
            'expected_improvement': 'Should route to solar system video instead of PDFs'
        },
        {
            'query': 'black holes physics',
            'expected_improvement': 'Should maintain good routing to black holes video'
        },
        {
            'query': 'solar panel technology',
            'expected_improvement': 'Should maintain PDF routing (not break)'
        },
        {
            'query': 'what does the video explain',
            'expected_improvement': 'Should maintain video routing (not break)'
        },
        {
            'query': 'energy storage batteries',
            'expected_improvement': 'Should maintain PDF routing (not break)'
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        query = test['query']
        expected = test['expected_improvement']
        
        print(f"Test {i}: '{query}'")
        print(f"Expected: {expected}")
        
        # Original routing
        try:
            orig_pdfs, orig_videos, orig_explanation = original_router.route_query(query)
            print(f"Original: {len(orig_pdfs)} PDFs, {len(orig_videos)} videos")
            print(f"  -> {orig_explanation}")
        except Exception as e:
            print(f"Original ERROR: {e}")
            orig_pdfs, orig_videos = [], []
        
        # Enhanced routing
        try:
            enh_pdfs, enh_videos, enh_explanation = enhanced_router.route_query(query)
            print(f"Enhanced: {len(enh_pdfs)} PDFs, {len(enh_videos)} videos")
            print(f"  -> {enh_explanation}")
            
            # Show improvement analysis
            if len(enh_videos) > len(orig_videos) and 'video' in expected.lower():
                print(f"  ✅ IMPROVEMENT: More videos found as expected")
            elif len(enh_pdfs) == len(orig_pdfs) and len(enh_videos) == len(orig_videos):
                print(f"  ✅ STABLE: Results unchanged (good for working cases)")
            elif len(enh_videos) > 0 and 'quantum' in query:
                print(f"  ✅ IMPROVEMENT: Found videos for quantum query")
            elif len(enh_videos) > 0 and 'solar system' in query:
                print(f"  ✅ IMPROVEMENT: Found videos for solar system query")
            else:
                print(f"  ⚠️ CHANGED: Results different, needs verification")
                
        except Exception as e:
            print(f"Enhanced ERROR: {e}")
        
        print("-" * 60)

def test_specific_improvements():
    """Test specific improvements for problem queries"""
    
    print("\n=== SPECIFIC IMPROVEMENT TESTS ===\n")
    
    content_store = UnifiedContentStore()
    enhanced_router = EnhancedNERFuzzyRouter(content_store)
    
    # Focus on the queries that were failing
    problem_queries = [
        'quantum physics energy',
        'solar system planets', 
        'planetary exploration',
        'physics concepts explained',
        'space exploration missions'
    ]
    
    for i, query in enumerate(problem_queries, 1):
        print(f"Problem Query {i}: '{query}'")
        
        try:
            pdfs, videos, explanation = enhanced_router.route_query(query)
            
            print(f"Results: {len(pdfs)} PDFs, {len(videos)} videos")
            print(f"Explanation: {explanation}")
            
            # Check for specific improvements
            if videos:
                print(f"Videos found:")
                for video_url in videos:
                    # Find video details
                    for item in content_store.get_all_content():
                        if item.content_type == 'youtube' and item.source_path == video_url:
                            video_id = video_url.split('=')[-1] if '=' in video_url else 'unknown'
                            
                            if 'quantum' in query and video_id == 'qJZ1Ez28C-A':
                                print(f"  ✅ CORRECT: Found quantum physics video")
                            elif 'solar system' in query and video_id == 'Tf_KKGk6FNY':
                                print(f"  ✅ CORRECT: Found solar system video")
                            elif 'space' in query and video_id in ['5iA7wZfxglE', 'Tf_KKGk6FNY']:
                                print(f"  ✅ CORRECT: Found space-related video")
                            else:
                                print(f"  - Found: {item.title}")
                            break
            
            # Get debug info
            debug = enhanced_router.get_routing_debug_info(query)
            print(f"Query intent: {debug['query_intent']['primary']}")
            print(f"Video preference: {debug['query_intent']['prefers_video']}")
            
        except Exception as e:
            print(f"ERROR: {e}")
        
        print("-" * 50)

def test_backward_compatibility():
    """Ensure enhanced router doesn't break existing good functionality"""
    
    print("\n=== BACKWARD COMPATIBILITY TEST ===\n")
    
    content_store = UnifiedContentStore()
    enhanced_router = EnhancedNERFuzzyRouter(content_store)
    
    # Queries that were working well before
    working_queries = [
        'black holes',
        'what are black holes', 
        'solar panel technology',
        'energy storage batteries',
        'what does the video discuss',
        'space mysteries',
        'carbon sequestration'
    ]
    
    for i, query in enumerate(working_queries, 1):
        print(f"Compatibility Test {i}: '{query}'")
        
        try:
            pdfs, videos, explanation = enhanced_router.route_query(query)
            
            print(f"Results: {len(pdfs)} PDFs, {len(videos)} videos")
            print(f"Explanation: {explanation}")
            
            # Basic sanity checks
            if 'black holes' in query and videos:
                print(f"  ✅ GOOD: Black holes query finds videos")
            elif 'solar panel' in query and pdfs:
                print(f"  ✅ GOOD: Solar panel query finds PDFs")
            elif 'battery' in query and pdfs:
                print(f"  ✅ GOOD: Battery query finds PDFs")
            elif 'video' in query and videos:
                print(f"  ✅ GOOD: Video query finds videos")
            elif len(pdfs) > 0 or len(videos) > 0:
                print(f"  ✅ GOOD: Query finds relevant content")
            else:
                print(f"  ⚠️ WARNING: No results found")
            
        except Exception as e:
            print(f"ERROR: {e}")
        
        print("-" * 40)

def main():
    print("ENHANCED NER ROUTER TESTING")
    print("="*60)
    
    # Test 1: Compare original vs enhanced
    compare_routing_results()
    
    # Test 2: Test specific improvements
    test_specific_improvements()
    
    # Test 3: Ensure backward compatibility
    test_backward_compatibility()
    
    print("\n" + "="*60)
    print("TESTING COMPLETE")
    print("Review results to verify:")
    print("1. Enhanced router improves problem queries")
    print("2. Existing functionality is preserved")
    print("3. No regressions in working queries")

if __name__ == "__main__":
    main()