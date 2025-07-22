#!/usr/bin/env python3
"""
Comprehensive system check - verify routing, content accuracy, and anti-hallucination
"""

import sys
sys.path.append('src')

from storage.unified_content_store import UnifiedContentStore
from storage.ner_fuzzy_router import NERFuzzyRouter

def check_all_content():
    """Check all stored content details"""
    
    print("=== COMPREHENSIVE CONTENT VERIFICATION ===\n")
    
    content_store = UnifiedContentStore()
    all_content = content_store.get_all_content()
    
    pdfs = [item for item in all_content if item.content_type == 'pdf']
    videos = [item for item in all_content if item.content_type == 'youtube']
    
    print(f"SYSTEM STATUS:")
    print(f"Total items: {len(all_content)}")
    print(f"PDFs: {len(pdfs)}")
    print(f"Videos: {len(videos)}")
    print()
    
    # Check each video in detail
    print("VIDEO CONTENT VERIFICATION:")
    print("="*50)
    
    for i, video in enumerate(videos, 1):
        print(f"Video {i}: {video.title}")
        print(f"URL: {video.source_path}")
        print(f"Content length: {len(video.full_text)} chars")
        print(f"Keywords extracted: {len(video.keywords)}")
        print(f"Topics assigned: {len(video.topic_assignments)}")
        
        # Show actual content samples
        content = video.full_text
        print(f"Content start: '{content[:150]}...'")
        print(f"Content end: '...{content[-150:]}'")
        
        # Check for key terms based on our analysis
        video_id = video.source_path.split('=')[-1] if '=' in video.source_path else 'unknown'
        
        if video_id == '5iA7wZfxglE':  # Space mysteries
            terms = ['space', 'universe', 'dark', 'matter', 'mystery']
        elif video_id == 'rJLtT0QXoPo':  # Black holes
            terms = ['black', 'holes', 'time', 'gravitational']
        elif video_id == 'Tf_KKGk6FNY':  # Solar system
            terms = ['planet', 'solar', 'nasa', 'mission']
        elif video_id == 'qJZ1Ez28C-A':  # Quantum physics
            terms = ['physics', 'quantum', 'energy', 'light', 'paths']
        else:
            terms = ['the', 'and', 'is']  # Generic
        
        print(f"Term verification for expected content:")
        for term in terms:
            count = content.lower().count(term)
            print(f"  '{term}': {count} occurrences")
        
        print("-" * 40)
    
    return videos

def test_routing_accuracy(videos):
    """Test routing accuracy for specific queries"""
    
    print("\n=== ROUTING ACCURACY TEST ===\n")
    
    content_store = UnifiedContentStore()
    ner_router = NERFuzzyRouter(content_store)
    
    # Test specific queries that should route to specific videos
    test_cases = [
        {
            'query': 'black holes physics',
            'expected_video': 'rJLtT0QXoPo',
            'reason': 'Should find black holes video'
        },
        {
            'query': 'space mysteries universe',
            'expected_video': '5iA7wZfxglE', 
            'reason': 'Should find space mysteries video'
        },
        {
            'query': 'quantum physics energy',
            'expected_video': 'qJZ1Ez28C-A',
            'reason': 'Should find quantum physics video'
        },
        {
            'query': 'planetary exploration solar system',
            'expected_video': 'Tf_KKGk6FNY',
            'reason': 'Should find solar system video'
        },
        {
            'query': 'solar panel technology',
            'expected_video': None,  # Should route to PDF, not video
            'reason': 'Should route to solar panel PDF, not video'
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        query = test['query']
        expected = test['expected_video']
        reason = test['reason']
        
        print(f"Test {i}: '{query}'")
        print(f"Expected: {reason}")
        
        try:
            pdf_files, youtube_urls, explanation = ner_router.route_query(query)
            
            print(f"Result: {len(pdf_files)} PDFs, {len(youtube_urls)} videos")
            print(f"Explanation: {explanation}")
            
            # Check if expected video was found
            if expected:
                video_found = any(expected in url for url in youtube_urls)
                status = "PASS" if video_found else "FAIL"
                print(f"Expected video found: {status}")
                
                if video_found:
                    # Verify the content actually contains relevant terms
                    for video in videos:
                        if expected in video.source_path:
                            content = video.full_text.lower()
                            query_words = query.lower().split()
                            matches = sum(1 for word in query_words if word in content and len(word) > 3)
                            print(f"Content verification: {matches}/{len([w for w in query_words if len(w) > 3])} query terms found in video")
                            break
            else:
                # Should not route to videos
                video_found = len(youtube_urls) > 0
                status = "PASS" if not video_found else "FAIL"
                print(f"Correctly avoided videos: {status}")
            
        except Exception as e:
            print(f"ERROR: {e}")
        
        print("-" * 50)

def test_content_retrieval():
    """Test actual content retrieval to check for hallucination"""
    
    print("\n=== CONTENT RETRIEVAL VERIFICATION ===\n")
    
    content_store = UnifiedContentStore()
    ner_router = NERFuzzyRouter(content_store)
    
    # Test queries and verify actual content
    verification_tests = [
        {
            'query': 'what are black holes',
            'verify_terms': ['black', 'hole', 'event', 'horizon'],
            'avoid_terms': ['solar panel', 'battery', 'carbon capture']
        },
        {
            'query': 'space mysteries explained',
            'verify_terms': ['space', 'universe', 'mystery'],
            'avoid_terms': ['black hole', 'planet', 'quantum']
        },
        {
            'query': 'quantum physics concepts',
            'verify_terms': ['quantum', 'physics', 'energy'],
            'avoid_terms': ['black hole', 'dark matter', 'solar system']
        }
    ]
    
    for i, test in enumerate(verification_tests, 1):
        query = test['query']
        verify_terms = test['verify_terms']
        avoid_terms = test['avoid_terms']
        
        print(f"Verification Test {i}: '{query}'")
        
        try:
            pdf_files, youtube_urls, explanation = ner_router.route_query(query)
            
            print(f"Routed to: {len(pdf_files)} PDFs, {len(youtube_urls)} videos")
            
            if youtube_urls:
                print(f"Content verification:")
                for url in youtube_urls:
                    # Find the video content
                    for video in content_store.get_all_content():
                        if video.content_type == 'youtube' and video.source_path == url:
                            content = video.full_text.lower()
                            
                            print(f"  Video: {video.title}")
                            
                            # Check for expected terms
                            verify_found = []
                            for term in verify_terms:
                                count = content.count(term.lower())
                                if count > 0:
                                    verify_found.append(f"{term}({count})")
                            
                            # Check for terms that should NOT be there
                            avoid_found = []
                            for term in avoid_terms:
                                count = content.count(term.lower())
                                if count > 0:
                                    avoid_found.append(f"{term}({count})")
                            
                            print(f"    Expected terms found: {verify_found}")
                            print(f"    Unexpected terms: {avoid_found}")
                            
                            # Show actual content sample
                            if verify_found:
                                # Find context around first verified term
                                first_term = verify_terms[0].lower()
                                pos = content.find(first_term)
                                if pos >= 0:
                                    context = video.full_text[max(0, pos-100):pos+100]
                                    print(f"    Content context: '...{context}...'")
                            
                            break
            
        except Exception as e:
            print(f"ERROR: {e}")
        
        print("-" * 50)

def test_cross_domain_intelligence():
    """Test if system can distinguish between similar terms in different contexts"""
    
    print("\n=== CROSS-DOMAIN INTELLIGENCE TEST ===\n")
    
    content_store = UnifiedContentStore()
    ner_router = NERFuzzyRouter(content_store)
    
    # Test contextual understanding
    context_tests = [
        {
            'query': 'solar technology efficiency',
            'expected': 'PDF',
            'reason': 'Solar panel technology context'
        },
        {
            'query': 'solar system planets',
            'expected': 'VIDEO',
            'reason': 'Solar system astronomy context'
        },
        {
            'query': 'energy storage batteries',
            'expected': 'PDF', 
            'reason': 'Practical energy storage'
        },
        {
            'query': 'energy physics quantum',
            'expected': 'VIDEO',
            'reason': 'Theoretical physics energy'
        }
    ]
    
    for i, test in enumerate(context_tests, 1):
        query = test['query']
        expected = test['expected']
        reason = test['reason']
        
        print(f"Context Test {i}: '{query}'")
        print(f"Expected: {expected} - {reason}")
        
        try:
            pdf_files, youtube_urls, explanation = ner_router.route_query(query)
            
            result_type = ""
            if len(pdf_files) > 0 and len(youtube_urls) == 0:
                result_type = "PDF"
            elif len(youtube_urls) > 0 and len(pdf_files) == 0:
                result_type = "VIDEO"
            elif len(pdf_files) > 0 and len(youtube_urls) > 0:
                result_type = "MIXED"
            else:
                result_type = "NONE"
            
            status = "PASS" if result_type == expected else f"FAIL (got {result_type})"
            
            print(f"Result: {result_type} - {status}")
            print(f"Details: {len(pdf_files)} PDFs, {len(youtube_urls)} videos")
            print(f"Explanation: {explanation}")
            
        except Exception as e:
            print(f"ERROR: {e}")
        
        print("-" * 40)

def main():
    print("COMPREHENSIVE SYSTEM VERIFICATION")
    print("="*60)
    
    # 1. Check all content
    videos = check_all_content()
    
    # 2. Test routing accuracy
    test_routing_accuracy(videos)
    
    # 3. Test content retrieval
    test_content_retrieval()
    
    # 4. Test cross-domain intelligence
    test_cross_domain_intelligence()
    
    print("\n" + "="*60)
    print("COMPREHENSIVE CHECK COMPLETE")
    print("Review the results above to verify:")
    print("1. All videos are stored with correct content")
    print("2. Routing matches expected videos")
    print("3. Content retrieval is accurate (no hallucination)")
    print("4. Context-aware routing works correctly")

if __name__ == "__main__":
    main()