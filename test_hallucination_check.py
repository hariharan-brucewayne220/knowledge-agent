#!/usr/bin/env python3
"""
Test if the system is using actual stored video content or hallucinating
"""

import sys
sys.path.append('src')

from storage.unified_content_store import UnifiedContentStore
from storage.ner_fuzzy_router import NERFuzzyRouter

def check_actual_video_content():
    """Check what content is actually stored in the videos"""
    
    print("=== Checking Actual Stored Video Content ===\n")
    
    content_store = UnifiedContentStore()
    all_content = content_store.get_all_content()
    videos = [item for item in all_content if item.content_type == 'youtube']
    
    for i, video in enumerate(videos, 1):
        print(f"Video {i}: {video.title}")
        print(f"URL: {video.source_path}")
        print(f"Content length: {len(video.full_text)} characters")
        print(f"Keywords extracted: {video.keywords[:10]}")
        print(f"Topics: {video.topic_assignments}")
        
        # Show actual content excerpts
        print(f"\nACTUAL CONTENT EXCERPTS:")
        content = video.full_text
        
        # Show first 300 characters
        print(f"Beginning: '{content[:300]}...'")
        
        # Show middle section
        middle_start = len(content) // 2
        print(f"Middle: '{content[middle_start:middle_start+300]}...'")
        
        # Show end section  
        print(f"End: '{content[-300:]}'")
        
        # Look for specific terms
        print(f"\nTERM ANALYSIS:")
        terms_to_check = ['black', 'holes', 'space', 'time', 'universe', 'gravitational', 'mystery', 'cosmic']
        for term in terms_to_check:
            count = content.lower().count(term)
            if count > 0:
                print(f"  '{term}': appears {count} times")
                
                # Show context where term appears
                term_pos = content.lower().find(term)
                if term_pos > 0:
                    context_start = max(0, term_pos - 50)
                    context_end = min(len(content), term_pos + 50)
                    context = content[context_start:context_end]
                    print(f"    Context: '...{context}...'")
        
        print("="*60)

def test_specific_content_retrieval():
    """Test if queries actually retrieve specific video content"""
    
    print("\n=== Testing Specific Content Retrieval ===\n")
    
    content_store = UnifiedContentStore()
    ner_router = NERFuzzyRouter(content_store)
    
    # Get the black holes video content
    videos = [item for item in content_store.get_all_content() if item.content_type == 'youtube']
    black_holes_video = None
    space_video = None
    
    for video in videos:
        if 'rJLtT0QXoPo' in video.source_path:
            black_holes_video = video
        elif '5iA7wZfxglE' in video.source_path:
            space_video = video
    
    if black_holes_video:
        print(f"BLACK HOLES VIDEO ANALYSIS:")
        print(f"Title: {black_holes_video.title}")
        
        # Check for actual black hole content
        content = black_holes_video.full_text.lower()
        black_hole_mentions = content.count('black hole')
        black_mentions = content.count('black')
        holes_mentions = content.count('holes')
        
        print(f"'black hole' appears {black_hole_mentions} times")
        print(f"'black' appears {black_mentions} times") 
        print(f"'holes' appears {holes_mentions} times")
        
        # Find specific black hole contexts
        if 'black' in content:
            black_pos = content.find('black')
            context = black_holes_video.full_text[max(0, black_pos-100):black_pos+100]
            print(f"Context around 'black': '{context}'")
        
        print()
    
    if space_video:
        print(f"SPACE MYSTERIES VIDEO ANALYSIS:")
        print(f"Title: {space_video.title}")
        
        content = space_video.full_text.lower()
        space_mentions = content.count('space')
        mystery_mentions = content.count('mystery')
        universe_mentions = content.count('universe')
        
        print(f"'space' appears {space_mentions} times")
        print(f"'mystery' appears {mystery_mentions} times")
        print(f"'universe' appears {universe_mentions} times")
        
        # Show space content context
        if 'space' in content:
            space_pos = content.find('space')
            context = space_video.full_text[max(0, space_pos-100):space_pos+100]
            print(f"Context around 'space': '{context}'")
        
        print()

def test_routing_with_content_verification():
    """Test routing and verify it uses actual stored content"""
    
    print("=== Routing with Content Verification ===\n")
    
    content_store = UnifiedContentStore()
    ner_router = NERFuzzyRouter(content_store)
    
    # Test specific queries and check if routing matches actual content
    test_cases = [
        ("black holes", "Should find black holes video if it actually contains 'black' content"),
        ("space mysteries", "Should find space video if it actually contains 'space' content"),
        ("time and space", "Should find videos that actually discuss time/space"),
        ("gravitational effects", "Should only match if videos actually mention gravity")
    ]
    
    for query, expectation in test_cases:
        print(f"Query: '{query}'")
        print(f"Expectation: {expectation}")
        
        pdf_files, youtube_urls, explanation = ner_router.route_query(query)
        
        print(f"Result: {len(pdf_files)} PDFs, {len(youtube_urls)} videos")
        print(f"Explanation: {explanation}")
        
        # Verify that routed videos actually contain relevant content
        if youtube_urls:
            print(f"CONTENT VERIFICATION:")
            for url in youtube_urls:
                video = None
                for item in content_store.get_all_content():
                    if item.content_type == 'youtube' and item.source_path == url:
                        video = item
                        break
                
                if video:
                    content = video.full_text.lower()
                    query_words = query.lower().split()
                    
                    for word in query_words:
                        if len(word) > 2:
                            count = content.count(word)
                            print(f"  Video '{video.title}' contains '{word}': {count} times")
                            
                            if count > 0:
                                # Show actual context
                                word_pos = content.find(word)
                                if word_pos >= 0:
                                    context = video.full_text[max(0, word_pos-80):word_pos+80]
                                    print(f"    Context: '...{context}...'")
        
        print("-" * 50)

if __name__ == "__main__":
    # Check what content is actually stored
    check_actual_video_content()
    
    # Test specific content retrieval
    test_specific_content_retrieval()
    
    # Test routing with verification
    test_routing_with_content_verification()