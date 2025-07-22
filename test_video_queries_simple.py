#!/usr/bin/env python3
"""
Test video queries without unicode issues
"""

import sys
sys.path.append('src')

from storage.unified_content_store import UnifiedContentStore
from storage.ner_fuzzy_router import NERFuzzyRouter

def test_black_holes_video():
    """Test queries for the black holes video"""
    
    print("=== Testing Black Holes Video Queries ===\n")
    
    # Initialize router
    content_store = UnifiedContentStore()
    ner_router = NERFuzzyRouter(content_store)
    
    # Show current content
    all_content = content_store.get_all_content()
    videos = [item for item in all_content if item.content_type == 'youtube']
    
    print(f"Current content: {len(all_content)} items")
    print(f"Videos available: {len(videos)}")
    for i, video in enumerate(videos, 1):
        print(f"  {i}. {video.title} ({len(video.full_text)} chars)")
    print()
    
    # Test targeted queries for black holes video
    black_hole_queries = [
        "black holes",
        "what are black holes",
        "gravitational effects",
        "time and space",
        "cosmic phenomena", 
        "universe mysteries",
        "black holes explained",
        "gravitational physics",
        "space time",
        "astronomical phenomena"
    ]
    
    print("Testing Black Holes Video Queries:")
    print("-" * 50)
    
    for i, query in enumerate(black_hole_queries, 1):
        try:
            pdf_files, youtube_urls, explanation = ner_router.route_query(query)
            
            video_found = len(youtube_urls) > 0
            status = "INCLUDES VIDEO" if video_found else "PDF ONLY"
            
            print(f"{i:2d}. {status}: '{query}'")
            print(f"    -> {len(pdf_files)} PDFs, {len(youtube_urls)} videos")
            
            # Check if black holes video specifically was found
            black_hole_video_found = any('rJLtT0QXoPo' in url for url in youtube_urls)
            if black_hole_video_found:
                print(f"    -> SUCCESS: Black holes video included!")
            
            print(f"    -> {explanation}")
            print()
            
        except Exception as e:
            print(f"{i:2d}. ERROR: '{query}' - {e}")
            print()
    
    # Test mixed queries that might include both videos
    print("Testing Mixed/General Queries:")
    print("-" * 50)
    
    mixed_queries = [
        "space science",
        "cosmic mysteries", 
        "universe explained",
        "what do the videos discuss",
        "space phenomena",
        "scientific explanations"
    ]
    
    for i, query in enumerate(mixed_queries, 1):
        try:
            pdf_files, youtube_urls, explanation = ner_router.route_query(query)
            
            print(f"{i}. Query: '{query}'")
            print(f"   -> {len(pdf_files)} PDFs, {len(youtube_urls)} videos")
            print(f"   -> {explanation}")
            print()
            
        except Exception as e:
            print(f"{i}. ERROR: '{query}' - {e}")
            print()

if __name__ == "__main__":
    test_black_holes_video()