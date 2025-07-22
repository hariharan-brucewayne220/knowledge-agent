#!/usr/bin/env python3
"""
Simple test for video upload and queries (no unicode characters)
"""

import asyncio
import sys
import json
import httpx
import time

sys.path.append('src')

from storage.unified_content_store import UnifiedContentStore
from storage.ner_fuzzy_router import NERFuzzyRouter

async def test_video_upload(youtube_url):
    """Test video upload via API"""
    
    print("=== Testing Video Upload ===\n")
    
    # Check current content
    content_store = UnifiedContentStore()
    initial_content = content_store.get_all_content()
    initial_videos = [item for item in initial_content if item.content_type == 'youtube']
    
    print(f"Initial content: {len(initial_content)} items")
    print(f"Initial videos: {len(initial_videos)}")
    print()
    
    # Test video upload
    print(f"Testing video upload: {youtube_url}")
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "http://localhost:8080/api/parse-input",
                json={"input": youtube_url}
            )
            
            if response.status_code == 200:
                result = response.json()
                print("SUCCESS: Parse-input API call successful")
                print(f"YouTube URLs detected: {len(result.get('parsed', {}).get('youtube_urls', []))}")
                print(f"Processing started: {result.get('processing_started', False)}")
                
                # Show video analysis
                youtube_analysis = result.get('youtube_analysis', [])
                for analysis in youtube_analysis:
                    print(f"Video: {analysis.get('title', 'Unknown')}")
                    print(f"Captions available: {analysis.get('captions_available', False)}")
                    print(f"Download needed: {analysis.get('download_needed', False)}")
                
                # Wait for processing
                if result.get('processing_started', False):
                    print("\nWaiting for video processing (30 seconds)...")
                    await asyncio.sleep(30)
                
                return True
                
            else:
                print(f"ERROR: Parse-input API failed: {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
    except httpx.ConnectError:
        print("ERROR: Could not connect to app server on localhost:8080")
        return False
    except Exception as e:
        print(f"ERROR: API call failed: {e}")
        return False

async def test_queries_after_upload():
    """Test queries after video upload"""
    
    print("\n=== Testing Queries After Upload ===\n")
    
    # Check content store again
    content_store = UnifiedContentStore()
    all_content = content_store.get_all_content()
    videos = [item for item in all_content if item.content_type == 'youtube']
    
    print(f"Total content: {len(all_content)} items")
    print(f"Videos found: {len(videos)}")
    
    if videos:
        print("Video content:")
        for video in videos:
            print(f"  - {video.title}")
            print(f"    Keywords: {len(video.keywords)} extracted")
            print(f"    Topics: {len(video.topic_assignments)} assigned")
    else:
        print("No videos found in content store")
        return False
    
    # Test NER routing with videos
    print("\n=== Testing NER Router with Video Content ===")
    ner_router = NERFuzzyRouter(content_store)
    
    test_queries = [
        "video content",
        "solar energy technology", 
        "renewable energy",
        "energy storage",
        "according to video"
    ]
    
    for query in test_queries:
        try:
            pdf_files, youtube_urls, explanation = ner_router.route_query(query)
            
            print(f"Query: '{query}'")
            print(f"  PDFs: {len(pdf_files)}, Videos: {len(youtube_urls)}")
            print(f"  Explanation: {explanation}")
            
            if youtube_urls:
                print("  SUCCESS: Query routed to video content")
            
            print()
            
        except Exception as e:
            print(f"ERROR: Query failed: {query} - {e}")
    
    return True

async def test_research_with_video():
    """Test complete research query that should include video"""
    
    print("=== Testing Research Query with Video ===\n")
    
    research_query = "what are renewable energy technologies"
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "http://localhost:8080/api/research",
                json={
                    "query": research_query,
                    "use_openai": False,
                    "pdf_files": [],
                    "youtube_urls": []
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                print("SUCCESS: Research query completed")
                
                sources_used = result.get('sources_used', [])
                video_sources = [s for s in sources_used if 'video' in s.lower() or 'youtube' in s.lower()]
                
                print(f"Total sources used: {len(sources_used)}")
                print(f"Video sources: {len(video_sources)}")
                
                if video_sources:
                    print("SUCCESS: Video content included in research")
                    for source in video_sources:
                        print(f"  - {source}")
                else:
                    print("WARNING: No video content in research results")
                
                # Show answer preview
                answer = result.get('answer', '')
                print(f"\nAnswer preview: {answer[:200]}...")
                return True
                
            else:
                print(f"ERROR: Research query failed: {response.status_code}")
                return False
                
    except Exception as e:
        print(f"ERROR: Research query failed: {e}")
        return False

async def main():
    youtube_url = "https://www.youtube.com/watch?v=5iA7wZfxglE"
    
    print(f"Testing with YouTube URL: {youtube_url}")
    print("="*60)
    
    # Step 1: Test video upload
    upload_success = await test_video_upload(youtube_url)
    if not upload_success:
        print("Video upload failed, stopping test")
        return
    
    # Step 2: Test queries
    query_success = await test_queries_after_upload()
    if not query_success:
        print("Query testing failed")
        return
    
    # Step 3: Test research
    research_success = await test_research_with_video()
    if not research_success:
        print("Research testing failed")
        return
    
    print("="*60)
    print("ALL TESTS COMPLETED")

if __name__ == "__main__":
    asyncio.run(main())