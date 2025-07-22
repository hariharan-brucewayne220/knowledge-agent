#!/usr/bin/env python3
"""
Test video upload and query functionality
"""

import asyncio
import sys
import json
import httpx
import time

sys.path.append('src')

from storage.unified_content_store import UnifiedContentStore
from storage.ner_fuzzy_router import NERFuzzyRouter

async def test_video_upload_and_queries(youtube_url):
    """Test video upload via API and subsequent queries"""
    
    print("=== Testing Video Upload and Query Functionality ===\n")
    
    # First, check current content
    content_store = UnifiedContentStore()
    initial_content = content_store.get_all_content()
    initial_videos = [item for item in initial_content if item.content_type == 'youtube']
    
    print(f"Initial content: {len(initial_content)} items")
    print(f"Initial videos: {len(initial_videos)}")
    for video in initial_videos:
        print(f"  - {video.title}")
    print()
    
    # Test the video upload via API
    print(f"Testing video upload: {youtube_url}")
    print("Calling /api/parse-input endpoint...")
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Test parse-input endpoint
            response = await client.post(
                "http://localhost:8080/api/parse-input",
                json={"input": youtube_url}
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Parse-input API call successful")
                print(f"   YouTube URLs detected: {len(result.get('parsed', {}).get('youtube_urls', []))}")
                print(f"   Processing started: {result.get('processing_started', False)}")
                
                # Show YouTube analysis
                youtube_analysis = result.get('youtube_analysis', [])
                for analysis in youtube_analysis:
                    print(f"   Video: {analysis.get('title', 'Unknown')}")
                    print(f"   Captions available: {analysis.get('captions_available', False)}")
                    print(f"   Download needed: {analysis.get('download_needed', False)}")
                
                # Wait for processing to complete
                if result.get('processing_started', False):
                    print("\\n⏳ Waiting for video processing to complete...")
                    await wait_for_video_processing(youtube_url, max_wait=300)  # 5 minutes max
                
            else:
                print(f"❌ Parse-input API failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return
                
    except httpx.ConnectError:
        print("❌ Could not connect to app server. Is it running on localhost:8080?")
        print("   Start the app with: python app.py (should be running on port 8080)")
        return
    except Exception as e:
        print(f"❌ Error calling API: {e}")
        return
    
    # Check if video was added to content store
    print("\\n=== Checking Content Store After Upload ===")
    content_store = UnifiedContentStore()  # Reload to get fresh data
    final_content = content_store.get_all_content()
    final_videos = [item for item in final_content if item.content_type == 'youtube']
    
    print(f"Final content: {len(final_content)} items")
    print(f"Final videos: {len(final_videos)}")
    
    new_videos = [v for v in final_videos if v not in initial_videos]
    if new_videos:
        print(f"\\n✅ New video(s) added: {len(new_videos)}")
        for video in new_videos:
            print(f"  - {video.title}")
            print(f"    URL: {video.source_path}")
            print(f"    Keywords: {len(video.keywords)} extracted")
            print(f"    Topics: {len(video.topic_assignments)} assigned")
    else:
        print("\\n⚠️ No new videos found in content store")
        return
    
    # Test queries with the new video content
    print("\\n=== Testing Queries with Video Content ===")
    ner_router = NERFuzzyRouter(content_store)
    
    # General video-related queries
    test_queries = [
        "video about energy",
        "explain solar technology", 
        "battery storage systems",
        "renewable energy solutions",
        "carbon capture technology",
        "what does the video discuss",
        "according to the video"
    ]
    
    for query in test_queries:
        try:
            pdf_files, youtube_urls, explanation = ner_router.route_query(query)
            
            video_found = len(youtube_urls) > 0
            status = "✅ INCLUDES VIDEO" if video_found else "📄 PDF ONLY"
            
            print(f"{status}: '{query}'")
            print(f"   PDFs: {len(pdf_files)}, Videos: {len(youtube_urls)}")
            if youtube_urls:
                for url in youtube_urls:
                    # Find video title
                    video_title = "Unknown"
                    for video in final_videos:
                        if video.source_path == url:
                            video_title = video.title
                            break
                    print(f"   Video: {video_title}")
            print(f"   Explanation: {explanation}")
            print()
            
        except Exception as e:
            print(f"❌ Query failed: {query} - {e}")
            print()
    
    # Test a complete research query
    print("=== Testing Complete Research Query ===")
    research_query = "what are the latest developments in renewable energy technology"
    
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
                print(f"✅ Research query successful")
                print(f"Query: {research_query}")
                
                # Check if video content was used
                sources_used = result.get('sources_used', [])
                video_sources = [s for s in sources_used if 'video' in s.lower() or 'youtube' in s.lower()]
                
                print(f"Total sources: {len(sources_used)}")
                print(f"Video sources: {len(video_sources)}")
                
                if video_sources:
                    print("✅ Video content was included in research")
                    for source in video_sources:
                        print(f"   - {source}")
                else:
                    print("⚠️ No video content found in research results")
                
                # Show answer preview
                answer = result.get('answer', '')
                print(f"\\nAnswer preview: {answer[:300]}...")
                
            else:
                print(f"❌ Research query failed: {response.status_code}")
                print(f"Response: {response.text}")
                
    except Exception as e:
        print(f"❌ Research query error: {e}")

async def wait_for_video_processing(youtube_url, max_wait=300):
    """Wait for video processing to complete"""
    
    from urllib.parse import urlparse, parse_qs
    
    # Extract video ID
    def extract_video_id(url):
        if 'youtube.com/watch' in url:
            parsed = urlparse(url)
            return parse_qs(parsed.query).get('v', [None])[0]
        elif 'youtu.be/' in url:
            return url.split('youtu.be/')[-1].split('?')[0]
        return None
    
    video_id = extract_video_id(youtube_url)
    if not video_id:
        print("Could not extract video ID")
        return
    
    waited = 0
    check_interval = 10  # Check every 10 seconds
    
    while waited < max_wait:
        # Check if video is now in content store
        content_store = UnifiedContentStore()
        videos = [item for item in content_store.get_all_content() if item.content_type == 'youtube']
        
        # Look for our video ID in the stored videos
        video_found = any(video_id in video.source_path for video in videos)
        
        if video_found:
            print(f"✅ Video processing completed after {waited} seconds")
            return
        
        print(f"   Still processing... ({waited}/{max_wait} seconds)")
        await asyncio.sleep(check_interval)
        waited += check_interval
    
    print(f"⚠️ Video processing did not complete within {max_wait} seconds")

if __name__ == "__main__":
    # Get YouTube URL from command line or use default
    import sys
    
    if len(sys.argv) > 1:
        youtube_url = sys.argv[1]
    else:
        print("Usage: python test_video_upload_and_queries.py <youtube_url>")
        print("\\nExample YouTube URLs to test:")
        print("- https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        print("- https://youtu.be/dQw4w9WgXcQ")
        print("\\nProvide a YouTube URL as an argument to test video upload functionality.")
        sys.exit(1)
    
    print(f"Testing with YouTube URL: {youtube_url}")
    asyncio.run(test_video_upload_and_queries(youtube_url))