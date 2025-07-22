#!/usr/bin/env python3
"""
Check video processing status and wait for completion
"""

import asyncio
import sys
import time

sys.path.append('src')

from storage.unified_content_store import UnifiedContentStore

async def monitor_video_processing(max_wait=600):  # 10 minutes
    """Monitor video processing progress"""
    
    print("=== Monitoring Video Processing ===\n")
    
    waited = 0
    check_interval = 15  # Check every 15 seconds
    
    print("Initial content check...")
    content_store = UnifiedContentStore()
    initial_content = content_store.get_all_content()
    initial_videos = [item for item in initial_content if item.content_type == 'youtube']
    
    print(f"Initial videos: {len(initial_videos)}")
    for video in initial_videos:
        print(f"  - {video.title} ({video.source_path})")
    
    print(f"\nWaiting for new video processing (checking every {check_interval}s, max {max_wait}s)...")
    
    while waited < max_wait:
        await asyncio.sleep(check_interval)
        waited += check_interval
        
        # Reload content store to get fresh data
        content_store = UnifiedContentStore()
        current_content = content_store.get_all_content()
        current_videos = [item for item in current_content if item.content_type == 'youtube']
        
        print(f"[{waited}s] Checking... {len(current_content)} total items, {len(current_videos)} videos")
        
        # Check for new videos
        new_videos = []
        for video in current_videos:
            if video not in initial_videos:
                new_videos.append(video)
        
        if new_videos:
            print(f"\nSUCCESS: New video(s) processed after {waited} seconds!")
            for video in new_videos:
                print(f"  - {video.title}")
                print(f"    URL: {video.source_path}")
                print(f"    Keywords: {len(video.keywords)} extracted")
                print(f"    Topics: {len(video.topic_assignments)} assigned")
                print(f"    Content length: {len(video.full_text)} chars")
            
            # Test a quick query
            print(f"\n=== Quick Query Test ===")
            from storage.ner_fuzzy_router import NERFuzzyRouter
            ner_router = NERFuzzyRouter(content_store)
            
            test_queries = ["video content", "renewable energy", "solar technology"]
            for query in test_queries:
                try:
                    pdf_files, youtube_urls, explanation = ner_router.route_query(query)
                    video_found = len(youtube_urls) > 0
                    status = "INCLUDES VIDEO" if video_found else "PDF ONLY"
                    print(f"{status}: '{query}' -> {len(pdf_files)} PDFs, {len(youtube_urls)} videos")
                except Exception as e:
                    print(f"Query error: {e}")
            
            return True
        
        # Show what we have so far
        if len(current_content) > len(initial_content):
            print(f"  Content increased from {len(initial_content)} to {len(current_content)} items")
        
        # Check if there are any temporary files or processing indicators
        try:
            import os
            temp_files = []
            for root, dirs, files in os.walk('.'):
                for file in files:
                    if any(x in file.lower() for x in ['temp', 'download', 'audio', '.wav', '.mp3']):
                        temp_files.append(os.path.join(root, file))
            
            if temp_files:
                print(f"  Found {len(temp_files)} temporary files (processing may be ongoing)")
                
        except:
            pass
    
    print(f"\nTIMEOUT: Video processing did not complete within {max_wait} seconds")
    return False

async def main():
    success = await monitor_video_processing()
    if success:
        print("\nVideo processing completed successfully!")
    else:
        print("\nVideo processing did not complete or failed.")
        
        # Show final state
        print("\nFinal content state:")
        content_store = UnifiedContentStore()
        all_content = content_store.get_all_content()
        videos = [item for item in all_content if item.content_type == 'youtube']
        print(f"Total items: {len(all_content)}")
        print(f"Videos: {len(videos)}")
        
        if videos:
            for video in videos:
                print(f"  - {video.title}")

if __name__ == "__main__":
    asyncio.run(main())