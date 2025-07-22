#!/usr/bin/env python3
"""
Manually process the YouTube video and store it in content store
"""

import asyncio
import sys

sys.path.append('src')

from storage.unified_content_store import UnifiedContentStore
from agents.youtube_agent import YouTubeAgent
from storage.ner_fuzzy_router import NERFuzzyRouter

async def manually_process_video():
    """Manually process the YouTube video"""
    
    youtube_url = "https://www.youtube.com/watch?v=5iA7wZfxglE"
    print(f"=== Manually Processing Video ===")
    print(f"URL: {youtube_url}")
    
    # Step 1: Get video transcript using YouTube agent
    print("\n1. Getting video transcript...")
    youtube_agent = YouTubeAgent()
    
    try:
        captions_result = await youtube_agent.execute_action('get_captions', {'url': youtube_url})
        
        if not captions_result.success:
            print(f"ERROR: Failed to get captions: {captions_result.error}")
            return False
        
        output = captions_result.output
        video_info = output.get('video_info', {})
        transcript = output.get('transcript', '')
        segments = output.get('segments', [])
        
        print(f"SUCCESS: Got video transcript")
        print(f"Title: {video_info.get('title', 'Unknown')}")
        print(f"Transcript length: {len(transcript)} characters")
        print(f"Preview: {transcript[:200]}...")
        
    except Exception as e:
        print(f"ERROR: Exception getting captions: {e}")
        return False
    
    # Step 2: Store in unified content store
    print("\n2. Storing in content store...")
    
    try:
        content_store = UnifiedContentStore()
        
        # Create content item data
        title = video_info.get('title', 'YouTube Video')
        if not title or title == 'YouTube Video':
            title = f"Video {youtube_url.split('v=')[1][:11]}"
        
        # Store the video content
        success = content_store.add_youtube_content(
            url=youtube_url,
            title=title,
            transcript_segments=segments,
            metadata={
                'video_id': youtube_url.split('v=')[1][:11] if 'v=' in youtube_url else 'unknown',
                'duration': video_info.get('duration', 'Unknown'),
                'source': 'youtube_captions',
                'transcript_length': len(transcript)
            }
        )
        
        if success:
            print(f"SUCCESS: Stored video in content store")
        else:
            print(f"ERROR: Failed to store video")
            return False
            
    except Exception as e:
        print(f"ERROR: Exception storing video: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 3: Verify storage
    print("\n3. Verifying storage...")
    
    try:
        content_store = UnifiedContentStore()  # Reload
        all_content = content_store.get_all_content()
        videos = [item for item in all_content if item.content_type == 'youtube']
        
        print(f"Total content items: {len(all_content)}")
        print(f"Videos: {len(videos)}")
        
        target_video = None
        for video in videos:
            if '5iA7wZfxglE' in video.source_path or '5iA7wZfxglE' in video.id:
                target_video = video
                break
        
        if target_video:
            print(f"SUCCESS: Found our video in content store")
            print(f"Title: {target_video.title}")
            print(f"Content length: {len(target_video.full_text)} chars")
            print(f"Keywords: {len(target_video.keywords)} extracted")
            print(f"Topics: {len(target_video.topic_assignments)} assigned")
        else:
            print(f"WARNING: Video not found in content store")
            return False
            
    except Exception as e:
        print(f"ERROR: Exception verifying storage: {e}")
        return False
    
    # Step 4: Test NER routing
    print("\n4. Testing NER routing with video...")
    
    try:
        ner_router = NERFuzzyRouter(content_store)
        
        test_queries = [
            "space mysteries",
            "unsolved mysteries",
            "universe questions", 
            "video content",
            "what does the video discuss"
        ]
        
        for query in test_queries:
            pdf_files, youtube_urls, explanation = ner_router.route_query(query)
            
            video_found = len(youtube_urls) > 0
            status = "INCLUDES VIDEO" if video_found else "PDF ONLY"
            
            print(f"{status}: '{query}'")
            print(f"  PDFs: {len(pdf_files)}, Videos: {len(youtube_urls)}")
            print(f"  Explanation: {explanation}")
            
            if youtube_urls:
                for url in youtube_urls:
                    if '5iA7wZfxglE' in url:
                        print(f"  SUCCESS: Our video was included!")
            print()
            
    except Exception as e:
        print(f"ERROR: NER routing test failed: {e}")
        return False
    
    print("=== Manual Video Processing Complete ===")
    return True

if __name__ == "__main__":
    success = asyncio.run(manually_process_video())
    if success:
        print("\nSUCCESS: Video processing completed successfully!")
        print("You can now test queries that should include video content.")
    else:
        print("\nFAILED: Video processing failed.")