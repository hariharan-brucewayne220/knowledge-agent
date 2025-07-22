#!/usr/bin/env python3
"""
Process video #3 and analyze content
"""

import asyncio
import sys

sys.path.append('src')

from storage.unified_content_store import UnifiedContentStore
from agents.youtube_agent import YouTubeAgent
from storage.ner_fuzzy_router import NERFuzzyRouter

async def process_video_3():
    """Process video #3 with URL: https://www.youtube.com/watch?v=Tf_KKGk6FNY"""
    
    youtube_url = "https://www.youtube.com/watch?v=Tf_KKGk6FNY"
    
    print(f"=== Processing Video #3 ===")
    print(f"URL: {youtube_url}")
    
    # Check current state
    print(f"\n1. Checking current content...")
    content_store = UnifiedContentStore()
    initial_content = content_store.get_all_content()
    initial_videos = [item for item in initial_content if item.content_type == 'youtube']
    
    print(f"Current content: {len(initial_content)} items")
    print(f"Current videos: {len(initial_videos)}")
    for i, video in enumerate(initial_videos, 1):
        print(f"  {i}. {video.title}")
    
    # Step 2: Get video transcript
    print(f"\n2. Getting video transcript...")
    youtube_agent = YouTubeAgent()
    
    try:
        captions_result = await youtube_agent.execute_action('get_captions', {'url': youtube_url})
        
        if not captions_result.success:
            print(f"ERROR: Failed to get captions: {captions_result.error}")
            return None
        
        output = captions_result.output
        video_info = output.get('video_info', {})
        transcript = output.get('transcript', '')
        segments = output.get('segments', [])
        
        print(f"SUCCESS: Got video transcript")
        print(f"Title: {video_info.get('title', 'Unknown')}")
        print(f"Transcript length: {len(transcript)} characters")
        print(f"Segments: {len(segments)}")
        
        # Show content preview
        print(f"\nContent preview:")
        print(f"'{transcript[:400]}...'")
        
    except Exception as e:
        print(f"ERROR: Exception getting captions: {e}")
        return None
    
    # Step 3: Analyze content themes
    print(f"\n3. Analyzing content themes...")
    
    # Look for key topics in transcript
    words = transcript.lower().split()
    word_freq = {}
    for word in words:
        if len(word) > 4 and word.isalpha():
            word_freq[word] = word_freq.get(word, 0) + 1
    
    top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:12]
    print(f"Top keywords: {[word for word, count in top_words]}")
    
    # Check for specific scientific topics
    science_topics = {
        'energy': transcript.lower().count('energy'),
        'solar': transcript.lower().count('solar'),
        'battery': transcript.lower().count('battery'),
        'carbon': transcript.lower().count('carbon'),
        'climate': transcript.lower().count('climate'),
        'technology': transcript.lower().count('technology'),
        'renewable': transcript.lower().count('renewable'),
        'electricity': transcript.lower().count('electricity'),
        'power': transcript.lower().count('power'),
        'physics': transcript.lower().count('physics'),
        'science': transcript.lower().count('science'),
        'research': transcript.lower().count('research')
    }
    
    relevant_topics = {topic: count for topic, count in science_topics.items() if count > 0}
    print(f"Science topics found: {relevant_topics}")
    
    # Step 4: Store in content store
    print(f"\n4. Storing in content store...")
    
    try:
        title = video_info.get('title', f'Video Tf_KKGk6FNY')
        if not title or 'Unknown' in title:
            title = "Video Tf_KKGk6FNY"
        
        success = content_store.add_youtube_content(
            url=youtube_url,
            title=title,
            transcript_segments=segments,
            metadata={
                'video_id': 'Tf_KKGk6FNY',
                'duration': video_info.get('duration', 'Unknown'),
                'source': 'youtube_captions',
                'transcript_length': len(transcript),
                'batch_number': 3,
                'science_topics': relevant_topics
            }
        )
        
        if success:
            print(f"SUCCESS: Stored video in content store")
        else:
            print(f"ERROR: Failed to store video")
            return None
            
    except Exception as e:
        print(f"ERROR: Exception storing video: {e}")
        return None
    
    # Step 5: Verify storage
    print(f"\n5. Verifying storage...")
    
    content_store = UnifiedContentStore()  # Reload
    final_content = content_store.get_all_content()
    final_videos = [item for item in final_content if item.content_type == 'youtube']
    
    print(f"Final content: {len(final_content)} items")
    print(f"Final videos: {len(final_videos)}")
    
    new_video = None
    for video in final_videos:
        if 'Tf_KKGk6FNY' in video.source_path:
            new_video = video
            break
    
    if new_video:
        print(f"SUCCESS: Found new video in content store")
        print(f"Title: {new_video.title}")
        print(f"Content length: {len(new_video.full_text)} chars")
        print(f"Keywords: {len(new_video.keywords)} extracted")
        print(f"Topics: {len(new_video.topic_assignments)} assigned")
        print(f"Sample keywords: {new_video.keywords[:8]}")
    else:
        print(f"WARNING: New video not found in content store")
        return None
    
    # Step 6: Generate targeted queries based on actual content
    print(f"\n6. Generating targeted queries...")
    
    queries = []
    
    # Add topic-based queries
    for topic, count in relevant_topics.items():
        if count > 1:
            queries.append(f"{topic} technology")
            queries.append(f"what is {topic}")
    
    # Add general video queries
    queries.extend([
        "video content",
        "what does the video explain",
        "according to the video",
        "technical explanation"
    ])
    
    # Add keyword-based queries
    for word, count in top_words[:4]:
        if count > 2:
            queries.append(f"{word} information")
    
    print(f"Generated {len(queries)} targeted queries:")
    for i, query in enumerate(queries[:10], 1):  # Show first 10
        print(f"  {i}. {query}")
    
    return {
        'video_id': 'Tf_KKGk6FNY',
        'title': title,
        'url': youtube_url,
        'transcript_length': len(transcript),
        'segments': len(segments),
        'keywords': [word for word, count in top_words[:10]],
        'science_topics': relevant_topics,
        'test_queries': queries,
        'content_preview': transcript[:600]
    }

async def test_video_3_routing(video_info):
    """Test routing for video #3"""
    
    print(f"\n=== Testing Video #3 Routing ===")
    
    content_store = UnifiedContentStore()
    ner_router = NERFuzzyRouter(content_store)
    
    # Test key queries
    priority_queries = [
        "technology explained",
        "energy systems", 
        "scientific research",
        "what does the video discuss",
        "technical information"
    ]
    
    # Add science topic queries if found
    if video_info['science_topics']:
        for topic in list(video_info['science_topics'].keys())[:3]:
            priority_queries.append(f"{topic} technology")
    
    print(f"Testing {len(priority_queries)} priority queries:")
    
    for i, query in enumerate(priority_queries, 1):
        try:
            pdf_files, youtube_urls, explanation = ner_router.route_query(query)
            
            video_found = len(youtube_urls) > 0
            video_3_found = any('Tf_KKGk6FNY' in url for url in youtube_urls)
            
            status = "INCLUDES VIDEO" if video_found else "PDF ONLY"
            if video_3_found:
                status += " (Video #3)"
            
            print(f"  {i}. {status}: '{query}'")
            print(f"     -> {len(pdf_files)} PDFs, {len(youtube_urls)} videos")
            print(f"     -> {explanation}")
            print()
            
        except Exception as e:
            print(f"  {i}. ERROR: '{query}' - {e}")
            print()

async def main():
    print("Processing Video #3...")
    video_info = await process_video_3()
    
    if video_info:
        print(f"\n" + "="*60)
        print(f"VIDEO #3 PROCESSING COMPLETE")
        print(f"Title: {video_info['title']}")
        print(f"Content: {video_info['transcript_length']} chars, {video_info['segments']} segments")
        print(f"Science topics found: {len(video_info['science_topics'])}")
        
        if video_info['science_topics']:
            print(f"Main topics: {list(video_info['science_topics'].keys())[:5]}")
        
        # Test routing
        await test_video_3_routing(video_info)
        
        # Show updated content store
        print(f"=== Updated Content Store ===")
        content_store = UnifiedContentStore()
        all_content = content_store.get_all_content()
        videos = [item for item in all_content if item.content_type == 'youtube']
        
        print(f"Total content: {len(all_content)} items")
        print(f"Videos: {len(videos)}")
        for i, video in enumerate(videos, 1):
            marker = " <- NEW" if 'Tf_KKGk6FNY' in video.source_path else ""
            print(f"  {i}. {video.title}{marker}")
        
        print(f"\nReady for video #4!")
        
    else:
        print(f"\nVideo #3 processing failed")

if __name__ == "__main__":
    asyncio.run(main())